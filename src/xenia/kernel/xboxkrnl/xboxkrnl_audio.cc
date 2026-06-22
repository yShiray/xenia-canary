/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#include "xenia/apu/audio_system.h"
#include "xenia/base/clock.h"
#include "xenia/base/logging.h"
#include "xenia/cpu/processor.h"
#include "xenia/emulator.h"
#include "xenia/kernel/kernel_flags.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_private.h"
#include "xenia/kernel/xthread.h"
#include "xenia/xbox.h"

DECLARE_uint32(audio_flag);

// The in-guest NUI voice MEC pump (see comment block below) runs heavy guest DSP
// (beamforming / echo-cancellation) on a host thread at this cadence. That work
// competes with other guest threads for execution bandwidth and is a suspect for
// the JD cinematic video freeze-and-recover (the guest libvpx decode thread gets
// starved while the rest of the scene keeps rendering at 60 fps). The voice
// command is not used mid-cinematic, so this cvar lets us throttle or fully
// disable the pump to isolate/relieve the contention. <= 0 disables the pump
// entirely; default 8 keeps the previous behaviour.
DEFINE_int32(
    xam_voice_mec_pump_interval_ms, 8,
    "Interval (ms) for the in-guest NUI voice MEC render-driver pump that runs "
    "mic beamforming/echo-cancellation guest code on a host thread. Lower = the "
    "in-guest mic ring is fed more often (original behaviour was 8). Raise it to "
    "reduce CPU/scheduler contention with other guest threads (e.g. the video "
    "decode thread during cinematics). Set to 0 to disable the pump entirely "
    "(voice command stops working, but it is unused during cinematics).",
    "Kernel");

namespace xe {
namespace kernel {
namespace xboxkrnl {

// ---------------------------------------------------------------------------
// MEC (Microphone Echo Cancellation) render-driver client.
//
// The in-guest NuiAudio mic-array pipeline (NUIAUD static lib) does NOT run on
// its own polling thread. It registers a callback here via
// XAudioRegisterRenderDriverMECClient and relies on the audio render driver to
// invoke that callback every audio frame. Inside the callback the in-guest code
// (MEC::MECPumpProcess) pulls the raw Kinect mic via XamVoiceGetMicArrayAudioEx,
// runs beamforming/echo-cancellation, and hands processed audio to the NuiSpeech
// engine (which fires the "Just Dance" voice command).
//
// Historically this export was a no-op stub, so the callback was never invoked,
// the mic never produced data, and the speech engine waited forever. We drive
// the callback from a dedicated guest-capable host thread at the audio-frame
// cadence, mirroring the hardware render-driver tick.
// ---------------------------------------------------------------------------
namespace {
std::mutex g_mec_mutex;
uint32_t g_mec_callback = 0;
uint32_t g_mec_callback_arg = 0;
std::atomic<bool> g_mec_running{false};
object_ref<XHostThread> g_mec_thread;

constexpr uint32_t kMecHandle = 0x4D454300;  // "MEC\0"
// Mic frame is 512 samples; ~10 ms at 48 kHz raw capture. The pump cadence is
// configurable via xam_voice_mec_pump_interval_ms (default 8) so it can be
// throttled to relieve contention with other guest threads. This fallback is
// only used if the cvar is somehow non-positive while the pump is running.
constexpr int64_t kMecIntervalMsDefault = 8;

void StopMecDriver() {
  object_ref<XHostThread> thread;
  {
    std::lock_guard<std::mutex> lock(g_mec_mutex);
    g_mec_running.store(false, std::memory_order_release);
    thread = g_mec_thread;
    g_mec_thread.reset();
    g_mec_callback = 0;
    g_mec_callback_arg = 0;
  }
  if (thread) {
    thread->Wait(0, 0, 0, nullptr);
  }
}

void MecDriverLoop() {
  XELOGI("[voice] MEC driver thread started (callback={:08X} arg={:08X})",
         g_mec_callback, g_mec_callback_arg);
  uint64_t ticks = 0;
  auto* self = XThread::GetCurrentThread();
  while (g_mec_running.load(std::memory_order_acquire)) {
    uint32_t callback = g_mec_callback;
    uint32_t callback_arg = g_mec_callback_arg;
    if (callback && self) {
      uint64_t args[] = {callback_arg};
      kernel_state()->processor()->Execute(self->thread_state(), callback, args,
                                           xe::countof(args));
      if (cvars::xam_nui_log_voice && (ticks % 250) == 0) {
        XELOGI("[voice] MEC callback driven (tick {})", ticks);
      }
      ++ticks;
    }
    const int32_t interval_ms = cvars::xam_voice_mec_pump_interval_ms;
    std::this_thread::sleep_for(std::chrono::milliseconds(
        interval_ms > 0 ? interval_ms : kMecIntervalMsDefault));
  }
  XELOGI("[voice] MEC driver thread exiting after {} ticks", ticks);
}
}  // namespace

dword_result_t XAudioGetSpeakerConfig_entry(lpdword_t config_ptr) {
  kernel_state()->xconfig()->ReadSetting(
      XCONFIG_USER_CATEGORY,
      XCONFIG_USER_CATEGORY_ENTRIES::XCONFIG_USER_AUDIO_FLAGS, config_ptr);
  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(XAudioGetSpeakerConfig, kAudio, kImplemented);

dword_result_t XAudioGetVoiceCategoryVolumeChangeMask_entry(
    lpunknown_t driver_ptr, lpdword_t out_ptr) {
  assert_true((driver_ptr.guest_address() & 0xFFFF0000) == 0x41550000);

  xe::threading::NanoSleep(1000);

  // Checking these bits to see if any voice volume changed.
  // I think.
  *out_ptr = 0;
  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT2(XAudioGetVoiceCategoryVolumeChangeMask, kAudio, kStub,
                         kHighFrequency);

dword_result_t XAudioGetVoiceCategoryVolume_entry(dword_t unk,
                                                  lpfloat_t out_ptr) {
  // Expects a floating point single. Volume %?
  *out_ptr = 1.0f;

  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT2(XAudioGetVoiceCategoryVolume, kAudio, kStub,
                         kHighFrequency);

dword_result_t XAudioEnableDucker_entry(dword_t unk) { return X_ERROR_SUCCESS; }
DECLARE_XBOXKRNL_EXPORT1(XAudioEnableDucker, kAudio, kStub);

dword_result_t XAudioRegisterRenderDriverClient_entry(lpdword_t callback_ptr,
                                                      lpdword_t driver_ptr) {
  if (!callback_ptr) {
    return X_E_INVALIDARG;
  }

  uint32_t callback = callback_ptr[0];

  if (!callback) {
    return X_E_INVALIDARG;
  }
  uint32_t callback_arg = callback_ptr[1];

  auto audio_system = kernel_state()->emulator()->audio_system();

  size_t index;
  auto result = audio_system->RegisterClient(callback, callback_arg, &index);
  if (XFAILED(result)) {
    return result;
  }

  assert_true(!(index & ~0x0000FFFF));
  *driver_ptr = 0x41550000 | (static_cast<uint32_t>(index) & 0x0000FFFF);
  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(XAudioRegisterRenderDriverClient, kAudio,
                         kImplemented);

dword_result_t XAudioUnregisterRenderDriverClient_entry(
    lpunknown_t driver_ptr) {
  assert_true((driver_ptr.guest_address() & 0xFFFF0000) == 0x41550000);

  auto audio_system = kernel_state()->emulator()->audio_system();
  audio_system->UnregisterClient(driver_ptr.guest_address() & 0x0000FFFF);
  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(XAudioUnregisterRenderDriverClient, kAudio,
                         kImplemented);

dword_result_t XAudioSubmitRenderDriverFrame_entry(lpunknown_t driver_ptr,
                                                   lpunknown_t samples_ptr) {
  assert_true((driver_ptr.guest_address() & 0xFFFF0000) == 0x41550000);

  auto audio_system = kernel_state()->emulator()->audio_system();
  auto samples =
      kernel_state()->memory()->TranslateVirtual<float*>(samples_ptr);
  audio_system->SubmitFrame(driver_ptr.guest_address() & 0x0000FFFF, samples);

  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT2(XAudioSubmitRenderDriverFrame, kAudio, kImplemented,
                         kHighFrequency);

dword_result_t XAudioRegisterRenderDriverMECClient_entry(
    lpdword_t callback_ptr, lpdword_t driver_ptr) {
  // callback_ptr matches the regular render-driver client descriptor:
  // [0] = guest callback function, [1] = callback argument.
  uint32_t callback = 0;
  uint32_t callback_arg = 0;
  if (callback_ptr) {
    callback = callback_ptr[0];
    callback_arg = callback_ptr[1];
  }

  XELOGI(
      "[voice] XAudioRegisterRenderDriverMECClient: desc={:08X} callback={:08X} "
      "arg={:08X}",
      callback_ptr.guest_address(), callback, callback_arg);

  if (driver_ptr) {
    *driver_ptr = kMecHandle;
  }

  // Tear down any previous driver before starting a fresh one.
  StopMecDriver();

  // xam_voice_mec_pump_interval_ms == 0 disables the pump entirely (diagnostic /
  // contention relief for the cinematic video freeze). We still register the
  // client and hand back a handle so the guest's voice setup succeeds; we just
  // never spin the host pump thread.
  if (callback && cvars::xam_voice_mec_pump_interval_ms <= 0) {
    XELOGI(
        "[voice] MEC pump DISABLED via xam_voice_mec_pump_interval_ms=0; "
        "voice command will not receive mic data this run");
  }

  if (callback && cvars::xam_voice_mec_pump_interval_ms > 0) {
    std::lock_guard<std::mutex> lock(g_mec_mutex);
    g_mec_callback = callback;
    g_mec_callback_arg = callback_arg;
    g_mec_running.store(true, std::memory_order_release);
    g_mec_thread = object_ref<XHostThread>(new XHostThread(
        kernel_state(), 128 * 1024, 0,
        []() {
          MecDriverLoop();
          return 0;
        },
        kernel_state()->GetSystemProcess()));
    g_mec_thread->set_name("NUI MEC Pump");
    g_mec_thread->Create();
  }

  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(XAudioRegisterRenderDriverMECClient, kAudio,
                         kImplemented);

dword_result_t XAudioUnregisterRenderDriverMECClient_entry(dword_t driver) {
  XELOGI("[voice] XAudioUnregisterRenderDriverMECClient: driver={:08X}",
         driver.value());
  StopMecDriver();
  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(XAudioUnregisterRenderDriverMECClient, kAudio,
                         kImplemented);

dword_result_t XAudioCaptureRenderDriverFrame_entry(
    unknown_t driver, unknown_t samples, unknown_t r5, unknown_t r6,
    unknown_t r7, unknown_t r8) {
  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT2(XAudioCaptureRenderDriverFrame, kAudio, kStub,
                         kHighFrequency);

dword_result_t XAudioGetRenderDriverTic_entry() {
  return static_cast<uint32_t>(Clock::QueryGuestTickCount());
}
DECLARE_XBOXKRNL_EXPORT2(XAudioGetRenderDriverTic, kAudio, kStub,
                         kHighFrequency);

dword_result_t XAudioGetUnderrunCount_entry(unknown_t driver,
                                            lpdword_t out_count) {
  if (out_count) {
    *out_count = 0;
  }
  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(XAudioGetUnderrunCount, kAudio, kStub);

dword_result_t DrvSetMicArrayStartCallback_entry(unknown_t callback,
                                                 unknown_t context) {
  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(DrvSetMicArrayStartCallback, kAudio, kStub);

dword_result_t DrvSetAudioLatencyCallback_entry(unknown_t callback,
                                                unknown_t context) {
  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(DrvSetAudioLatencyCallback, kAudio, kStub);

dword_result_t XAudioSetProcessFrameCallback_entry(unknown_t callback,
                                                   unknown_t context) {
  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(XAudioSetProcessFrameCallback, kAudio, kStub);

}  // namespace xboxkrnl
}  // namespace kernel
}  // namespace xe

DECLARE_XBOXKRNL_EMPTY_REGISTER_EXPORTS(Audio);
