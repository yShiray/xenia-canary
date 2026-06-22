/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <atomic>
#include <cstdio>

#include "xenia/base/logging.h"
#include "xenia/base/memory.h"
#include "xenia/kernel/kernel_flags.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xam/xam_private.h"
#include "xenia/xbox.h"

namespace xe {
namespace kernel {
namespace xam {

bool IsNuiReady();

// Diagnostic: dump a guest pointer's first bytes so the in-guest NuiAudio
// mic-array ABI (XMICARRAY_AUDIO struct, sample buffers) can be captured from a
// running title without guessing. Gated by --xam_nui_log_voice.
static void LogGuestBytes(const char* label, uint32_t guest_addr,
                          uint32_t byte_count) {
  if (!guest_addr) {
    XELOGI("  {} = 0 (null)", label);
    return;
  }
  // Only peek addresses that look like mapped guest virtual memory.
  if (guest_addr < 0x10000u || guest_addr >= 0xC0000000u) {
    XELOGI("  {} = {:08X} (not a peekable pointer)", label, guest_addr);
    return;
  }
  auto* p = kernel_memory()->TranslateVirtual<const uint8_t*>(guest_addr);
  if (!p) {
    XELOGI("  {} = {:08X} (untranslatable)", label, guest_addr);
    return;
  }
  char hex[3 * 64 + 1];
  uint32_t n = byte_count > 64 ? 64 : byte_count;
  for (uint32_t i = 0; i < n; ++i) {
    std::snprintf(hex + i * 3, 4, "%02X ", p[i]);
  }
  XELOGI("  {} = {:08X} -> {}", label, guest_addr, hex);
}

static void LogVoiceCall(const char* name, uint32_t r3, uint32_t r4,
                         uint32_t r5, uint32_t r6, uint32_t r7, uint32_t r8) {
  if (!cvars::xam_nui_log_voice) {
    return;
  }
  XELOGI("[voice] {}: r3={:08X} r4={:08X} r5={:08X} r6={:08X} r7={:08X} r8={:08X}",
         name, r3, r4, r5, r6, r7, r8);
  LogGuestBytes("r3*", r3, 64);
  LogGuestBytes("r4*", r4, 64);
  LogGuestBytes("r5*", r5, 64);
}

dword_result_t XamVoiceIsActiveProcess_entry() {
  return IsNuiReady() ? 1 : 0;
}
DECLARE_XAM_EXPORT1(XamVoiceIsActiveProcess, kNone, kStub);

dword_result_t XamVoiceCreate_entry(dword_t user_index,
                                    dword_t max_attached_packets,  // 0xF
                                    lpdword_t out_voice_ptr) {
  // Null out the ptr.
  out_voice_ptr.Zero();
  return X_ERROR_ACCESS_DENIED;
}
DECLARE_XAM_EXPORT1(XamVoiceCreate, kNone, kStub);

dword_result_t XamVoiceClose_entry(lpunknown_t voice_ptr) { return 0; }
DECLARE_XAM_EXPORT1(XamVoiceClose, kNone, kStub);

dword_result_t XamVoiceHeadsetPresent_entry(lpunknown_t voice_ptr) { return 0; }
DECLARE_XAM_EXPORT1(XamVoiceHeadsetPresent, kNone, kStub);

dword_result_t XamVoiceSubmitPacket_entry(lpdword_t unk1, dword_t unk2,
                                          lpdword_t unk3) {
  // also may return 0xD000009D
  return 0x800700AA;
}
DECLARE_XAM_EXPORT1(XamVoiceSubmitPacket, kNone, kStub);

dword_result_t XamVoiceGetMicArrayStatus_entry() {
  if (cvars::xam_nui_log_voice) {
    static std::atomic<uint32_t> once{0};
    if (once.fetch_add(1) == 0) {
      XELOGI("[voice] XamVoiceGetMicArrayStatus called (NuiReady={})",
             IsNuiReady() ? 1 : 0);
    }
  }
  return IsNuiReady() ? 1 : 0;
}
DECLARE_XAM_EXPORT1(XamVoiceGetMicArrayStatus, kNone, kStub);

dword_result_t XamVoiceGetBatteryStatus_entry(dword_t user_index,
                                              lpdword_t out_status) {
  if (out_status) {
    *out_status = IsNuiReady() ? 1 : 0;
  }
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamVoiceGetBatteryStatus, kNone, kStub);

dword_result_t XamVoiceSetAudioCaptureRoutine_entry(dword_t r3, dword_t r4,
                                                    dword_t r5) {
  LogVoiceCall("XamVoiceSetAudioCaptureRoutine", r3, r4, r5, 0, 0, 0);
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamVoiceSetAudioCaptureRoutine, kNone, kStub);

dword_result_t XamVoiceGetDirectionalData_entry(unknown_t r3, lpvoid_t out_ptr,
                                                dword_t out_size) {
  if (out_ptr && out_size) {
    out_ptr.Zero(out_size);
  }
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamVoiceGetDirectionalData, kNone, kStub);

dword_result_t XamVoiceGetMicArrayAudio_entry(dword_t r3, dword_t r4, dword_t r5,
                                              dword_t r6, dword_t r7,
                                              dword_t r8) {
  static std::atomic<uint32_t> call_count{0};
  if (call_count.fetch_add(1) < 16) {
    LogVoiceCall("XamVoiceGetMicArrayAudio", r3, r4, r5, r6, r7, r8);
  }
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT2(XamVoiceGetMicArrayAudio, kNone, kStub, kHighFrequency);

dword_result_t XamVoiceRecordUserPrivileges_entry(unknown_t r3, unknown_t r4,
                                                  unknown_t r5) {
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamVoiceRecordUserPrivileges, kNone, kStub);

dword_result_t XamVoiceSetMicArrayIdleUsers_entry(unknown_t r3, unknown_t r4) {
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamVoiceSetMicArrayIdleUsers, kNone, kStub);

dword_result_t XamVoiceMuteMicArray_entry(unknown_t r3) {
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamVoiceMuteMicArray, kNone, kStub);

dword_result_t XamVoiceGetMicArrayUnderrunStatus_entry(dword_t r3, dword_t r4,
                                                       dword_t r5, dword_t r6,
                                                       dword_t r7, dword_t r8) {
  // The in-guest MEC RenderDriverCallback calls this every tick to learn how
  // many new mic samples are available. r3 and r4 are out-pointers (consecutive
  // stack slots) into which the function writes the producer status; the
  // callback accumulates that count and pulls a frame via
  // XamVoiceGetMicArrayAudioEx once it crosses the pipeline threshold (>=256
  // samples for speech). Returning 0 with the out-params untouched (the old
  // stub) made the callback see "no new audio" forever.
  //
  // Experiment: report a full mic frame (512 samples) available each tick by
  // writing the count into the out-pointers. Validated via the log: if this is
  // the right gate, XamVoiceGetMicArrayAudioEx starts firing from the MEC pump.
  static std::atomic<uint32_t> call_count{0};
  if (call_count.fetch_add(1) < 16) {
    LogVoiceCall("XamVoiceGetMicArrayUnderrunStatus", r3, r4, r5, r6, r7, r8);
  }
  constexpr uint32_t kSamplesAvailable = 512;
  if (r3) {
    auto* p = kernel_memory()->TranslateVirtual<uint8_t*>(r3);
    if (p) {
      xe::store_and_swap<uint32_t>(p, kSamplesAvailable);
    }
  }
  if (r4) {
    auto* p = kernel_memory()->TranslateVirtual<uint8_t*>(r4);
    if (p) {
      xe::store_and_swap<uint32_t>(p, kSamplesAvailable);
    }
  }
  return 0;
}
DECLARE_XAM_EXPORT2(XamVoiceGetMicArrayUnderrunStatus, kNone, kStub,
                    kHighFrequency);

dword_result_t XamVoiceGetMicArrayAudioEx_entry(dword_t r3, dword_t r4,
                                                dword_t r5, dword_t r6,
                                                dword_t r7, dword_t r8) {
  static std::atomic<uint32_t> call_count{0};
  uint32_t n = call_count.fetch_add(1);
  if (n < 16) {
    LogVoiceCall("XamVoiceGetMicArrayAudioEx", r3, r4, r5, r6, r7, r8);
  } else if (cvars::xam_nui_log_voice && (n % 500) == 0) {
    // Proof the in-guest mic pump is being driven continuously (not just the
    // 2 init-time probes).
    XELOGI("[voice] XamVoiceGetMicArrayAudioEx call #{}", n);
  }
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT2(XamVoiceGetMicArrayAudioEx, kNone, kStub, kHighFrequency);

dword_result_t XamVoiceDisableMicArray_entry(unknown_t r3) {
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamVoiceDisableMicArray, kNone, kStub);

dword_result_t XamVoiceSetMicArrayBeamAngle_entry(float_t angle) {
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamVoiceSetMicArrayBeamAngle, kNone, kStub);

dword_result_t XamVoiceGetMicArrayFilenameDesc_entry(dword_t r3, dword_t r4,
                                                     dword_t r5) {
  LogVoiceCall("XamVoiceGetMicArrayFilenameDesc", r3, r4, r5, 0, 0, 0);
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamVoiceGetMicArrayFilenameDesc, kNone, kStub);

}  // namespace xam
}  // namespace kernel
}  // namespace xe

DECLARE_XAM_EMPTY_REGISTER_EXPORTS(Voice);
