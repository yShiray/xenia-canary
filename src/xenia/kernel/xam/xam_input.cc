/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <atomic>

#include "xenia/base/logging.h"
#include "xenia/emulator.h"
#include "xenia/hid/input.h"
#include "xenia/hid/input_system.h"
#include "xenia/kernel/kernel_flags.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xam/xam_private.h"
#include "xenia/xbox.h"

namespace xe {
namespace kernel {
namespace xam {

using xe::hid::X_INPUT_CAPABILITIES;
using xe::hid::X_INPUT_FLAG;
using xe::hid::X_INPUT_KEYSTROKE;
using xe::hid::X_INPUT_STATE;
using xe::hid::X_INPUT_VIBRATION;

bool IsNuiReady();
uint32_t GetNuiDeviceStatus();
uint32_t GetNuiTrackedSkeletonCount();
uint32_t GetNuiBestTrackingId();
uint32_t GetNuiSkeletonFrameNumber();
uint32_t GetEngagedNuiTrackingId();
void SetEngagedNuiTrackingId(uint32_t tracking_id);

bool IsNonGamepadInputQuery(uint32_t flags) {
  return (flags & X_INPUT_FLAG::X_INPUT_FLAG_ANYDEVICE) != 0 &&
         (flags & X_INPUT_FLAG::X_INPUT_FLAG_GAMEPAD) == 0;
}

bool IsKeyboardInputQuery(uint32_t flags) {
  return (flags & X_INPUT_FLAG::X_INPUT_FLAG_KEYBOARD) != 0;
}

bool IsNuiInputQuery(uint32_t flags) {
  if ((flags & X_INPUT_FLAG::X_INPUT_FLAG_UNKNOWN2) != 0) {
    return true;
  }
  if (!IsNonGamepadInputQuery(flags)) {
    return false;
  }
  constexpr uint32_t kNonNuiDeviceFlags =
      X_INPUT_FLAG::X_INPUT_FLAG_KEYBOARD | X_INPUT_FLAG::X_INPUT_FLAG_UNKNOWN;
  return (flags & kNonNuiDeviceFlags) == 0;
}

uint32_t ResolveNuiUserIndex(uint32_t user_index, uint32_t flags) {
  if ((user_index & XUserIndexAny) == XUserIndexAny ||
      (flags & X_INPUT_FLAG::X_INPUT_FLAG_ANY_USER)) {
    return 0;
  }
  return user_index;
}

uint32_t ResolveNuiPresenceTrackingId(uint32_t tracked_count,
                                      uint32_t tracking_id) {
  return tracked_count != 0 && tracking_id != 0 ? tracking_id : 0;
}

void LogInputCall(const char* api, uint32_t user_index, uint32_t flags,
                  uint32_t result) {
  static std::atomic<uint32_t> log_count{0};
  const uint32_t slot = log_count.fetch_add(1, std::memory_order_relaxed);
  if (slot < 128) {
    XELOGI("{}: user={:08X}, flags={:08X}, result={:08X}", api, user_index,
           flags, result);
  }
}

void LogNuiInputState(uint32_t user_index, uint32_t flags,
                      uint32_t tracked_count, uint32_t tracking_id,
                      uint32_t packet_number, uint16_t buttons,
                      uint8_t left_trigger, uint8_t right_trigger,
                      uint32_t result) {
  static std::atomic<uint32_t> log_count{0};
  static std::atomic<uint32_t> last_flags{0xFFFFFFFFu};
  static std::atomic<uint32_t> last_tracked_count{0xFFFFFFFFu};
  static std::atomic<uint32_t> last_tracking_id{0xFFFFFFFFu};
  static std::atomic<uint32_t> last_buttons{0xFFFFFFFFu};
  static std::atomic<uint32_t> last_left_trigger{0xFFFFFFFFu};
  static std::atomic<uint32_t> last_right_trigger{0xFFFFFFFFu};
  static std::atomic<uint32_t> last_result{0xFFFFFFFFu};
  const uint32_t slot = log_count.fetch_add(1, std::memory_order_relaxed);
  const bool changed =
      last_flags.exchange(flags, std::memory_order_relaxed) != flags ||
      last_tracked_count.exchange(tracked_count, std::memory_order_relaxed) !=
          tracked_count ||
      last_tracking_id.exchange(tracking_id, std::memory_order_relaxed) !=
          tracking_id ||
      last_buttons.exchange(buttons, std::memory_order_relaxed) != buttons ||
      last_left_trigger.exchange(left_trigger, std::memory_order_relaxed) !=
          left_trigger ||
      last_right_trigger.exchange(right_trigger, std::memory_order_relaxed) !=
          right_trigger ||
      last_result.exchange(result, std::memory_order_relaxed) != result;
  if (changed || slot < 32) {
    XELOGI("XamInputGetState/NUI state: user={:08X}, flags={:08X}, "
           "tracked_count={}, tracking_id={}, packet={}, buttons={:04X}, "
           "lt={:02X}, rt={:02X}, result={:08X}",
           user_index, flags, tracked_count, tracking_id, packet_number,
           buttons, left_trigger, right_trigger, result);
  }
}

void LogNuiKeystrokePresence(uint32_t flags, uint32_t tracked_count,
                             uint32_t tracking_id, uint32_t stroke_flags,
                             uint32_t result) {
  static std::atomic<uint32_t> log_count{0};
  static std::atomic<uint32_t> last_flags{0xFFFFFFFFu};
  static std::atomic<uint32_t> last_tracked_count{0xFFFFFFFFu};
  static std::atomic<uint32_t> last_tracking_id{0xFFFFFFFFu};
  static std::atomic<uint32_t> last_stroke_flags{0xFFFFFFFFu};
  static std::atomic<uint32_t> last_result{0xFFFFFFFFu};
  const uint32_t slot = log_count.fetch_add(1, std::memory_order_relaxed);
  const bool changed =
      last_flags.exchange(flags, std::memory_order_relaxed) != flags ||
      last_tracked_count.exchange(tracked_count, std::memory_order_relaxed) !=
          tracked_count ||
      last_tracking_id.exchange(tracking_id, std::memory_order_relaxed) !=
          tracking_id ||
      last_stroke_flags.exchange(stroke_flags, std::memory_order_relaxed) !=
          stroke_flags ||
      last_result.exchange(result, std::memory_order_relaxed) != result;
  if (changed || slot < 32) {
    XELOGI("XamInputGetKeystroke/NUI presence: flags={:08X}, "
           "tracked_count={}, tracking_id={}, stroke_flags={:04X}, "
           "result={:08X}",
           flags, tracked_count, tracking_id, stroke_flags, result);
  }
}

void LogInputRawCall(const char* api, uint32_t r3, uint32_t r4, uint32_t r5,
                     uint32_t r6, uint32_t r7, uint32_t r8,
                     uint32_t result) {
  static std::atomic<uint32_t> log_count{0};
  const uint32_t slot = log_count.fetch_add(1, std::memory_order_relaxed);
  if (slot < 64) {
    XELOGI("{}: r3={:08X}, r4={:08X}, r5={:08X}, r6={:08X}, r7={:08X}, "
           "r8={:08X}, result={:08X}",
           api, r3, r4, r5, r6, r7, r8, result);
  }
}

bool TryHandleNuiInputCapabilities(uint32_t flags,
                                   pointer_t<X_INPUT_CAPABILITIES> caps,
                                   uint32_t* result) {
  // NUI only answers non-gamepad NUI queries. A plain gamepad query (e.g.
  // flags == 0) must fall through to the real gamepad path, otherwise the
  // Kinect masquerades as a connected pad on every user slot.
  if (!IsNonGamepadInputQuery(flags) || !IsNuiInputQuery(flags)) {
    static std::atomic<uint32_t> rejected_log_count{0};
    const uint32_t call =
        rejected_log_count.fetch_add(1, std::memory_order_relaxed) + 1;
    if (call <= 16) {
      XELOGI("XamInputGetCapabilitiesEx/NUI reject flags={:08X}", flags);
    }
    return false;
  }

  if (!GetNuiDeviceStatus()) {
    *result = X_ERROR_DEVICE_NOT_CONNECTED;
    return true;
  }

  caps.Zero();
  caps->type = 1;
  caps->sub_type = 1;
  caps->flags = xe::hid::X_INPUT_CAPS_VOICE_SUPPORTED |
                xe::hid::X_INPUT_CAPS_PMD_SUPPORTED |
                xe::hid::X_INPUT_CAPS_NO_NAVIGATION;
  *result = X_ERROR_SUCCESS;
  return true;
}

bool TryHandleNuiInputState(uint32_t user_index, uint32_t flags,
                            pointer_t<X_INPUT_STATE> input_state,
                            uint32_t* result) {
  // Only handle genuine NUI queries. Plain gamepad polls (flags == 0) fall
  // through so absent user slots report DEVICE_NOT_CONNECTED as on hardware.
  if (!IsNonGamepadInputQuery(flags) || !IsNuiInputQuery(flags)) {
    return false;
  }

  if (!GetNuiDeviceStatus()) {
    *result = X_ERROR_DEVICE_NOT_CONNECTED;
    return true;
  }

  const uint32_t actual_user_index = ResolveNuiUserIndex(user_index, flags);
  const bool valid_user = actual_user_index < XUserMaxUserCount;
  const uint32_t tracked_count = GetNuiTrackedSkeletonCount();
  const uint32_t tracking_id = GetNuiBestTrackingId();
  const uint32_t presence_tracking_id =
      ResolveNuiPresenceTrackingId(tracked_count, tracking_id);
  const bool has_nui_user = presence_tracking_id != 0;
  const bool has_skeleton = valid_user && has_nui_user;
  const uint32_t packet_number = GetNuiSkeletonFrameNumber();
  uint16_t buttons = 0;
  uint8_t left_trigger = 0;
  uint8_t right_trigger = 0;

  if (has_skeleton) {
    buttons = xe::hid::X_INPUT_GAMEPAD_GUIDE;
    left_trigger = 0x40;
    right_trigger = tracked_count != 0 ? 0xFF : 0x80;
  }

  if (input_state) {
    input_state.Zero();
    if (valid_user && has_skeleton) {
      input_state->packet_number = packet_number ? packet_number : 1;
      input_state->gamepad.buttons = buttons;
      input_state->gamepad.left_trigger = left_trigger;
      input_state->gamepad.right_trigger = right_trigger;
    }
  }

  if (has_nui_user) {
    SetEngagedNuiTrackingId(presence_tracking_id);
  }

  *result = valid_user ? X_ERROR_SUCCESS : X_ERROR_DEVICE_NOT_CONNECTED;
  LogNuiInputState(user_index, flags, tracked_count, presence_tracking_id,
                   has_skeleton ? packet_number : 0, buttons, left_trigger,
                   right_trigger, *result);
  return true;
}

bool TryHandleNuiKeystroke(uint32_t flags, pointer_t<X_INPUT_KEYSTROKE> stroke,
                           uint32_t* result) {
  if (!IsNuiInputQuery(flags)) {
    return false;
  }

  if (!GetNuiDeviceStatus()) {
    *result = X_ERROR_DEVICE_NOT_CONNECTED;
    return true;
  }

  stroke.Zero();
  const uint32_t tracked_count = GetNuiTrackedSkeletonCount();
  const uint32_t tracking_id = GetNuiBestTrackingId();
  const uint32_t presence_tracking_id =
      ResolveNuiPresenceTrackingId(tracked_count, tracking_id);
  static std::atomic<uint32_t> last_reported_tracking_id{0};
  const uint32_t previous_tracking_id =
      last_reported_tracking_id.load(std::memory_order_relaxed);

  if (presence_tracking_id && previous_tracking_id != presence_tracking_id) {
    last_reported_tracking_id.store(presence_tracking_id,
                                    std::memory_order_relaxed);
    stroke->user_index = 0;
    stroke->flags = xe::hid::X_INPUT_KEYSTROKE_KEYDOWN;
    SetEngagedNuiTrackingId(presence_tracking_id);
    *result = X_ERROR_SUCCESS;
    LogNuiKeystrokePresence(flags, tracked_count, presence_tracking_id,
                            uint32_t(stroke->flags), *result);
    return true;
  }

  if (!presence_tracking_id && previous_tracking_id != 0) {
    last_reported_tracking_id.store(0, std::memory_order_relaxed);
    stroke->user_index = 0;
    stroke->flags = xe::hid::X_INPUT_KEYSTROKE_KEYUP;
    *result = X_ERROR_SUCCESS;
    LogNuiKeystrokePresence(flags, tracked_count, presence_tracking_id,
                            uint32_t(stroke->flags), *result);
    return true;
  }

  stroke->user_index = 0xFF;
  *result = X_ERROR_EMPTY;
  LogNuiKeystrokePresence(flags, tracked_count, presence_tracking_id, 0,
                          *result);
  return true;
}

dword_result_t XAutomationpUnbindController_entry(dword_t user_index) {
  if (user_index >= XUserMaxUserCount) {
    return 0;
  }

  return 1;
}
DECLARE_XAM_EXPORT1(XAutomationpUnbindController, kInput, kStub);

void XamResetInactivity_entry() {
  // Do we need to do anything?
}
DECLARE_XAM_EXPORT1(XamResetInactivity, kInput, kStub);

dword_result_t XamEnableInactivityProcessing_entry(dword_t inactivity_index,
                                                   dword_t enable) {
  // Enables/disables screen saver and auto shutoff
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamEnableInactivityProcessing, kInput, kStub);

dword_result_t XamInputGetCapabilitiesEx_entry(
    dword_t unk, dword_t user_index, dword_t flags,
    pointer_t<X_INPUT_CAPABILITIES> caps) {
  if (unk > 1) {
    return X_ERROR_NOT_SUPPORTED;
  }

  // Fail-safe check
  if (!caps) {
    return X_ERROR_BAD_ARGUMENTS;
  }

  uint32_t nui_result = X_ERROR_DEVICE_NOT_CONNECTED;
  if (TryHandleNuiInputCapabilities(flags, caps, &nui_result)) {
    LogInputCall("XamInputGetCapabilitiesEx/NUI", user_index, flags,
                 nui_result);
    return nui_result;
  }

  caps.Zero();

  if ((flags & X_INPUT_FLAG::X_INPUT_FLAG_ANY_USER) != 0) {
    // should trap
  }

  if ((flags & 4) != 0) {
    // should trap
  }

  uint32_t actual_user_index = user_index;
  if ((actual_user_index & XUserIndexAny) == XUserIndexAny ||
      (flags & X_INPUT_FLAG::X_INPUT_FLAG_ANY_USER)) {
    // Always pin user to 0.
    actual_user_index = 0;
  }

  uint32_t actual_flags = flags;
  if (!flags) {
    actual_flags = X_INPUT_FLAG::X_INPUT_FLAG_GAMEPAD |
                   X_INPUT_FLAG::X_INPUT_FLAG_KEYBOARD;
  }

  auto input_system = kernel_state()->emulator()->input_system();
  auto lock = input_system->lock();
  const uint32_t result =
      input_system->GetCapabilities(actual_user_index, actual_flags, caps);
  LogInputCall("XamInputGetCapabilitiesEx", actual_user_index, actual_flags,
               result);
  return result;
}
DECLARE_XAM_EXPORT1(XamInputGetCapabilitiesEx, kInput, kSketchy);

// https://msdn.microsoft.com/en-us/library/windows/desktop/microsoft.directx_sdk.reference.xinputgetcapabilities(v=vs.85).aspx
dword_result_t XamInputGetCapabilities_entry(
    dword_t user_index, dword_t flags, pointer_t<X_INPUT_CAPABILITIES> caps) {
  // chrispy: actually, it appears that caps is never checked for null, it is
  // memset at the start regardless
  return XamInputGetCapabilitiesEx_entry(1, user_index, flags, caps);
}
DECLARE_XAM_EXPORT1(XamInputGetCapabilities, kInput, kSketchy);

// https://msdn.microsoft.com/en-us/library/windows/desktop/microsoft.directx_sdk.reference.xinputgetstate(v=vs.85).aspx
dword_result_t XamInputGetState_entry(dword_t user_index, dword_t flags,
                                      pointer_t<X_INPUT_STATE> input_state) {
  if (input_state) {
    memset((void*)input_state.host_address(), 0, sizeof(X_INPUT_STATE));
  }

  uint32_t nui_result = X_ERROR_DEVICE_NOT_CONNECTED;
  if (TryHandleNuiInputState(user_index, flags, input_state, &nui_result)) {
    LogInputCall("XamInputGetState/NUI", user_index, flags, nui_result);
    return nui_result;
  }

  uint32_t actual_user_index = user_index;
  // chrispy: change this, logic is not right
  if ((actual_user_index & XUserIndexAny) == XUserIndexAny ||
      (flags & X_INPUT_FLAG::X_INPUT_FLAG_ANY_USER)) {
    // Always pin user to 0.
    actual_user_index = 0;
  }

  if (actual_user_index >= XUserMaxUserCount) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  if (kernel_state()->xam_state()->IsUIActive()) {
    return X_ERROR_SUCCESS;
  }

  // Games call this with a NULL state ptr, probably as a query.

  X_RESULT result;
  auto input_system = kernel_state()->emulator()->input_system();
  {
    auto lock = input_system->lock();
    result = input_system->GetState(
        actual_user_index, !flags ? X_INPUT_FLAG::X_INPUT_FLAG_GAMEPAD : flags,
        input_state);
  }

  if (input_state && result == X_ERROR_SUCCESS) {
    if (auto patch = kernel_state()->xmp_volume_patch()) {
      patch->OnInputPoll(input_state->packet_number);
    }
  }

  return result;
}
DECLARE_XAM_EXPORT2(XamInputGetState, kInput, kImplemented, kHighFrequency);

// https://msdn.microsoft.com/en-us/library/windows/desktop/microsoft.directx_sdk.reference.xinputsetstate(v=vs.85).aspx
dword_result_t XamInputSetState_entry(
    dword_t user_index,
    dword_t flags, /* flags, as far as i can see, is not used*/
    pointer_t<X_INPUT_VIBRATION> vibration) {
  if (user_index >= XUserMaxUserCount) {
    return X_E_DEVICE_NOT_CONNECTED;
  }
  if (!vibration) {
    return X_ERROR_BAD_ARGUMENTS;
  }

  auto input_system = kernel_state()->emulator()->input_system();
  auto lock = input_system->lock();
  return input_system->SetState(user_index, vibration);
}
DECLARE_XAM_EXPORT1(XamInputSetState, kInput, kImplemented);

dword_result_t XamInputRawState_entry(unknown_t r3, unknown_t r4, unknown_t r5,
                                      unknown_t r6, unknown_t r7,
                                      unknown_t r8) {
  const uint32_t result =
      GetNuiDeviceStatus() ? X_ERROR_SUCCESS : X_ERROR_DEVICE_NOT_CONNECTED;
  LogInputRawCall("XamInputRawState", uint32_t(r3), uint32_t(r4), uint32_t(r5),
                  uint32_t(r6), uint32_t(r7), uint32_t(r8), result);
  return result;
}
DECLARE_XAM_EXPORT1(XamInputRawState, kInput, kStub);

// https://msdn.microsoft.com/en-us/library/windows/desktop/microsoft.directx_sdk.reference.xinputgetkeystroke(v=vs.85).aspx
dword_result_t XamInputGetKeystroke_entry(
    dword_t user_index, dword_t flags, pointer_t<X_INPUT_KEYSTROKE> keystroke) {
  // https://github.com/CodeAsm/ffplay360/blob/master/Common/AtgXime.cpp
  // user index = index or XUSER_INDEX_ANY
  // flags = XINPUT_FLAG_GAMEPAD (| _ANYUSER | _ANYDEVICE)

  if (!keystroke) {
    return X_ERROR_BAD_ARGUMENTS;
  }

  uint32_t nui_result = X_ERROR_DEVICE_NOT_CONNECTED;
  if (TryHandleNuiKeystroke(flags, keystroke, &nui_result)) {
    LogInputCall("XamInputGetKeystroke/NUI", user_index, flags, nui_result);
    return nui_result;
  }

  uint32_t actual_user_index = user_index;
  if ((actual_user_index & XUserIndexAny) == XUserIndexAny ||
      (flags & X_INPUT_FLAG::X_INPUT_FLAG_ANY_USER)) {
    // Always pin user to 0.
    actual_user_index = 0;
  }

  auto input_system = kernel_state()->emulator()->input_system();
  auto lock = input_system->lock();
  return input_system->GetKeystroke(user_index, flags, keystroke);
}
DECLARE_XAM_EXPORT1(XamInputGetKeystroke, kInput, kImplemented);

// Same as non-ex, just takes a pointer to user index.
dword_result_t XamInputGetKeystrokeEx_entry(
    lpdword_t user_index_ptr, dword_t flags,
    pointer_t<X_INPUT_KEYSTROKE> keystroke) {
  if (!keystroke) {
    return X_ERROR_BAD_ARGUMENTS;
  }

  uint32_t nui_result = X_ERROR_DEVICE_NOT_CONNECTED;
  if (TryHandleNuiKeystroke(flags, keystroke, &nui_result)) {
    if (nui_result == X_ERROR_SUCCESS && user_index_ptr) {
      *user_index_ptr = keystroke->user_index;
    }
    LogInputCall("XamInputGetKeystrokeEx/NUI",
                 user_index_ptr ? uint32_t(*user_index_ptr) : 0xFFFFFFFFu,
                 flags, nui_result);
    return nui_result;
  }

  keystroke.Zero();

  if (kernel_state()->xam_state()->IsUIActive()) {
    return X_ERROR_SUCCESS;
  }

  uint32_t user_index = *user_index_ptr;
  auto input_system = kernel_state()->emulator()->input_system();
  auto lock = input_system->lock();
  if ((user_index & XUserIndexAny) == XUserIndexAny) {
    // Always pin user to 0.
    user_index = 0;
  }

  if (flags & X_INPUT_FLAG::X_INPUT_FLAG_ANY_USER) {
    // That flag means we should iterate over every connected controller and
    // check which one have pending request.
    auto result = X_ERROR_DEVICE_NOT_CONNECTED;
    for (uint32_t i = 0; i < XUserMaxUserCount; i++) {
      auto result = input_system->GetKeystroke(i, flags, keystroke);

      // Return result from first user that have pending request
      if (result == X_ERROR_SUCCESS) {
        *user_index_ptr = keystroke->user_index;
        return result;
      }
    }
    return result;
  }

  auto result = input_system->GetKeystroke(user_index, flags, keystroke);

  if (XSUCCEEDED(result)) {
    *user_index_ptr = keystroke->user_index;
  }
  return result;
}
DECLARE_XAM_EXPORT1(XamInputGetKeystrokeEx, kInput, kImplemented);

X_HRESULT_result_t XamUserGetDeviceContext_entry(dword_t user_index,
                                                 dword_t device_type,
                                                 lpdword_t out_ptr) {
  if (cvars::xam_nui_fake_connected) {
    XELOGI("KINECT: XamUserGetDeviceContext user={} device_type={:08X} "
           "out_ptr={:08X}",
           user_index.value(), device_type.value(), out_ptr.guest_address());
  }

  // Games check the result - usually with some masking.
  // If this function fails they assume zero, so let's fail AND
  // set zero just to be safe.
  *out_ptr = 0;
  if (kernel_state()->xam_state()->IsUserSignedIn(user_index) ||
      (user_index & XUserIndexAny) == XUserIndexAny) {
    *out_ptr = (uint32_t)user_index;
    return X_E_SUCCESS;
  } else {
    return X_E_DEVICE_NOT_CONNECTED;
  }
}
DECLARE_XAM_EXPORT1(XamUserGetDeviceContext, kInput, kStub);

X_HRESULT_result_t XamInputNonControllerGetRawEx_entry(
    dword_t device_id, lpdword_t buffer_ptr, lpdword_t buffer_length_ptr,
    lpword_t state_ptr) {
  if (cvars::xam_nui_fake_connected) {
    XELOGI("KINECT: XamInputNonControllerGetRawEx device_id={:08X} "
           "buffer_ptr={:08X} length_ptr={:08X} state_ptr={:08X}",
           device_id.value(), buffer_ptr.guest_address(),
           buffer_length_ptr.guest_address(), state_ptr.guest_address());
  }

  if (device_id != 5 && device_id != 6) {
    return X_ERROR_INVALID_PARAMETER;
  }
  if (!state_ptr || !buffer_length_ptr || !buffer_ptr) {
    return X_ERROR_INVALID_PARAMETER;
  }

  if (*buffer_length_ptr == 0 || *buffer_length_ptr > hid::kPortalBufferSize) {
    return X_ERROR_INVALID_PARAMETER;
  }

  auto portal = kernel_state()->emulator()->input_system()->GetPortal();
  if (!portal) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  uint32_t bytes_read = *buffer_length_ptr;
  uint16_t state = 0;

  const auto result = portal->Read(
      {kernel_memory()->TranslateVirtual(buffer_ptr.guest_address()),
       *buffer_length_ptr},
      bytes_read, state);

  if (XSUCCEEDED(result)) {
    *buffer_length_ptr = bytes_read;
    *state_ptr = state;
  }
  return result;
}
DECLARE_XAM_EXPORT1(XamInputNonControllerGetRawEx, kInput, kSketchy);

X_HRESULT_result_t XamInputNonControllerSetRawEx_entry(dword_t device_id,
                                                       lpdword_t buffer_ptr,
                                                       dword_t buffer_length) {
  if (cvars::xam_nui_fake_connected) {
    XELOGI("KINECT: XamInputNonControllerSetRawEx device_id={:08X} "
           "buffer_ptr={:08X} buffer_length={:08X}",
           device_id.value(), buffer_ptr.guest_address(),
           buffer_length.value());
  }

  if (device_id != 5 && device_id != 6) {
    return X_ERROR_INVALID_PARAMETER;
  }
  if (!buffer_ptr || !buffer_length || buffer_length > hid::kPortalBufferSize) {
    return X_ERROR_INVALID_PARAMETER;
  }

  auto portal = kernel_state()->emulator()->input_system()->GetPortal();
  if (!portal) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  return portal->Write(
      {kernel_memory()->TranslateVirtual(buffer_ptr.guest_address()),
       buffer_length});
}
DECLARE_XAM_EXPORT1(XamInputNonControllerSetRawEx, kInput, kSketchy);

X_HRESULT_result_t XamInputNonControllerGetRaw_entry(
    lpword_t state_ptr, lpdword_t buffer_length_ptr, lpdword_t buffer_ptr) {
  return XamInputNonControllerGetRawEx_entry(5, buffer_ptr, buffer_length_ptr,
                                             state_ptr);
}
DECLARE_XAM_EXPORT1(XamInputNonControllerGetRaw, kInput, kSketchy);

X_HRESULT_result_t XamInputNonControllerSetRaw_entry(dword_t buffer_length,
                                                     lpdword_t buffer_ptr) {
  // Normally there are handled separatelly with different first param, but
  // whatever.
  return XamInputNonControllerSetRawEx_entry(5, buffer_ptr, buffer_length);
}
DECLARE_XAM_EXPORT1(XamInputNonControllerSetRaw, kInput, kSketchy);

}  // namespace xam
}  // namespace kernel
}  // namespace xe

DECLARE_XAM_EMPTY_REGISTER_EXPORTS(Input);
