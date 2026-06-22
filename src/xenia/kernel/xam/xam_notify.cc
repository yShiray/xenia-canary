/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <mutex>
#include <unordered_map>

#include "xenia/base/logging.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xam/xam_private.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_threading.h"
#include "xenia/kernel/xnotifylistener.h"
#include "xenia/kernel/xthread.h"
#include "xenia/xbox.h"

namespace xe {
namespace kernel {
namespace xam {

// Provided by xam_nui.cc.
uint32_t GetNuiTrackedSkeletonCount();
uint32_t GetNuiBestTrackingId();
uint32_t GetNuiSkeletonFrameNumber();
uint32_t GetNuiDeviceStatus();
bool IsNuiReady();

namespace {

// NUI system notification IDs. JD2019 (and other Kinect titles) gate boot/menu
// progress on these arriving once a player is present and bound. NOTE: 0x0008001E
// is NOT the hardware-status ID - that one is an audio/speaker-config
// notification. The NUI hardware-status ID is 0x00060019, which fits the NUI
// notification block (0x0006001A NUIPAUSE, 0x0006001B UIAPPROACH, 0x0006001D
// BINDINGCHANGED) and decodes to mask_index=0, version=6.
constexpr XNotificationID kXnSysNuiUiApproach = XNotificationID(0x0006001B);
constexpr XNotificationID kXnSysNuiBindingChanged = XNotificationID(0x0006001D);
constexpr XNotificationID kXnSysNuiHardwareStatusChanged =
    XNotificationID(0x00060019);
constexpr uint32_t kNuiStatusConnected = 1;
constexpr uint32_t kNuiHardwareStatusConnected = 1;
constexpr uint32_t kNuiHardwareStatusReady = 2;
constexpr uint32_t kNuiDefaultUserIndex = 0;

bool IsNotificationAllowed(uint64_t areas, uint32_t max_version,
                           XNotificationID notification_id) {
  XNotificationKey key(notification_id);
  return (areas & (1ull << key.mask_index)) != 0 && key.version <= max_version;
}

// The title only advances past its Kinect-gated screens when its system listener
// observes a present, bound player. When the title's own queue is empty we stand
// in the matching NUI notifications, driven entirely by the live NUI state so the
// real sensor (when a body is tracked) and the synthetic stand-in player behave
// identically. Each is edge-triggered per listener handle so it is emitted once
// per state change rather than every poll.
bool TrySynthesizeNuiUiApproach(uint32_t handle, uint64_t areas,
                                uint32_t max_version, uint32_t match_id,
                                uint32_t* id, uint32_t* param) {
  if (match_id && match_id != kXnSysNuiUiApproach) {
    return false;
  }
  if (!IsNotificationAllowed(areas, max_version, kXnSysNuiUiApproach)) {
    return false;
  }

  const uint32_t tracked_count = GetNuiTrackedSkeletonCount();
  const uint32_t tracking_id = GetNuiBestTrackingId();
  const uint32_t frame_number = GetNuiSkeletonFrameNumber();

  static std::mutex mutex;
  static std::unordered_map<uint32_t, uint32_t> last_tracking_by_handle;
  if (!tracked_count || !tracking_id || !frame_number) {
    std::lock_guard<std::mutex> lock(mutex);
    last_tracking_by_handle.erase(handle);
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(mutex);
    uint32_t& last_tracking = last_tracking_by_handle[handle];
    if (last_tracking == tracking_id) {
      return false;
    }
    last_tracking = tracking_id;
  }

  *id = kXnSysNuiUiApproach;
  *param = 0;
  XELOGI(
      "XNotifyGetNext: synthesized XN_SYS_NUIUIAPPROACH frame={} "
      "tracked_count={} tracking_id={} param={:08X}",
      frame_number, tracked_count, tracking_id, *param);
  return true;
}

bool TrySynthesizeNuiBindingChanged(uint32_t handle, uint64_t areas,
                                    uint32_t max_version, uint32_t match_id,
                                    uint32_t* id, uint32_t* param) {
  if (match_id && match_id != kXnSysNuiBindingChanged) {
    return false;
  }
  if (!IsNotificationAllowed(areas, max_version, kXnSysNuiBindingChanged)) {
    return false;
  }

  const uint32_t device_status = GetNuiDeviceStatus();
  const bool ready = device_status == kNuiStatusConnected && IsNuiReady();
  const uint32_t tracked_count = GetNuiTrackedSkeletonCount();
  const uint32_t tracking_id = GetNuiBestTrackingId();
  const uint32_t frame_number = GetNuiSkeletonFrameNumber();
  const bool active = ready && tracked_count && tracking_id && frame_number;

  static std::mutex mutex;
  static std::unordered_map<uint32_t, uint32_t> last_tracking_by_handle;
  {
    std::lock_guard<std::mutex> lock(mutex);
    uint32_t& last_tracking = last_tracking_by_handle[handle];
    if (!active) {
      last_tracking = 0;
      return false;
    }
    if (last_tracking == tracking_id) {
      return false;
    }
    last_tracking = tracking_id;
  }

  *id = kXnSysNuiBindingChanged;
  *param = kNuiDefaultUserIndex;
  XELOGI(
      "XNotifyGetNext: synthesized XN_SYS_NUIBINDINGCHANGED frame={} "
      "tracked_count={} tracking_id={} user_index={} param={:08X}",
      frame_number, tracked_count, tracking_id, kNuiDefaultUserIndex, *param);
  return true;
}

bool TrySynthesizeNuiHardwareStatusChanged(uint32_t handle, uint64_t areas,
                                           uint32_t max_version,
                                           uint32_t match_id, uint32_t* id,
                                           uint32_t* param) {
  if (match_id && match_id != kXnSysNuiHardwareStatusChanged) {
    return false;
  }
  if (!IsNotificationAllowed(areas, max_version,
                             kXnSysNuiHardwareStatusChanged)) {
    return false;
  }

  const uint32_t device_status = GetNuiDeviceStatus();
  const bool ready = device_status == kNuiStatusConnected && IsNuiReady();
  uint32_t hardware_status =
      device_status == kNuiStatusConnected
          ? kNuiHardwareStatusConnected | (ready ? kNuiHardwareStatusReady : 0)
          : 0;
  const uint32_t tracked_count = GetNuiTrackedSkeletonCount();
  const uint32_t tracking_id = GetNuiBestTrackingId();
  const uint32_t frame_number = GetNuiSkeletonFrameNumber();

  static std::mutex mutex;
  static std::unordered_map<uint32_t, uint64_t> last_status_key_by_handle;
  {
    std::lock_guard<std::mutex> lock(mutex);
    const uint64_t status_key =
        (uint64_t(hardware_status) << 32) |
        (tracked_count != 0 ? uint64_t(tracking_id) : 0);
    uint64_t& last_status_key = last_status_key_by_handle[handle];
    if (last_status_key == status_key ||
        (last_status_key == 0 && status_key == 0)) {
      return false;
    }
    last_status_key = status_key;
  }

  *id = kXnSysNuiHardwareStatusChanged;
  *param = hardware_status;
  XELOGI(
      "XNotifyGetNext: synthesized XN_SYS_NUIHARDWARESTATUSCHANGED "
      "status={:08X} ready={} tracked_count={} tracking_id={} frame={}",
      hardware_status, ready ? 1 : 0, tracked_count, tracking_id, frame_number);
  return true;
}

}  // namespace

uint32_t xeXamNotifyCreateListener(uint64_t mask, uint32_t is_system,
                                   uint32_t max_version) {
  assert_true(max_version < 11);

  if (max_version > 10) {
    max_version = 10;
  }

  auto listener =
      object_ref<XNotifyListener>(new XNotifyListener(kernel_state()));
  listener->Initialize(mask, is_system, max_version);

  // Handle ref is incremented, so return that.
  uint32_t handle = listener->handle();

  return handle;
}

dword_result_t XamNotifyCreateListener_entry(qword_t mask,
                                             dword_t max_version) {
  auto thread = kernel::XThread::GetCurrentThread();
  auto ctx = thread->thread_state()->context();
  auto type = xboxkrnl::xeKeGetCurrentProcessType(ctx);
  return xeXamNotifyCreateListener(mask, type == 2, max_version);
}
DECLARE_XAM_EXPORT1(XamNotifyCreateListener, kNone, kImplemented);

dword_result_t XamNotifyCreateListenerInternal_entry(qword_t mask,
                                                     dword_t is_system,
                                                     dword_t max_version) {
  return xeXamNotifyCreateListener(mask, is_system, max_version);
}
DECLARE_XAM_EXPORT1(XamNotifyCreateListenerInternal, kNone, kImplemented);

// https://github.com/CodeAsm/ffplay360/blob/master/Common/AtgSignIn.cpp
dword_result_t XNotifyGetNext_entry(dword_t handle, dword_t match_id,
                                    lpdword_t id_ptr, lpdword_t param_ptr) {
  if (param_ptr) {
    *param_ptr = 0;
  }

  if (!id_ptr) {
    return 0;
  }
  *id_ptr = 0;

  // Grab listener.
  auto listener =
      kernel_state()->object_table()->LookupObject<XNotifyListener>(handle);
  if (!listener) {
    return 0;
  }

  const uint64_t areas = listener->mask();
  const uint32_t max_version = listener->max_version();

  bool dequeued = false;
  uint32_t id = 0;
  uint32_t param = 0;
  if (match_id) {
    // Asking for a specific notification
    id = match_id;
    dequeued = listener->DequeueNotification(match_id, &param);
  } else {
    // Just get next.
    dequeued = listener->DequeueNotification(&id, &param);
  }

  // When the title's own queue is empty, stand in the NUI system notifications a
  // Kinect title needs to detect a present and bound player. These are gated on
  // the listener's subscription mask/version, so non-NUI listeners are untouched.
  if (!dequeued) {
    dequeued = TrySynthesizeNuiHardwareStatusChanged(
        uint32_t(handle), areas, max_version, uint32_t(match_id), &id, &param);
  }
  if (!dequeued) {
    dequeued = TrySynthesizeNuiBindingChanged(
        uint32_t(handle), areas, max_version, uint32_t(match_id), &id, &param);
  }
  if (!dequeued) {
    dequeued = TrySynthesizeNuiUiApproach(uint32_t(handle), areas, max_version,
                                          uint32_t(match_id), &id, &param);
  }

  *id_ptr = dequeued ? id : 0;
  // param_ptr may be null - 555307F0 Demo explicitly passes nullptr in the
  // code.
  // https://github.com/xenia-project/xenia/pull/1577
  if (param_ptr) {
    *param_ptr = dequeued ? param : 0;
  }
  return dequeued ? 1 : 0;
}
DECLARE_XAM_EXPORT2(XNotifyGetNext, kNone, kImplemented, kHighFrequency);

dword_result_t XNotifyDelayUI_entry(dword_t delay_ms) {
  // Ignored.
  return 0;
}
DECLARE_XAM_EXPORT1(XNotifyDelayUI, kNone, kStub);

void XNotifyPositionUI_entry(dword_t position) {
  kernel_state()->notification_position_ = position;
  // Ignored.
}
DECLARE_XAM_EXPORT1(XNotifyPositionUI, kNone, kStub);

dword_result_t XNotifyBroadcast_entry(dword_t notification, dword_t data) {
  kernel_state()->BroadcastNotification(notification, data);

  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XNotifyBroadcast, kNone, kStub);

}  // namespace xam
}  // namespace kernel
}  // namespace xe

DECLARE_XAM_EMPTY_REGISTER_EXPORTS(Notify);
