/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/logging.h"
#include "xenia/kernel/kernel_flags.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_private.h"
#include "xenia/xbox.h"

namespace xe {
namespace kernel {
namespace xam {
bool IsNuiReady();
}  // namespace xam

namespace xboxkrnl {

dword_result_t GetNatalDeviceResult() {
  return cvars::xam_nui_fake_connected || xam::IsNuiReady()
             ? X_STATUS_SUCCESS
             : X_STATUS_NOT_FOUND;
}

#define XBOXKRNL_USBCAM_STUB(name)                                      \
  dword_result_t name##_entry(unknown_t r3, unknown_t r4, unknown_t r5,  \
                              unknown_t r6) {                           \
    XELOGI("KINECT: " #name " {:08X} {:08X} {:08X} {:08X}", r3.value(),  \
           r4.value(), r5.value(), r6.value());                         \
    return GetNatalDeviceResult();                                      \
  }                                                                     \
  DECLARE_XBOXKRNL_EXPORT1(name, kNone, kStub)

XBOXKRNL_USBCAM_STUB(XUsbcamSetCaptureMode);
XBOXKRNL_USBCAM_STUB(XUsbcamGetConfig);
XBOXKRNL_USBCAM_STUB(XUsbcamSetConfig);
XBOXKRNL_USBCAM_STUB(XUsbcamReadFrame);
XBOXKRNL_USBCAM_STUB(XUsbcamSnapshot);
XBOXKRNL_USBCAM_STUB(XUsbcamSetView);
XBOXKRNL_USBCAM_STUB(XUsbcamGetView);
XBOXKRNL_USBCAM_STUB(XUsbcamDestroy);
XBOXKRNL_USBCAM_STUB(XUsbcamReset);

dword_result_t PsCamDeviceRequest_entry(unknown_t request, unknown_t input,
                                        unknown_t input_size,
                                        unknown_t output,
                                        unknown_t output_size) {
  XELOGI("KINECT: PsCamDeviceRequest request={:08X} input={:08X} "
         "input_size={:08X} output={:08X} output_size={:08X}",
         request.value(), input.value(), input_size.value(), output.value(),
         output_size.value());
  return GetNatalDeviceResult();
}
DECLARE_XBOXKRNL_EXPORT1(PsCamDeviceRequest, kNone, kStub);

dword_result_t MicDeviceRequest_entry(unknown_t request, unknown_t input,
                                      unknown_t input_size, unknown_t output,
                                      unknown_t output_size) {
  XELOGI("KINECT: MicDeviceRequest request={:08X} input={:08X} "
         "input_size={:08X} output={:08X} output_size={:08X}",
         request.value(), input.value(), input_size.value(), output.value(),
         output_size.value());
  return GetNatalDeviceResult();
}
DECLARE_XBOXKRNL_EXPORT1(MicDeviceRequest, kNone, kStub);

dword_result_t RmcDeviceRequest_entry(unknown_t request, unknown_t input,
                                      unknown_t input_size, unknown_t output,
                                      unknown_t output_size) {
  XELOGI("KINECT: RmcDeviceRequest request={:08X} input={:08X} "
         "input_size={:08X} output={:08X} output_size={:08X}",
         request.value(), input.value(), input_size.value(), output.value(),
         output_size.value());
  return GetNatalDeviceResult();
}
DECLARE_XBOXKRNL_EXPORT1(RmcDeviceRequest, kNone, kStub);

dword_result_t XUsbcamCreate_entry(dword_t buffer,
                                   dword_t buffer_size,  // 0x4B000 640x480?
                                   lpdword_t handle_out) {
  XELOGI("KINECT: XUsbcamCreate buffer={:08X} buffer_size={:08X} "
         "handle_ptr={:08X}",
         buffer.value(), buffer_size.value(), handle_out.guest_address());
  // This function should return success.
  // It looks like it only allocates space for usbcam support.
  // returning error code might cause games to initialize incorrectly.
  // "Carcassonne" initalization function checks for result from this
  // function. If value is different than 0 instead of loading
  // rest of the game it returns from initalization function and tries
  // to run game normally which causes crash, due to uninitialized data.
  return X_STATUS_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(XUsbcamCreate, kNone, kStub);

dword_result_t XUsbcamGetState_entry() {
  XELOGI("KINECT: XUsbcamGetState");
  // 0 = not connected.
  // 1 = initialized
  // 2 = connected
  return GetNatalDeviceResult() == X_STATUS_SUCCESS ? 2 : 0;
}
DECLARE_XBOXKRNL_EXPORT1(XUsbcamGetState, kNone, kStub);

dword_result_t UsbdGetNatalHub_entry() {
  XELOGI("KINECT: UsbdGetNatalHub");
  return GetNatalDeviceResult() == X_STATUS_SUCCESS ? 0x4E554930 : 0;
}
DECLARE_XBOXKRNL_EXPORT1(UsbdGetNatalHub, kNone, kStub);

dword_result_t UsbdGetNatalHardwareVersion_entry() {
  XELOGI("KINECT: UsbdGetNatalHardwareVersion");
  return GetNatalDeviceResult() == X_STATUS_SUCCESS ? 0x00010000 : 0;
}
DECLARE_XBOXKRNL_EXPORT1(UsbdGetNatalHardwareVersion, kNone, kStub);

#undef XBOXKRNL_USBCAM_STUB

}  // namespace xboxkrnl
}  // namespace kernel
}  // namespace xe

DECLARE_XBOXKRNL_EMPTY_REGISTER_EXPORTS(Usbcam);
