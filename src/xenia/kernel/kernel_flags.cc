/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/kernel/kernel_flags.h"

DEFINE_bool(headless, false,
            "Don't display any UI, using defaults for prompts as needed.",
            "UI");
DEFINE_bool(log_high_frequency_kernel_calls, false,
            "Log kernel calls with the kHighFrequency tag.", "Logging");
DEFINE_bool(xam_nui_fake_connected, false,
            "Expose a synthetic connected Kinect/NUI device to titles.",
            "Kernel");
DEFINE_bool(xam_nui_log_voice, false,
            "Log XamVoice mic-array calls (args + pointed-to memory) so the "
            "in-guest NuiAudio/NuiSpeech mic-array ABI can be captured from a "
            "running title. Diagnostic for the Kinect voice-command pipeline.",
            "Kernel");
DEFINE_bool(xam_nui_use_real_kinect, false,
            "Use the Windows Kinect v1 SDK as the XAM NUI device backend.",
            "Kernel");
DEFINE_string(xam_nui_backend, "auto",
              "Kinect/NUI backend. Use: [auto, sdk, real, fake, disabled].",
              "Kernel");
DEFINE_int32(xam_nui_initial_skeleton_wait_ms, 250,
             "Milliseconds to wait for a real Kinect skeleton during backend "
             "startup before the background poller takes over. Use 0 to "
             "disable.",
             "Kernel");
DEFINE_int32(xam_nui_skeleton_tracking_flags, 0,
             "Windows Kinect v1 SDK NuiSkeletonTrackingEnable flags. The "
             "default uses automatic tracking, matching the Xbox 360 NUI SDK.",
             "Kernel");
DEFINE_bool(
    xam_nui_color_crop_to_view, false,
    "Crop the Kinect colour buffer to the head-following digital-zoom "
    "view area inside Xenia (the SDK behaviour: titles that display the "
    "colour feed as-is, like Just Dance's face-follow thumbnail, expect "
    "the colour stream already cropped/zoomed onto the player's head). "
    "Set false to hand the title the full uncropped colour frame for "
    "titles that crop it themselves.",
    "Kernel");
DEFINE_int32(xam_nui_color_crop_zoom_percent, 150,
             "Software zoom used for the Kinect colour face-follow crop. "
             "100 keeps the full frame, 150 frames the whole head, and 200 "
             "matches the Xbox NUI 2x digital zoom.",
             "Kernel");
DEFINE_bool(xam_nui_synthesize_head_for_facecam, false,
            "Give Kinect titles a continuous head joint for the colour "
            "face-follow thumbnail. Host skeletal tracking flickers between "
            "fully-tracked and position-only, so most frames carry no joints "
            "and titles (e.g. Just Dance) that centre the colour crop on the "
            "HEAD joint stick on a fixed point. When set, position-only bodies "
            "are presented as tracked with HEAD and SHOULDER_CENTER joints "
            "inferred above the body centre (the title sizes the digital zoom "
            "from the head<->shoulder span, so both are needed for it to "
            "magnify); other joints stay NOT_TRACKED so dance scoring still "
            "ignores them. Disable if it disturbs engagement or scoring.",
            "Kernel");
DEFINE_double(xam_nui_facecam_head_drop, 0.0,
              "Slide the colour face-cam crop DOWN onto the face. Titles centre "
              "the thumbnail on the HEAD joint, which sits at the top of the "
              "head, so the face lands low with empty headroom above. This is "
              "off by default because moving HEAD/SHOULDER_CENTER in real "
              "tracked skeletons can perturb dance scoring; nonzero values are "
              "only applied to synthesized position-only facecam continuity "
              "bodies. 0 = off (raw scoring joints).",
              "Kernel");
DEFINE_int32(xam_nui_color_view_y_offset, 0,
             "Vertical pixel bias for the Kinect colour face-follow crop. The "
             "skeleton head joint projects to the top of the head, so the 2x "
             "digital-zoom crop centres above the face; this shifts it down "
             "(positive = down) to frame the face. Tune per camera height/tilt "
             "(0 = centre exactly on the head joint).",
             "Kernel");
DEFINE_int32(
    xam_nui_color_view_x_offset, 0,
    "Horizontal pixel bias for the Kinect colour face-follow crop "
    "(positive = right). Usually 0; tune if the framing is off-centre.",
    "Kernel");
DEFINE_bool(xam_nui_silhouette_segment, true,
            "Background-remove the 640x480 silhouette depth buffer: keep depth "
            "only where the Kinect marked a player (low 3 player-index bits "
            "set), zero everywhere else. The Xbox NUI silhouette converter / "
            "ps_mumo_silhouette shader paints whatever has depth, so without "
            "this the whole room is drawn instead of just the dancer. Only "
            "affects the silhouette buffer, not skeleton tracking. Disable to "
            "feed the raw full-room depth.",
            "Kernel");
DEFINE_bool(xam_nui_dc3_camera_dispatch, false,
            "EXPERIMENTAL (Dance Central 3 / title 373307D9 only, opt-in). DC3 "
            "uses a PUSH camera model: it registers depth buffers via _NUICAM "
            "opcode 0x05 and expects the driver to deliver each frame by writing "
            "a descriptor to its NUI context+0xA0 and signalling the event at "
            "context+0x38 (proven by disassembling its frame callback 829CE1C0; "
            "same structure as the working Just Dance 2019 dispatcher). DC3 never "
            "pulls (XStudio 0x1004 is called 0 times), so without this the "
            "silhouette panel colorizes an all-zero buffer (renders empty). This "
            "synthesizes that frame delivery each depth frame. Off by default "
            "because it pokes the title's NUI dispatcher; toggle on to test, off "
            "if DC3 misbehaves.",
            "Kernel");
DEFINE_bool(xam_nui_depth_test_pattern, false,
            "DECISIVE DIAGNOSTIC: fill the GPU-sampled NUI depth title surface "
            "(phys 0x102B9000 for DC3) with a bright full-frame gradient where "
            "every pixel has depth + player-index bit set, ignoring the real "
            "Kinect data. If the title then draws a visible gradient where the "
            "silhouette belongs, the title IS sampling our surface and only the "
            "data/format needs work; if the screen stays black, the title reads "
            "a DIFFERENT surface and we must hook that instead. Turn off for "
            "normal play.",
            "Kernel");
DEFINE_bool(xam_nui_jd4_camera_experiment, true,
            "EXPERIMENTAL (Just Dance 4 / title 555308B5 only). JD4 drives the "
            "camera with a push/DMA model and polls _NUICAM opcode 0x24 for a "
            "ready frame. This makes 0x24 report a frame ready in a rotating "
            "buffer so JD4 starts consuming the host-filled silhouette buffers. "
            "The reply layout is a best-guess; toggle off if JD4 misbehaves.",
            "Kernel");
DEFINE_int32(
    xam_nui_log_skeleton_values, 0,
    "Diagnostic: dump the full real-Kinect skeleton the host hands the title "
    "every N published frames (0 = off, 30 ~= once per second). Logs the bound "
    "body's tracking id/state, all 20 joints (index, per-joint tracking state, "
    "x/y/z metres), the body centre, vNormalToGravity and vFloorClipPlane, plus "
    "a per-frame head-joint jitter delta so raw SDK noise is visible. Use this "
    "to verify the values feeding dance scoring (Just Dance / Dance Central) "
    "rather than guessing at tracking instability.",
    "Kernel");
DEFINE_bool(
    xam_nui_gravity_from_floor, true,
    "Feed titles a real gravity-up vector (vNormalToGravity) derived from the "
    "Kinect floor clip plane when the SDK leaves vNormalToGravity zero (it "
    "almost always does). Xbox Kinect titles level the skeleton with "
    "NuiTransformMatrixLevel(vNormalToGravity) before scoring hand/spine/head "
    "positions (see the XDK AtgNuiRelativeCoordinates sample); handing them a "
    "flat (0,1,0) when the camera is tilted leaves every pose rotated by the "
    "camera tilt, so correctly performed moves score wrong. The floor normal is "
    "low-pass smoothed and the last valid normal is held across frames where "
    "the SDK drops the floor plane. Disable to fall back to a flat (0,1,0) up "
    "vector.",
    "Kernel");
DEFINE_bool(
    xam_nui_gravity_from_accelerometer, false,
    "Allow the Kinect's 3-axis accelerometer to seed vNormalToGravity before a "
    "measured floor clip plane is available. Off by default to match the known "
    "working DC1/DC3 scoring path: once a real floor normal has been seen, "
    "scoring holds that floor-derived up vector across floor-plane dropouts "
    "instead of drifting toward accelerometer bias. Enable only for diagnostics "
    "or setups where the SDK never reports a usable floor plane.",
    "Kernel");
DEFINE_bool(xam_nui_drive_tracked_skeletons, false,
            "Drive the Windows Kinect v1 SDK body segmentation the same way "
            "the official Kinect SDK 'Segmentation' sample does: enable "
            "TITLE_SETS_TRACKED_SKELETONS and, every skeleton frame, ask the "
            "SDK to fully track the closest detected bodies. Full tracking is "
            "what makes the depth+player stream carry the per-pixel player "
            "index (body silhouette); without it player_pixels stays 0 and "
            "titles get a whole-room silhouette instead of a body cutout. "
            "Disable to fall back to pure automatic tracking.",
            "Kernel");
DEFINE_bool(
    xam_nui_extrapolate_floor_plane, true,
    "Emulate the Xbox NUI_INITIALIZE_FLAG_EXTRAPOLATE_FLOOR_PLANE (XDK "
    "nuiapi.h 0x800), which the desktop Kinect v1 SDK does not implement. On "
    "real hardware the runtime reports a floor clip plane derived from the "
    "camera tilt even with NO body in view, so a title's start-up 'lower the "
    "sensor to scan the floor / set the field of view' calibration can finish "
    "and raise the camera. The Windows SDK leaves vFloorClipPlane zero until a "
    "skeleton is tracked, so that calibration lowers and then hangs forever. "
    "When set, the host synthesizes the floor plane from the real elevation "
    "angle and an assumed camera height whenever (and only when) the SDK left "
    "it empty; a tracked body's real floor plane is never overridden. Disable "
    "to pass the SDK's raw (usually zero) floor plane through unchanged.",
    "Kernel");
DEFINE_double(
    xam_nui_camera_height_meters, 1.68,
    "Assumed Kinect sensor height above the floor, in metres, used to "
    "synthesize the floor clip plane distance (D) when "
    "xam_nui_extrapolate_floor_plane fills in a plane the SDK left empty. The "
    "desktop SDK cannot measure the height without a tracked body, so this is "
    "the fallback the title calibrates its field of view against (the official "
    "Kinect 'assume 5.5 ft off the floor' default is 1.68). Set it to your "
    "real sensor height for the most accurate start-up framing.",
    "Kernel");
DEFINE_bool(
    xam_nui_tilt_complete_notify, false,
    "Fire an asynchronous XAM tilt-move-complete notification (message "
    "0x0002B00A) to the title's registered tilt callback after the camera "
    "elevation motor settles. The proven-working reference behaviour (and the "
    "real console for the floor-scan calibration) delivers no such callback - "
    "the statically linked NUI runtime drives its own floor search off the "
    "live skeleton/floor-clip-plane stream - so this is OFF by default and "
    "XamNuiCameraElevationSetAngle simply forwards the angle to the motor. "
    "Enable only to experiment with the callback path.",
    "Kernel");
DEFINE_bool(
    xam_nui_lock_elevation, true,
    "Keep the Kinect tilt motor locked in its mounted position and never let a "
    "title physically move it. Titles tilt the camera down at start-up to scan "
    "the floor; that motion is disruptive and, on the desktop SDK, can leave "
    "the sensor stuck pointing at the floor. When set, XamNuiCameraElevation"
    "SetAngle does not drive the motor - it accepts the commanded angle "
    "virtually and XamNuiCameraElevationGetAngle reports that value back, so a "
    "title's calibration still completes (paired with "
    "xam_nui_extrapolate_floor_plane, which supplies the floor plane at the "
    "commanded angle) while the real sensor stays put. Disable to allow titles "
    "to physically tilt the sensor again.",
    "Kernel");
