/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_KERNEL_FLAGS_H_
#define XENIA_KERNEL_KERNEL_FLAGS_H_
#include "xenia/base/cvar.h"

DECLARE_bool(headless);
DECLARE_bool(log_high_frequency_kernel_calls);
DECLARE_bool(xam_nui_fake_connected);
DECLARE_bool(xam_nui_log_voice);
DECLARE_bool(xam_nui_use_real_kinect);
DECLARE_string(xam_nui_backend);
DECLARE_int32(xam_nui_initial_skeleton_wait_ms);
DECLARE_int32(xam_nui_skeleton_tracking_flags);
DECLARE_bool(xam_nui_drive_tracked_skeletons);
DECLARE_bool(xam_nui_silhouette_segment);
DECLARE_bool(xam_nui_dc3_camera_dispatch);
DECLARE_bool(xam_nui_depth_test_pattern);
DECLARE_bool(xam_nui_jd4_camera_experiment);
DECLARE_bool(xam_nui_color_crop_to_view);
DECLARE_int32(xam_nui_color_crop_zoom_percent);
DECLARE_int32(xam_nui_color_view_y_offset);
DECLARE_int32(xam_nui_color_view_x_offset);
DECLARE_bool(xam_nui_synthesize_head_for_facecam);
DECLARE_double(xam_nui_facecam_head_drop);
DECLARE_int32(xam_nui_log_skeleton_values);
DECLARE_bool(xam_nui_gravity_from_floor);
DECLARE_bool(xam_nui_gravity_from_accelerometer);
DECLARE_bool(xam_nui_extrapolate_floor_plane);
DECLARE_double(xam_nui_camera_height_meters);
DECLARE_bool(xam_nui_tilt_complete_notify);
DECLARE_bool(xam_nui_lock_elevation);

#endif  // XENIA_KERNEL_KERNEL_FLAGS_H_
