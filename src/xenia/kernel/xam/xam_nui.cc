/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "xenia/base/logging.h"
#include "xenia/memory.h"

#if XE_PLATFORM_WIN32
#include "xenia/base/platform_win.h"
#endif  // XE_PLATFORM_WIN32

#include "xenia/emulator.h"
#include "xenia/cpu/processor.h"
#include "xenia/gpu/command_processor.h"
#include "xenia/gpu/graphics_system.h"
#include "xenia/kernel/kernel_flags.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xam/xam_private.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_threading.h"
#include "xenia/kernel/xevent.h"
#include "xenia/kernel/xthread.h"
#include "xenia/ui/imgui_dialog.h"
#include "xenia/ui/imgui_drawer.h"
#include "xenia/ui/window.h"
#include "xenia/ui/windowed_app_context.h"
#include "xenia/xbox.h"

namespace xe {
namespace kernel {
namespace xam {
// https://web.cs.ucdavis.edu/~okreylos/ResDev/Kinect/MainPage.html

constexpr uint32_t kNuiInvalidTrackingId = 0;
constexpr uint32_t kNuiSyntheticTrackingId = 1;
constexpr uint32_t kNuiSkeletonCount = 6;
constexpr uint32_t kNuiSkeletonPositionCount = 20;
constexpr uint32_t kNuiSkeletonNotTracked = 0;
constexpr uint32_t kNuiSkeletonPositionOnly = 1;
constexpr uint32_t kNuiSkeletonTracked = 2;

// NUI_SKELETON_POSITION_INDEX order (Kinect v1). Used by the skeleton-value
// diagnostic so dumped joints are human-readable.
constexpr const char* kNuiJointNames[kNuiSkeletonPositionCount] = {
    "HIP_CENTER",    "SPINE",         "SHOULDER_CENTER", "HEAD",
    "SHOULDER_LEFT", "ELBOW_LEFT",    "WRIST_LEFT",      "HAND_LEFT",
    "SHOULDER_RIGHT","ELBOW_RIGHT",   "WRIST_RIGHT",     "HAND_RIGHT",
    "HIP_LEFT",      "KNEE_LEFT",     "ANKLE_LEFT",      "FOOT_LEFT",
    "HIP_RIGHT",     "KNEE_RIGHT",    "ANKLE_RIGHT",     "FOOT_RIGHT"};
// NUI_SKELETON_POSITION_TRACKING_STATE: 0=NOT_TRACKED 1=INFERRED 2=TRACKED.
constexpr const char* kNuiJointStateNames[3] = {"NOT", "INF", "TRK"};
constexpr uint32_t kNuiImagePlayerIndexMask = 0x7;
constexpr float kNuiDepthFocalLength320x240 = 285.63f;
constexpr uint32_t kNuiImageDigitalZoom1x = 0;
constexpr uint32_t kNuiImageDigitalZoom2x = 1;
constexpr uint32_t kNuiSkeletonTrackingFlagSuppressNoFrameData = 0x00000001;
constexpr uint32_t kNuiSkeletonTrackingFlagTitleSetsTrackedSkeletons =
    0x00000002;
constexpr uint32_t kNuiSkeletonTrackingFlagEnableSeatedSupport = 0x00000004;
constexpr uint32_t kNuiSkeletonTrackingValidFlags =
    kNuiSkeletonTrackingFlagSuppressNoFrameData |
    kNuiSkeletonTrackingFlagTitleSetsTrackedSkeletons |
    kNuiSkeletonTrackingFlagEnableSeatedSupport;
constexpr uint32_t kNuiInvalidEnrollmentIndex = 0xFFFFFFFFu;
constexpr uint32_t kNuiInvalidUserIndex = 0x000000FFu;
constexpr uint32_t kNuiHardwareStatusConnected = 0x00000001;
constexpr uint32_t kNuiHardwareStatusReady = 0x00000002;
constexpr uint32_t kNuiHardwareStatusActive = 0x00000008;

void NotifyGpuMemoryWritten(uint32_t guest_address, uint32_t length) {
  if (!guest_address || !length || !kernel_state()) {
    return;
  }
  auto* memory = kernel_state()->memory();
  const uint32_t physical_address = memory->GetPhysicalAddress(guest_address);
  auto* emulator = kernel_state()->emulator();
  auto* graphics_system = emulator ? emulator->graphics_system() : nullptr;
  auto* command_processor =
      graphics_system ? graphics_system->command_processor() : nullptr;
  if (physical_address == UINT32_MAX || !command_processor) {
    return;
  }

  // Host-side camera copies bypass guest CPU write watches. Explicitly
  // invalidate the GPU upload so each new face-follow crop reaches the title.
  command_processor->TracePlaybackWroteMemory(physical_address, length);
}

struct X_NUI_VECTOR4 {
  xe::be<float> x;
  xe::be<float> y;
  xe::be<float> z;
  xe::be<float> w;
};
static_assert(sizeof(X_NUI_VECTOR4) == 16, "NUI Vector4 size must match XDK");

struct X_NUI_SKELETON_DATA {
  xe::be<int32_t> tracking_state;
  xe::be<uint32_t> tracking_id;
  xe::be<uint32_t> enrollment_index;
  xe::be<uint32_t> user_index;
  X_NUI_VECTOR4 position;
  X_NUI_VECTOR4 skeleton_positions[kNuiSkeletonPositionCount];
  xe::be<int32_t> skeleton_position_tracking_state[kNuiSkeletonPositionCount];
  xe::be<uint32_t> quality_flags;
  xe::be<uint32_t> padding[3];
};
static_assert(sizeof(X_NUI_SKELETON_DATA) == 448,
              "NUI skeleton data size must match XDK");

#pragma pack(push, 16)
struct X_NUI_SKELETON_FRAME {
  xe::be<int64_t> timestamp;
  xe::be<uint32_t> frame_number;
  xe::be<uint32_t> flags;
  X_NUI_VECTOR4 floor_clip_plane;
  X_NUI_VECTOR4 normal_to_gravity;
  X_NUI_SKELETON_DATA skeleton_data[kNuiSkeletonCount];
};
#pragma pack(pop)
static_assert(sizeof(X_NUI_SKELETON_FRAME) == 2736,
              "NUI skeleton frame size must match XDK");

struct X_NUI_FITNESS_BODY_PROFILE_RECORD {
  xe::be<float> height;
  xe::be<float> weight;
  xe::be<uint16_t> year_of_birth;
  xe::be<uint16_t> reserved;
  xe::be<uint32_t> gender;
};
static_assert(sizeof(X_NUI_FITNESS_BODY_PROFILE_RECORD) == 16,
              "NUI fitness body profile record size must match XDK");

constexpr uint32_t kXamNuiFrameNoData = 0x83010001u;
constexpr uint32_t kNuiD3DDepthWidth = 320;
constexpr uint32_t kNuiD3DDepthHeight = 240;
constexpr uint32_t kNuiD3DColorWidth = 640;
constexpr uint32_t kNuiD3DColorHeight = 480;
constexpr uint32_t kNuiD3DColorPitch = kNuiD3DColorWidth * 4;
constexpr uint32_t kNuiD3DColorYuvPitch = kNuiD3DColorWidth * 2;
constexpr uint32_t kNuiD3DDepthStridePixels = 384;
constexpr uint32_t kNuiD3DDepthPitch =
    kNuiD3DDepthStridePixels * sizeof(uint16_t);
constexpr uint32_t kNuiD3DMiniDepthWidth = 80;
constexpr uint32_t kNuiD3DMiniDepthHeight = 60;
constexpr uint32_t kNuiD3DMiniDepthStridePixels = 128;
constexpr uint32_t kNuiD3DMiniDepthPitch =
    kNuiD3DMiniDepthStridePixels * sizeof(uint16_t);
constexpr uint32_t kNuiD3DCommonTypeTexture = 3;
constexpr uint32_t kNuiD3DCommonD3DCreated = 0x00100000;
constexpr uint32_t kNuiD3DCommonCpuCachedMemory = 0x00200000;
constexpr uint32_t kNuiD3DFlushInitialValue = 0xFFFF0000;
constexpr uint32_t kNuiD3DTextureObjectSize = 128;
constexpr uint32_t kNuiD3DTextureIdentifier = 0x4E554954;  // "NUIT"
constexpr uint32_t kNuiD3DTextureFormatOffset = 0x1C;
constexpr uint32_t kNuiGpuTextureFetchTypeTexture = 2;
constexpr uint32_t kNuiGpuTextureSignUnsigned = 0;
constexpr uint32_t kNuiGpuTextureClampToEdge = 2;
constexpr uint32_t kNuiGpuTextureFormat8888 = 6;
constexpr uint32_t kNuiGpuTextureFormatYuy2 = 11;
constexpr uint32_t kNuiGpuTextureFormat16 = 24;
constexpr uint32_t kNuiGpuEndian8In16 = 1;
constexpr uint32_t kNuiGpuEndian8In32 = 2;
constexpr uint32_t kNuiGpuRequestSize256Bit = 0;
constexpr uint32_t kNuiGpuNumFormatFraction = 0;
constexpr uint32_t kNuiGpuNumFormatInteger = 1;
constexpr uint32_t kNuiGpuSwizzleABGR =
    0 | (1u << 3) | (2u << 6) | (3u << 9);
constexpr uint32_t kNuiGpuSwizzleORGB =
    2 | (1u << 3) | (0u << 6) | (5u << 9);
constexpr uint32_t kNuiGpuTextureFilterPoint = 0;
constexpr uint32_t kNuiGpuMipFilterBaseMap = 2;
constexpr uint32_t kNuiGpuAnisoDisabled = 0;
constexpr uint32_t kNuiGpuDimension2D = 1;

struct X_D3DLOCKED_RECT {
  xe::be<int32_t> pitch;
  xe::be<uint32_t> bits;
};
static_assert(sizeof(X_D3DLOCKED_RECT) == 8,
              "D3DLOCKED_RECT size must match XDK");

struct X_RECT {
  xe::be<int32_t> left;
  xe::be<int32_t> top;
  xe::be<int32_t> right;
  xe::be<int32_t> bottom;
};
static_assert(sizeof(X_RECT) == 16, "RECT size must match XDK");

#if XE_PLATFORM_WIN32
constexpr DWORD kNuiInitializeFlagUsesDepthAndPlayerIndex = 0x00000001;
constexpr DWORD kNuiInitializeFlagUsesColor = 0x00000002;
constexpr DWORD kNuiInitializeFlagUsesSkeleton = 0x00000008;
constexpr DWORD kNuiInitializeFlags =
    kNuiInitializeFlagUsesDepthAndPlayerIndex |
    kNuiInitializeFlagUsesColor |
    kNuiInitializeFlagUsesSkeleton;
constexpr DWORD kNuiSkeletonTrackingAutomatic = 0;
constexpr DWORD kNuiSkeletonPollWaitMs = 40;
constexpr DWORD kNuiImageStreamFrameLimit = 2;
constexpr int kNuiImageTypeDepthAndPlayerIndex = 0;
constexpr int kNuiImageTypeColor = 1;
constexpr int kNuiImageTypeColorYuv = 2;
constexpr int kNuiImageTypeDepth = 3;
constexpr int kNuiImageTypeDepthAndPlayerIndexInColorSpace = 4;
constexpr int kNuiImageTypeDepthInColorSpace = 5;
constexpr int kNuiImageTypeColorInDepthSpace = 6;
constexpr int kNuiImageTypeDepthAndPlayerIndex80x60 = 7;
constexpr int kNuiImageTypeDepth80x60 = 8;
constexpr uint32_t kNuiImageTypeCount = 9;
constexpr int kNuiImageResolution320x240 = 1;
constexpr int kNuiImageResolution640x480 = 2;
constexpr int kNuiImageResolution80x60 = 0;
constexpr uint32_t kNuiDepthWidth = 320;
constexpr uint32_t kNuiDepthHeight = 240;
constexpr uint32_t kNuiColorWidth = 640;
constexpr uint32_t kNuiColorHeight = 480;
constexpr HRESULT kNuiFrameNoData = static_cast<HRESULT>(0x83010001u);

bool EnsureKinectComInitializedForCurrentThread() {
  struct ThreadComState {
    ThreadComState() {
      hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
      should_uninitialize = hr == S_OK || hr == S_FALSE;
    }
    ~ThreadComState() {
      if (should_uninitialize) {
        CoUninitialize();
      }
    }

    HRESULT hr = E_FAIL;
    bool should_uninitialize = false;
  };

  thread_local ThreadComState com_state;
  if (FAILED(com_state.hr) && com_state.hr != RPC_E_CHANGED_MODE) {
    static std::atomic_bool logged_failure{false};
    if (!logged_failure.exchange(true, std::memory_order_relaxed)) {
      XELOGW("Kinect v1 COM initialization failed: {:08X}",
             static_cast<uint32_t>(com_state.hr));
    }
    return false;
  }
  return true;
}

std::string GetNuiBackendName() {
  std::string backend = cvars::xam_nui_backend;
  std::transform(backend.begin(), backend.end(), backend.begin(),
                 [](unsigned char c) { return char(std::tolower(c)); });
  return backend;
}

bool IsRealKinectBackendRequested() {
  const std::string backend = GetNuiBackendName();
  if (backend == "fake" || backend == "stub" || backend == "disabled") {
    return false;
  }
  if (cvars::xam_nui_use_real_kinect) {
    return true;
  }
  return backend == "auto" || backend == "sdk" || backend == "real" ||
         backend == "kinect" || backend == "kinect_sdk";
}

struct NuiVector4 {
  FLOAT x;
  FLOAT y;
  FLOAT z;
  FLOAT w;
};
static_assert(sizeof(NuiVector4) == 16, "NUI Vector4 size must match SDK");

struct NuiSkeletonData {
  int32_t tracking_state;
  DWORD tracking_id;
  DWORD enrollment_index;
  DWORD user_index;
  NuiVector4 position;
  NuiVector4 skeleton_positions[kNuiSkeletonPositionCount];
  int32_t skeleton_position_tracking_state[kNuiSkeletonPositionCount];
  DWORD quality_flags;
};
static_assert(sizeof(NuiSkeletonData) == 436,
              "NUI skeleton data size must match SDK");

#pragma pack(push, 16)
struct NuiSkeletonFrame {
  LARGE_INTEGER timestamp;
  DWORD frame_number;
  DWORD flags;
  NuiVector4 floor_clip_plane;
  NuiVector4 normal_to_gravity;
  NuiSkeletonData skeleton_data[kNuiSkeletonCount];
};
#pragma pack(pop)
static_assert(sizeof(NuiSkeletonFrame) == 2664,
              "NUI skeleton frame size must match SDK");

struct NuiImageViewArea {
  int32_t digital_zoom;
  LONG center_x;
  LONG center_y;
};
static_assert(sizeof(NuiImageViewArea) == 12,
              "NUI image view area size must match SDK");

struct NuiLockedRect {
  INT pitch;
  int size;
  BYTE* bits;
};
static_assert(sizeof(NuiLockedRect) == 16,
              "NUI locked rect size must match SDK");

struct NuiSurfaceDesc {
  UINT width;
  UINT height;
};
static_assert(sizeof(NuiSurfaceDesc) == 8,
              "NUI surface desc size must match SDK");

struct NuiFrameTexture : public IUnknown {
  virtual int STDMETHODCALLTYPE BufferLen() = 0;
  virtual int STDMETHODCALLTYPE Pitch() = 0;
  virtual HRESULT STDMETHODCALLTYPE LockRect(UINT level, NuiLockedRect* rect,
                                             RECT* source_rect,
                                             DWORD flags) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetLevelDesc(UINT level,
                                                 NuiSurfaceDesc* desc) = 0;
  virtual HRESULT STDMETHODCALLTYPE UnlockRect(UINT level) = 0;
};

struct NuiDepthImagePixel {
  uint16_t player_index;
  uint16_t depth;
};
static_assert(sizeof(NuiDepthImagePixel) == 4,
              "NUI_DEPTH_IMAGE_PIXEL size must match Kinect SDK");

struct NuiImageFrame {
  LARGE_INTEGER timestamp;
  DWORD frame_number;
  int32_t image_type;
  int32_t resolution;
  NuiFrameTexture* frame_texture;
  DWORD frame_flags;
  NuiImageViewArea view_area;
};
static_assert(sizeof(NuiImageFrame) == 48,
              "NUI image frame size must match SDK on Win64");

struct INuiSensor {
  virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid,
                                                  void** object) = 0;
  virtual ULONG STDMETHODCALLTYPE AddRef() = 0;
  virtual ULONG STDMETHODCALLTYPE Release() = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiInitialize(DWORD flags) = 0;
  virtual void STDMETHODCALLTYPE NuiShutdown() = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiSetFrameEndEvent(HANDLE event,
                                                        DWORD flags) = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiImageStreamOpen(
      int image_type, int resolution, DWORD frame_flags, DWORD frame_limit,
      HANDLE next_frame_event, HANDLE* stream_handle) = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiImageStreamSetImageFrameFlags(
      HANDLE stream, DWORD flags) = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiImageStreamGetImageFrameFlags(
      HANDLE stream, DWORD* flags) = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiImageStreamGetNextFrame(
      HANDLE stream, DWORD wait_ms, void* image_frame) = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiImageStreamReleaseFrame(
      HANDLE stream, void* image_frame) = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiImageGetColorPixelCoordinatesFromDepthPixel(
      int color_resolution, const void* view_area, LONG depth_x, LONG depth_y,
      USHORT depth_value, LONG* color_x, LONG* color_y) = 0;
  virtual HRESULT STDMETHODCALLTYPE
  NuiImageGetColorPixelCoordinatesFromDepthPixelAtResolution(
      int color_resolution, int depth_resolution, const void* view_area,
      LONG depth_x, LONG depth_y, USHORT depth_value, LONG* color_x,
      LONG* color_y) = 0;
  virtual HRESULT STDMETHODCALLTYPE
  NuiImageGetColorPixelCoordinateFrameFromDepthPixelFrameAtResolution(
      int color_resolution, int depth_resolution, DWORD depth_value_count,
      USHORT* depth_values, DWORD color_coordinate_count,
      LONG* color_coordinates) = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiCameraElevationSetAngle(LONG angle) = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiCameraElevationGetAngle(LONG* angle) = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiSkeletonTrackingEnable(
      HANDLE next_frame_event, DWORD flags) = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiSkeletonTrackingDisable() = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiSkeletonSetTrackedSkeletons(
      DWORD* tracking_ids) = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiSkeletonGetNextFrame(
      DWORD wait_ms, NuiSkeletonFrame* skeleton_frame) = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiTransformSmooth(
      NuiSkeletonFrame* skeleton_frame, const void* smoothing_params) = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiGetAudioSource(void** audio_source) = 0;
  virtual int STDMETHODCALLTYPE NuiInstanceIndex() = 0;
  virtual wchar_t* STDMETHODCALLTYPE NuiDeviceConnectionId() = 0;
  virtual wchar_t* STDMETHODCALLTYPE NuiUniqueId() = 0;
  virtual wchar_t* STDMETHODCALLTYPE NuiAudioArrayId() = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiStatus() = 0;
  virtual DWORD STDMETHODCALLTYPE NuiInitializationFlags() = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiGetCoordinateMapper(void** mapping) = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiImageFrameGetDepthImagePixelFrameTexture(
      HANDLE stream, void* image_frame, BOOL* near_mode,
      NuiFrameTexture** frame_texture) = 0;
  // The following three slots are unused by Xenia but MUST be present so that
  // NuiAccelerometerGetCurrentReading lands on the correct vtable index for the
  // Kinect v1.8 SDK INuiSensor (see SDK NuiSensor.h). Pointer parameters are
  // left opaque (void**) because we never call these.
  virtual HRESULT STDMETHODCALLTYPE NuiGetColorCameraSettings(
      void** camera_settings) = 0;
  virtual BOOL STDMETHODCALLTYPE NuiGetForceInfraredEmitterOff() = 0;
  virtual HRESULT STDMETHODCALLTYPE NuiSetForceInfraredEmitterOff(
      BOOL force_off) = 0;
  // Reads the sensor's 3-axis accelerometer: the true gravity-down vector in
  // camera space (+Y up, +Z forward), ~(0,-1,0) when level, independent of the
  // tilt motor, the elevation lock, and whether a body is tracked. This is the
  // physical "which way is down" the real Xbox runtime uses to level skeletons.
  virtual HRESULT STDMETHODCALLTYPE NuiAccelerometerGetCurrentReading(
      NuiVector4* reading) = 0;
};

class KinectV1SkeletonRuntime {
 public:
  ~KinectV1SkeletonRuntime() { Shutdown(); }

  bool IsEnabled() const { return IsRealKinectBackendRequested(); }

  bool EnsureStarted() {
    if (!IsEnabled() || !EnsureKinectComInitializedForCurrentThread()) {
      return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (started_) {
      return true;
    }
    if (!LoadSdk()) {
      return false;
    }

    if (nui_get_sensor_count_) {
      int sensor_count = 0;
      const HRESULT count_hr = nui_get_sensor_count_(&sensor_count);
      XELOGI("Kinect v1 NuiGetSensorCount: hr={:08X} count={}",
             static_cast<uint32_t>(count_hr), sensor_count);
      sensor_count_ = SUCCEEDED(count_hr) ? sensor_count : 0;
    }
    if (sensor_count_ > 0) {
      OpenBestSensor();
    }

    const HRESULT init_hr =
        sensor_ ? sensor_->NuiInitialize(kNuiInitializeFlags)
                : nui_initialize_(kNuiInitializeFlags);
    if (FAILED(init_hr)) {
      LogFailureOnce("NuiInitialize failed", init_hr);
      ReleaseSensor();
      return false;
    }

    started_ = true;
    XELOGI("Kinect v1 SDK initialized for XAM NUI skeleton/depth/color tracking");
    EnableSkeletonTracking(DefaultSkeletonTrackingFlags());
    OpenDepthStream();
    OpenColorStream();
    PollSkeletonFrameOnce(static_cast<DWORD>(
        std::max<int32_t>(0, cvars::xam_nui_initial_skeleton_wait_ms)));
    StartSkeletonPoller();
    StartColorPoller();
    return true;
  }

  bool IsReady() { return EnsureStarted(); }

  void Shutdown() {
    StopSkeletonPoller();
    StopColorPoller();
    StopDc3DispatchThread();
    StopTiltDispatchThread();
    EnsureKinectComInitializedForCurrentThread();
    std::lock_guard<std::mutex> lock(mutex_);
    if (sensor_) {
      sensor_->NuiSkeletonTrackingDisable();
    }
    if (started_) {
      if (sensor_) {
        sensor_->NuiShutdown();
      } else if (nui_shutdown_) {
        nui_shutdown_();
      }
      XELOGI("Kinect v1 SDK shut down");
    }
    ReleaseSensor();
    if (module_) {
      FreeLibrary(module_);
      module_ = nullptr;
    }
    depth_stream_ = nullptr;
    color_stream_ = nullptr;
    CloseKinectEvent();
    {
      std::lock_guard<std::mutex> event_lock(guest_event_mutex_);
      image_frame_event_handles_.clear();
      color_frame_event_handles_.clear();
    }
    skeleton_frame_event_handle_.store(0, std::memory_order_relaxed);
    frame_end_event_handle_.store(0, std::memory_order_relaxed);
    latest_skeleton_frame_valid_.store(false, std::memory_order_relaxed);
    latest_skeleton_frame_number_.store(0, std::memory_order_relaxed);
    last_delivered_skeleton_frame_.store(0, std::memory_order_relaxed);
    tracked_skeleton_count_.store(0, std::memory_order_relaxed);
    best_tracking_id_.store(0, std::memory_order_relaxed);
    selected_host_player_index_.store(0, std::memory_order_relaxed);
    ResetDepthPlayerIndexRemap();
    skeleton_tracking_enabled_ = false;
    skeleton_tracking_flags_ = kNuiSkeletonTrackingAutomatic;
    last_driven_tracked_ids_[0] = 0;
    last_driven_tracked_ids_[1] = 0;
    guest_set_tracked_skeletons_.store(false, std::memory_order_relaxed);
    latest_depth_valid_.store(false, std::memory_order_relaxed);
    pending_depth_timestamp_.store(0, std::memory_order_relaxed);
    pending_depth_frame_number_.store(0, std::memory_order_relaxed);
    latest_depth_timestamp_.store(0, std::memory_order_relaxed);
    latest_depth_frame_number_.store(0, std::memory_order_relaxed);
    depth_guest_valid_.store(false, std::memory_order_relaxed);
    latest_depth_color_space_valid_.store(false, std::memory_order_relaxed);
    latest_color_valid_.store(false, std::memory_order_relaxed);
    pending_color_timestamp_.store(0, std::memory_order_relaxed);
    pending_color_frame_number_.store(0, std::memory_order_relaxed);
    latest_color_timestamp_.store(0, std::memory_order_relaxed);
    latest_color_frame_number_.store(0, std::memory_order_relaxed);
    color_guest_valid_.store(false, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> color_view_lock(color_view_mutex_);
      color_view_locked_ = false;
      color_view_center_x_ = 0.0f;
      color_view_center_y_ = 0.0f;
      color_view_last_real_head_time_ = {};
    }
    sensor_count_ = 0;
    started_ = false;
  }

  uint32_t TrackedSkeletonCount() const {
    return tracked_skeleton_count_.load(std::memory_order_relaxed);
  }

  uint32_t BestTrackingId() const {
    return best_tracking_id_.load(std::memory_order_relaxed);
  }

  uint32_t BestSkeletonIndex() const {
    return tracked_skeleton_count_.load(std::memory_order_relaxed)
               ? 0
               : 0xFFFFFFFFu;
  }

  uint32_t LatestSkeletonFrameNumber() const {
    return latest_skeleton_frame_number_.load(std::memory_order_relaxed);
  }

  void SetSkeletonFrameEventHandle(uint32_t event_handle) {
    const uint32_t previous =
        skeleton_frame_event_handle_.exchange(event_handle,
                                              std::memory_order_relaxed);
    if (previous != event_handle) {
      XELOGI("Kinect v1 skeleton frame event handle set: {:08X}",
             event_handle);
    }
  }

  void ResetSkeletonFrameEvent() {
    ResetGuestFrameEvent(
        skeleton_frame_event_handle_.load(std::memory_order_relaxed));
  }

  void RegisterImageFrameEventHandle(uint32_t event_handle) {
    RegisterGuestFrameEvent(image_frame_event_handles_, event_handle,
                            "image");
  }

  void SignalImageFrameEvent() {
    SignalGuestFrameEvents(image_frame_event_handles_);
  }

  void RegisterColorFrameEventHandle(uint32_t event_handle) {
    RegisterGuestFrameEvent(color_frame_event_handles_, event_handle,
                            "color");
  }

  void ConfigureFrameEndEvent(uint32_t event_handle, uint32_t flags) {
    const uint32_t previous =
        frame_end_event_handle_.exchange(event_handle,
                                         std::memory_order_relaxed);
    if (previous != event_handle) {
      XELOGI("Kinect v1 NUI frame-end event handle set: {:08X} flags={:08X}",
             event_handle, flags);
    }
  }

  void SignalColorFrameEvent() {
    SignalGuestFrameEvents(color_frame_event_handles_);
  }

  void SignalFrameEndEvent() {
    SignalGuestFrameEvent(
        frame_end_event_handle_.load(std::memory_order_relaxed));
  }

  void ResetFrameEvent(uint32_t event_handle) {
    ResetGuestFrameEvent(event_handle);
  }

  void ResetGuestEventRegistrations() {
    std::lock_guard<std::mutex> lock(guest_event_mutex_);
    image_frame_event_handles_.clear();
    color_frame_event_handles_.clear();
    skeleton_frame_event_handle_.store(0, std::memory_order_relaxed);
    frame_end_event_handle_.store(0, std::memory_order_relaxed);
  }

  bool CopyNextSkeletonFrame(X_NUI_SKELETON_FRAME* frame, uint32_t wait_ms) {
    if (!frame || !EnsureStarted()) {
      return false;
    }
    std::lock_guard<std::mutex> delivery_lock(skeleton_delivery_mutex_);
    const uint32_t last_delivered =
        last_delivered_skeleton_frame_.load(std::memory_order_relaxed);
    if (latest_skeleton_frame_number_.load(std::memory_order_acquire) ==
            last_delivered &&
        wait_ms != 0) {
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
      do {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
          break;
        }
        const auto remaining_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline -
                                                                  now)
                .count();
        PollSkeletonFrameOnce(static_cast<DWORD>(
            std::clamp<int64_t>(remaining_ms, 1, kNuiSkeletonPollWaitMs)));
      } while (latest_skeleton_frame_number_.load(std::memory_order_acquire) ==
                   last_delivered &&
               std::chrono::steady_clock::now() < deadline);
    }

    const uint32_t available =
        latest_skeleton_frame_number_.load(std::memory_order_acquire);
    if (!latest_skeleton_frame_valid_.load(std::memory_order_acquire) ||
        available == last_delivered) {
      return false;
    }
    {
      std::lock_guard<std::mutex> lock(latest_skeleton_frame_mutex_);
      *frame = latest_skeleton_frame_;
    }
    last_delivered_skeleton_frame_.store(available,
                                         std::memory_order_release);
    return true;
  }

  bool WaitForTrackedSkeletonFrame(uint32_t wait_ms) {
    if (!EnsureStarted()) {
      return false;
    }
    if (TrackedSkeletonCount() != 0 && LatestSkeletonFrameNumber() != 0) {
      return true;
    }
    if (wait_ms == 0) {
      return false;
    }

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
    do {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        break;
      }
      const auto remaining_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
              .count();
      PollSkeletonFrameOnce(static_cast<DWORD>(
          std::clamp<int64_t>(remaining_ms, 1, kNuiSkeletonPollWaitMs)));
      if (TrackedSkeletonCount() != 0 && LatestSkeletonFrameNumber() != 0) {
        return true;
      }
    } while (std::chrono::steady_clock::now() < deadline);
    return TrackedSkeletonCount() != 0 && LatestSkeletonFrameNumber() != 0;
  }

  HRESULT ConfigureSkeletonTracking(uint32_t event_handle, uint32_t flags) {
    if (!EnsureStarted()) {
      return E_FAIL;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const HRESULT hr = EnableSkeletonTracking(static_cast<DWORD>(flags));
    if (SUCCEEDED(hr)) {
      SetSkeletonFrameEventHandle(event_handle);
    }
    return hr;
  }

  HRESULT ConfigureTrackedSkeletons(uint32_t first_tracking_id,
                                    uint32_t second_tracking_id) {
    if (!EnsureStarted()) {
      return E_FAIL;
    }
    // The title is driving its own skeleton selection; stop the host
    // segmentation driver from fighting it.
    guest_set_tracked_skeletons_.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mutex_);
    return SetTrackedSkeletons(first_tracking_id, second_tracking_id);
  }

  // Record a JD4 _NUICAM opcode 0x05 frame-buffer registration. Only the type-4
  // 640x480x16bpp depth+player buffers (the silhouette, size >= 0x96000) are
  // captured; deduplicated and capped. The depth pump fills these each frame.
  void RegisterJd4CameraBuffer(uint32_t type, uint32_t phys_addr,
                               uint32_t size) {
    constexpr uint32_t kSilhouetteBytes = 640u * 480u * sizeof(uint16_t);
    if (!phys_addr || size < kSilhouetteBytes) {
      return;
    }
    // type 4 = NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX_IN_COLOR_SPACE (the mask);
    // type 2 = NUI_IMAGE_TYPE_COLOR_YUV (the camera the silhouette is masked
    // from). DC3's silhouette descriptor (ctx+0x80) references BOTH a colour
    // (type-2) and a depth (type-4) buffer, so both must be filled or the
    // silhouette has the mask but no colour -> renders empty.
    std::array<std::atomic<uint32_t>, kJd4MaxCameraBuffers>* buffers = nullptr;
    std::atomic<uint32_t>* count_ref = nullptr;
    const char* label = nullptr;
    if (type == 4u) {
      buffers = &jd4_depth_buffers_;
      count_ref = &jd4_depth_buffer_count_;
      label = "depth";
    } else if (type == 2u) {
      buffers = &jd4_color_buffers_;
      count_ref = &jd4_color_buffer_count_;
      label = "color";
    } else {
      return;
    }
    const uint32_t count = count_ref->load(std::memory_order_relaxed);
    for (uint32_t i = 0; i < count; ++i) {
      if ((*buffers)[i].load(std::memory_order_relaxed) == phys_addr) {
        return;  // already registered
      }
    }
    if (count >= kJd4MaxCameraBuffers) {
      return;
    }
    (*buffers)[count].store(phys_addr, std::memory_order_relaxed);
    count_ref->store(count + 1, std::memory_order_release);
    XELOGI("JD4 camera {} buffer #{} registered: phys={:08X} size={:X}", label,
           count + 1, phys_addr, size);
  }

  bool HasDepthFrame() const {
    return depth_guest_valid_.load(std::memory_order_acquire);
  }

  int64_t DepthTimestamp() const {
    return latest_depth_timestamp_.load(std::memory_order_relaxed);
  }

  uint32_t DepthFrameNumber() const {
    return latest_depth_frame_number_.load(std::memory_order_acquire);
  }

  uint32_t DepthGuestAddress() const {
    return HasDepthFrame() ? depth_guest_addr_.load(std::memory_order_relaxed)
                           : 0;
  }

  uint32_t Depth384GuestAddress() const {
    return HasDepthFrame()
               ? depth384_guest_addr_.load(std::memory_order_relaxed)
               : 0;
  }

  uint32_t DepthOnly384GuestAddress() const {
    return HasDepthFrame()
               ? depth_only384_guest_addr_.load(std::memory_order_relaxed)
               : 0;
  }

  uint32_t DepthColorSpace384GuestAddress() const {
    return HasDepthFrame()
               ? depth_color384_guest_addr_.load(std::memory_order_relaxed)
               : 0;
  }

  uint32_t DepthColorSpaceOnly384GuestAddress() const {
    return HasDepthFrame()
               ? depth_color_only384_guest_addr_.load(
                     std::memory_order_relaxed)
               : 0;
  }

  uint32_t MiniDepthGuestAddress() const {
    return HasDepthFrame()
               ? mini_depth_guest_addr_.load(std::memory_order_relaxed)
               : 0;
  }

  uint32_t Mini128DepthGuestAddress() const {
    return HasDepthFrame()
               ? mini128_depth_guest_addr_.load(std::memory_order_relaxed)
               : 0;
  }

  uint32_t Mini128DepthOnlyGuestAddress() const {
    return HasDepthFrame()
               ? mini128_depth_only_guest_addr_.load(std::memory_order_relaxed)
               : 0;
  }

  bool HasColorFrame() const {
    return color_guest_valid_.load(std::memory_order_acquire);
  }

  int64_t ColorTimestamp() const {
    return latest_color_timestamp_.load(std::memory_order_relaxed);
  }

  uint32_t ColorFrameNumber() const {
    return latest_color_frame_number_.load(std::memory_order_acquire);
  }

  uint32_t ColorGuestAddress() const {
    return HasColorFrame() ? color_guest_addr_.load(std::memory_order_relaxed)
                           : 0;
  }

  uint32_t ColorYuvGuestAddress() const {
    return HasColorFrame()
               ? color_yuv_guest_addr_.load(std::memory_order_relaxed)
               : 0;
  }

  bool CopyColorViewArea(uint32_t* digital_zoom, int32_t* center_x,
                         int32_t* center_y) {
    if (!digital_zoom || !center_x || !center_y) {
      return false;
    }
    *digital_zoom = kNuiImageDigitalZoom1x;
    *center_x = 0;
    *center_y = 0;
    // When Xenia is not pre-cropping the colour buffer, report a full-frame
    // view area (zoom 1x, centre 0,0) to match the uncropped image the title
    // gets. The title then crops the full colour itself from the skeleton head,
    // which is what makes the face-follow thumbnail track. Only synthesise a
    // head-following digital-zoom view area when we actually crop the buffer.
    if (!cvars::xam_nui_color_crop_to_view) {
      return false;
    }
    return BuildColorViewAreaFromSkeleton(digital_zoom, center_x, center_y);
  }

  HRESULT GetElevationAngle(int32_t* angle) {
    if (!angle) {
      return E_INVALIDARG;
    }
    // With the elevation locked, once a title has issued a (virtual) tilt
    // command, report that commanded angle back instead of the unmoved motor so
    // titles that poll for the move to finish see it complete. Before any
    // command we fall through to the real motor read so the actual mounted
    // angle seeds the extrapolated floor plane.
    if (cvars::xam_nui_lock_elevation &&
        elevation_virtual_valid_.load(std::memory_order_relaxed)) {
      *angle = last_elevation_angle_.load(std::memory_order_relaxed);
      return S_OK;
    }
    if (!EnsureStarted()) {
      return E_FAIL;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    LONG sdk_angle = 0;
    const HRESULT hr =
        sensor_ ? sensor_->NuiCameraElevationGetAngle(&sdk_angle)
                : nui_camera_elevation_get_angle_
                      ? nui_camera_elevation_get_angle_(&sdk_angle)
                      : E_NOTIMPL;
    if (SUCCEEDED(hr)) {
      *angle = sdk_angle;
    }
    return hr;
  }

  // Reads the physical gravity-up unit vector from the sensor accelerometer
  // (camera space, +Y up). Unlike the tilt motor angle, this reflects the real
  // mounted orientation regardless of the elevation lock or commanded angle, so
  // it tracks the sensor being physically re-aimed. Only available through the
  // COM sensor interface; the legacy C-export fallback has no accelerometer.
  bool ReadAccelerometerUp(float out_up[3]) {
    if (!cvars::xam_nui_gravity_from_accelerometer) {
      return false;
    }
    if (!EnsureStarted()) {
      return false;
    }
    NuiVector4 reading = {0.0f, 0.0f, 0.0f, 0.0f};
    HRESULT hr = E_FAIL;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!sensor_) {
        if (!accel_diag_logged_.exchange(true, std::memory_order_relaxed)) {
          XELOGW(
              "Kinect accelerometer unavailable: no COM sensor interface "
              "(legacy C-export path) -> gravity falls back to floor plane");
        }
        return false;
      }
      hr = sensor_->NuiAccelerometerGetCurrentReading(&reading);
    }
    if (FAILED(hr)) {
      if (!accel_diag_logged_.exchange(true, std::memory_order_relaxed)) {
        XELOGW(
            "Kinect NuiAccelerometerGetCurrentReading failed: hr={:08X} -> "
            "gravity falls back to floor plane",
            static_cast<uint32_t>(hr));
      }
      return false;
    }
    // The reading points along gravity (down); the up vector is its negation.
    const float dx = reading.x, dy = reading.y, dz = reading.z;
    const float len = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (len < 0.5f || len > 2.0f) {
      // Magnitude should be ~1 g at rest; reject implausible / transient reads
      // (e.g. the sensor being bumped) so a jolt cannot corrupt the up vector.
      return false;
    }
    out_up[0] = -dx / len;
    out_up[1] = -dy / len;
    out_up[2] = -dz / len;
    if (!accel_diag_logged_.exchange(true, std::memory_order_relaxed)) {
      const float tilt =
          std::atan2(std::sqrt(out_up[0] * out_up[0] + out_up[2] * out_up[2]),
                     out_up[1]) *
          57.2957795f;
      XELOGI(
          "Kinect accelerometer online: raw=({:.3f},{:.3f},{:.3f}) "
          "up=({:.3f},{:.3f},{:.3f}) tilt={:.1f}deg (now driving "
          "vNormalToGravity for skeleton leveling)",
          dx, dy, dz, out_up[0], out_up[1], out_up[2], tilt);
    }
    return true;
  }

  HRESULT SetElevationAngle(int32_t angle) {
    if (angle < -27 || angle > 27) {
      return E_INVALIDARG;
    }
    if (!EnsureStarted()) {
      return E_FAIL;
    }
    // Keep the physical sensor in its mounted position. Titles' start-up
    // calibration tilts the camera down to scan the floor; that motion is
    // disruptive and, on the desktop SDK, leaves the camera stuck pointing at
    // the floor. When xam_nui_lock_elevation is set, never drive the motor --
    // accept the commanded angle virtually so the title's calibration still
    // completes (GetElevationAngle reports it back and the floor plane is
    // extrapolated at that angle) while the real sensor never moves.
    if (cvars::xam_nui_lock_elevation) {
      last_elevation_angle_.store(angle, std::memory_order_relaxed);
      elevation_virtual_valid_.store(true, std::memory_order_relaxed);
      if (cvars::xam_nui_tilt_complete_notify) {
        QueueTiltAngleNotification(angle);
      }
      return S_OK;
    }
    HRESULT hr = E_NOTIMPL;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      hr = sensor_ ? sensor_->NuiCameraElevationSetAngle(angle)
                   : nui_camera_elevation_set_angle_
                         ? nui_camera_elevation_set_angle_(angle)
                         : E_NOTIMPL;
    }
    if (SUCCEEDED(hr)) {
      // Track the commanded angle so the floor-plane extrapolation (which
      // emulates NUI_INITIALIZE_FLAG_EXTRAPOLATE_FLOOR_PLANE) can rebuild the
      // floor for the title's scan even before the poller re-reads the motor.
      last_elevation_angle_.store(angle, std::memory_order_relaxed);
      // The real console / known-working reference delivers no tilt callback
      // for the floor-scan calibration; the NUI runtime drives its own search
      // off the skeleton/floor-clip-plane stream. Keep the dispatch path behind
      // a cvar (default off) so SetElevationAngle just moves the motor.
      if (cvars::xam_nui_tilt_complete_notify) {
        QueueTiltAngleNotification(angle);
      }
    }
    return hr;
  }

  void SetTiltCallback(uint32_t callback, uint32_t context) {
    tilt_callback_context_.store(context, std::memory_order_relaxed);
    tilt_callback_.store(callback, std::memory_order_release);
    if (callback) {
      EnsureTiltDispatchThread();
    }
  }

 private:
  void RegisterGuestFrameEvent(std::vector<uint32_t>& event_handles,
                               uint32_t event_handle, const char* kind) {
    if (!event_handle) {
      return;
    }
    std::lock_guard<std::mutex> lock(guest_event_mutex_);
    if (std::find(event_handles.begin(), event_handles.end(), event_handle) !=
        event_handles.end()) {
      return;
    }
    constexpr size_t kMaximumGuestFrameEvents = 32;
    if (event_handles.size() >= kMaximumGuestFrameEvents) {
      event_handles.erase(event_handles.begin());
    }
    event_handles.push_back(event_handle);
    XELOGI("Kinect v1 NUI {} frame event registered: {:08X}", kind,
           event_handle);
  }

  void SignalGuestFrameEvent(uint32_t event_handle) {
    if (!event_handle || !kernel_state()) {
      return;
    }
    auto event =
        kernel_state()->object_table()->LookupObject<XEvent>(event_handle);
    if (event) {
      event->Set(0, false);
    }
  }

  void SignalGuestFrameEvents(const std::vector<uint32_t>& event_handles) {
    std::vector<uint32_t> handles;
    {
      std::lock_guard<std::mutex> lock(guest_event_mutex_);
      handles = event_handles;
    }
    for (uint32_t event_handle : handles) {
      SignalGuestFrameEvent(event_handle);
    }
  }

  void ResetGuestFrameEvent(uint32_t event_handle) {
    if (!event_handle || !kernel_state()) {
      return;
    }
    auto event =
        kernel_state()->object_table()->LookupObject<XEvent>(event_handle);
    if (event) {
      event->Reset();
    }
  }

  using NuiInitializeFn = HRESULT(WINAPI*)(DWORD);
  using NuiShutdownFn = VOID(WINAPI*)();
  using NuiGetSensorCountFn = HRESULT(WINAPI*)(int*);
  using NuiCreateSensorByIndexFn = HRESULT(WINAPI*)(int, INuiSensor**);
  using NuiImageStreamOpenFn =
      HRESULT(WINAPI*)(int, int, DWORD, DWORD, HANDLE, HANDLE*);
  using NuiImageStreamGetNextFrameFn =
      HRESULT(WINAPI*)(HANDLE, DWORD, const NuiImageFrame**);
  using NuiImageStreamReleaseFrameFn =
      HRESULT(WINAPI*)(HANDLE, const NuiImageFrame*);
  using NuiImageGetColorPixelCoordinatesFromDepthPixelFn =
      HRESULT(WINAPI*)(int, const void*, LONG, LONG, USHORT, LONG*, LONG*);
  using NuiImageGetColorPixelCoordinateFrameFromDepthPixelFrameAtResolutionFn =
      HRESULT(WINAPI*)(int, int, DWORD, USHORT*, DWORD, LONG*);
  using NuiSkeletonTrackingEnableFn = HRESULT(WINAPI*)(HANDLE, DWORD);
  using NuiSkeletonGetNextFrameFn = HRESULT(WINAPI*)(DWORD, NuiSkeletonFrame*);
  using NuiSkeletonSetTrackedSkeletonsFn = HRESULT(WINAPI*)(DWORD*);
  using NuiCameraElevationGetAngleFn = HRESULT(WINAPI*)(LONG*);
  using NuiCameraElevationSetAngleFn = HRESULT(WINAPI*)(LONG);

  template <typename T>
  T LoadProc(const char* name) {
    return reinterpret_cast<T>(GetProcAddress(module_, name));
  }

  bool LoadSdk() {
    if (module_) {
      return nui_initialize_ && nui_shutdown_ && HasSkeletonFrameReader();
    }
    module_ = LoadLibraryW(L"Kinect10.dll");
    if (!module_) {
      LogFailureOnce("Kinect10.dll was not found",
                     HRESULT_FROM_WIN32(GetLastError()));
      return false;
    }
    nui_initialize_ = LoadProc<NuiInitializeFn>("NuiInitialize");
    nui_shutdown_ = LoadProc<NuiShutdownFn>("NuiShutdown");
    nui_get_sensor_count_ = LoadProc<NuiGetSensorCountFn>("NuiGetSensorCount");
    nui_create_sensor_by_index_ =
        LoadProc<NuiCreateSensorByIndexFn>("NuiCreateSensorByIndex");
    nui_image_stream_open_ =
        LoadProc<NuiImageStreamOpenFn>("NuiImageStreamOpen");
    nui_image_stream_get_next_frame_ =
        LoadProc<NuiImageStreamGetNextFrameFn>("NuiImageStreamGetNextFrame");
    nui_image_stream_release_frame_ =
        LoadProc<NuiImageStreamReleaseFrameFn>("NuiImageStreamReleaseFrame");
    nui_image_get_color_pixel_coordinates_from_depth_pixel_ =
        LoadProc<NuiImageGetColorPixelCoordinatesFromDepthPixelFn>(
            "NuiImageGetColorPixelCoordinatesFromDepthPixel");
    nui_image_get_color_pixel_coordinate_frame_from_depth_pixel_frame_at_resolution_ =
        LoadProc<
            NuiImageGetColorPixelCoordinateFrameFromDepthPixelFrameAtResolutionFn>(
            "NuiImageGetColorPixelCoordinateFrameFromDepthPixelFrameAtResolution");
    nui_skeleton_tracking_enable_ =
        LoadProc<NuiSkeletonTrackingEnableFn>("NuiSkeletonTrackingEnable");
    nui_skeleton_get_next_frame_ =
        LoadProc<NuiSkeletonGetNextFrameFn>("NuiSkeletonGetNextFrame");
    nui_skeleton_set_tracked_skeletons_ =
        LoadProc<NuiSkeletonSetTrackedSkeletonsFn>(
            "NuiSkeletonSetTrackedSkeletons");
    nui_camera_elevation_get_angle_ =
        LoadProc<NuiCameraElevationGetAngleFn>("NuiCameraElevationGetAngle");
    nui_camera_elevation_set_angle_ =
        LoadProc<NuiCameraElevationSetAngleFn>("NuiCameraElevationSetAngle");
    if (!nui_initialize_ || !nui_shutdown_ || !HasSkeletonFrameReader()) {
      LogFailureOnce("Kinect10.dll is missing required NUI skeleton exports",
                     E_FAIL);
      return false;
    }
    return true;
  }

  bool OpenBestSensor() {
    if (!nui_create_sensor_by_index_ || sensor_count_ <= 0) {
      return false;
    }
    for (int i = 0; i < sensor_count_; ++i) {
      INuiSensor* candidate = nullptr;
      const HRESULT create_hr = nui_create_sensor_by_index_(i, &candidate);
      if (FAILED(create_hr) || !candidate) {
        continue;
      }
      const HRESULT status_hr = candidate->NuiStatus();
      XELOGI("Kinect v1 sensor[{}] status={:08X}", i,
             static_cast<uint32_t>(status_hr));
      if (SUCCEEDED(status_hr)) {
        sensor_ = candidate;
        return true;
      }
      candidate->Release();
    }
    return false;
  }

  void ReleaseSensor() {
    if (sensor_) {
      sensor_->Release();
      sensor_ = nullptr;
    }
  }

  bool HasSkeletonFrameReader() const {
    return sensor_ || nui_skeleton_get_next_frame_;
  }

  DWORD DefaultSkeletonTrackingFlags() const {
    const int32_t configured_flags = cvars::xam_nui_skeleton_tracking_flags;
    DWORD flags = configured_flags >= 0 ? static_cast<DWORD>(configured_flags)
                                        : kNuiSkeletonTrackingAutomatic;
    // Host-driven body segmentation needs TITLE_SETS_TRACKED_SKELETONS so the
    // per-frame NuiSkeletonSetTrackedSkeletons selection actually takes effect
    // (in automatic mode the SDK ignores it). This mirrors the Kinect SDK
    // "Segmentation" sample, which enables this exact flag.
    if (cvars::xam_nui_drive_tracked_skeletons) {
      flags |= kNuiSkeletonTrackingFlagTitleSetsTrackedSkeletons;
    }
    return flags;
  }

  HRESULT EnableSkeletonTracking(DWORD tracking_flags) {
    if ((!sensor_ && !nui_skeleton_tracking_enable_) ||
        !HasSkeletonFrameReader()) {
      return E_FAIL;
    }
    const DWORD sdk_tracking_flags =
        tracking_flags & kNuiSkeletonTrackingValidFlags;
    if (sdk_tracking_flags != tracking_flags) {
      static std::atomic<uint32_t> sanitized_log_count{0};
      const uint32_t n =
          sanitized_log_count.fetch_add(1, std::memory_order_relaxed) + 1;
      if (n <= 8 || (n & (n - 1)) == 0) {
        XELOGW(
            "Kinect v1 skeleton tracking sanitized title flags {:08X} -> "
            "{:08X}",
            tracking_flags, sdk_tracking_flags);
      }
      tracking_flags = sdk_tracking_flags;
    }
    if (skeleton_tracking_enabled_) {
      if (skeleton_tracking_flags_ != tracking_flags) {
        static std::atomic<uint32_t> reconfigure_log_count{0};
        const uint32_t n =
            reconfigure_log_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n <= 8 || (n & (n - 1)) == 0) {
          XELOGI(
              "Kinect v1 skeleton tracking update {:08X} ignored; keeping "
              "active flags {:08X}",
              tracking_flags, skeleton_tracking_flags_);
        }
      }
      return S_OK;
    }
    if (!skeleton_event_) {
      skeleton_event_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    }
    const HRESULT hr =
        sensor_ ? sensor_->NuiSkeletonTrackingEnable(skeleton_event_,
                                                     tracking_flags)
                : nui_skeleton_tracking_enable_(skeleton_event_,
                                                tracking_flags);
    if (FAILED(hr)) {
      if (skeleton_tracking_enabled_) {
        static std::atomic<uint32_t> rejected_log_count{0};
        const uint32_t n =
            rejected_log_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n <= 8 || (n & (n - 1)) == 0) {
          XELOGW(
              "Kinect v1 skeleton tracking flags {:08X} rejected ({:08X}); "
              "keeping active flags {:08X}",
              tracking_flags, static_cast<uint32_t>(hr),
              skeleton_tracking_flags_);
        }
        return S_OK;
      }
      XELOGW("Kinect v1 skeleton tracking enable failed: {:08X}",
             static_cast<uint32_t>(hr));
    } else {
      skeleton_tracking_enabled_ = true;
      skeleton_tracking_flags_ = tracking_flags;
      XELOGI("Kinect v1 skeleton tracking enabled with flags {:08X}",
             tracking_flags);
    }
    return hr;
  }

  HRESULT SetTrackedSkeletons(uint32_t first_tracking_id,
                              uint32_t second_tracking_id = 0,
                              bool quiet = false) {
    if (!sensor_ && !nui_skeleton_set_tracked_skeletons_) {
      return E_FAIL;
    }
    DWORD ids[2] = {static_cast<DWORD>(first_tracking_id),
                    static_cast<DWORD>(second_tracking_id)};
    const HRESULT hr = sensor_
                           ? sensor_->NuiSkeletonSetTrackedSkeletons(ids)
                           : nui_skeleton_set_tracked_skeletons_(ids);
    if (SUCCEEDED(hr)) {
      if (!quiet) {
        XELOGI("Kinect v1 SetTrackedSkeletons({}, {})", first_tracking_id,
               second_tracking_id);
      }
    } else {
      static std::atomic_bool logged{false};
      if (!logged.exchange(true, std::memory_order_relaxed)) {
        XELOGW(
            "Kinect v1 NuiSkeletonSetTrackedSkeletons({}, {}) failed: {:08X}",
            first_tracking_id, second_tracking_id, static_cast<uint32_t>(hr));
      }
    }
    return hr;
  }

  // Mirror the Kinect SDK "Segmentation" sample: every skeleton frame, pick the
  // closest detected bodies (NUI detects bodies as position-only automatically;
  // including those here) and ask the SDK to fully track them. Full skeletal
  // tracking is what makes the depth+player stream carry the per-pixel player
  // index used by titles to cut out the body silhouette. Without this the SDK
  // leaves bodies position-only, player_pixels stays 0, and titles paint the
  // whole room instead of the dancer. Runs on the skeleton poller thread.
  void DriveTrackedSkeletons(const NuiSkeletonFrame& frame) {
    if (!cvars::xam_nui_drive_tracked_skeletons ||
        guest_set_tracked_skeletons_.load(std::memory_order_relaxed)) {
      return;
    }
    uint32_t closest_ids[2] = {0, 0};
    float closest_dist_sq[2] = {std::numeric_limits<float>::max(),
                                std::numeric_limits<float>::max()};
    for (uint32_t i = 0; i < kNuiSkeletonCount; ++i) {
      const auto& skeleton = frame.skeleton_data[i];
      if (skeleton.tracking_state == kNuiSkeletonNotTracked ||
          !skeleton.tracking_id) {
        continue;
      }
      const float x = skeleton.position.x;
      const float y = skeleton.position.y;
      const float z = skeleton.position.z;
      if (z <= 0.0f) {
        continue;
      }
      const float dist_sq = x * x + y * y + z * z;
      if (dist_sq < closest_dist_sq[0]) {
        closest_dist_sq[1] = closest_dist_sq[0];
        closest_ids[1] = closest_ids[0];
        closest_dist_sq[0] = dist_sq;
        closest_ids[0] = skeleton.tracking_id;
      } else if (dist_sq < closest_dist_sq[1]) {
        closest_dist_sq[1] = dist_sq;
        closest_ids[1] = skeleton.tracking_id;
      }
    }
    // Hysteresis: if the body we are already tracking is still present, keep it
    // even when another body is momentarily a touch closer, so the id (and the
    // joint solve) stay stable instead of ping-ponging between bodies.
    if (last_driven_tracked_ids_[0]) {
      for (uint32_t i = 0; i < kNuiSkeletonCount; ++i) {
        const auto& s = frame.skeleton_data[i];
        if (s.tracking_state != kNuiSkeletonNotTracked &&
            s.tracking_id == last_driven_tracked_ids_[0]) {
          closest_ids[1] = (closest_ids[0] != last_driven_tracked_ids_[0])
                               ? closest_ids[0]
                               : closest_ids[1];
          closest_ids[0] = last_driven_tracked_ids_[0];
          break;
        }
      }
    }
    // Momentary detection gap (no body this frame): re-assert the body we last
    // selected instead of clearing to "track nobody" (0,0), which would abort
    // the in-progress full-skeletal solve and make the SDK re-acquire the body
    // with a fresh id every gap (the tracked_count 1<->0 flicker that strips
    // the joints, starving the face-cam and dance scoring).
    if (closest_ids[0] == 0) {
      closest_ids[0] = last_driven_tracked_ids_[0];
      closest_ids[1] = last_driven_tracked_ids_[1];
    }
    if (closest_ids[0] == 0) {
      return;  // no body has been selected yet
    }
    const bool changed = closest_ids[0] != last_driven_tracked_ids_[0] ||
                         closest_ids[1] != last_driven_tracked_ids_[1];
    last_driven_tracked_ids_[0] = closest_ids[0];
    last_driven_tracked_ids_[1] = closest_ids[1];
    // Re-issue EVERY frame, exactly like the Kinect SDK "Segmentation" sample.
    // In TITLE_SETS mode the SDK keeps full-skeletal tracking alive only while
    // the title keeps asserting which body to track; asserting only on change
    // lets the SDK drop the body between calls (which is why tracked_count kept
    // flickering 1<->0 with a fresh id each time). Stay quiet in the log unless
    // the selection actually changed.
    SetTrackedSkeletons(closest_ids[0], closest_ids[1], !changed);
  }

  void OpenDepthStream() {
    if (depth_stream_ ||
        ((!sensor_ && !nui_image_stream_open_) || !HasImageFrameReader())) {
      return;
    }
    if (!depth_event_) {
      depth_event_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    }
    // Keep the host depth stream continuously producing depth and player data.
    const DWORD frame_flags = 0;
    const HRESULT hr =
        sensor_ ? sensor_->NuiImageStreamOpen(
                      kNuiImageTypeDepthAndPlayerIndex,
                      kNuiImageResolution320x240, frame_flags,
                      kNuiImageStreamFrameLimit, depth_event_, &depth_stream_)
                : nui_image_stream_open_(
                      kNuiImageTypeDepthAndPlayerIndex,
                      kNuiImageResolution320x240, frame_flags,
                      kNuiImageStreamFrameLimit, depth_event_, &depth_stream_);
    if (FAILED(hr)) {
      XELOGW("Kinect v1 depth stream open failed: {:08X}",
             static_cast<uint32_t>(hr));
      depth_stream_ = nullptr;
    } else {
      XELOGI("Kinect v1 depth+player stream opened: handle={:016X} flags={:08X}",
             static_cast<uint64_t>(reinterpret_cast<uintptr_t>(depth_stream_)),
             frame_flags);
    }
  }

  void OpenColorStream() {
    if (color_stream_ ||
        ((!sensor_ && !nui_image_stream_open_) || !HasImageFrameReader())) {
      return;
    }
    if (!color_event_) {
      color_event_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    }
    // Keep the host color stream continuously producing frames.
    const DWORD frame_flags = 0;
    const HRESULT hr =
        sensor_ ? sensor_->NuiImageStreamOpen(
                      kNuiImageTypeColor, kNuiImageResolution640x480,
                      frame_flags, kNuiImageStreamFrameLimit, color_event_,
                      &color_stream_)
                : nui_image_stream_open_(
                      kNuiImageTypeColor, kNuiImageResolution640x480,
                      frame_flags, kNuiImageStreamFrameLimit, color_event_,
                      &color_stream_);
    if (FAILED(hr)) {
      XELOGW("Kinect v1 color stream open failed: {:08X}",
             static_cast<uint32_t>(hr));
      color_stream_ = nullptr;
    } else {
      XELOGI("Kinect v1 color stream opened: handle={:016X} flags={:08X}",
             static_cast<uint64_t>(reinterpret_cast<uintptr_t>(color_stream_)),
             frame_flags);
    }
  }

  HRESULT ReadSkeletonFrame(DWORD wait_ms, NuiSkeletonFrame* frame) {
    return sensor_ ? sensor_->NuiSkeletonGetNextFrame(wait_ms, frame)
                   : nui_skeleton_get_next_frame_(wait_ms, frame);
  }

  void ResetDepthPlayerIndexRemap() {
    for (auto& mapped : depth_player_index_remap_) {
      mapped.store(0, std::memory_order_relaxed);
    }
    depth_single_guest_player_index_.store(0, std::memory_order_relaxed);
  }

  void PublishDepthPlayerIndexRemap(const NuiSkeletonFrame& frame,
                                    uint32_t bound_tracking_id) {
    std::array<uint32_t, kNuiImagePlayerIndexMask + 1> remap{};
    uint32_t tracked_count = 0;
    uint32_t single_guest_player_index = 0;
    uint32_t next_guest_player_index = 2;
    // Xbox samples use SkeletonData slot + 1 as the segmentation index.
    // ConvertSkeletonFrameToGuest publishes the bound body in guest slot 0, so
    // remap host depth bits into that same guest skeleton slot space.
    auto map_skeleton = [&](const NuiSkeletonData& skeleton,
                            uint32_t host_player_index,
                            uint32_t guest_player_index) {
      if (skeleton.tracking_state != kNuiSkeletonTracked ||
          !guest_player_index ||
          guest_player_index > kNuiImagePlayerIndexMask) {
        return false;
      }
      if (!host_player_index || host_player_index > kNuiImagePlayerIndexMask) {
        return false;
      }
      remap[host_player_index] = guest_player_index;
      if (!single_guest_player_index) {
        single_guest_player_index = guest_player_index;
      }
      ++tracked_count;
      return true;
    };

    if (bound_tracking_id != kNuiInvalidTrackingId) {
      for (uint32_t i = 0; i < kNuiSkeletonCount; ++i) {
        const auto& skeleton = frame.skeleton_data[i];
        if (skeleton.tracking_id == bound_tracking_id &&
            map_skeleton(skeleton, i + 1, 1)) {
          break;
        }
      }
    }

    for (uint32_t i = 0; i < kNuiSkeletonCount; ++i) {
      const auto& skeleton = frame.skeleton_data[i];
      if (skeleton.tracking_state != kNuiSkeletonTracked) {
        continue;
      }
      if (bound_tracking_id != kNuiInvalidTrackingId &&
          skeleton.tracking_id == bound_tracking_id) {
        continue;
      }
      while (next_guest_player_index <= kNuiImagePlayerIndexMask) {
        bool used = false;
        for (uint32_t mapped : remap) {
          used = used || mapped == next_guest_player_index;
        }
        if (!used) {
          break;
        }
        ++next_guest_player_index;
      }
      if (!map_skeleton(skeleton, i + 1, next_guest_player_index)) {
        continue;
      }
      ++next_guest_player_index;
    }
    for (uint32_t i = 0; i < remap.size(); ++i) {
      depth_player_index_remap_[i].store(remap[i], std::memory_order_relaxed);
    }
    depth_single_guest_player_index_.store(
        tracked_count == 1 ? single_guest_player_index : 0,
        std::memory_order_relaxed);
  }

  uint32_t RemapDepthPlayerIndex(uint32_t raw_player_index) const {
    if (!raw_player_index || raw_player_index > kNuiImagePlayerIndexMask) {
      return raw_player_index;
    }
    const uint32_t mapped =
        depth_player_index_remap_[raw_player_index].load(
            std::memory_order_relaxed);
    if (mapped) {
      return mapped;
    }
    const uint32_t single_guest_player_index =
        depth_single_guest_player_index_.load(std::memory_order_relaxed);
    return single_guest_player_index ? single_guest_player_index
                                     : raw_player_index;
  }

  bool GetDepthToColorCoordinates(const std::vector<uint16_t>& depth,
                                  std::vector<LONG>& coordinates) {
    const DWORD depth_count = kNuiDepthWidth * kNuiDepthHeight;
    if (depth.size() != depth_count) {
      return false;
    }
    if (coordinates.size() != depth_count * 2) {
      coordinates.assign(depth_count * 2, 0);
    }
    USHORT* depth_values =
        const_cast<USHORT*>(reinterpret_cast<const USHORT*>(depth.data()));
    const HRESULT hr =
        sensor_
            ? sensor_
                  ->NuiImageGetColorPixelCoordinateFrameFromDepthPixelFrameAtResolution(
                      kNuiImageResolution320x240, kNuiImageResolution320x240,
                      depth_count, depth_values, depth_count * 2,
                      coordinates.data())
            : nui_image_get_color_pixel_coordinate_frame_from_depth_pixel_frame_at_resolution_
                  ? nui_image_get_color_pixel_coordinate_frame_from_depth_pixel_frame_at_resolution_(
                        kNuiImageResolution320x240,
                        kNuiImageResolution320x240, depth_count, depth_values,
                        depth_count * 2, coordinates.data())
                  : E_NOTIMPL;
    if (FAILED(hr)) {
      static std::atomic_bool logged{false};
      if (!logged.exchange(true, std::memory_order_relaxed)) {
        XELOGW("Kinect v1 depth/color coordinate mapping unavailable: {:08X}",
               static_cast<uint32_t>(hr));
      }
      return false;
    }
    return true;
  }

  static bool ShouldReplaceMappedDepth(uint16_t current, uint16_t candidate) {
    const uint32_t candidate_depth = candidate >> 3;
    if (!candidate_depth) {
      return false;
    }
    const uint32_t current_depth = current >> 3;
    if (!current_depth) {
      return true;
    }
    const bool candidate_player = (candidate & kNuiImagePlayerIndexMask) != 0;
    const bool current_player = (current & kNuiImagePlayerIndexMask) != 0;
    if (candidate_player != current_player) {
      return candidate_player;
    }
    return candidate_depth < current_depth;
  }

  static bool StoreMappedDepthPixel(std::vector<uint16_t>& color_space_depth,
                                    int32_t x, int32_t y,
                                    uint16_t packed_depth) {
    if (x < 0 || x >= static_cast<int32_t>(kNuiDepthWidth) || y < 0 ||
        y >= static_cast<int32_t>(kNuiDepthHeight)) {
      return false;
    }
    auto& dst = color_space_depth[y * kNuiDepthWidth + x];
    if (!ShouldReplaceMappedDepth(dst, packed_depth)) {
      return false;
    }
    dst = packed_depth;
    return true;
  }

  uint32_t RebuildDepthColorSpaceFrame(const std::vector<uint16_t>& depth,
                                       std::vector<uint16_t>& color_space_depth,
                                       uint32_t* mapped_player_pixels) {
    if (mapped_player_pixels) {
      *mapped_player_pixels = 0;
    }
    const size_t depth_count = kNuiDepthWidth * kNuiDepthHeight;
    if (depth.size() != depth_count) {
      color_space_depth.clear();
      return 0;
    }
    color_space_depth.assign(depth_count, 0);
    if (!GetDepthToColorCoordinates(depth, depth_to_color_coordinates_)) {
      color_space_depth = depth;
      return 0;
    }

    uint32_t mapped_pixels = 0;
    uint32_t player_pixels = 0;
    // The depth-in-color-space (type 4) frame is what titles overlay on the
    // colour image to cut out the dancer (it is the player-mask source for the
    // Just Dance silhouette). When segmenting, only map player pixels so the
    // mask is the body, not the whole room.
    const bool segment = cvars::xam_nui_silhouette_segment;
    for (uint32_t y = 0; y < kNuiDepthHeight; ++y) {
      for (uint32_t x = 0; x < kNuiDepthWidth; ++x) {
        const uint32_t source_index = y * kNuiDepthWidth + x;
        const uint16_t packed = depth[source_index];
        if (!(packed >> 3)) {
          continue;
        }
        if (segment && !(packed & kNuiImagePlayerIndexMask)) {
          continue;  // background pixel -> excluded from the silhouette mask
        }
        const int32_t color_x = depth_to_color_coordinates_[source_index * 2];
        const int32_t color_y =
            depth_to_color_coordinates_[source_index * 2 + 1];
        const bool player = (packed & kNuiImagePlayerIndexMask) != 0;
        if (StoreMappedDepthPixel(color_space_depth, color_x, color_y,
                                  packed)) {
          ++mapped_pixels;
          if (player) {
            ++player_pixels;
          }
        }
      }
    }
    if (mapped_player_pixels) {
      *mapped_player_pixels = player_pixels;
    }
    return mapped_pixels;
  }

  static bool ProjectSkeletonPointToDepth(const X_NUI_VECTOR4& point,
                                          int32_t* depth_x, int32_t* depth_y,
                                          uint16_t* packed_depth) {
    const float x = point.x;
    const float y = point.y;
    const float z = point.z;
    if (!depth_x || !depth_y || !packed_depth || z <= 1e-4f) {
      return false;
    }
    // Matches the XDK NuiTransformSkeletonToDepthImage inline helper for the
    // 320x240 depth stream.
    const int32_t px = static_cast<int32_t>(
        160.0f + x * kNuiDepthFocalLength320x240 / z);
    const int32_t py = static_cast<int32_t>(
        120.0f - y * kNuiDepthFocalLength320x240 / z);
    const uint16_t depth_mm = static_cast<uint16_t>(z * 1000.0f);
    *depth_x = px;
    *depth_y = py;
    *packed_depth = static_cast<uint16_t>(depth_mm << 3);
    return true;
  }

  struct DepthPlayerStats {
    uint32_t depth_pixels = 0;
    uint32_t player_pixels = 0;
    uint32_t player_mask = 0;
  };

  static DepthPlayerStats CountDepthPlayerStats(
      const std::vector<uint16_t>& depth) {
    DepthPlayerStats stats;
    for (const uint16_t packed : depth) {
      if (packed >> 3) {
        ++stats.depth_pixels;
      }
      const uint32_t player = packed & kNuiImagePlayerIndexMask;
      if (player) {
        ++stats.player_pixels;
        stats.player_mask |= 1u << player;
      }
    }
    return stats;
  }

  bool GetColorCoordinatesForSkeletonPoint(const X_NUI_VECTOR4& point,
                                           int32_t* color_x,
                                           int32_t* color_y) {
    if (!color_x || !color_y || point.z <= 1e-4f) {
      return false;
    }

    int32_t depth_x = 0;
    int32_t depth_y = 0;
    uint16_t packed_depth = 0;
    if (!ProjectSkeletonPointToDepth(point, &depth_x, &depth_y,
                                     &packed_depth)) {
      return false;
    }
    HRESULT hr = E_NOTIMPL;
    if (EnsureKinectComInitializedForCurrentThread()) {
      LONG mapped_x = 0;
      LONG mapped_y = 0;
      hr = sensor_ ? sensor_->NuiImageGetColorPixelCoordinatesFromDepthPixel(
                         kNuiImageResolution640x480, nullptr, depth_x, depth_y,
                         packed_depth, &mapped_x, &mapped_y)
           : nui_image_get_color_pixel_coordinates_from_depth_pixel_
               ? nui_image_get_color_pixel_coordinates_from_depth_pixel_(
                     kNuiImageResolution640x480, nullptr, depth_x, depth_y,
                     packed_depth, &mapped_x, &mapped_y)
               : E_NOTIMPL;
      if (SUCCEEDED(hr)) {
        LogColorMapDiag(depth_x, depth_y, packed_depth, mapped_x, mapped_y,
                        true);
        *color_x = mapped_x;
        *color_y = mapped_y;
        return true;
      }
    }
    // Fallback when the SDK depth->colour registration is unavailable: the
    // 640x480 colour image is the 320x240 depth image scaled 2x.
    *color_x = depth_x * 2;
    *color_y = depth_y * 2;
    LogColorMapDiag(depth_x, depth_y, packed_depth, *color_x, *color_y, false);
    return true;
  }

  static void LogColorMapDiag(int32_t depth_x, int32_t depth_y,
                              uint16_t packed_depth, int32_t color_x,
                              int32_t color_y, bool sdk) {
    static std::atomic<uint32_t> n{0};
    const uint32_t i = n.fetch_add(1, std::memory_order_relaxed) + 1;
    if (i <= 8 || i % 120 == 0) {
      XELOGI(
          "Kinect v1 head->colour map #{}: src=sdk?{} depth=({}, {}) mm={} -> "
          "colour=({}, {})",
          i, sdk ? 1 : 0, depth_x, depth_y, packed_depth >> 3, color_x,
          color_y);
    }
  }

  bool BuildColorViewAreaFromSkeleton(uint32_t* digital_zoom,
                                      int32_t* center_x, int32_t* center_y) {
    if (!digital_zoom || !center_x || !center_y) {
      return false;
    }

    X_NUI_SKELETON_FRAME frame = {};
    const bool frame_valid =
        latest_skeleton_frame_valid_.load(std::memory_order_acquire);
    if (frame_valid) {
      std::lock_guard<std::mutex> lock(latest_skeleton_frame_mutex_);
      frame = latest_skeleton_frame_;
    }

    bool have_real_head = false;
    int32_t target_center_x = 0;
    int32_t target_center_y = 0;
    const uint32_t best_tracking_id =
        best_tracking_id_.load(std::memory_order_relaxed);
    for (uint32_t i = 0; frame_valid && i < kNuiSkeletonCount; ++i) {
      const auto& skeleton = frame.skeleton_data[i];
      if (skeleton.tracking_state != kNuiSkeletonTracked) {
        continue;
      }
      if (best_tracking_id && skeleton.tracking_id != best_tracking_id) {
        continue;
      }
      constexpr uint32_t kHeadJoint = 3;
      constexpr int32_t kInferred = 1;  // NUI_SKELETON_POSITION_INFERRED
      // The SDK's own position-only continuity joint is marked inferred and is
      // unreliable (its body-centre estimate can be over half a frame from the
      // real head), so by default only snap the face crop to a TRACKED head.
      // But when xam_nui_synthesize_head_for_facecam is on, an inferred head is
      // our deliberate synthesized estimate (body centre + fixed offset) that
      // exists precisely to keep the crop following through the position-only
      // frames the real Kinect spends most of its time in. Accept it then, or
      // the host crop disables itself (view-area valid=0) >90% of the time and
      // the colour feed never zooms onto the player.
      const int32_t head_state =
          skeleton.skeleton_position_tracking_state[kHeadJoint];
      const bool head_usable =
          head_state == kNuiSkeletonTracked ||
          (cvars::xam_nui_synthesize_head_for_facecam &&
           head_state == kInferred);
      if (!head_usable) {
        continue;
      }
      int32_t head_color_x = 0;
      int32_t head_color_y = 0;
      if (!GetColorCoordinatesForSkeletonPoint(
              skeleton.skeleton_positions[kHeadJoint], &head_color_x,
              &head_color_y)) {
        continue;
      }
      // The skeleton HEAD joint projects to the top of the head, which sits
      // high in the colour frame; a tight crop centred there clamps to the top
      // edge and shows above the face. Bias the centre down (and allow X trim)
      // so the crop frames the face. Tunable per camera height/tilt via cvars.
      target_center_x = std::clamp<int32_t>(
          head_color_x - 320 + cvars::xam_nui_color_view_x_offset, -319, 319);
      target_center_y = std::clamp<int32_t>(
          head_color_y - 240 + cvars::xam_nui_color_view_y_offset, -239, 239);
      have_real_head = true;
      break;
    }

    constexpr auto kHeadLockHold = std::chrono::milliseconds(700);
    constexpr float kHeadJitterDeadZonePixels = 6.0f;
    constexpr float kHeadFollowSmoothing = 0.25f;
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> color_view_lock(color_view_mutex_);
    if (have_real_head) {
      if (!color_view_locked_) {
        color_view_center_x_ = static_cast<float>(target_center_x);
        color_view_center_y_ = static_cast<float>(target_center_y);
        color_view_locked_ = true;
      } else {
        const auto follow_axis = [=](float current, float target) {
          const float delta = target - current;
          return std::abs(delta) <= kHeadJitterDeadZonePixels
                     ? current
                     : current + delta * kHeadFollowSmoothing;
        };
        color_view_center_x_ = follow_axis(color_view_center_x_,
                                           static_cast<float>(target_center_x));
        color_view_center_y_ = follow_axis(color_view_center_y_,
                                           static_cast<float>(target_center_y));
      }
      color_view_last_real_head_time_ = now;
    } else if (!color_view_locked_ ||
               now - color_view_last_real_head_time_ > kHeadLockHold) {
      color_view_locked_ = false;
      return false;
    }

    *digital_zoom = kNuiImageDigitalZoom2x;
    *center_x = std::clamp<int32_t>(
        static_cast<int32_t>(std::lround(color_view_center_x_)), -319, 319);
    *center_y = std::clamp<int32_t>(
        static_cast<int32_t>(std::lround(color_view_center_y_)), -239, 239);
    return true;
  }

  bool HasImageFrameReader() const {
    return sensor_ ||
           (nui_image_stream_get_next_frame_ &&
            nui_image_stream_release_frame_);
  }

  HRESULT ReadImageFrame(HANDLE stream, DWORD wait_ms, NuiImageFrame* storage,
                         const NuiImageFrame** frame) {
    if (!frame) {
      return E_POINTER;
    }
    *frame = nullptr;
    if (sensor_) {
      const HRESULT hr =
          sensor_->NuiImageStreamGetNextFrame(stream, wait_ms, storage);
      if (SUCCEEDED(hr)) {
        *frame = storage;
      }
      return hr;
    }
    return nui_image_stream_get_next_frame_(stream, wait_ms, frame);
  }

  void ReleaseImageFrame(HANDLE stream, const NuiImageFrame* frame) {
    if (!frame) {
      return;
    }
    if (sensor_) {
      sensor_->NuiImageStreamReleaseFrame(stream,
                                          const_cast<NuiImageFrame*>(frame));
    } else if (nui_image_stream_release_frame_) {
      nui_image_stream_release_frame_(stream, frame);
    }
  }

  bool PollDepthFrameOnce(DWORD wait_ms) {
    if (!depth_stream_ || !HasImageFrameReader() ||
        !EnsureKinectComInitializedForCurrentThread()) {
      return false;
    }

    NuiImageFrame frame_storage = {};
    const NuiImageFrame* frame = nullptr;
    const HRESULT hr =
        ReadImageFrame(depth_stream_, wait_ms, &frame_storage, &frame);
    if (hr == S_FALSE || hr == kNuiFrameNoData) {
      return false;
    }
    if (FAILED(hr)) {
      static std::atomic<uint32_t> failure_count{0};
      const uint32_t n =
          failure_count.fetch_add(1, std::memory_order_relaxed) + 1;
      if (n <= 8 || (n & (n - 1)) == 0) {
        XELOGW("Kinect v1 depth NuiImageStreamGetNextFrame failed #{}: {:08X}",
               n, static_cast<uint32_t>(hr));
      }
      return false;
    }
    if (!frame || !frame->frame_texture) {
      ReleaseImageFrame(depth_stream_, frame);
      return false;
    }
    const int64_t frame_timestamp = frame->timestamp.QuadPart;
    const uint32_t frame_number = frame->frame_number;

    NuiLockedRect rect = {};
    NuiFrameTexture* locked_texture = frame->frame_texture;
    NuiFrameTexture* depth_pixel_texture = nullptr;
    BOOL near_mode = FALSE;
    bool depth_pixel_frame = false;
    HRESULT lock_hr = E_FAIL;

    if (sensor_) {
      const HRESULT pixel_hr =
          sensor_->NuiImageFrameGetDepthImagePixelFrameTexture(
              depth_stream_, const_cast<NuiImageFrame*>(frame), &near_mode,
              &depth_pixel_texture);
      if (SUCCEEDED(pixel_hr) && depth_pixel_texture) {
        lock_hr = depth_pixel_texture->LockRect(0, &rect, nullptr, 0);
        if (SUCCEEDED(lock_hr) && rect.bits &&
            rect.pitch >=
                static_cast<INT>(kNuiDepthWidth * sizeof(NuiDepthImagePixel))) {
          locked_texture = depth_pixel_texture;
          depth_pixel_frame = true;
        } else {
          if (SUCCEEDED(lock_hr)) {
            depth_pixel_texture->UnlockRect(0);
          }
          depth_pixel_texture->Release();
          depth_pixel_texture = nullptr;
          rect = {};
        }
      }
    }

    if (!depth_pixel_frame) {
      lock_hr = frame->frame_texture->LockRect(0, &rect, nullptr, 0);
    }
    if (FAILED(lock_hr) || !rect.bits ||
        rect.pitch < static_cast<INT>(
                         kNuiDepthWidth *
                         (depth_pixel_frame ? sizeof(NuiDepthImagePixel)
                                            : sizeof(uint16_t)))) {
      if (depth_pixel_texture) {
        depth_pixel_texture->Release();
      }
      ReleaseImageFrame(depth_stream_, frame);
      return false;
    }

    uint32_t player_pixels = 0;
    uint32_t remapped_player_pixels = 0;
    uint32_t color_space_mapped_pixels = 0;
    uint32_t color_space_player_pixels = 0;
    uint32_t raw_player_mask = 0;
    uint32_t guest_player_mask = 0;
    uint32_t depth_pixels = 0;
    uint16_t center_depth_mm = 0;
    {
      std::lock_guard<std::mutex> lock(latest_depth_mutex_);
      if (latest_depth_.size() != kNuiDepthWidth * kNuiDepthHeight) {
        latest_depth_.assign(kNuiDepthWidth * kNuiDepthHeight, 0);
      }
      for (uint32_t y = 0; y < kNuiDepthHeight; ++y) {
        auto* dst = &latest_depth_[y * kNuiDepthWidth];
        if (depth_pixel_frame) {
          const auto* src = reinterpret_cast<const NuiDepthImagePixel*>(
              rect.bits + static_cast<size_t>(y) * rect.pitch);
          for (uint32_t x = 0; x < kNuiDepthWidth; ++x) {
            const uint32_t raw_player =
                src[x].player_index & kNuiImagePlayerIndexMask;
            const uint32_t guest_player =
                RemapDepthPlayerIndex(raw_player) & kNuiImagePlayerIndexMask;
            const uint16_t raw =
                static_cast<uint16_t>((src[x].depth << 3) | raw_player);
            const uint16_t packed = static_cast<uint16_t>(
                (raw & ~uint16_t(kNuiImagePlayerIndexMask)) | guest_player);
            dst[x] = packed;
            if (raw_player) {
              raw_player_mask |= 1u << raw_player;
            }
            if (guest_player) {
              guest_player_mask |= 1u << guest_player;
            }
            if (raw_player && raw_player != guest_player) {
              ++remapped_player_pixels;
            }
          }
        } else {
          const auto* src = reinterpret_cast<const uint16_t*>(
              rect.bits + static_cast<size_t>(y) * rect.pitch);
          for (uint32_t x = 0; x < kNuiDepthWidth; ++x) {
            const uint16_t raw = src[x];
            const uint32_t raw_player = raw & kNuiImagePlayerIndexMask;
            const uint32_t guest_player =
                RemapDepthPlayerIndex(raw_player) & kNuiImagePlayerIndexMask;
            const uint16_t packed = static_cast<uint16_t>(
                (raw & ~uint16_t(kNuiImagePlayerIndexMask)) | guest_player);
            dst[x] = packed;
            if (raw_player) {
              raw_player_mask |= 1u << raw_player;
            }
            if (guest_player) {
              guest_player_mask |= 1u << guest_player;
            }
            if (raw_player && raw_player != guest_player) {
              ++remapped_player_pixels;
            }
          }
        }
      }
      DepthPlayerStats stats = CountDepthPlayerStats(latest_depth_);
      depth_pixels = stats.depth_pixels;
      player_pixels = stats.player_pixels;
      guest_player_mask = stats.player_mask;

      color_space_mapped_pixels = RebuildDepthColorSpaceFrame(
          latest_depth_, latest_depth_color_space_, &color_space_player_pixels);
      latest_depth_color_space_valid_.store(
          !latest_depth_color_space_.empty(), std::memory_order_relaxed);
      center_depth_mm =
          latest_depth_[(kNuiDepthHeight / 2) * kNuiDepthWidth +
                        (kNuiDepthWidth / 2)] >>
          3;
      latest_depth_valid_.store(true, std::memory_order_relaxed);
    }

    pending_depth_timestamp_.store(frame_timestamp, std::memory_order_relaxed);
    pending_depth_frame_number_.store(frame_number, std::memory_order_relaxed);
    locked_texture->UnlockRect(0);
    if (depth_pixel_texture) {
      depth_pixel_texture->Release();
    }
    ReleaseImageFrame(depth_stream_, frame);
    const uint32_t n =
        depth_frame_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n <= 8 || (n & (n - 1)) == 0) {
      XELOGI(
          "Kinect v1 depth read #{}: frame={} pitch={} size={} "
          "depth_pixels={} player_pixels={} remapped={} color_space={} "
          "color_player={} raw_mask={:02X} guest_mask={:02X} center_mm={} "
          "pixel_frame={} near={}",
          n, frame_number, rect.pitch, rect.size, depth_pixels, player_pixels,
          remapped_player_pixels, color_space_mapped_pixels,
          color_space_player_pixels, raw_player_mask, guest_player_mask,
          center_depth_mm, depth_pixel_frame ? 1 : 0, near_mode ? 1 : 0);
    } else if (remapped_player_pixels || color_space_player_pixels) {
      static std::atomic<uint32_t> player_update_log_count{0};
      const uint32_t update_log =
          player_update_log_count.fetch_add(1, std::memory_order_relaxed) + 1;
      if (update_log <= 16 || (update_log & (update_log - 1)) == 0) {
        XELOGI(
            "Kinect v1 depth player update #{}: player_pixels={} remapped={} "
            "color_space={} color_player={} raw_mask={:02X} guest_mask={:02X}",
            n, player_pixels, remapped_player_pixels, color_space_mapped_pixels,
            color_space_player_pixels, raw_player_mask, guest_player_mask);
      }
    }
    return true;
  }

  // JD2019 GPU "fence" for the NUI depth path: gfxC+0x24AAC bit4 (mask 0x10). On
  // real hardware the GPU command-0x55 completion handler (0x82B62200) sets it;
  // Xenia never delivers that packet, so it stays 0 and the engine per-frame
  // commit gate (0x82B53F18, needs bit4 SET) never runs the gfxC committer. We
  // emulate it: SET on publish, CLEAR at the start of the next publish so the
  // produce-side gate (needs bit4 CLEAR) can alternate. gfxC+0x24AB0 is the gfxC
  // critsec MUTANT - do NOT touch it. Gated on gfxC != 0 and device+0x2988.
  void DriveJd2019GfxFence(xe::Memory* memory, bool set) {
    constexpr uint32_t kJdNuiDevice = 0x82FE3110u;
    auto* gfxc_p = memory->TranslateVirtual<uint8_t*>(0x82FF1C84u);
    if (!gfxc_p) {
      return;
    }
    const uint32_t gfxc = xe::load_and_swap<uint32_t>(gfxc_p);
    if (!gfxc) {
      return;
    }
    auto* present_p = memory->TranslateVirtual<uint8_t*>(kJdNuiDevice + 0x2988u);
    if (!present_p || !xe::load_and_swap<uint32_t>(present_p)) {
      return;
    }
    auto* aa8 = memory->TranslateVirtual<uint8_t*>(gfxc + 0x24AA8u);
    auto* aac = memory->TranslateVirtual<uint8_t*>(gfxc + 0x24AACu);
    if (!aa8 || !aac) {
      return;
    }
    const uint32_t bits = xe::load_and_swap<uint32_t>(aac);
    if (set) {
      xe::store_and_swap<uint32_t>(aa8, 2u);
      xe::store_and_swap<uint32_t>(aac, bits | 0x10u);
    } else {
      xe::store_and_swap<uint32_t>(aa8, 0u);
      xe::store_and_swap<uint32_t>(aac, bits & ~0x10u);
    }
  }

  // Deliver a JD2019 depth frame to the guest's camera dispatcher. JD2019's
  // per-stream producer (0x82B49590, which would issue cmd 0x1004 and fill
  // device+0xA0) is STATICALLY DEAD CODE in the linked image, so the title can
  // never self-drive its silhouette. Xenia synthesizes the transfer descriptor
  // directly into device+0xA0 (exactly what 0x82B49590 would have written) and
  // signals CONSUME (device+0x38) then PRODUCE (device+0x58). The guest
  // consume->commit chain then runs untouched, so the silhouette TextureDyn
  // creator (0x82B439B8) finally finds a ready gfxC entry. Self-guarded: only
  // acts for the JD2019 title family with the camera marked ready (device+0x80)
  // and the 640x480 + 384x240 depth buffers published; a safe no-op otherwise.
  void SignalJd2019CameraDepthEvent(xe::Memory* memory) {
    auto* state = kernel_state();
    if (!state || !memory) {
      return;
    }
    if (state->title_id() != 0x555308D9u) {
      // EXACT JD2019 title id only. The whole 0x555308xx "Just Dance" family
      // shares this id space (e.g. Just Dance 4 = 0x555308B5), but the device
      // struct @0x82FE3110 and the reversed dispatcher addresses below are
      // JD2019-specific. For any other Just Dance title that guest address is
      // unmapped, so TranslateVirtual hands back a pointer into uncommitted
      // guest space and the load below faults (host AV). Gate tightly.
      return;
    }
    constexpr uint32_t kJdNuiDevice = 0x82FE3110u;
    auto* ready_p = memory->TranslateVirtual<uint8_t*>(kJdNuiDevice + 0x80u);
    if (!ready_p || !xe::load_and_swap<uint32_t>(ready_p)) {
      return;  // camera init not done yet
    }
    const uint32_t depth384 =
        depth384_guest_addr_.load(std::memory_order_relaxed);  // 384x240/768
    const uint32_t depth640 =
        depth640_guest_addr_.load(std::memory_order_relaxed);  // 640x480/1280
    if (!depth384 || !depth640) {
      return;  // no real depth published yet
    }

    // Clear the previous frame's GPU fence (begin-of-submit).
    DriveJd2019GfxFence(memory, false);

    // Lazily allocate the persistent transfer descriptor (>=0x90 bytes).
    uint32_t desc = jd2019_transfer_desc_addr_.load(std::memory_order_relaxed);
    if (!desc) {
      desc = memory->SystemHeapAlloc(0x90u);
      if (!desc) {
        return;
      }
      if (auto* dz = memory->TranslateVirtual<uint8_t*>(desc)) {
        std::memset(dz, 0, 0x90u);
      }
      jd2019_transfer_desc_addr_.store(desc, std::memory_order_relaxed);
      XELOGI("JD2019 NUI transfer DESC allocated at {:08X}", desc);
    }
    auto* d = memory->TranslateVirtual<uint8_t*>(desc);
    if (!d) {
      return;
    }

    // Fill DESC big-endian (mirrors what producer 0x82B49590 writes for DEPTH):
    //  +0x00 flags 0xE4 (0x80 consumer gate, 0x20|0x40 NEW_FRAME, 0x04 always),
    //  +0x04 must be >= 0, +0x48/+0x58/+0x5C/+0x60 timestamps, +0x64 the guest
    //  depth buffer the consumer forwards (the converter reads it), +0x8C dst 0.
    const uint32_t t =
        jd2019_desc_timestamp_.fetch_add(1, std::memory_order_relaxed) + 1;
    xe::store_and_swap<uint32_t>(d + 0x00, 0x000000E4u);
    xe::store_and_swap<uint32_t>(d + 0x04, 0u);
    xe::store_and_swap<uint32_t>(d + 0x48, t);
    xe::store_and_swap<uint32_t>(d + 0x58, t);
    xe::store_and_swap<uint32_t>(d + 0x5C, t);
    xe::store_and_swap<uint32_t>(d + 0x60, t);
    xe::store_and_swap<uint32_t>(d + 0x64, depth640);  // 640x480/1280 converter
    xe::store_and_swap<uint32_t>(d + 0x8C, 0u);

    // Publish device+0xA0 = desc (the sole field consume reads).
    auto* a0 = memory->TranslateVirtual<uint8_t*>(kJdNuiDevice + 0xA0u);
    const uint32_t prev_a0 = a0 ? xe::load_and_swap<uint32_t>(a0) : 0u;
    if (a0) {
      xe::store_and_swap<uint32_t>(a0, desc);
    }

    // SET the GPU fence so the engine per-frame commit gate runs.
    DriveJd2019GfxFence(memory, true);

    // Signal CONSUME (device+0x38): runs the consume chain + ENQUEUE that fills
    // the active-stream ring (processed first, lower wait index).
    auto* ev38 = memory->TranslateVirtual<uint8_t*>(kJdNuiDevice + 0x38u);
    if (ev38 && xe::load_and_swap<uint32_t>(ev38)) {
      xboxkrnl::xeKeSetEvent(reinterpret_cast<X_KEVENT*>(ev38), 1, 0);
    }
    // Signal PRODUCE (device+0x58): runs the produce handler that ALLOCATES the
    // converter's working buffers from the just-enqueued ring entry. Without it
    // the converter 0x82B94210 faults on an unallocated buffer.
    auto* ev58 = memory->TranslateVirtual<uint8_t*>(kJdNuiDevice + 0x58u);
    if (ev58 && xe::load_and_swap<uint32_t>(ev58)) {
      xboxkrnl::xeKeSetEvent(reinterpret_cast<X_KEVENT*>(ev58), 1, 0);
    }

    static std::atomic<uint32_t> sig_log{0};
    const uint32_t sn = sig_log.fetch_add(1, std::memory_order_relaxed) + 1;
    if (sn <= 12 || (sn & (sn - 1)) == 0) {
      XELOGI(
          "JD2019 depth inject #{} desc={:08X} prev[+0xA0]={:08X} "
          "depth640={:08X} consumed_last={}",
          sn, desc, prev_a0, depth640, (prev_a0 != desc) ? 1 : 0);
    }
  }

  // Emulate the NUI camera "driver" DELIVERING a depth frame to DC3. PROVEN from
  // the XDK (XStudioApi.h): the camera service delivers a frame by CALLING the
  // title's registered stream callback. DC3 registers callback 0x829CE1C0 via
  // _NUICAM opcode 0x05 (req[5]); that callback writes the frame descriptor to
  // NUI-context+0xA0, KeSetEvents context+0x38, AND itself issues
  // XamXStudioRequest(0x1004) + reads the GPU fence -- i.e. the real per-frame
  // processing lives INSIDE the callback, so merely poking ctx+0xA0/+0x38 from
  // the host did nothing (DC3 calls 0x1004 zero times). The only faithful fix is
  // to actually CALL 0x829CE1C0 each frame. Guest functions must run on a guest
  // thread, so a dedicated XHostThread (guest context) is woken per depth frame
  // and Executes the callback with r3 = the NUI context (0x8311C8A0, the value
  // the callback recomputes for itself), r4 = 0 (status >= 0), r5 = 0x96000 (the
  // 640x480x16bpp frame size it validates). Opt-in (cvar) + title-gated +
  // self-guarded so it cannot disturb DC3's working tracking unless enabled.
  void EnsureDc3DispatchThread() {
    if (dc3_dispatch_thread_) {
      return;
    }
    dc3_dispatch_running_.store(true, std::memory_order_release);
    dc3_dispatch_thread_ = object_ref<XHostThread>(new XHostThread(
        kernel_state(), 128 * 1024, 0,
        [this]() {
          auto* thread = XThread::GetCurrentThread();
          while (dc3_dispatch_running_.load(std::memory_order_acquire)) {
            {
              std::unique_lock<std::mutex> lk(dc3_dispatch_mutex_);
              dc3_dispatch_cond_.wait(lk, [this] {
                return dc3_frame_pending_.load(std::memory_order_relaxed) ||
                       !dc3_dispatch_running_.load(std::memory_order_acquire);
              });
              if (!dc3_dispatch_running_.load(std::memory_order_acquire)) {
                break;
              }
              dc3_frame_pending_.store(0, std::memory_order_relaxed);
            }
            // Call DC3's _NUICAM type-4 depth callback as the driver would.
            // r3 = a SYNTHESIZED request object whose +0x10 points to a writable
            // 0x90 frame descriptor (the only field the callback dereferences
            // from r3 -- 829CE208 `lwz r30,0x10(r29)`). The callback recomputes
            // the global NUI context (0x8311C8A0) itself, so we don't pass it.
            const uint32_t r3 = dc3_callback_r3_.load(std::memory_order_acquire);
            if (!r3) {
              continue;
            }
            // The hardware driver pre-populates the frame descriptor's BUFFER
            // field before calling the callback; our zeroed descriptor left it 0
            // (dump showed desc[+64]=0), so the consumer had no depth to render.
            // Set desc+0x64 = the registered depth buffer (the callback does not
            // overwrite +0x64, so it persists through the call).
            {
              auto* mem0 = kernel_state()->memory();
              const uint32_t descp0 =
                  dc3_callback_desc_.load(std::memory_order_relaxed);
              const uint32_t buf0 =
                  jd4_depth_buffers_[0].load(std::memory_order_relaxed);
              if (descp0 && buf0) {
                if (auto* p =
                        mem0->TranslateVirtual<uint8_t*>(descp0 + 0x64u)) {
                  xe::store_and_swap<uint32_t>(p, buf0);
                }
              }
            }
            uint64_t args[] = {static_cast<uint64_t>(r3), 0ull, 0x96000ull};
            kernel_state()->processor()->Execute(thread->thread_state(),
                                                 0x829CE1C0u, args, 3);
            // Deliver the COLOUR frame too (type-2 callback 0x829CE568): the
            // silhouette = colour masked by depth, so both must arrive. Point its
            // descriptor +0x64 at the registered colour buffer (0xBDD30000).
            const uint32_t cr3 =
                dc3_color_r3_.load(std::memory_order_acquire);
            const uint32_t cbuf =
                jd4_color_buffers_[0].load(std::memory_order_relaxed);
            if (cr3 && cbuf) {
              auto* memc = kernel_state()->memory();
              const uint32_t cdescp =
                  dc3_color_desc_.load(std::memory_order_relaxed);
              if (cdescp) {
                if (auto* p =
                        memc->TranslateVirtual<uint8_t*>(cdescp + 0x64u)) {
                  xe::store_and_swap<uint32_t>(p, cbuf);
                }
              }
              uint64_t cargs[] = {static_cast<uint64_t>(cr3), 0ull, 0x96000ull};
              kernel_state()->processor()->Execute(thread->thread_state(),
                                                   0x829CE568u, cargs, 3);
            }
            static std::atomic<uint32_t> n{0};
            const uint32_t c = n.fetch_add(1, std::memory_order_relaxed) + 1;
            if (c <= 4 || c % 300 == 0) {
              // Dump what the callback produced so we can see what the downstream
              // consumer would read (descriptor fields + the linked ctx+0xA0).
              auto* mem = kernel_state()->memory();
              const uint32_t descp =
                  dc3_callback_desc_.load(std::memory_order_relaxed);
              auto rd = [&](uint32_t a) -> uint32_t {
                auto* p = mem->TranslateVirtual<uint8_t*>(a);
                return p ? xe::load_and_swap<uint32_t>(p) : 0xFFFFFFFFu;
              };
              XELOGI(
                  "DC3 callback #{}: desc[+00]={:08X} [+04]={:08X} [+58]={:08X} "
                  "[+5C]={:08X} [+60]={:08X} [+64]={:08X} ctx[+A0]={:08X} "
                  "ctx[+38evt]={:08X}",
                  c, rd(descp + 0x00), rd(descp + 0x04), rd(descp + 0x58),
                  rd(descp + 0x5C), rd(descp + 0x60), rd(descp + 0x64),
                  rd(0x8311C8A0u + 0xA0u), rd(0x8311C8A0u + 0x38u));
              // The consumer (thread 0x12) drains a lockfree queue at ctx+0x2FE8
              // and the producer (0x829CF0D8) pulls frame SLOTS from the array at
              // ctx+0x2FF0 indexed by ctx+0x3008. Dump that state to learn the
              // slot/frame-node structure (and whether DC3 bound a slot to our
              // buffer 0xBDF00000), so we can enqueue a real frame node.
              constexpr uint32_t kCtx = 0x8311C8A0u;
              XELOGI(
                  "DC3 queue/slots #{}: q[2FE8]={:08X} idx[3008]={:08X} "
                  "flag[2FFC]={:08X} slots[2FF0..]={:08X} {:08X} {:08X} {:08X}",
                  c, rd(kCtx + 0x2FE8), rd(kCtx + 0x3008), rd(kCtx + 0x2FFC),
                  rd(kCtx + 0x2FF0), rd(kCtx + 0x2FF4), rd(kCtx + 0x2FF8),
                  rd(kCtx + 0x2FFC));
              const uint32_t idx = rd(kCtx + 0x3008);
              const uint32_t slot =
                  (idx < 8u) ? rd(kCtx + 0x2FF0 + idx * 4u) : 0u;
              if (slot && slot != 0xFFFFFFFFu) {
                XELOGI(
                    "DC3 slot[idx={}]={:08X}: +0x1C={:08X} +0x34={:08X} "
                    "+0x3D10={:08X} +0x3D14={:08X} +0x3D18={:08X}",
                    idx, slot, rd(slot + 0x1C), rd(slot + 0x34),
                    rd(slot + 0x3D10), rd(slot + 0x3D14), rd(slot + 0x3D18));
              }
              // DECISIVE one-shot scan: does DC3 ever reference our depth buffer
              // (0xBDF00000) in its slot objects / NUI context? If found, DC3
              // reads our buffer (issue is downstream); if NEVER, the slot->buffer
              // link is missing (the precise fix).
              static std::atomic<uint32_t> scanned{0};
              uint32_t expz = 0;
              const uint32_t target =
                  jd4_depth_buffers_[0].load(std::memory_order_relaxed);
              if (target &&
                  scanned.compare_exchange_strong(expz, 1,
                                                  std::memory_order_relaxed)) {
                uint32_t hits = 0;
                for (uint32_t scan = slot ? slot : kCtx;
                     scan < (slot ? slot : kCtx) + 0x4000u && hits < 8;
                     scan += 4) {
                  if (rd(scan) == target) {
                    XELOGI("DC3 buffer {:08X} referenced @ {:08X} (slot+0x{:X})",
                           target, scan, scan - (slot ? slot : kCtx));
                    ++hits;
                  }
                }
                // also scan the context block 0x8311C8A0..+0x8000
                for (uint32_t scan = kCtx; scan < kCtx + 0x8000u && hits < 16;
                     scan += 4) {
                  if (rd(scan) == target) {
                    XELOGI("DC3 buffer {:08X} referenced @ {:08X} (ctx+0x{:X})",
                           target, scan, scan - kCtx);
                    ++hits;
                  }
                }
                XELOGI("DC3 buffer-ref scan done: target={:08X} hits={}", target,
                       hits);
                // Dump the 3 stream descriptors (ctx+0x80, +0xC0, +0x100; the
                // buffer sits at +0xC of each). Reveals type/dims/stride/format =
                // how DC3 interprets the buffer (and which is the silhouette).
                for (uint32_t d = 0; d < 3; ++d) {
                  const uint32_t base = kCtx + 0x80u + d * 0x40u;
                  XELOGI(
                      "DC3 streamdesc[{}] @ctx+0x{:X}: {:08X} {:08X} {:08X} "
                      "{:08X} | buf={:08X} {:08X} {:08X} {:08X} | {:08X} "
                      "{:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X}",
                      d, 0x80u + d * 0x40u, rd(base + 0x00), rd(base + 0x04),
                      rd(base + 0x08), rd(base + 0x0C), rd(base + 0x10),
                      rd(base + 0x14), rd(base + 0x18), rd(base + 0x1C),
                      rd(base + 0x20), rd(base + 0x24), rd(base + 0x28),
                      rd(base + 0x2C), rd(base + 0x30), rd(base + 0x34),
                      rd(base + 0x38), rd(base + 0x3C));
                }
              }
            }
          }
          return 0;
        },
        kernel_state()->GetSystemProcess()));
    dc3_dispatch_thread_->set_name("DC3 NUI Camera Dispatch");
    dc3_dispatch_thread_->Create();
  }

  void EnsureTiltDispatchThread() {
    if (tilt_dispatch_thread_ || !kernel_state()) {
      return;
    }
    tilt_dispatch_running_.store(true, std::memory_order_release);
    tilt_dispatch_thread_ = object_ref<XHostThread>(new XHostThread(
        kernel_state(), 128 * 1024, 0,
        [this]() {
          // The Detroit NUI runtime handles 0x2B00A as the physical tilt
          // movement completion and reads a single signed angle from the
          // payload. 0x2B00B is a separate auto-tilt request with a larger
          // payload, so sending it here leaves floor search stuck at -25.
          constexpr uint32_t kNuiTiltMoveCompleteNotification = 0x0002B00Au;
          auto* thread = XThread::GetCurrentThread();
          while (tilt_dispatch_running_.load(std::memory_order_acquire)) {
            uint32_t serial = 0;
            int32_t requested_angle = 0;
            {
              std::unique_lock<std::mutex> lock(tilt_dispatch_mutex_);
              tilt_dispatch_cond_.wait(lock, [this] {
                return tilt_notification_pending_.load(
                           std::memory_order_relaxed) ||
                       !tilt_dispatch_running_.load(std::memory_order_acquire);
              });
              if (!tilt_dispatch_running_.load(std::memory_order_acquire)) {
                break;
              }
              tilt_notification_pending_.store(false,
                                               std::memory_order_relaxed);
              serial =
                  tilt_command_serial_.load(std::memory_order_relaxed);
              requested_angle =
                  tilt_requested_angle_.load(std::memory_order_relaxed);
            }

            // XAM reports the completion after the physical motor settles.
            // Waiting here lets the statically linked NUI runtime continue its
            // floor-search state machine instead of leaving the camera at -25.
            int32_t completed_angle = requested_angle;
            int32_t previous_angle = std::numeric_limits<int32_t>::min();
            uint32_t stable_samples = 0;
            for (uint32_t sample = 0; sample < 24; ++sample) {
              std::this_thread::sleep_for(std::chrono::milliseconds(250));
              if (!tilt_dispatch_running_.load(std::memory_order_acquire) ||
                  serial !=
                      tilt_command_serial_.load(std::memory_order_acquire)) {
                break;
              }
              int32_t sampled_angle = requested_angle;
              if (SUCCEEDED(GetElevationAngle(&sampled_angle))) {
                completed_angle = sampled_angle;
              }
              stable_samples =
                  completed_angle == previous_angle ? stable_samples + 1 : 0;
              previous_angle = completed_angle;
              if (stable_samples >= 2 &&
                  std::abs(completed_angle - requested_angle) <= 1) {
                break;
              }
            }
            if (!tilt_dispatch_running_.load(std::memory_order_acquire) ||
                serial != tilt_command_serial_.load(std::memory_order_acquire)) {
              continue;
            }

            const uint32_t callback =
                tilt_callback_.load(std::memory_order_acquire);
            if (!callback || !kernel_state()) {
              continue;
            }
            auto* memory = kernel_state()->memory();
            uint32_t data_address =
                tilt_callback_data_address_.load(std::memory_order_acquire);
            if (!data_address) {
              data_address = memory->SystemHeapAlloc(sizeof(uint32_t));
              if (!data_address) {
                continue;
              }
              tilt_callback_data_address_.store(data_address,
                                                std::memory_order_release);
            }
            if (auto* data =
                    memory->TranslateVirtual<uint8_t*>(data_address)) {
              xe::store_and_swap<uint32_t>(
                  data, static_cast<uint32_t>(completed_angle));
            } else {
              continue;
            }

            XELOGI(
                "Kinect v1 tilt move complete: requested={} completed={} "
                "callback={:08X}",
                requested_angle, completed_angle, callback);
            uint64_t args[] = {kNuiTiltMoveCompleteNotification, data_address};
            kernel_state()->processor()->Execute(thread->thread_state(),
                                                 callback, args,
                                                 xe::countof(args));
          }
          return 0;
        },
        kernel_state()->GetSystemProcess()));
    tilt_dispatch_thread_->set_name("NUI Tilt Dispatch");
    tilt_dispatch_thread_->Create();
  }

  void QueueTiltAngleNotification(int32_t angle) {
    if (!tilt_callback_.load(std::memory_order_acquire)) {
      return;
    }
    EnsureTiltDispatchThread();
    tilt_requested_angle_.store(angle, std::memory_order_relaxed);
    tilt_command_serial_.fetch_add(1, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(tilt_dispatch_mutex_);
      tilt_notification_pending_.store(true, std::memory_order_relaxed);
    }
    tilt_dispatch_cond_.notify_one();
  }

  void StopTiltDispatchThread() {
    if (!tilt_dispatch_thread_) {
      return;
    }
    tilt_dispatch_running_.store(false, std::memory_order_release);
    tilt_dispatch_cond_.notify_all();
    tilt_dispatch_thread_->Wait(0, 0, 0, nullptr);
  }

  void SignalDc3CameraDepthEvent(xe::Memory* memory) {
    auto* state = kernel_state();
    if (!state || !memory || !cvars::xam_nui_dc3_camera_dispatch ||
        state->title_id() != 0x373307D9u) {
      return;
    }
    if (!jd4_depth_buffer_count_.load(std::memory_order_acquire)) {
      return;  // DC3 hasn't registered its push buffers yet
    }
    // The consume event (context+0x38) must be an initialised X_KEVENT (its first
    // dword is non-zero) before DC3's dispatcher is ready to consume a frame.
    auto* ev38 = memory->TranslateVirtual<uint8_t*>(0x8311C8A0u + 0x38u);
    if (!ev38 || !xe::load_and_swap<uint32_t>(ev38)) {
      return;
    }
    // Lazily synthesize the callback's r3: a request object whose +0x10 points to
    // a zeroed 0x90 frame descriptor. 829CE1C0 writes the descriptor (flags,
    // timestamps), links it to the global NUI context+0xA0, fires the consume
    // event and pulls via XStudio 0x1004 -- all from a VALID descriptor, instead
    // of the garbage [0x8311C8A0+0x10] that faulted the previous attempt.
    if (!dc3_callback_r3_.load(std::memory_order_acquire)) {
      const uint32_t desc = memory->SystemHeapAlloc(0x100u);
      const uint32_t obj = memory->SystemHeapAlloc(0x100u);
      if (!desc || !obj) {
        return;
      }
      if (auto* dz = memory->TranslateVirtual<uint8_t*>(desc)) {
        std::memset(dz, 0, 0x100u);
      }
      if (auto* oz = memory->TranslateVirtual<uint8_t*>(obj)) {
        std::memset(oz, 0, 0x100u);
      }
      if (auto* p10 = memory->TranslateVirtual<uint8_t*>(obj + 0x10u)) {
        xe::store_and_swap<uint32_t>(p10, desc);
      }
      dc3_callback_desc_.store(desc, std::memory_order_relaxed);
      dc3_callback_r3_.store(obj, std::memory_order_release);
      XELOGI("DC3 synthetic callback r3={:08X} (desc={:08X})", obj, desc);
    }
    // Second synthetic r3 for the COLOUR (type-2) callback 0x829CE568 -- the
    // silhouette is the colour camera masked by the depth, so the colour frame
    // must also be delivered (its descriptor +0x64 = the registered colour
    // buffer 0xBDD30000).
    if (!dc3_color_r3_.load(std::memory_order_acquire)) {
      const uint32_t desc = memory->SystemHeapAlloc(0x100u);
      const uint32_t obj = memory->SystemHeapAlloc(0x100u);
      if (desc && obj) {
        if (auto* dz = memory->TranslateVirtual<uint8_t*>(desc)) {
          std::memset(dz, 0, 0x100u);
        }
        if (auto* oz = memory->TranslateVirtual<uint8_t*>(obj)) {
          std::memset(oz, 0, 0x100u);
        }
        if (auto* p10 = memory->TranslateVirtual<uint8_t*>(obj + 0x10u)) {
          xe::store_and_swap<uint32_t>(p10, desc);
        }
        dc3_color_desc_.store(desc, std::memory_order_relaxed);
        dc3_color_r3_.store(obj, std::memory_order_release);
        XELOGI("DC3 synthetic COLOR callback r3={:08X} (desc={:08X})", obj,
               desc);
      }
    }
    EnsureDc3DispatchThread();
    {
      std::lock_guard<std::mutex> lk(dc3_dispatch_mutex_);
      dc3_frame_pending_.store(1, std::memory_order_relaxed);
    }
    dc3_dispatch_cond_.notify_one();
  }

  void StopDc3DispatchThread() {
    if (!dc3_dispatch_thread_) {
      return;
    }
    // Signal the loop to exit; the thread observes it and returns. The object_ref
    // is released when the runtime is destroyed (don't drop it while it may still
    // be mid-Execute of a guest callback).
    dc3_dispatch_running_.store(false, std::memory_order_release);
    dc3_dispatch_cond_.notify_all();
  }

  void PublishDepthToGuest() {
    if (!latest_depth_valid_.load(std::memory_order_relaxed) ||
        !kernel_state()) {
      return;
    }
    auto* memory = kernel_state()->memory();
    const uint32_t flat_size = kNuiDepthWidth * kNuiDepthHeight *
                               static_cast<uint32_t>(sizeof(uint16_t));
    uint32_t flat_addr =
        depth_guest_addr_.load(std::memory_order_relaxed);
    if (!flat_addr) {
      flat_addr = memory->SystemHeapAlloc(flat_size);
      if (!flat_addr) {
        return;
      }
      depth_guest_addr_.store(flat_addr, std::memory_order_relaxed);
      XELOGI("Kinect v1 depth guest buffer 320x240 at {:08X}", flat_addr);
    }

    constexpr uint32_t stride_width = 384;
    constexpr uint32_t stride_size =
        stride_width * kNuiDepthHeight * sizeof(uint16_t);
    uint32_t stride_addr =
        depth384_guest_addr_.load(std::memory_order_relaxed);
    if (!stride_addr) {
      stride_addr = memory->SystemHeapAlloc(stride_size);
      if (!stride_addr) {
        return;
      }
      depth384_guest_addr_.store(stride_addr, std::memory_order_relaxed);
      XELOGI("Kinect v1 depth guest buffer 384x240 pitch 768 at {:08X}",
             stride_addr);
    }
    uint32_t depth_only_stride_addr =
        depth_only384_guest_addr_.load(std::memory_order_relaxed);
    if (!depth_only_stride_addr) {
      depth_only_stride_addr = memory->SystemHeapAlloc(stride_size);
      if (!depth_only_stride_addr) {
        return;
      }
      depth_only384_guest_addr_.store(depth_only_stride_addr,
                                      std::memory_order_relaxed);
      XELOGI(
          "Kinect v1 depth-only guest buffer 384x240 pitch 768 at {:08X}",
          depth_only_stride_addr);
    }
    uint32_t depth_color_stride_addr =
        depth_color384_guest_addr_.load(std::memory_order_relaxed);
    if (!depth_color_stride_addr) {
      depth_color_stride_addr = memory->SystemHeapAlloc(stride_size);
      if (depth_color_stride_addr) {
        depth_color384_guest_addr_.store(depth_color_stride_addr,
                                         std::memory_order_relaxed);
        XELOGI(
            "Kinect v1 depth color-space guest buffer 384x240 pitch 768 at "
            "{:08X}",
            depth_color_stride_addr);
      }
    }
    uint32_t depth_color_only_stride_addr =
        depth_color_only384_guest_addr_.load(std::memory_order_relaxed);
    if (!depth_color_only_stride_addr) {
      depth_color_only_stride_addr = memory->SystemHeapAlloc(stride_size);
      if (depth_color_only_stride_addr) {
        depth_color_only384_guest_addr_.store(depth_color_only_stride_addr,
                                              std::memory_order_relaxed);
        XELOGI(
            "Kinect v1 depth-only color-space guest buffer 384x240 pitch 768 "
            "at {:08X}",
            depth_color_only_stride_addr);
      }
    }

    constexpr uint32_t mini_width = 80;
    constexpr uint32_t mini_height = 60;
    constexpr uint32_t mini_stride_width = 128;
    uint32_t mini_addr =
        mini_depth_guest_addr_.load(std::memory_order_relaxed);
    if (!mini_addr) {
      mini_addr =
          memory->SystemHeapAlloc(mini_width * mini_height * sizeof(uint16_t));
      if (mini_addr) {
        mini_depth_guest_addr_.store(mini_addr, std::memory_order_relaxed);
        XELOGI("Kinect v1 mini depth guest buffer 80x60 at {:08X}",
               mini_addr);
      }
    }
    uint32_t mini128_addr =
        mini128_depth_guest_addr_.load(std::memory_order_relaxed);
    if (!mini128_addr) {
      mini128_addr = memory->SystemHeapAlloc(mini_stride_width * mini_height *
                                             sizeof(uint16_t));
      if (mini128_addr) {
        mini128_depth_guest_addr_.store(mini128_addr,
                                        std::memory_order_relaxed);
        XELOGI("Kinect v1 mini depth guest buffer 128x60 pitch 256 at {:08X}",
               mini128_addr);
      }
    }
    uint32_t mini128_depth_only_addr =
        mini128_depth_only_guest_addr_.load(std::memory_order_relaxed);
    if (!mini128_depth_only_addr) {
      mini128_depth_only_addr =
          memory->SystemHeapAlloc(mini_stride_width * mini_height *
                                  sizeof(uint16_t));
      if (mini128_depth_only_addr) {
        mini128_depth_only_guest_addr_.store(mini128_depth_only_addr,
                                             std::memory_order_relaxed);
        XELOGI(
            "Kinect v1 mini depth-only guest buffer 128x60 pitch 256 at {:08X}",
            mini128_depth_only_addr);
      }
    }

    auto* flat = memory->TranslateVirtual<uint8_t*>(flat_addr);
    auto* stride = memory->TranslateVirtual<uint8_t*>(stride_addr);
    auto* depth_only_stride =
        memory->TranslateVirtual<uint8_t*>(depth_only_stride_addr);
    auto* depth_color_stride =
        depth_color_stride_addr
            ? memory->TranslateVirtual<uint8_t*>(depth_color_stride_addr)
            : nullptr;
    auto* depth_color_only_stride =
        depth_color_only_stride_addr
            ? memory->TranslateVirtual<uint8_t*>(depth_color_only_stride_addr)
            : nullptr;
    auto* mini = mini_addr ? memory->TranslateVirtual<uint8_t*>(mini_addr)
                           : nullptr;
    auto* mini128 = mini128_addr
                        ? memory->TranslateVirtual<uint8_t*>(mini128_addr)
                        : nullptr;
    auto* mini128_depth_only =
        mini128_depth_only_addr
            ? memory->TranslateVirtual<uint8_t*>(mini128_depth_only_addr)
            : nullptr;
    if (!flat || !stride || !depth_only_stride) {
      return;
    }

    std::lock_guard<std::mutex> lock(latest_depth_mutex_);
    if (latest_depth_.size() != kNuiDepthWidth * kNuiDepthHeight) {
      return;
    }
    // DECISIVE PROBE (xam_nui_depth_test_pattern): overwrite the SOURCE depth so
    // a bright full-frame gradient propagates into EVERY downstream buffer at
    // once -- the CPU/LockRect buffers (flat/stride/depth_color384, which the
    // title's AtgNuiVisualization::SetDepthTexture LockRect-reads) AND the GPU
    // title surface. Every pixel carries depth + player-index bit 1, so it
    // survives segmentation. If the title then shows a gradient where the
    // silhouette belongs, it IS consuming our depth and only data/format
    // remains; if it stays black, the title reads a surface we are not feeding.
    if (cvars::xam_nui_depth_test_pattern) {
      for (uint32_t i = 0; i < latest_depth_.size(); ++i) {
        const uint32_t x = i % kNuiDepthWidth;
        const uint32_t y = i / kNuiDepthWidth;
        const uint16_t depth_mm = static_cast<uint16_t>(800 + ((x + y) & 0x3FF));
        latest_depth_[i] = static_cast<uint16_t>((depth_mm << 3) | 0x1);
      }
      latest_depth_color_space_ = latest_depth_;
      latest_depth_color_space_valid_.store(true, std::memory_order_relaxed);
    }
    for (uint32_t y = 0; y < kNuiDepthHeight; ++y) {
      uint8_t* flat_row = flat + y * kNuiDepthWidth * sizeof(uint16_t);
      uint8_t* stride_row = stride + y * stride_width * sizeof(uint16_t);
      uint8_t* depth_only_stride_row =
          depth_only_stride + y * stride_width * sizeof(uint16_t);
      std::memset(stride_row, 0, stride_width * sizeof(uint16_t));
      std::memset(depth_only_stride_row, 0,
                  stride_width * sizeof(uint16_t));
      const bool segment = cvars::xam_nui_silhouette_segment;
      for (uint32_t x = 0; x < kNuiDepthWidth; ++x) {
        const uint16_t raw = latest_depth_[y * kNuiDepthWidth + x];
        const uint16_t depth_only = raw & ~uint16_t(0x7);
        // The depth+player (type 0) buffer doubles as the player-mask source for
        // the silhouette. When segmenting, drop background pixels so the mask is
        // the body, not the whole room; the depth-only buffer keeps full depth.
        const uint16_t packed =
            (segment && !(raw & uint16_t(kNuiImagePlayerIndexMask))) ? 0 : raw;
        flat_row[x * 2] = static_cast<uint8_t>(packed >> 8);
        flat_row[x * 2 + 1] = static_cast<uint8_t>(packed & 0xFF);
        stride_row[x * 2] = static_cast<uint8_t>(packed >> 8);
        stride_row[x * 2 + 1] = static_cast<uint8_t>(packed & 0xFF);
        depth_only_stride_row[x * 2] =
            static_cast<uint8_t>(depth_only >> 8);
        depth_only_stride_row[x * 2 + 1] =
            static_cast<uint8_t>(depth_only & 0xFF);
      }
    }
    if (depth_color_stride && depth_color_only_stride &&
        latest_depth_color_space_valid_.load(std::memory_order_relaxed) &&
        latest_depth_color_space_.size() == kNuiDepthWidth * kNuiDepthHeight) {
      for (uint32_t y = 0; y < kNuiDepthHeight; ++y) {
        uint8_t* color_row =
            depth_color_stride + y * stride_width * sizeof(uint16_t);
        uint8_t* color_only_row =
            depth_color_only_stride + y * stride_width * sizeof(uint16_t);
        std::memset(color_row, 0, stride_width * sizeof(uint16_t));
        std::memset(color_only_row, 0, stride_width * sizeof(uint16_t));
        for (uint32_t x = 0; x < kNuiDepthWidth; ++x) {
          const uint16_t packed =
              latest_depth_color_space_[y * kNuiDepthWidth + x];
          const uint16_t depth_only = packed & ~uint16_t(0x7);
          color_row[x * 2] = static_cast<uint8_t>(packed >> 8);
          color_row[x * 2 + 1] = static_cast<uint8_t>(packed & 0xFF);
          color_only_row[x * 2] = static_cast<uint8_t>(depth_only >> 8);
          color_only_row[x * 2 + 1] =
              static_cast<uint8_t>(depth_only & 0xFF);
        }
      }
    }
    // The GPU samples THIS physical surface for the DC3/DC1 silhouette, NOT the
    // SystemHeapAlloc'd depth_color384 buffer above: the title's texture fetch
    // constant base (e.g. 0x302B9000) masks to a physical page (0x102B9000) that
    // d3d12_shared_memory backs + registers on first sample (log line
    // "NUI depth surface backed + registered for runtime fill"). Two things must
    // hold every frame for the silhouette to render:
    //  (1) it must carry the SAME segmented, color-space depth+player the type-4
    //      stream expects (filling it with raw full-room depth was wrong), and
    //  (2) it must be invalidated so the GPU texture cache re-uploads it. The
    //      host fills it through the writable physical alias (TranslatePhysical),
    //      which never page-faults, so the normal guest-CPU-write watch never
    //      fires and the GPU copy froze on the first (empty) upload -> black.
    const uint32_t title_depth_addr = memory->nui_title_depth_surface_addr();
    if (title_depth_addr &&
        memory->nui_title_depth_surface_size() >= stride_size) {
      if (auto* title = memory->TranslatePhysical<uint8_t*>(title_depth_addr)) {
        const bool have_color_space =
            latest_depth_color_space_valid_.load(std::memory_order_relaxed) &&
            latest_depth_color_space_.size() ==
                kNuiDepthWidth * kNuiDepthHeight;
        for (uint32_t y = 0; y < kNuiDepthHeight; ++y) {
          uint8_t* title_row = title + y * stride_width * sizeof(uint16_t);
          std::memset(title_row, 0, stride_width * sizeof(uint16_t));
          for (uint32_t x = 0; x < kNuiDepthWidth; ++x) {
            const uint16_t packed =
                have_color_space
                    ? latest_depth_color_space_[y * kNuiDepthWidth + x]
                    : latest_depth_[y * kNuiDepthWidth + x];
            title_row[x * 2] = static_cast<uint8_t>(packed >> 8);
            title_row[x * 2 + 1] = static_cast<uint8_t>(packed & 0xFF);
          }
        }
      }
      // title_depth_addr is already a physical guest address (< 0x20000000), so
      // NotifyGpuMemoryWritten -> GetPhysicalAddress returns it unchanged and
      // invalidates the EXACT page the GPU samples, forcing a per-frame
      // re-upload. (Invalidating the 0x30xxxxxx virtual buffers instead, as the
      // prior fix did, mis-translated to the wrong physical page and never
      // touched this surface -> the texture stayed at its first empty upload.)
      NotifyGpuMemoryWritten(title_depth_addr, stride_size);
      // Diagnostic: prove the surface is filled + invalidated, and how many
      // player pixels reached it, so the live log shows the GPU silhouette
      // texture refreshing instead of freezing on upload #1.
      static std::atomic<uint32_t> title_fill_log{0};
      const uint32_t tfn =
          title_fill_log.fetch_add(1, std::memory_order_relaxed) + 1;
      if (tfn <= 4 || tfn % 300 == 0) {
        uint32_t player_px = 0;
        if (latest_depth_color_space_valid_.load(std::memory_order_relaxed) &&
            latest_depth_color_space_.size() ==
                kNuiDepthWidth * kNuiDepthHeight) {
          for (uint16_t v : latest_depth_color_space_) {
            if (v & kNuiImagePlayerIndexMask) ++player_px;
          }
        }
        XELOGI(
            "NUI title depth surface fill #{}: phys={:08X} size={:X} "
            "color_space_player_px={}",
            tfn, title_depth_addr, stride_size, player_px);
      }
    }
    // Dense 640x480 / 1280-pitch depth+player buffer for the Just Dance 2019
    // silhouette converter (0x82B94210), which walks 0x96000 bytes (1280 * 480).
    // Nearest-neighbor 2x upscale of the 320x240 source, big-endian USHORT
    // (depth_mm<<3)|player like the other depth buffers. Over-allocate one extra
    // row-group (0x2000) so the converter's trailing 128-bit reads stay in
    // bounds. Title-agnostic: any consumer of a color-resolution depth map works.
    {
      constexpr uint32_t kBigW = 640u;
      constexpr uint32_t kBigH = 480u;
      constexpr uint32_t kBigBytes = kBigW * kBigH * sizeof(uint16_t);  // 0x96000
      uint32_t addr640 = depth640_guest_addr_.load(std::memory_order_relaxed);
      if (!addr640) {
        addr640 = memory->SystemHeapAlloc(kBigBytes + 0x2000u);
        if (addr640) {
          depth640_guest_addr_.store(addr640, std::memory_order_relaxed);
          XELOGI(
              "Kinect v1 silhouette depth (640x480, 1280 pitch, {:X} bytes) at "
              "{:08X}",
              kBigBytes, addr640);
        }
      }
      auto* d640 =
          addr640 ? memory->TranslateVirtual<uint8_t*>(addr640) : nullptr;
      if (d640) {
        // Background removal: the Xbox NUI silhouette converter /
        // ps_mumo_silhouette shader paints every pixel that carries depth, so a
        // full-room depth map shows the whole room. Real hardware hands the
        // converter pre-segmented depth (zero outside the player). Keep depth
        // only where the player-index bits (low 3) are set; zero elsewhere.
        const bool segment = cvars::xam_nui_silhouette_segment;
        std::memset(d640, 0, kBigBytes + 0x2000u);
        for (uint32_t y = 0; y < kBigH; ++y) {
          uint8_t* row = d640 + y * kBigW * sizeof(uint16_t);  // 1280-byte pitch
          const uint32_t sy = y >> 1;                          // 480 -> 240
          for (uint32_t x = 0; x < kBigW; ++x) {
            uint16_t v =
                latest_depth_[sy * kNuiDepthWidth + (x >> 1)];  // 640 -> 320
            if (segment && !(v & uint16_t(kNuiImagePlayerIndexMask))) {
              v = 0;  // not a player pixel -> remove from the silhouette
            }
            row[x * 2] = static_cast<uint8_t>(v >> 8);
            row[x * 2 + 1] = static_cast<uint8_t>(v & 0xFF);
          }
        }
        // JD4 PUSH camera model: copy this freshly built 640x480 depth+player
        // silhouette into every physical buffer JD4 registered via _NUICAM 0x05.
        // JD4 never calls the pull-fetch path (0x1004/0x1402), so without this
        // it receives no camera/depth and never shows the player. The same frame
        // is written to all (triple) buffers so it doesn't matter which one JD4
        // treats as current. No buffers are registered for any other title, so
        // this loop is a no-op everywhere else.
        const uint32_t jd4_count =
            jd4_depth_buffer_count_.load(std::memory_order_acquire);
        // DC3's type-4 push buffers are DEPTH_AND_PLAYER_INDEX_IN_COLOR_SPACE, so
        // build a 640x480 COLOR-SPACE depth (parallax-mapped to the colour
        // camera) from latest_depth_color_space_ (320x240, segmented), instead of
        // the raw depth-camera-space d640 (which JD2019 wants). DC3 references and
        // CPU-reads these buffers (proven by the buffer-ref scan), so the body
        // pixels must land at colour-space positions or the silhouette is empty.
        const uint8_t* push_src = d640;
        const bool dc3_cs =
            jd4_count &&
            cvars::xam_nui_dc3_camera_dispatch &&
            latest_depth_color_space_valid_.load(std::memory_order_relaxed) &&
            latest_depth_color_space_.size() == kNuiDepthWidth * kNuiDepthHeight;
        if (dc3_cs) {
          if (dc3_cs640_.size() != kBigBytes) {
            dc3_cs640_.assign(kBigBytes, 0);
          }
          for (uint32_t y = 0; y < kBigH; ++y) {
            uint8_t* row = dc3_cs640_.data() + y * kBigW * sizeof(uint16_t);
            const uint32_t sy = y >> 1;
            for (uint32_t x = 0; x < kBigW; ++x) {
              const uint16_t v =
                  latest_depth_color_space_[sy * kNuiDepthWidth + (x >> 1)];
              row[x * 2] = static_cast<uint8_t>(v >> 8);
              row[x * 2 + 1] = static_cast<uint8_t>(v & 0xFF);
            }
          }
          push_src = dc3_cs640_.data();
        }
        uint32_t filled = 0;
        for (uint32_t i = 0; i < jd4_count; ++i) {
          const uint32_t phys =
              jd4_depth_buffers_[i].load(std::memory_order_relaxed);
          if (!phys) {
            continue;
          }
          auto* jd4_heap = memory->LookupHeap(phys);
          if (!jd4_heap || jd4_heap != memory->LookupHeap(phys + kBigBytes - 1)) {
            continue;  // not a committed range; skip rather than fault
          }
          if (auto* jd4_dst = memory->TranslateVirtual<uint8_t*>(phys)) {
            std::memcpy(jd4_dst, push_src, kBigBytes);
            ++filled;
          }
        }
        // PROBE: confirm the DC3 push buffers (0xBDF00000...) actually get the
        // segmented depth each frame (non-zero player pixels in the source d640).
        if (jd4_count) {
          static std::atomic<uint32_t> filln{0};
          const uint32_t fc = filln.fetch_add(1, std::memory_order_relaxed) + 1;
          if (fc <= 4 || fc % 300 == 0) {
            uint32_t nz = 0;
            for (uint32_t p = 0; p < kBigBytes; p += 2) {
              if (d640[p] || d640[p + 1]) ++nz;
            }
            XELOGI(
                "DC3 push-buffer fill #{}: count={} filled={} buf0={:08X} "
                "nonzero_px={}/{}",
                fc, jd4_count, filled,
                jd4_depth_buffers_[0].load(std::memory_order_relaxed), nz,
                kBigBytes / 2);
          }
        }
      }
    }
    if (mini || mini128 || mini128_depth_only) {
      if (mini128) {
        std::memset(mini128, 0, mini_stride_width * mini_height *
                                  sizeof(uint16_t));
      }
      if (mini128_depth_only) {
        std::memset(mini128_depth_only, 0,
                    mini_stride_width * mini_height * sizeof(uint16_t));
      }
      for (uint32_t my = 0; my < mini_height; ++my) {
        for (uint32_t mx = 0; mx < mini_width; ++mx) {
          uint16_t best = 0xFFFF;
          uint16_t best_player = 0xFFFF;
          for (uint32_t dy = 0; dy < 4; ++dy) {
            for (uint32_t dx = 0; dx < 4; ++dx) {
              const uint16_t packed =
                  latest_depth_[(my * 4 + dy) * kNuiDepthWidth +
                                (mx * 4 + dx)];
              if (packed && packed < best) {
                best = packed;
              }
              if ((packed & kNuiImagePlayerIndexMask) &&
                  (packed >> 3) &&
                  ((packed >> 3) < (best_player >> 3))) {
                best_player = packed;
              }
            }
          }
          if (best_player != 0xFFFF) {
            best = best_player;
          }
          if (best == 0xFFFF) {
            best = 0;
          }
          if (mini) {
            const uint32_t index = my * mini_width + mx;
            mini[index * 2] = static_cast<uint8_t>(best >> 8);
            mini[index * 2 + 1] = static_cast<uint8_t>(best & 0xFF);
          }
          if (mini128) {
            const uint32_t index = my * mini_stride_width + mx;
            mini128[index * 2] = static_cast<uint8_t>(best >> 8);
            mini128[index * 2 + 1] = static_cast<uint8_t>(best & 0xFF);
          }
          if (mini128_depth_only) {
            const uint16_t depth_only = best & ~uint16_t(0x7);
            const uint32_t index = my * mini_stride_width + mx;
            mini128_depth_only[index * 2] =
                static_cast<uint8_t>(depth_only >> 8);
            mini128_depth_only[index * 2 + 1] =
                static_cast<uint8_t>(depth_only & 0xFF);
          }
        }
      }
    }
    // NOTE: the GPU-sampled silhouette texture is the physical title surface
    // invalidated above (NotifyGpuMemoryWritten(title_depth_addr)). The
    // 0x30xxxxxx SystemHeapAlloc'd buffers below are filled for CPU/LockRect
    // consumers and other titles, but the GPU never samples them directly
    // (their fetch base masks to the physical surface), so they need no GPU
    // invalidation here.
    latest_depth_timestamp_.store(
        pending_depth_timestamp_.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    latest_depth_frame_number_.store(
        pending_depth_frame_number_.load(std::memory_order_relaxed),
        std::memory_order_release);
    depth_guest_valid_.store(true, std::memory_order_release);
    SignalImageFrameEvent();
    // Drive JD2019's silhouette dispatcher with this freshly published depth
    // frame (self-gated; a no-op for every other title).
    SignalJd2019CameraDepthEvent(memory);
    // Same for DC3's push camera dispatcher (opt-in cvar + title-gated +
    // self-guarded; no-op for every other title or until DC3's dispatcher is up).
    SignalDc3CameraDepthEvent(memory);
  }

  static void CopyColorWithViewArea(const std::vector<uint8_t>& source,
                                    uint8_t* destination,
                                    uint32_t digital_zoom, int32_t center_x,
                                    int32_t center_y) {
    if (digital_zoom != kNuiImageDigitalZoom2x) {
      std::memcpy(destination, source.data(), source.size());
      return;
    }

    const int32_t zoom_percent =
        std::clamp(cvars::xam_nui_color_crop_zoom_percent, 100, 200);
    const int32_t crop_width = kNuiColorWidth * 100 / zoom_percent;
    const int32_t crop_height = kNuiColorHeight * 100 / zoom_percent;
    const int32_t center_abs_x =
        std::clamp<int32_t>(kNuiColorWidth / 2 + center_x, crop_width / 2,
                            kNuiColorWidth - crop_width / 2);
    const int32_t center_abs_y =
        std::clamp<int32_t>(kNuiColorHeight / 2 + center_y, crop_height / 2,
                            kNuiColorHeight - crop_height / 2);
    const int32_t left =
        std::clamp<int32_t>(center_abs_x - crop_width / 2, 0,
                            kNuiColorWidth - crop_width);
    const int32_t top =
        std::clamp<int32_t>(center_abs_y - crop_height / 2, 0,
                            kNuiColorHeight - crop_height);

    for (uint32_t y = 0; y < kNuiColorHeight; ++y) {
      const uint32_t src_y =
          static_cast<uint32_t>(top + y * crop_height / kNuiColorHeight);
      const uint8_t* src_row =
          source.data() + src_y * kNuiColorWidth * 4;
      uint8_t* dst_row = destination + y * kNuiColorWidth * 4;
      for (uint32_t x = 0; x < kNuiColorWidth; ++x) {
        const uint32_t src_x =
            static_cast<uint32_t>(left + x * crop_width / kNuiColorWidth);
        std::memcpy(dst_row + x * 4, src_row + src_x * 4, 4);
      }
    }
  }

  static void CopyHostBgraToGuestXrgb(const uint8_t* source,
                                      uint8_t* destination,
                                      size_t pixel_count) {
    // The Windows Kinect SDK locks color frames as little-endian X8R8G8B8
    // (B, G, R, X bytes). Xbox 360 NUI exposes the same DWORD format to a
    // big-endian CPU, so its locked texture bytes are X, R, G, B.
    for (size_t i = 0; i < pixel_count; ++i) {
      destination[i * 4 + 0] = 0xFF;
      destination[i * 4 + 1] = source[i * 4 + 2];
      destination[i * 4 + 2] = source[i * 4 + 1];
      destination[i * 4 + 3] = source[i * 4 + 0];
    }
  }

  bool PollColorFrameOnce(DWORD wait_ms) {
    if (!color_stream_ || !HasImageFrameReader() ||
        !EnsureKinectComInitializedForCurrentThread()) {
      return false;
    }

    NuiImageFrame frame_storage = {};
    const NuiImageFrame* frame = nullptr;
    const HRESULT hr =
        ReadImageFrame(color_stream_, wait_ms, &frame_storage, &frame);
    if (hr == S_FALSE || hr == kNuiFrameNoData) {
      return false;
    }
    if (FAILED(hr) || !frame || !frame->frame_texture) {
      return false;
    }

    NuiLockedRect rect = {};
    const HRESULT lock_hr =
        frame->frame_texture->LockRect(0, &rect, nullptr, 0);
    if (FAILED(lock_hr) || !rect.bits ||
        rect.pitch < static_cast<INT>(kNuiColorWidth * 4)) {
      ReleaseImageFrame(color_stream_, frame);
      return false;
    }

    {
      std::lock_guard<std::mutex> lock(latest_color_mutex_);
      if (latest_color_.size() != kNuiColorWidth * kNuiColorHeight * 4) {
        latest_color_.assign(kNuiColorWidth * kNuiColorHeight * 4, 0);
      }
      for (uint32_t y = 0; y < kNuiColorHeight; ++y) {
        std::memcpy(&latest_color_[y * kNuiColorWidth * 4],
                    rect.bits + static_cast<size_t>(y) * rect.pitch,
                    kNuiColorWidth * 4);
      }
      latest_color_valid_.store(true, std::memory_order_relaxed);
    }

    const uint32_t frame_number = frame->frame_number;
    pending_color_timestamp_.store(frame->timestamp.QuadPart,
                                   std::memory_order_relaxed);
    pending_color_frame_number_.store(frame_number, std::memory_order_relaxed);
    frame->frame_texture->UnlockRect(0);
    ReleaseImageFrame(color_stream_, frame);
    const uint32_t n =
        color_frame_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n <= 8 || (n & (n - 1)) == 0) {
      XELOGI("Kinect v1 color read #{}: frame={} pitch={} size={}", n,
             frame_number, rect.pitch, rect.size);
    }
    return true;
  }

  void PublishColorToGuest() {
    if (!latest_color_valid_.load(std::memory_order_relaxed) ||
        !kernel_state()) {
      return;
    }
    auto* memory = kernel_state()->memory();
    uint32_t addr = color_guest_addr_.load(std::memory_order_relaxed);
    if (!addr) {
      addr = memory->SystemHeapAlloc(kNuiColorWidth * kNuiColorHeight * 4);
      if (!addr) {
        return;
      }
      color_guest_addr_.store(addr, std::memory_order_relaxed);
      XELOGI("Kinect v1 color guest buffer 640x480 pitch 2560 at {:08X}",
             addr);
    }
    uint32_t yuv_addr =
        color_yuv_guest_addr_.load(std::memory_order_relaxed);
    if (!yuv_addr) {
      yuv_addr = memory->SystemHeapAlloc(kNuiColorWidth * kNuiColorHeight * 2);
      if (yuv_addr) {
        color_yuv_guest_addr_.store(yuv_addr, std::memory_order_relaxed);
        XELOGI("Kinect v1 YUV color guest buffer 640x480 pitch 1280 at {:08X}",
               yuv_addr);
      }
    }
    auto* dst = memory->TranslateVirtual<uint8_t*>(addr);
    if (!dst) {
      return;
    }
    auto* yuv_dst =
        yuv_addr ? memory->TranslateVirtual<uint8_t*>(yuv_addr) : nullptr;
    {
      std::lock_guard<std::mutex> lock(latest_color_mutex_);
      if (latest_color_.size() != kNuiColorWidth * kNuiColorHeight * 4) {
        return;
      }
      uint32_t digital_zoom = kNuiImageDigitalZoom1x;
      int32_t view_center_x = 0;
      int32_t view_center_y = 0;
      const bool view_area_valid =
          CopyColorViewArea(&digital_zoom, &view_center_x, &view_center_y);
      // Diagnostic: prove whether the head-following crop centre actually moves
      // (face-follow) or is stuck. Logged once per second-ish.
      {
        static std::atomic<uint32_t> view_log{0};
        const uint32_t vn =
            view_log.fetch_add(1, std::memory_order_relaxed) + 1;
        if (vn <= 8 || vn % 60 == 0) {
          XELOGI(
              "Kinect v1 colour view-area #{}: valid={} zoom={} center=({}, "
              "{}) crop={}",
              vn, view_area_valid ? 1 : 0, digital_zoom, view_center_x,
              view_center_y, cvars::xam_nui_color_crop_to_view ? 1 : 0);
        }
      }
      std::vector<uint8_t> viewed_color;
      const uint8_t* color_source = latest_color_.data();
      // Crop the colour to the head-following digital-zoom view area (the SDK
      // behaviour) when enabled; titles that display the feed as-is (Just
      // Dance's face thumbnail) need the stream pre-cropped onto the head.
      if (cvars::xam_nui_color_crop_to_view && view_area_valid &&
          digital_zoom == kNuiImageDigitalZoom2x) {
        viewed_color.resize(latest_color_.size());
        CopyColorWithViewArea(latest_color_, viewed_color.data(),
                              digital_zoom, view_center_x, view_center_y);
        color_source = viewed_color.data();
      }
      CopyHostBgraToGuestXrgb(color_source, dst,
                              kNuiColorWidth * kNuiColorHeight);
      if (yuv_dst) {
        auto clamp_byte = [](int value) -> uint8_t {
          return static_cast<uint8_t>(std::clamp(value, 0, 255));
        };
        for (uint32_t y = 0; y < kNuiColorHeight; ++y) {
          const uint8_t* src_row = color_source + y * kNuiColorWidth * 4;
          uint8_t* dst_row = yuv_dst + y * kNuiD3DColorYuvPitch;
          for (uint32_t x = 0; x < kNuiColorWidth; x += 2) {
            const uint8_t* p0 = src_row + x * 4;
            const uint8_t* p1 = src_row + (x + 1) * 4;
            const int b0 = p0[0], g0 = p0[1], r0 = p0[2];
            const int b1 = p1[0], g1 = p1[1], r1 = p1[2];
            const int y0 = ((66 * r0 + 129 * g0 + 25 * b0 + 128) >> 8) + 16;
            const int y1 = ((66 * r1 + 129 * g1 + 25 * b1 + 128) >> 8) + 16;
            const int r = (r0 + r1 + 1) / 2;
            const int g = (g0 + g1 + 1) / 2;
            const int b = (b0 + b1 + 1) / 2;
            const int u =
                ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
            const int v =
                ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
            const uint32_t out = x * 2;
            // NUI COLOR_YUV is exposed as UYVY on Xbox 360. The matching
            // D3DFMT_YUY2 texture fetch uses 8-in-16 endian swapping.
            dst_row[out + 0] = clamp_byte(u);
            dst_row[out + 1] = clamp_byte(y0);
            dst_row[out + 2] = clamp_byte(v);
            dst_row[out + 3] = clamp_byte(y1);
          }
        }
      }
    }
    NotifyGpuMemoryWritten(addr, kNuiColorWidth * kNuiColorHeight * 4);
    if (yuv_addr) {
      NotifyGpuMemoryWritten(yuv_addr, kNuiColorWidth * kNuiColorHeight * 2);
    }
    // Mirror the UYVY colour into the title's GPU-sampled colour surface: the
    // colour fetch base (0x301A3xxx) masks to a physical page (0x101A3xxx) that
    // d3d12_shared_memory backs + registers (log "NUI color surface backed").
    // THAT physical page is what the camera shader samples, not the
    // SystemHeapAlloc'd yuv buffer, so it must be filled + invalidated here or
    // the colour camera stays black (was: "Invalid upload range 000101A3").
    const uint32_t title_color_addr = memory->nui_title_color_surface_addr();
    const uint32_t yuv_bytes = kNuiColorHeight * kNuiD3DColorYuvPitch;
    if (yuv_dst && title_color_addr &&
        memory->nui_title_color_surface_size() >= yuv_bytes) {
      if (auto* title_color =
              memory->TranslatePhysical<uint8_t*>(title_color_addr)) {
        std::memcpy(title_color, yuv_dst, yuv_bytes);
      }
      NotifyGpuMemoryWritten(title_color_addr, yuv_bytes);
      static std::atomic<uint32_t> title_color_log{0};
      const uint32_t tcn =
          title_color_log.fetch_add(1, std::memory_order_relaxed) + 1;
      if (tcn <= 4 || tcn % 300 == 0) {
        XELOGI("NUI title color surface fill #{}: phys={:08X} size={:X}", tcn,
               title_color_addr, yuv_bytes);
      }
    }
    // Fill DC3's registered type-2 (NUI_IMAGE_TYPE_COLOR_YUV) push buffers with
    // the live YUV camera. PROVEN: DC3's silhouette descriptor (ctx+0x80)
    // references a COLOUR buffer (e.g. 0xBDD30000) AND a depth buffer
    // (0xBDF00000); the silhouette is the colour camera masked by the
    // depth/player index. We filled depth but never colour -> empty silhouette.
    // Same 640x480 UYVY / 1280-pitch layout as yuv_dst (0x96000 bytes).
    if (yuv_dst) {
      const uint32_t color_count =
          jd4_color_buffer_count_.load(std::memory_order_acquire);
      uint32_t cfilled = 0;
      for (uint32_t i = 0; i < color_count; ++i) {
        const uint32_t phys =
            jd4_color_buffers_[i].load(std::memory_order_relaxed);
        if (!phys) {
          continue;
        }
        auto* heap = memory->LookupHeap(phys);
        if (!heap || heap != memory->LookupHeap(phys + yuv_bytes - 1)) {
          continue;
        }
        if (auto* dst = memory->TranslateVirtual<uint8_t*>(phys)) {
          std::memcpy(dst, yuv_dst, yuv_bytes);
          ++cfilled;
        }
      }
      if (color_count) {
        static std::atomic<uint32_t> ccn{0};
        const uint32_t c = ccn.fetch_add(1, std::memory_order_relaxed) + 1;
        if (c <= 4 || c % 300 == 0) {
          XELOGI(
              "DC3 color push-buffer fill #{}: count={} filled={} buf0={:08X}",
              c, color_count, cfilled,
              jd4_color_buffers_[0].load(std::memory_order_relaxed));
        }
      }
    }
    latest_color_timestamp_.store(
        pending_color_timestamp_.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    latest_color_frame_number_.store(
        pending_color_frame_number_.load(std::memory_order_relaxed),
        std::memory_order_release);
    color_guest_valid_.store(true, std::memory_order_release);
    SignalColorFrameEvent();
  }

  bool PollSkeletonFrameOnce(DWORD wait_ms) {
    if (!HasSkeletonFrameReader() ||
        !EnsureKinectComInitializedForCurrentThread()) {
      return false;
    }

    std::lock_guard<std::mutex> lock(skeleton_poll_mutex_);
    NuiSkeletonFrame frame = {};
    const HRESULT hr = ReadSkeletonFrame(wait_ms, &frame);
    if (hr == S_FALSE || hr == kNuiFrameNoData) {
      const uint32_t no_data =
          skeleton_no_data_count_.fetch_add(1, std::memory_order_relaxed) + 1;
      if (no_data == 1 || no_data % 1000 == 0) {
        XELOGI("Kinect v1 skeleton polling: no frame yet ({}) hr={:08X}",
               no_data, static_cast<uint32_t>(hr));
      }
      return false;
    }
    if (FAILED(hr)) {
      XELOGW("Kinect v1 NuiSkeletonGetNextFrame failed: {:08X}",
             static_cast<uint32_t>(hr));
      return false;
    }

    uint32_t tracked_count = 0;
    uint32_t position_only_count = 0;
    uint32_t best_tracking_id = kNuiInvalidTrackingId;
    uint32_t best_skeleton_index = std::numeric_limits<uint32_t>::max();
    for (uint32_t i = 0; i < kNuiSkeletonCount; ++i) {
      auto& skeleton = frame.skeleton_data[i];
      if (skeleton.tracking_state == kNuiSkeletonPositionOnly) {
        ++position_only_count;
      }
      if (skeleton.tracking_state != kNuiSkeletonTracked) {
        continue;
      }
      ++tracked_count;
      if (best_tracking_id == kNuiInvalidTrackingId) {
        best_tracking_id = skeleton.tracking_id;
        best_skeleton_index = i;
      }
    }

    // Ask the SDK to fully track (and therefore segment) the closest detected
    // bodies. This is what populates the depth player index / body silhouette.
    DriveTrackedSkeletons(frame);

    X_NUI_SKELETON_FRAME guest_frame = {};
    PublishDepthPlayerIndexRemap(frame, best_tracking_id);
    ConvertSkeletonFrameToGuest(frame, &guest_frame, best_tracking_id);
    {
      std::lock_guard<std::mutex> latest_lock(latest_skeleton_frame_mutex_);
      latest_skeleton_frame_ = guest_frame;
    }
    latest_skeleton_frame_valid_.store(true, std::memory_order_release);
    tracked_skeleton_count_.store(tracked_count, std::memory_order_relaxed);
    best_tracking_id_.store(tracked_count ? best_tracking_id : 0,
                            std::memory_order_relaxed);
    uint32_t selected_host_player_index = 0;
    if (tracked_count &&
        best_skeleton_index != std::numeric_limits<uint32_t>::max()) {
      selected_host_player_index = best_skeleton_index + 1;
    }
    selected_host_player_index_.store(selected_host_player_index,
                                      std::memory_order_relaxed);
    const uint32_t published =
        latest_skeleton_frame_number_.fetch_add(1,
                                                std::memory_order_relaxed) + 1;
    const bool tracked_count_changed =
        tracked_count != last_logged_tracked_count_.exchange(
                             tracked_count, std::memory_order_relaxed);
    const bool position_only_count_changed =
        position_only_count != last_logged_position_only_count_.exchange(
                                   position_only_count,
                                   std::memory_order_relaxed);
    const bool host_player_index_changed =
        selected_host_player_index != last_logged_host_player_index_.exchange(
                                          selected_host_player_index,
                                          std::memory_order_relaxed);
    if (tracked_count_changed || position_only_count_changed ||
        host_player_index_changed) {
      XELOGI(
          "Kinect v1 skeleton frame: tracked_count={} position_only={} "
          "best_host_id={} "
          "host_player_index={}",
          tracked_count, position_only_count, best_tracking_id,
          selected_host_player_index);
    }
    if (published == 1 || published % 300 == 0) {
      XELOGI("Kinect v1 skeleton guest frame published: published={} sdk={}",
             published, frame.frame_number);
    }
    AccumulateHeadJitter(frame, best_skeleton_index);
    DumpSkeletonValues(frame, best_skeleton_index, published);
    SignalSkeletonFrameEvent();
    return true;
  }

  // Track the HEAD joint's frame-to-frame displacement for the tracked body,
  // every frame, so the NUI-VALUES dump can report the true raw jitter (the
  // per-dump head_dpos samples 30 frames apart and so mixes in real movement).
  void AccumulateHeadJitter(const NuiSkeletonFrame& frame,
                            uint32_t best_skeleton_index) {
    if (cvars::xam_nui_log_skeleton_values <= 0) {
      return;
    }
    if (best_skeleton_index >= kNuiSkeletonCount) {
      last_frame_head_valid_ = false;
      return;
    }
    const auto& head = frame.skeleton_data[best_skeleton_index].skeleton_positions[3];
    if (last_frame_head_valid_) {
      const float dx = head.x - last_frame_head_[0];
      const float dy = head.y - last_frame_head_[1];
      const float dz = head.z - last_frame_head_[2];
      const double d = std::sqrt(double(dx * dx + dy * dy + dz * dz));
      if (jitter_count_ == 0) {
        jitter_min_ = jitter_max_ = d;
      } else {
        jitter_min_ = std::min(jitter_min_, d);
        jitter_max_ = std::max(jitter_max_, d);
      }
      jitter_sum_ += d;
      ++jitter_count_;
    }
    last_frame_head_[0] = head.x;
    last_frame_head_[1] = head.y;
    last_frame_head_[2] = head.z;
    last_frame_head_valid_ = true;
  }

  // Diagnostic: dump the exact skeleton the host hands the title, so the values
  // feeding dance scoring (joints / vNormalToGravity / vFloorClipPlane) can be
  // verified instead of guessed at. Gated behind xam_nui_log_skeleton_values
  // (N = every N published frames; 0 = off). Logs the raw host floats, which
  // ConvertSkeletonFrameToGuest copies verbatim into the guest frame.
  void DumpSkeletonValues(const NuiSkeletonFrame& frame,
                          uint32_t best_skeleton_index, uint32_t published) {
    const int32_t interval = cvars::xam_nui_log_skeleton_values;
    if (interval <= 0 || (published % static_cast<uint32_t>(interval)) != 0) {
      return;
    }
    const auto& g = frame.normal_to_gravity;
    const auto& f = frame.floor_clip_plane;
    if (best_skeleton_index >= kNuiSkeletonCount) {
      XELOGI(
          "NUI-VALUES f#{}: NO TRACKED BODY | gravity=({:.3f},{:.3f},{:.3f},"
          "{:.3f}) floor=({:.3f},{:.3f},{:.3f},{:.3f})",
          published, g.x, g.y, g.z, g.w, f.x, f.y, f.z, f.w);
      return;
    }
    const auto& s = frame.skeleton_data[best_skeleton_index];
    const auto& head = s.skeleton_positions[3];  // HEAD
    // Per-frame head jitter: distance the HEAD joint moved since the last dump
    // interval. A still player should show only millimetre-scale raw SDK noise.
    const float dx = head.x - last_dump_head_[0];
    const float dy = head.y - last_dump_head_[1];
    const float dz = head.z - last_dump_head_[2];
    const float head_jitter = std::sqrt(dx * dx + dy * dy + dz * dz);
    last_dump_head_[0] = head.x;
    last_dump_head_[1] = head.y;
    last_dump_head_[2] = head.z;

    const double jit_avg = jitter_count_ ? jitter_sum_ / jitter_count_ : 0.0;
    // grav_out = the up vector we actually hand the title (derived from the
    // floor plane when the SDK leaves vNormalToGravity zero). tilt = angle of
    // grav_out off straight up, i.e. the camera tilt the title now levels out.
    float gl = std::sqrt(gravity_up_[0] * gravity_up_[0] +
                         gravity_up_[1] * gravity_up_[1] +
                         gravity_up_[2] * gravity_up_[2]);
    const float ginv = gl > 1e-6f ? 1.0f / gl : 0.0f;
    const float gox = gravity_up_[0] * ginv, goy = gravity_up_[1] * ginv,
                goz = gravity_up_[2] * ginv;
    const float tilt_deg =
        gravity_up_valid_
            ? std::acos(std::clamp(goy, -1.0f, 1.0f)) * 57.29578f
            : 0.0f;
    // grav_src tells which source actually drove grav_out this frame: floor =
    // the measured floor-plane normal, held = the previous floor-derived normal
    // held across a dropout, ACCEL/extrap = startup fallback, sdk = SDK gravity.
    const bool accel_v = accel_up_valid_.load(std::memory_order_relaxed);
    const bool floor_present =
        std::sqrt(f.x * f.x + f.y * f.y + f.z * f.z) >= 1e-3f;
    const bool sdk_gravity = std::fabs(g.x) > 1e-6f ||
                             std::fabs(g.y) > 1e-6f ||
                             std::fabs(g.z) > 1e-6f;
    const char* grav_src =
        sdk_gravity
            ? "sdk"
            : (floor_present && cvars::xam_nui_gravity_from_floor)
                  ? "floor"
                  : gravity_up_valid_
                        ? "held"
                        : (cvars::xam_nui_gravity_from_accelerometer && accel_v)
                              ? "ACCEL"
                              : "extrap";
    XELOGI(
        "NUI-VALUES f#{}: id={} state={} center=({:.3f},{:.3f},{:.3f}) "
        "sdk_grav=({:.3f},{:.3f},{:.3f},{:.3f}) grav_out=({:.3f},{:.3f},{:.3f}) "
        "grav_src={} accel_up=({:.3f},{:.3f},{:.3f}) "
        "tilt={:.1f}deg floor=({:.3f},{:.3f},{:.3f},{:.3f}) head_dpos={:.4f}m "
        "perframe_jitter[min={:.4f} avg={:.4f} max={:.4f}]m n={} quality={:08X}",
        published, s.tracking_id, s.tracking_state, s.position.x, s.position.y,
        s.position.z, g.x, g.y, g.z, g.w, gox, goy, goz, grav_src,
        accel_up_x_.load(std::memory_order_relaxed),
        accel_up_y_.load(std::memory_order_relaxed),
        accel_up_z_.load(std::memory_order_relaxed), tilt_deg, f.x, f.y, f.z,
        f.w, head_jitter, jitter_min_, jit_avg, jitter_max_, jitter_count_,
        s.quality_flags);
    jitter_sum_ = 0.0;
    jitter_min_ = 0.0;
    jitter_max_ = 0.0;
    jitter_count_ = 0;
    for (uint32_t j = 0; j < kNuiSkeletonPositionCount; ++j) {
      const auto& p = s.skeleton_positions[j];
      const int32_t st = s.skeleton_position_tracking_state[j];
      XELOGI("NUI-VALUES   [{:>2}] {:<15} {} ({:+.3f},{:+.3f},{:+.3f})", j,
             kNuiJointNames[j],
             (st >= 0 && st < 3) ? kNuiJointStateNames[st] : "???", p.x, p.y,
             p.z);
    }
    // Face-cam framing diagnostic: replicate the title's PlayerHeadImage crop
    // (centre = HEAD in colour, half-size from |HEAD-SHOULDER_CENTER| at 4:3 x
    // 0.65, with the title's top/bottom clamp) from the SAME head/shoulder this
    // dump shows, so the actual thumbnail framing is visible in the log without
    // changing what the title renders. Reads the (already head-drop-adjusted)
    // joints, so it reflects exactly what the title crops.
    {
      const auto& hp = s.skeleton_positions[3];
      const auto& sp = s.skeleton_positions[2];
      // Apply the same head-drop the guest conversion applies, so the logged
      // crop matches what the title actually renders.
      const float f_drop =
          static_cast<float>(cvars::xam_nui_facecam_head_drop);
      const float vx = f_drop * (hp.x - sp.x);
      const float vy = f_drop * (hp.y - sp.y);
      const float vz = f_drop * (hp.z - sp.z);
      X_NUI_VECTOR4 head_pt{}, sh_pt{};
      head_pt.x = hp.x - vx; head_pt.y = hp.y - vy; head_pt.z = hp.z - vz;
      head_pt.w = 1.0f;
      sh_pt.x = sp.x - vx; sh_pt.y = sp.y - vy; sh_pt.z = sp.z - vz;
      sh_pt.w = 1.0f;
      int32_t hcx = 0, hcy = 0, scx = 0, scy = 0;
      if (GetColorCoordinatesForSkeletonPoint(head_pt, &hcx, &hcy) &&
          GetColorCoordinatesForSkeletonPoint(sh_pt, &scx, &scy)) {
        const float kW = static_cast<float>(kNuiColorWidth);
        const float kH = static_cast<float>(kNuiColorHeight);
        float cu = hcx / kW, cv = hcy / kH;
        const float du = (hcx - scx) / kW, dv = (hcy - scy) / kH;
        float hw = std::sqrt(du * du + dv * dv);
        float hh = hw * (4.0f / 3.0f);
        hw *= 0.65f;
        hh *= 0.65f;
        bool clamped = false;
        if (cv - hh < 0.0f) { cv = hh; clamped = true; }
        else if (cv + hh > 1.0f) { cv = 1.0f - hh; clamped = true; }
        XELOGI(
            "NUI-FACECAM head_col=({},{}) shoulder_col=({},{}) of {}x{} -> "
            "cropV=[{:.0f}..{:.0f}] centreV={:.0f} halfH={:.0f}px clamped={} "
            "(face sits at the title's HEAD row {})",
            hcx, hcy, scx, scy, static_cast<int>(kW), static_cast<int>(kH),
            (cv - hh) * kH, (cv + hh) * kH, cv * kH, hh * kH, clamped ? 1 : 0,
            hcy);
      }
    }
  }

  static void WriteGuestVector(X_NUI_VECTOR4& guest_vector,
                               const NuiVector4& host_vector) {
    guest_vector.x = host_vector.x;
    guest_vector.y = host_vector.y;
    guest_vector.z = host_vector.z;
    guest_vector.w = host_vector.w;
  }

  void ConvertSkeletonFrameToGuest(
      const NuiSkeletonFrame& host_frame, X_NUI_SKELETON_FRAME* guest_frame,
      uint32_t bound_tracking_id) {
    std::memset(guest_frame, 0, sizeof(*guest_frame));
    guest_frame->timestamp = static_cast<int64_t>(host_frame.timestamp.QuadPart);
    guest_frame->frame_number = host_frame.frame_number;
    guest_frame->flags = host_frame.flags;

    // Emulate the Xbox NUI_INITIALIZE_FLAG_EXTRAPOLATE_FLOOR_PLANE (XDK
    // nuiapi.h 0x800), which the desktop Kinect v1 SDK does not implement. Xbox
    // titles' start-up "lower the sensor to scan the floor / set the field of
    // view" calibration reads vFloorClipPlane to locate the floor and then
    // raises the camera; on real hardware the runtime fills that plane from the
    // camera tilt even with no body in view. The Windows SDK instead leaves it
    // zero until a skeleton is tracked, so the calibration lowers and hangs
    // forever. When the SDK left the plane empty, rebuild it from the real
    // elevation angle and an assumed sensor height. A tracked body's real plane
    // is non-zero and is always passed through untouched, so gameplay scoring
    // (gravity-from-floor leveling below) is unaffected.
    NuiVector4 effective_floor = host_frame.floor_clip_plane;
    // True only when the SDK gave a real measured floor plane (a body is
    // tracked). That plane is the ground truth for which way is up in the room
    // and is the preferred leveling source for scoring; the accelerometer (a
    // coarser sensor with a few degrees of roll bias) and the extrapolated
    // plane are only used when no real plane is available.
    bool floor_is_real = false;
    {
      const float fl = std::sqrt(effective_floor.x * effective_floor.x +
                                 effective_floor.y * effective_floor.y +
                                 effective_floor.z * effective_floor.z);
      floor_is_real = fl >= 1e-3f;
      if (fl < 1e-3f && cvars::xam_nui_extrapolate_floor_plane) {
        // Kinect camera space: +Y up, +Z forward. The floor's up-normal at a
        // signed elevation angle a (a < 0 looks down) is (0, cos a, sin a); the
        // plane is Hessian-normal A x + B y + C z + D = 0 with D the positive
        // camera-to-floor (sensor height) distance. A tiny non-zero A keeps all
        // four components non-zero so titles that gate "floor found" on every
        // component being set still accept the plane.
        const float a =
            static_cast<float>(
                last_elevation_angle_.load(std::memory_order_relaxed)) *
            0.0174532925199f;  // degrees -> radians
        const float height = std::max(
            0.1f, static_cast<float>(cvars::xam_nui_camera_height_meters));
        effective_floor.x = 1e-4f;
        effective_floor.y = std::cos(a);
        effective_floor.z = std::sin(a);
        effective_floor.w = height;
        if (!floor_extrapolation_logged_.exchange(true,
                                                  std::memory_order_relaxed)) {
          XELOGI(
              "Kinect floor clip plane extrapolated for sensor calibration "
              "(no SDK floor; angle={} deg, height={:.2f} m): "
              "({:.4f},{:.3f},{:.3f},{:.3f})",
              last_elevation_angle_.load(std::memory_order_relaxed), height,
              effective_floor.x, effective_floor.y, effective_floor.z,
              effective_floor.w);
        }
      }
    }
    WriteGuestVector(guest_frame->floor_clip_plane, effective_floor);
    WriteGuestVector(guest_frame->normal_to_gravity,
                     host_frame.normal_to_gravity);

    // vNormalToGravity is the "up" vector titles feed to NuiTransformMatrixLevel
    // to level the skeleton against camera tilt before scoring hand/spine/head
    // positions (XDK AtgNuiRelativeCoordinates). The Windows Kinect v1 SDK
    // almost always leaves it zero. Handing the title a flat (0,1,0) when the
    // camera is tilted makes NuiTransformMatrixLevel the identity, so the
    // skeleton is never un-tilted and every pose is rotated by the camera tilt
    // -> correctly performed dance moves score wrong. The source priority for
    // the up vector is: (1) the SDK's real measured floor clip plane normal --
    // ground truth for the room's up, available whenever a body is tracked,
    // i.e. always during scoring, and it re-fits live as the camera is moved;
    // (2) the last measured floor-derived up vector held across frame dropouts;
    // (3) startup-only accelerometer / extrapolated-plane fallback before any
    // measured floor has been seen. The accelerometer is deliberately kept out
    // of the live scoring path: its roll axis can carry a few degrees of bias,
    // while the measured floor plane was the known-good DC1/DC3 score fix.
    {
      const float gx = host_frame.normal_to_gravity.x;
      const float gy = host_frame.normal_to_gravity.y;
      const float gz = host_frame.normal_to_gravity.z;
      const bool gravity_zero = std::fabs(gx) < 1e-6f &&
                                std::fabs(gy) < 1e-6f && std::fabs(gz) < 1e-6f;
      if (!gravity_zero) {
        // SDK supplied a real gravity vector; trust it (already copied above).
      } else if (cvars::xam_nui_gravity_from_accelerometer ||
                 cvars::xam_nui_gravity_from_floor) {
        // Pick the raw up-normal for this frame. During scoring, only a real
        // measured floor plane may move the held normal; if the SDK drops the
        // floor for a frame, keep the last known-good floor vector instead of
        // drifting toward accelerometer or virtual-tilt bias.
        float rnx = 0.0f, rny = 0.0f, rnz = 0.0f;
        bool have_raw = false;
        // 1) A real measured floor plane (body tracked) is ground truth for the
        //    room's up direction and the most accurate leveling source, so it
        //    wins whenever present. This is the path scoring takes, and it
        //    already follows the camera being moved (the SDK re-fits the plane
        //    from the live depth view each frame).
        if (floor_is_real && cvars::xam_nui_gravity_from_floor) {
          const float fx = effective_floor.x;
          const float fy = effective_floor.y;
          const float fz = effective_floor.z;
          const float len = std::sqrt(fx * fx + fy * fy + fz * fz);
          if (len > 1e-3f) {
            rnx = fx / len;
            rny = fy / len;
            rnz = fz / len;
            have_raw = true;
          }
        }
        // 2) Startup fallback only: before any measured floor has seeded the
        //    held normal, optionally use the accelerometer.
        if (!have_raw && !gravity_up_valid_ &&
            cvars::xam_nui_gravity_from_accelerometer &&
            accel_up_valid_.load(std::memory_order_relaxed)) {
          rnx = accel_up_x_.load(std::memory_order_relaxed);
          rny = accel_up_y_.load(std::memory_order_relaxed);
          rnz = accel_up_z_.load(std::memory_order_relaxed);
          have_raw = (rnx * rnx + rny * rny + rnz * rnz) > 1e-6f;
        }
        // 3) Startup last resort: the extrapolated floor plane.
        if (!have_raw && !gravity_up_valid_ &&
            cvars::xam_nui_gravity_from_floor) {
          const float fx = effective_floor.x;
          const float fy = effective_floor.y;
          const float fz = effective_floor.z;
          const float len = std::sqrt(fx * fx + fy * fy + fz * fz);
          if (len > 1e-3f) {
            rnx = fx / len;
            rny = fy / len;
            rnz = fz / len;
            have_raw = true;
          }
        }
        if (have_raw) {
          // Low-pass the chosen normal into the held up vector so the leveling
          // rotation does not jitter frame to frame.
          if (!gravity_up_valid_) {
            gravity_up_[0] = rnx;
            gravity_up_[1] = rny;
            gravity_up_[2] = rnz;
            gravity_up_valid_ = true;
          } else {
            constexpr float kAlpha = 0.1f;  // host-side pre-smoothing
            gravity_up_[0] += (rnx - gravity_up_[0]) * kAlpha;
            gravity_up_[1] += (rny - gravity_up_[1]) * kAlpha;
            gravity_up_[2] += (rnz - gravity_up_[2]) * kAlpha;
          }
        }
        if (gravity_up_valid_) {
          // Re-normalise (the lerp shortens the vector) and publish the held
          // up vector even on frames where the SDK dropped the floor plane.
          const float gl = std::sqrt(gravity_up_[0] * gravity_up_[0] +
                                     gravity_up_[1] * gravity_up_[1] +
                                     gravity_up_[2] * gravity_up_[2]);
          const float inv = gl > 1e-6f ? 1.0f / gl : 0.0f;
          guest_frame->normal_to_gravity.x = gravity_up_[0] * inv;
          guest_frame->normal_to_gravity.y = gravity_up_[1] * inv;
          guest_frame->normal_to_gravity.z = gravity_up_[2] * inv;
          guest_frame->normal_to_gravity.w = 0.0f;
        } else {
          // No gravity source seen yet; flat up until one becomes available.
          guest_frame->normal_to_gravity.x = 0.0f;
          guest_frame->normal_to_gravity.y = 1.0f;
          guest_frame->normal_to_gravity.z = 0.0f;
          guest_frame->normal_to_gravity.w = 0.0f;
        }
      } else {
        guest_frame->normal_to_gravity.x = 0.0f;
        guest_frame->normal_to_gravity.y = 1.0f;
        guest_frame->normal_to_gravity.z = 0.0f;
        guest_frame->normal_to_gravity.w = 0.0f;
      }
    }

    for (uint32_t i = 0; i < kNuiSkeletonCount; ++i) {
      auto& guest_skeleton = guest_frame->skeleton_data[i];
      guest_skeleton.tracking_state = kNuiSkeletonNotTracked;
      guest_skeleton.enrollment_index = kNuiInvalidEnrollmentIndex;
      guest_skeleton.user_index = kNuiInvalidUserIndex;
    }

    std::array<bool, kNuiSkeletonCount> copied{};
    uint32_t next_guest_index = 0;
    auto copy_skeleton = [&](uint32_t host_index, uint32_t guest_index,
                             bool bound) {
      const auto& host_skeleton = host_frame.skeleton_data[host_index];
      auto& guest_skeleton = guest_frame->skeleton_data[guest_index];
      guest_skeleton.tracking_state = host_skeleton.tracking_state;
      guest_skeleton.tracking_id = host_skeleton.tracking_id;
      if (bound) {
        guest_skeleton.enrollment_index = 0;
        guest_skeleton.user_index = 0;
      } else {
        guest_skeleton.enrollment_index = kNuiInvalidEnrollmentIndex;
        guest_skeleton.user_index = kNuiInvalidUserIndex;
      }
      WriteGuestVector(guest_skeleton.position, host_skeleton.position);
      for (uint32_t j = 0; j < kNuiSkeletonPositionCount; ++j) {
        WriteGuestVector(guest_skeleton.skeleton_positions[j],
                         host_skeleton.skeleton_positions[j]);
        guest_skeleton.skeleton_position_tracking_state[j] =
            host_skeleton.skeleton_position_tracking_state[j];
      }
      guest_skeleton.quality_flags = host_skeleton.quality_flags;
      // Face-cam continuity: titles (e.g. Just Dance) centre the colour
      // thumbnail on the HEAD joint of a FULLY TRACKED body, and size the
      // digital zoom from the HEAD<->SHOULDER_CENTER distance projected into
      // colour space (JD_PlayerHeadImage::getClipTextureDataFromBone). Host
      // tracking flickers tracked_count 1<->0, so most frames arrive
      // position-only (no joints) and the thumbnail sticks on a fixed point.
      // Present a position-only body as tracked with HEAD and SHOULDER_CENTER
      // joints inferred from the reported body centre, matching the standing
      // reference pose offsets (head ~0.78 m, shoulder-centre ~0.55 m above the
      // hip-centred body position). Both joints are required: with only HEAD
      // filled, the title's head<->shoulder span collapses to a near-zero/garbage
      // shoulder at the origin, so the zoom radius clamps to the full frame and
      // never magnifies. Only these two joints are filled; the rest stay
      // NOT_TRACKED so dance scoring, which needs the full body, still treats
      // this as a non-scoring skeleton.
      bool synthesized_facecam_body = false;
      if (cvars::xam_nui_synthesize_head_for_facecam && guest_index == 0 &&
          host_skeleton.tracking_state == kNuiSkeletonPositionOnly &&
          host_skeleton.position.z > 1e-4f) {
        synthesized_facecam_body = true;
        guest_skeleton.tracking_state = kNuiSkeletonTracked;
        constexpr uint32_t kShoulderCenterJoint = 2;
        constexpr uint32_t kHeadJoint = 3;
        constexpr int32_t kInferred = 1;  // NUI_SKELETON_POSITION_INFERRED
        // Keep the tuned head height (sets the crop centre); only ADD the
        // shoulder so the title's zoom radius has a valid second point.
        constexpr float kHeadAboveBody = 0.70f;
        // Standing-pose head/shoulder-centre gap (1.78 - 1.55 m); the title
        // sizes the digital zoom from this span, so it must stay realistic.
        constexpr float kHeadToShoulder = 0.23f;
        auto& head = guest_skeleton.skeleton_positions[kHeadJoint];
        head.x = host_skeleton.position.x;
        head.y = host_skeleton.position.y + kHeadAboveBody;
        head.z = host_skeleton.position.z;
        head.w = 1.0f;
        guest_skeleton.skeleton_position_tracking_state[kHeadJoint] = kInferred;
        auto& shoulder = guest_skeleton.skeleton_positions[kShoulderCenterJoint];
        shoulder.x = host_skeleton.position.x;
        shoulder.y = host_skeleton.position.y + kHeadAboveBody - kHeadToShoulder;
        shoulder.z = host_skeleton.position.z;
        shoulder.w = 1.0f;
        guest_skeleton.skeleton_position_tracking_state[kShoulderCenterJoint] =
            kInferred;
      }
      // Face-cam vertical framing for synthesized continuity bodies only. Do
      // not move real tracked HEAD/SHOULDER_CENTER joints: titles use those in
      // their leveled scoring pose, and the DC1/DC3 score fix depends on the
      // skeleton remaining raw apart from vNormalToGravity.
      if (synthesized_facecam_body &&
          cvars::xam_nui_facecam_head_drop > 0.0) {
        constexpr uint32_t kShoulderCenterJoint = 2;
        constexpr uint32_t kHeadJoint = 3;
        if (guest_skeleton.skeleton_position_tracking_state[kHeadJoint] != 0 &&
            guest_skeleton
                    .skeleton_position_tracking_state[kShoulderCenterJoint] !=
                0) {
          auto& head = guest_skeleton.skeleton_positions[kHeadJoint];
          auto& shoulder = guest_skeleton.skeleton_positions[kShoulderCenterJoint];
          const float f =
              static_cast<float>(cvars::xam_nui_facecam_head_drop);
          const float hx = head.x, hy = head.y, hz = head.z;
          const float sx = shoulder.x, sy = shoulder.y, sz = shoulder.z;
          // Shift vector = f * (head - shoulder), i.e. f of the neck length,
          // pointing up the neck; subtract it from both to slide the pair down.
          const float vx = f * (hx - sx);
          const float vy = f * (hy - sy);
          const float vz = f * (hz - sz);
          head.x = hx - vx;
          head.y = hy - vy;
          head.z = hz - vz;
          shoulder.x = sx - vx;
          shoulder.y = sy - vy;
          shoulder.z = sz - vz;
        }
      }
      copied[host_index] = true;
    };

    if (bound_tracking_id != kNuiInvalidTrackingId) {
      for (uint32_t i = 0; i < kNuiSkeletonCount; ++i) {
        const auto& host_skeleton = host_frame.skeleton_data[i];
        if (host_skeleton.tracking_state == kNuiSkeletonTracked &&
            host_skeleton.tracking_id == bound_tracking_id) {
          copy_skeleton(i, next_guest_index++, true);
          break;
        }
      }
    }

    for (uint32_t i = 0; i < kNuiSkeletonCount; ++i) {
      if (next_guest_index >= kNuiSkeletonCount) {
        break;
      }
      if (copied[i] ||
          host_frame.skeleton_data[i].tracking_state != kNuiSkeletonTracked) {
        continue;
      }
      copy_skeleton(i, next_guest_index++, false);
    }

    for (uint32_t i = 0; i < kNuiSkeletonCount; ++i) {
      if (next_guest_index >= kNuiSkeletonCount) {
        break;
      }
      if (copied[i] ||
          host_frame.skeleton_data[i].tracking_state !=
              kNuiSkeletonPositionOnly) {
        continue;
      }
      copy_skeleton(i, next_guest_index++, false);
    }
  }

  void StartSkeletonPoller() {
    if (!HasSkeletonFrameReader() ||
        poller_running_.exchange(true, std::memory_order_relaxed)) {
      return;
    }
    poller_thread_ = std::thread([this]() {
      EnsureKinectComInitializedForCurrentThread();
      XELOGI("Kinect v1 skeleton background polling started");
      // Seed the cached elevation angle from the physical motor once so the
      // extrapolated floor plane reflects the real sensor tilt even before the
      // title issues its first SetElevationAngle. Done here (off any held lock)
      // because GetElevationAngle takes the runtime mutex.
      {
        int32_t seed_angle = 0;
        if (SUCCEEDED(GetElevationAngle(&seed_angle))) {
          last_elevation_angle_.store(seed_angle, std::memory_order_relaxed);
        }
      }
      uint32_t elevation_refresh = 0;
      // Per-second delivery-health summary, gated behind the same diagnostic
      // cvar as the skeleton-value dump. Answers "is Xenia keeping up with the
      // Kinect constantly?": frames/s should hold ~30 and the per-frame delivery
      // period should stay near 33 ms. A high depth cost or a large period max
      // means depth work on this thread is stalling skeleton delivery (the
      // title then scores against a stale/bursty pose).
      uint32_t health_iters = 0;
      double health_depth_ms_sum = 0.0, health_depth_ms_max = 0.0;
      double health_skel_ms_max = 0.0;
      double health_period_ms_max = 0.0;
      auto health_window_start = std::chrono::steady_clock::now();
      auto prev_poll_start = health_window_start;
      while (poller_running_.load(std::memory_order_relaxed)) {
        const auto poll_start = std::chrono::steady_clock::now();
        const double period_ms = std::chrono::duration<double, std::milli>(
                                     poll_start - prev_poll_start)
                                     .count();
        prev_poll_start = poll_start;
        // Track the physical motor (~1 Hz, off any held lock) so the
        // extrapolated floor plane rotates as the sensor tilts during the
        // start-up floor scan and follows any manual tilt. Cheap relative to
        // the depth work below; throttled to keep skeleton delivery at ~30 Hz.
        if ((elevation_refresh++ % 30u) == 0u) {
          int32_t motor_angle = 0;
          if (SUCCEEDED(GetElevationAngle(&motor_angle))) {
            last_elevation_angle_.store(motor_angle,
                                        std::memory_order_relaxed);
          }
          // Refresh the physical gravity-up from the accelerometer at the same
          // ~1 Hz cadence so the skeleton leveling tracks the sensor's true
          // orientation even when the tilt motor is locked / reporting a
          // virtual angle. Holds the last good reading on a failed/rejected
          // sample.
          float up[3];
          if (ReadAccelerometerUp(up)) {
            accel_up_x_.store(up[0], std::memory_order_relaxed);
            accel_up_y_.store(up[1], std::memory_order_relaxed);
            accel_up_z_.store(up[2], std::memory_order_relaxed);
            accel_up_valid_.store(true, std::memory_order_relaxed);
          }
        }
        const bool skeleton_updated =
            PollSkeletonFrameOnce(kNuiSkeletonPollWaitMs);
        const auto after_skel = std::chrono::steady_clock::now();
        const bool depth_updated = PollDepthFrameOnce(0);
        if (depth_updated) {
          PublishDepthToGuest();
        }
        const auto after_depth = std::chrono::steady_clock::now();
        if (skeleton_updated || depth_updated) {
          SignalFrameEndEvent();
        }
        if (cvars::xam_nui_log_skeleton_values > 0) {
          const double skel_ms = std::chrono::duration<double, std::milli>(
                                     after_skel - poll_start)
                                     .count();
          const double depth_ms = std::chrono::duration<double, std::milli>(
                                      after_depth - after_skel)
                                      .count();
          ++health_iters;
          health_depth_ms_sum += depth_ms;
          health_depth_ms_max = std::max(health_depth_ms_max, depth_ms);
          health_skel_ms_max = std::max(health_skel_ms_max, skel_ms);
          // Ignore the very first period (prev_poll_start seeded at startup).
          if (health_iters > 1) {
            health_period_ms_max = std::max(health_period_ms_max, period_ms);
          }
          const double window_ms = std::chrono::duration<double, std::milli>(
                                       after_depth - health_window_start)
                                       .count();
          if (window_ms >= 1000.0) {
            XELOGI(
                "NUI-HEALTH: {} frames/s | depth avg={:.1f} max={:.1f}ms | "
                "skel max={:.1f}ms | deliver-period max={:.1f}ms (ideal ~33)",
                health_iters, health_depth_ms_sum / health_iters,
                health_depth_ms_max, health_skel_ms_max, health_period_ms_max);
            health_iters = 0;
            health_depth_ms_sum = health_depth_ms_max = 0.0;
            health_skel_ms_max = 0.0;
            health_period_ms_max = 0.0;
            health_window_start = after_depth;
          }
        }
        const auto elapsed = std::chrono::steady_clock::now() - poll_start;
        // Poll at the SDK's native skeleton rate (~30 Hz). Throttling the poller
        // slower than the sensor delivers drops skeleton frames, which makes the
        // real tracking flicker (tracked_count 1<->0) and the engagement gate
        // never latch. A 16 ms target keeps the loop from sleeping past one
        // sensor frame; PollSkeletonFrameOnce already blocks for the next frame.
        constexpr auto target = std::chrono::milliseconds(16);
        if (elapsed < target) {
          std::this_thread::sleep_for(target - elapsed);
        }
      }
      XELOGI("Kinect v1 skeleton background polling stopped");
    });
  }

  void StopSkeletonPoller() {
    if (!poller_running_.exchange(false, std::memory_order_relaxed)) {
      return;
    }
    if (poller_thread_.joinable()) {
      poller_thread_.join();
    }
  }

  void StartColorPoller() {
    if (!color_stream_ ||
        color_poller_running_.exchange(true, std::memory_order_relaxed)) {
      return;
    }
    color_poller_thread_ = std::thread([this]() {
      EnsureKinectComInitializedForCurrentThread();
      XELOGI("Kinect v1 color background polling started");
      while (color_poller_running_.load(std::memory_order_relaxed)) {
        if (PollColorFrameOnce(30)) {
          PublishColorToGuest();
        }
      }
      XELOGI("Kinect v1 color background polling stopped");
    });
  }

  void StopColorPoller() {
    if (!color_poller_running_.exchange(false, std::memory_order_relaxed)) {
      return;
    }
    if (color_poller_thread_.joinable()) {
      color_poller_thread_.join();
    }
  }

  void SignalSkeletonFrameEvent() {
    SignalGuestFrameEvent(
        skeleton_frame_event_handle_.load(std::memory_order_relaxed));
  }

  void CloseKinectEvent() {
    if (skeleton_event_) {
      CloseHandle(skeleton_event_);
      skeleton_event_ = nullptr;
    }
    if (depth_event_) {
      CloseHandle(depth_event_);
      depth_event_ = nullptr;
    }
    if (color_event_) {
      CloseHandle(color_event_);
      color_event_ = nullptr;
    }
  }

  static void LogFailureOnce(const char* message, HRESULT hr) {
    static std::mutex log_mutex;
    static std::string last_message;
    static HRESULT last_hr = S_OK;
    std::lock_guard<std::mutex> lock(log_mutex);
    if (last_message == message && last_hr == hr) {
      return;
    }
    last_message = message;
    last_hr = hr;
    XELOGW("Kinect v1 {}: {:08X}", message, static_cast<uint32_t>(hr));
  }

  mutable std::mutex latest_skeleton_frame_mutex_;
  std::mutex color_view_mutex_;
  std::mutex mutex_;
  std::mutex skeleton_poll_mutex_;
  std::mutex skeleton_delivery_mutex_;
  std::thread poller_thread_;
  std::thread color_poller_thread_;
  HMODULE module_ = nullptr;
  INuiSensor* sensor_ = nullptr;
  NuiInitializeFn nui_initialize_ = nullptr;
  NuiShutdownFn nui_shutdown_ = nullptr;
  NuiGetSensorCountFn nui_get_sensor_count_ = nullptr;
  NuiCreateSensorByIndexFn nui_create_sensor_by_index_ = nullptr;
  NuiImageStreamOpenFn nui_image_stream_open_ = nullptr;
  NuiImageStreamGetNextFrameFn nui_image_stream_get_next_frame_ = nullptr;
  NuiImageStreamReleaseFrameFn nui_image_stream_release_frame_ = nullptr;
  NuiImageGetColorPixelCoordinatesFromDepthPixelFn
      nui_image_get_color_pixel_coordinates_from_depth_pixel_ = nullptr;
  NuiImageGetColorPixelCoordinateFrameFromDepthPixelFrameAtResolutionFn
      nui_image_get_color_pixel_coordinate_frame_from_depth_pixel_frame_at_resolution_ =
          nullptr;
  NuiSkeletonTrackingEnableFn nui_skeleton_tracking_enable_ = nullptr;
  NuiSkeletonGetNextFrameFn nui_skeleton_get_next_frame_ = nullptr;
  NuiSkeletonSetTrackedSkeletonsFn nui_skeleton_set_tracked_skeletons_ = nullptr;
  NuiCameraElevationGetAngleFn nui_camera_elevation_get_angle_ = nullptr;
  NuiCameraElevationSetAngleFn nui_camera_elevation_set_angle_ = nullptr;
  HANDLE skeleton_event_ = nullptr;
  HANDLE depth_event_ = nullptr;
  HANDLE depth_stream_ = nullptr;
  HANDLE color_event_ = nullptr;
  HANDLE color_stream_ = nullptr;
  int sensor_count_ = 0;
  bool started_ = false;
  bool skeleton_tracking_enabled_ = false;
  DWORD skeleton_tracking_flags_ = kNuiSkeletonTrackingAutomatic;
  // Last body selection pushed to NuiSkeletonSetTrackedSkeletons by the host
  // segmentation driver. Touched only on the skeleton poller thread.
  uint32_t last_driven_tracked_ids_[2] = {0, 0};
  // Set once a title drives skeleton selection itself, so the host segmentation
  // driver backs off and defers to the title's choice.
  std::atomic_bool guest_set_tracked_skeletons_ = false;
  std::atomic_bool poller_running_ = false;
  std::atomic_bool color_poller_running_ = false;
  std::atomic<uint32_t> skeleton_no_data_count_ = 0;
  std::atomic<uint32_t> latest_skeleton_frame_number_ = 0;
  std::atomic<uint32_t> last_delivered_skeleton_frame_ = 0;
  std::atomic_bool latest_skeleton_frame_valid_ = false;
  std::atomic<uint32_t> skeleton_frame_event_handle_ = 0;
  std::atomic<uint32_t> frame_end_event_handle_ = 0;
  std::mutex guest_event_mutex_;
  std::vector<uint32_t> image_frame_event_handles_;
  std::vector<uint32_t> color_frame_event_handles_;
  std::atomic<uint32_t> tracked_skeleton_count_ = 0;
  std::atomic<uint32_t> best_tracking_id_ = kNuiInvalidTrackingId;
  std::atomic<uint32_t> selected_host_player_index_ = 0;
  std::array<std::atomic<uint32_t>, kNuiImagePlayerIndexMask + 1>
      depth_player_index_remap_{};
  std::atomic<uint32_t> depth_single_guest_player_index_ = 0;
  mutable std::mutex latest_depth_mutex_;
  std::vector<uint16_t> latest_depth_;
  std::vector<uint16_t> latest_depth_color_space_;
  std::vector<LONG> depth_to_color_coordinates_;
  std::atomic_bool latest_depth_valid_ = false;
  std::atomic_bool latest_depth_color_space_valid_ = false;
  std::atomic<int64_t> pending_depth_timestamp_ = 0;
  std::atomic<uint32_t> pending_depth_frame_number_ = 0;
  std::atomic<int64_t> latest_depth_timestamp_ = 0;
  std::atomic<uint32_t> latest_depth_frame_number_ = 0;
  std::atomic_bool depth_guest_valid_ = false;
  std::atomic<uint32_t> depth_frame_count_ = 0;
  std::atomic<uint32_t> depth_guest_addr_ = 0;
  std::atomic<uint32_t> depth384_guest_addr_ = 0;
  std::atomic<uint32_t> depth_only384_guest_addr_ = 0;
  std::atomic<uint32_t> depth_color384_guest_addr_ = 0;
  std::atomic<uint32_t> depth_color_only384_guest_addr_ = 0;
  std::atomic<uint32_t> mini_depth_guest_addr_ = 0;
  std::atomic<uint32_t> mini128_depth_guest_addr_ = 0;
  std::atomic<uint32_t> mini128_depth_only_guest_addr_ = 0;
  // 640x480, 1280-byte pitch (0x96000 bytes) big-endian USHORT depth+player
  // buffer. This is the dense buffer Just Dance 2019's silhouette converter
  // (0x82B94210) walks via the synthesized transfer descriptor desc+0x64. Any
  // title that wants a full-color-resolution depth+player map can also use it.
  std::atomic<uint32_t> depth640_guest_addr_ = 0;
  // Persistent JD2019 NUI transfer descriptor synthesized into device+0xA0.
  std::atomic<uint32_t> jd2019_transfer_desc_addr_ = 0;
  std::atomic<uint32_t> jd2019_desc_timestamp_ = 0;
  // DC3 push-camera frame-delivery dispatch: a dedicated guest-context thread
  // that calls DC3's _NUICAM depth callback (0x829CE1C0) when woken per frame.
  object_ref<XHostThread> dc3_dispatch_thread_;
  std::atomic<bool> dc3_dispatch_running_{false};
  std::atomic<uint32_t> dc3_frame_pending_{0};
  std::atomic<uint32_t> dc3_callback_r3_{0};
  std::atomic<uint32_t> dc3_callback_desc_{0};
  std::atomic<uint32_t> dc3_color_r3_{0};
  std::atomic<uint32_t> dc3_color_desc_{0};
  std::vector<uint8_t> dc3_cs640_;  // 640x480 color-space depth for DC3 push bufs
  std::mutex dc3_dispatch_mutex_;
  std::condition_variable dc3_dispatch_cond_;
  // XAM's tilt service reports asynchronous motor completion to the callback
  // registered by the statically linked NUI runtime. Without this notification,
  // its floor search moves to -25 degrees and never issues the return move.
  object_ref<XHostThread> tilt_dispatch_thread_;
  std::atomic<bool> tilt_dispatch_running_{false};
  std::atomic<bool> tilt_notification_pending_{false};
  std::atomic<uint32_t> tilt_callback_{0};
  std::atomic<uint32_t> tilt_callback_context_{0};
  std::atomic<uint32_t> tilt_callback_data_address_{0};
  std::atomic<uint32_t> tilt_command_serial_{0};
  std::atomic<int32_t> tilt_requested_angle_{0};
  std::mutex tilt_dispatch_mutex_;
  std::condition_variable tilt_dispatch_cond_;
  // Last known camera elevation angle (degrees), updated on every
  // SetElevationAngle and refreshed from the motor by the skeleton poller. Used
  // to synthesize the floor clip plane for titles whose start-up "scan the
  // floor" calibration reads a floor plane the Windows SDK never fills in.
  std::atomic<int32_t> last_elevation_angle_{0};
  // Set once a title issues a tilt command while xam_nui_lock_elevation is on,
  // so GetElevationAngle reports the commanded (virtual) angle instead of the
  // unmoved motor. Cleared (default) means "report the real mounted angle".
  std::atomic<bool> elevation_virtual_valid_{false};
  std::atomic<bool> floor_extrapolation_logged_{false};
  // Physical gravity-up unit vector from the sensor accelerometer, refreshed by
  // the skeleton poller. This is the live "which way is down" used to level the
  // skeleton for scoring; it follows the sensor being physically re-aimed,
  // which the (possibly locked / virtual) tilt motor angle does not.
  std::atomic<float> accel_up_x_{0.0f};
  std::atomic<float> accel_up_y_{1.0f};
  std::atomic<float> accel_up_z_{0.0f};
  std::atomic<bool> accel_up_valid_{false};
  // One-shot guard so the accelerometer status (online + value, or why it is
  // unavailable) is logged exactly once instead of every poll.
  std::atomic<bool> accel_diag_logged_{false};
  // JD4 (555308B5) drives the camera with a PUSH model: it registers physical
  // frame buffers via _NUICAM opcode 0x05 and never calls the pull-fetch path
  // (0x1004/0x1402), so the host must fill those buffers itself. These hold the
  // type-4 (640x480x16bpp depth+player = the silhouette, 0x96000 bytes) targets
  // JD4 registers (it triple-buffers). The depth pump copies the freshly built
  // silhouette into each one every frame.
  static constexpr uint32_t kJd4MaxCameraBuffers = 4;
  std::array<std::atomic<uint32_t>, kJd4MaxCameraBuffers> jd4_depth_buffers_{};
  std::atomic<uint32_t> jd4_depth_buffer_count_ = 0;
  std::array<std::atomic<uint32_t>, kJd4MaxCameraBuffers> jd4_color_buffers_{};
  std::atomic<uint32_t> jd4_color_buffer_count_ = 0;
  mutable std::mutex latest_color_mutex_;
  std::vector<uint8_t> latest_color_;
  std::atomic_bool latest_color_valid_ = false;
  std::atomic<int64_t> pending_color_timestamp_ = 0;
  std::atomic<uint32_t> pending_color_frame_number_ = 0;
  std::atomic<int64_t> latest_color_timestamp_ = 0;
  std::atomic<uint32_t> latest_color_frame_number_ = 0;
  std::atomic_bool color_guest_valid_ = false;
  std::atomic<uint32_t> color_frame_count_ = 0;
  std::atomic<uint32_t> color_guest_addr_ = 0;
  std::atomic<uint32_t> color_yuv_guest_addr_ = 0;
  bool color_view_locked_ = false;
  float color_view_center_x_ = 0.0f;
  float color_view_center_y_ = 0.0f;
  std::chrono::steady_clock::time_point color_view_last_real_head_time_{};
  std::atomic<uint32_t> last_logged_tracked_count_ =
      std::numeric_limits<uint32_t>::max();
  std::atomic<uint32_t> last_logged_position_only_count_ =
      std::numeric_limits<uint32_t>::max();
  std::atomic<uint32_t> last_logged_host_player_index_ =
      std::numeric_limits<uint32_t>::max();
  X_NUI_SKELETON_FRAME latest_skeleton_frame_ = {};
  // Last HEAD joint position seen by the skeleton-value diagnostic, for the
  // per-frame jitter delta. Only touched on the skeleton poller thread.
  float last_dump_head_[3] = {0.0f, 0.0f, 0.0f};
  // Per-frame (not per-dump) HEAD-joint jitter accumulators, updated every
  // tracked frame on the poller thread and summarised in each NUI-VALUES dump.
  // The min over a window approximates the raw SDK noise floor (frames where
  // the dancer is briefly still); the max is the fastest real movement.
  float last_frame_head_[3] = {0.0f, 0.0f, 0.0f};
  bool last_frame_head_valid_ = false;
  double jitter_sum_ = 0.0;
  double jitter_min_ = 0.0;
  double jitter_max_ = 0.0;
  uint32_t jitter_count_ = 0;
  // Smoothed gravity-up vector derived from the floor clip plane normal, held
  // across frames where the SDK drops the floor plane. Skeleton poller thread.
  float gravity_up_[3] = {0.0f, 1.0f, 0.0f};
  bool gravity_up_valid_ = false;
};

KinectV1SkeletonRuntime& GetKinectV1SkeletonRuntime() {
  static KinectV1SkeletonRuntime runtime;
  return runtime;
}
#endif  // XE_PLATFORM_WIN32

std::atomic_bool g_nui_force_device_off{false};
std::atomic_bool g_nui_chat_mic_enabled{false};
std::atomic_bool g_nui_automation_enabled{false};
std::atomic_bool g_natal_playback_enabled{false};
std::array<std::atomic_bool, XUserMaxUserCount> g_nui_biometric_enabled{};

bool IsNuiFakeConnected() {
  if (g_nui_force_device_off.load(std::memory_order_relaxed)) {
    return false;
  }
  return cvars::xam_nui_fake_connected ||
         (kernel_state()->xconfig()->ReadSetting<uint32_t>(
              X_CONFIG_CATEGORY::XCONFIG_USER_CATEGORY,
              XCONFIG_USER_RETAIL_FLAGS) &
          X_RETAIL_FLAGS::KinectInitialized);
}

bool IsNuiRealReady() {
  if (g_nui_force_device_off.load(std::memory_order_relaxed)) {
    return false;
  }
#if XE_PLATFORM_WIN32
  return GetKinectV1SkeletonRuntime().IsReady();
#else
  return false;
#endif
}

// Whether to stand in a single synthetic tracked player. This is intentionally
// limited to the fake backend/retail flag path. In auto/real modes reporting a
// player before a full NUI_SKELETON_FRAME exists lets titles advance into camera
// and scoring paths with invalid joints.
bool ShouldSynthesizeNuiPlayer() {
  return IsNuiFakeConnected();
}

uint32_t GetSyntheticTrackingId() {
  return ShouldSynthesizeNuiPlayer() ? kNuiSyntheticTrackingId : 0;
}

uint32_t g_engaged_nui_tracking_id = 0;

constexpr uint32_t kNuiIdentityEnrollSkeletonFlag = 1;
constexpr uint32_t kNuiIdentityNoEnrollmentIndex = 0xFFFFFFFFu;

uint32_t GetNuiDeviceStatus() {
  return (IsNuiRealReady() || IsNuiFakeConnected()) ? 1 : 0;
}

bool IsNuiReady() { return GetNuiDeviceStatus() != 0; }

uint32_t GetNuiTrackedSkeletonCount() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    const uint32_t tracked = runtime.TrackedSkeletonCount();
    if (tracked != 0) {
      return tracked;
    }
  }
#endif
  return ShouldSynthesizeNuiPlayer() ? 1 : 0;
}

// Blocks (up to timeout_ms) for the host sensor to deliver a tracked skeleton,
// then returns the tracked count. Falls back to the synthetic player count when
// synthesis is enabled and no real body arrives.
uint32_t WaitForNuiTrackedSkeleton(uint32_t timeout_ms) {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    runtime.WaitForTrackedSkeletonFrame(timeout_ms);
    const uint32_t tracked_count = runtime.TrackedSkeletonCount();
    return tracked_count != 0 || !ShouldSynthesizeNuiPlayer() ? tracked_count
                                                              : 1;
  }
#else
  (void)timeout_ms;
#endif
  return GetNuiTrackedSkeletonCount();
}

uint32_t GetNuiBestTrackingId() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    const uint32_t tracking_id = runtime.BestTrackingId();
    if (tracking_id != 0) {
      return tracking_id;
    }
  }
#endif
  return GetSyntheticTrackingId();
}

uint32_t GetNuiBestSkeletonIndex() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady() && runtime.TrackedSkeletonCount() != 0) {
    return runtime.BestSkeletonIndex();
  }
#endif
  return ShouldSynthesizeNuiPlayer() ? 0 : 0xFFFFFFFFu;
}

bool HasNuiIdentityPlayer(uint32_t tracking_id) {
  if (!IsNuiReady()) {
    return false;
  }
  const uint32_t best_tracking_id = GetNuiBestTrackingId();
  if (GetNuiTrackedSkeletonCount() == 0 ||
      best_tracking_id == kNuiInvalidTrackingId) {
    return false;
  }
  if (tracking_id == kNuiInvalidTrackingId ||
      tracking_id == kNuiIdentityNoEnrollmentIndex) {
    return true;
  }
  return tracking_id == best_tracking_id;
}

uint32_t GetNuiIdentityEnrollmentIndex(uint32_t tracking_id) {
  return HasNuiIdentityPlayer(tracking_id) ? 0 : kNuiIdentityNoEnrollmentIndex;
}

uint32_t GetNuiIdentityUserIndex(uint32_t tracking_id) {
  return HasNuiIdentityPlayer(tracking_id) ? 0 : 0xFFu;
}

uint32_t GetNuiIdentityEnrollmentFlags(uint32_t tracking_id) {
  return HasNuiIdentityPlayer(tracking_id) ? kNuiIdentityEnrollSkeletonFlag : 0;
}

uint32_t GetNuiCameraElevationAngle(int32_t* angle) {
  if (!angle) {
    return X_E_INVALIDARG;
  }
#if XE_PLATFORM_WIN32
  if (!g_nui_force_device_off.load(std::memory_order_relaxed)) {
    const HRESULT hr = GetKinectV1SkeletonRuntime().GetElevationAngle(angle);
    if (SUCCEEDED(hr)) {
      return X_STATUS_SUCCESS;
    }
  }
#endif
  if (IsNuiFakeConnected()) {
    *angle = 0;
    return X_STATUS_SUCCESS;
  }
  return X_E_DEVICE_NOT_CONNECTED;
}

uint32_t SetNuiCameraElevationAngle(int32_t angle) {
  if (angle < -27 || angle > 27) {
    return X_E_INVALIDARG;
  }
#if XE_PLATFORM_WIN32
  if (!g_nui_force_device_off.load(std::memory_order_relaxed)) {
    const HRESULT hr = GetKinectV1SkeletonRuntime().SetElevationAngle(angle);
    if (SUCCEEDED(hr)) {
      return X_STATUS_SUCCESS;
    }
  }
#endif
  return IsNuiFakeConnected() ? X_STATUS_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}

uint32_t GetNuiSkeletonFrameNumber() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    return runtime.LatestSkeletonFrameNumber();
  }
#endif
  static uint32_t frame_number = 1;
  return IsNuiReady() ? frame_number++ : 0;
}

int64_t GetNuiDepthTimestamp() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    return runtime.DepthTimestamp();
  }
#endif
  return 0;
}

uint32_t GetNuiDepthFrameNumber() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    return runtime.DepthFrameNumber();
  }
#endif
  return GetNuiSkeletonFrameNumber();
}

uint32_t GetNuiDepthGuestAddress() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    return runtime.DepthGuestAddress();
  }
#endif
  return 0;
}

uint32_t GetNuiDepth384GuestAddress() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    return runtime.Depth384GuestAddress();
  }
#endif
  return 0;
}

uint32_t GetNuiMini128GuestAddress() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    return runtime.Mini128DepthGuestAddress();
  }
#endif
  return 0;
}

void RegisterJd4CameraBuffer(uint32_t type, uint32_t phys_addr, uint32_t size) {
#if XE_PLATFORM_WIN32
  GetKinectV1SkeletonRuntime().RegisterJd4CameraBuffer(type, phys_addr, size);
#else
  (void)type;
  (void)phys_addr;
  (void)size;
#endif
}

uint32_t GetNuiMiniDepthGuestAddress() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    return runtime.MiniDepthGuestAddress();
  }
#endif
  return 0;
}

uint32_t GetNuiMini128DepthGuestAddress() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    return runtime.Mini128DepthGuestAddress();
  }
#endif
  return 0;
}

uint32_t GetNuiMini128DepthOnlyGuestAddress() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    return runtime.Mini128DepthOnlyGuestAddress();
  }
#endif
  return 0;
}

int64_t GetNuiColorTimestamp() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    return runtime.ColorTimestamp();
  }
#endif
  return 0;
}

uint32_t GetNuiColorFrameNumber() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    return runtime.ColorFrameNumber();
  }
#endif
  return GetNuiSkeletonFrameNumber();
}

uint32_t GetNuiColorGuestAddress() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    return runtime.ColorGuestAddress();
  }
#endif
  return 0;
}

uint32_t GetNuiDepthOnly384GuestAddress() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    return runtime.DepthOnly384GuestAddress();
  }
#endif
  return 0;
}

uint32_t GetNuiDepthColorSpace384GuestAddress() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    return runtime.DepthColorSpace384GuestAddress();
  }
#endif
  return 0;
}

uint32_t GetNuiDepthColorSpaceOnly384GuestAddress() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    return runtime.DepthColorSpaceOnly384GuestAddress();
  }
#endif
  return 0;
}

uint32_t GetNuiColorYuvGuestAddress() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    return runtime.ColorYuvGuestAddress();
  }
#endif
  return 0;
}

std::atomic<uint32_t> g_nui_image_texture_guest_addr{0};
std::atomic<uint32_t> g_nui_depth_only_image_texture_guest_addr{0};
std::atomic<uint32_t> g_nui_depth_color_space_image_texture_guest_addr{0};
std::atomic<uint32_t>
    g_nui_depth_color_space_only_image_texture_guest_addr{0};
std::atomic<uint32_t> g_nui_color_image_texture_guest_addr{0};
std::atomic<uint32_t> g_nui_color_yuv_image_texture_guest_addr{0};
std::atomic<uint32_t> g_nui_mini_depth_image_texture_guest_addr{0};
std::atomic<uint32_t> g_nui_mini_depth_only_image_texture_guest_addr{0};
std::array<std::atomic<uint32_t>, kNuiImageTypeCount>
    g_nui_image_frame_node_guest_addr_by_type{};

void WriteNuiDepthTextureFetch(uint8_t* texture, uint32_t depth_base,
                               uint32_t width, uint32_t height,
                               uint32_t stride_pixels) {
  const uint32_t pitch_field = stride_pixels >> 5;
  const uint32_t base_address_field = depth_base >> 12;
  const uint32_t width_minus_one = width - 1;
  const uint32_t height_minus_one = height - 1;

  uint32_t fetch[6] = {};
  fetch[0] = kNuiGpuTextureFetchTypeTexture |
             (kNuiGpuTextureSignUnsigned << 2) |
             (kNuiGpuTextureSignUnsigned << 4) |
             (kNuiGpuTextureSignUnsigned << 6) |
             (kNuiGpuTextureSignUnsigned << 8) |
             (kNuiGpuTextureClampToEdge << 10) |
             (kNuiGpuTextureClampToEdge << 13) |
             (kNuiGpuTextureClampToEdge << 16) | (pitch_field << 22);
  fetch[1] = kNuiGpuTextureFormat16 | (kNuiGpuEndian8In16 << 6) |
             (kNuiGpuRequestSize256Bit << 8) |
             (base_address_field << 12);
  fetch[2] = width_minus_one | (height_minus_one << 13);
  fetch[3] = kNuiGpuNumFormatInteger | (kNuiGpuSwizzleABGR << 1) |
             (kNuiGpuTextureFilterPoint << 19) |
             (kNuiGpuTextureFilterPoint << 21) |
             (kNuiGpuMipFilterBaseMap << 23) |
             (kNuiGpuAnisoDisabled << 25);
  fetch[4] = 0;
  fetch[5] = kNuiGpuDimension2D << 9;

  for (uint32_t i = 0; i < 6; ++i) {
    xe::store_and_swap<uint32_t>(
        texture + kNuiD3DTextureFormatOffset + i * sizeof(uint32_t), fetch[i]);
  }

  static std::atomic_bool logged{false};
  bool expected = false;
  if (logged.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
    XELOGI("Kinect v1 NUI D3DTexture fetch: depth={:08X} {}x{} stride={} "
           "fetch={:08X} {:08X} {:08X} {:08X} {:08X} {:08X}",
           depth_base, width, height, stride_pixels, fetch[0], fetch[1],
           fetch[2], fetch[3], fetch[4], fetch[5]);
  }
}

void WriteNuiColorTextureFetch(uint8_t* texture, uint32_t image_base,
                               bool yuv) {
  const uint32_t pitch_field = kNuiD3DColorWidth >> 5;
  const uint32_t base_address_field = image_base >> 12;

  uint32_t fetch[6] = {};
  fetch[0] = kNuiGpuTextureFetchTypeTexture |
             (kNuiGpuTextureSignUnsigned << 2) |
             (kNuiGpuTextureSignUnsigned << 4) |
             (kNuiGpuTextureSignUnsigned << 6) |
             (kNuiGpuTextureSignUnsigned << 8) |
             (kNuiGpuTextureClampToEdge << 10) |
             (kNuiGpuTextureClampToEdge << 13) |
             (kNuiGpuTextureClampToEdge << 16) | (pitch_field << 22);
  fetch[1] = (yuv ? kNuiGpuTextureFormatYuy2 : kNuiGpuTextureFormat8888) |
             ((yuv ? kNuiGpuEndian8In16 : kNuiGpuEndian8In32) << 6) |
             (kNuiGpuRequestSize256Bit << 8) | (base_address_field << 12);
  fetch[2] = (kNuiD3DColorWidth - 1) | ((kNuiD3DColorHeight - 1) << 13);
  fetch[3] = kNuiGpuNumFormatFraction |
             ((yuv ? kNuiGpuSwizzleABGR : kNuiGpuSwizzleORGB) << 1) |
             (kNuiGpuTextureFilterPoint << 19) |
             (kNuiGpuTextureFilterPoint << 21) |
             (kNuiGpuMipFilterBaseMap << 23) |
             (kNuiGpuAnisoDisabled << 25);
  fetch[4] = 0;
  fetch[5] = kNuiGpuDimension2D << 9;

  for (uint32_t i = 0; i < 6; ++i) {
    xe::store_and_swap<uint32_t>(
        texture + kNuiD3DTextureFormatOffset + i * sizeof(uint32_t), fetch[i]);
  }
}

uint32_t GetNuiDepthTextureGuestAddress(std::atomic<uint32_t>& slot,
                                        uint32_t depth, uint32_t width,
                                        uint32_t height,
                                        uint32_t stride_pixels,
                                        const char* description) {
  if (!depth || !kernel_state()) {
    return 0;
  }
  auto* memory = kernel_state()->memory();
  uint32_t addr = slot.load(std::memory_order_relaxed);
  if (!addr) {
    addr = memory->SystemHeapAlloc(kNuiD3DTextureObjectSize);
    if (!addr) {
      return 0;
    }
    slot.store(addr, std::memory_order_relaxed);
    XELOGI("Kinect v1 NUI {} D3DTexture sentinel allocated at {:08X}",
           description, addr);
  }

  auto* p = memory->TranslateVirtual<uint8_t*>(addr);
  if (!p) {
    return 0;
  }

  std::memset(p, 0, kNuiD3DTextureObjectSize);
  xe::store_and_swap<uint32_t>(
      p + 0x00, kNuiD3DCommonTypeTexture | kNuiD3DCommonD3DCreated |
                    kNuiD3DCommonCpuCachedMemory);
  xe::store_and_swap<uint32_t>(p + 0x04, 1);
  xe::store_and_swap<uint32_t>(p + 0x10, kNuiD3DTextureIdentifier);
  xe::store_and_swap<uint32_t>(p + 0x14, kNuiD3DFlushInitialValue);
  xe::store_and_swap<uint32_t>(p + 0x18, kNuiD3DFlushInitialValue);
  WriteNuiDepthTextureFetch(p, depth, width, height, stride_pixels);
  return addr;
}

uint32_t GetNuiImageTextureGuestAddress() {
  return GetNuiDepthTextureGuestAddress(
      g_nui_image_texture_guest_addr, GetNuiDepth384GuestAddress(),
      kNuiD3DDepthWidth, kNuiD3DDepthHeight, kNuiD3DDepthStridePixels,
      "depth+player");
}

uint32_t GetNuiDepthOnlyImageTextureGuestAddress() {
  return GetNuiDepthTextureGuestAddress(
      g_nui_depth_only_image_texture_guest_addr,
      GetNuiDepthOnly384GuestAddress(), kNuiD3DDepthWidth, kNuiD3DDepthHeight,
      kNuiD3DDepthStridePixels, "depth-only");
}

uint32_t GetNuiDepthColorSpaceImageTextureGuestAddress() {
  return GetNuiDepthTextureGuestAddress(
      g_nui_depth_color_space_image_texture_guest_addr,
      GetNuiDepthColorSpace384GuestAddress(), kNuiD3DDepthWidth,
      kNuiD3DDepthHeight, kNuiD3DDepthStridePixels, "depth+player color-space");
}

uint32_t GetNuiDepthColorSpaceOnlyImageTextureGuestAddress() {
  return GetNuiDepthTextureGuestAddress(
      g_nui_depth_color_space_only_image_texture_guest_addr,
      GetNuiDepthColorSpaceOnly384GuestAddress(), kNuiD3DDepthWidth,
      kNuiD3DDepthHeight, kNuiD3DDepthStridePixels, "depth-only color-space");
}

uint32_t GetNuiMiniDepthImageTextureGuestAddress() {
  return GetNuiDepthTextureGuestAddress(
      g_nui_mini_depth_image_texture_guest_addr,
      GetNuiMini128DepthGuestAddress(), kNuiD3DMiniDepthWidth,
      kNuiD3DMiniDepthHeight, kNuiD3DMiniDepthStridePixels,
      "mini depth+player");
}

uint32_t GetNuiMiniDepthOnlyImageTextureGuestAddress() {
  return GetNuiDepthTextureGuestAddress(
      g_nui_mini_depth_only_image_texture_guest_addr,
      GetNuiMini128DepthOnlyGuestAddress(), kNuiD3DMiniDepthWidth,
      kNuiD3DMiniDepthHeight, kNuiD3DMiniDepthStridePixels,
      "mini depth-only");
}

uint32_t GetNuiColorImageTextureGuestAddress() {
  const uint32_t color = GetNuiColorGuestAddress();
  if (!color || !kernel_state()) {
    return 0;
  }
  auto* memory = kernel_state()->memory();
  uint32_t addr =
      g_nui_color_image_texture_guest_addr.load(std::memory_order_relaxed);
  if (!addr) {
    addr = memory->SystemHeapAlloc(kNuiD3DTextureObjectSize);
    if (!addr) {
      return 0;
    }
    g_nui_color_image_texture_guest_addr.store(addr, std::memory_order_relaxed);
    XELOGI("Kinect v1 NUI color D3DTexture sentinel allocated at {:08X}",
           addr);
  }

  auto* p = memory->TranslateVirtual<uint8_t*>(addr);
  if (!p) {
    return 0;
  }
  std::memset(p, 0, kNuiD3DTextureObjectSize);
  xe::store_and_swap<uint32_t>(
      p + 0x00, kNuiD3DCommonTypeTexture | kNuiD3DCommonD3DCreated |
                    kNuiD3DCommonCpuCachedMemory);
  xe::store_and_swap<uint32_t>(p + 0x04, 1);
  xe::store_and_swap<uint32_t>(p + 0x10, kNuiD3DTextureIdentifier);
  xe::store_and_swap<uint32_t>(p + 0x14, kNuiD3DFlushInitialValue);
  xe::store_and_swap<uint32_t>(p + 0x18, kNuiD3DFlushInitialValue);
  WriteNuiColorTextureFetch(p, color, false);
  return addr;
}

uint32_t GetNuiColorYuvImageTextureGuestAddress() {
  const uint32_t color_yuv = GetNuiColorYuvGuestAddress();
  if (!color_yuv || !kernel_state()) {
    return 0;
  }
  auto* memory = kernel_state()->memory();
  uint32_t addr =
      g_nui_color_yuv_image_texture_guest_addr.load(std::memory_order_relaxed);
  if (!addr) {
    addr = memory->SystemHeapAlloc(kNuiD3DTextureObjectSize);
    if (!addr) {
      return 0;
    }
    g_nui_color_yuv_image_texture_guest_addr.store(
        addr, std::memory_order_relaxed);
    XELOGI("Kinect v1 NUI YUV color D3DTexture sentinel allocated at {:08X}",
           addr);
  }

  auto* p = memory->TranslateVirtual<uint8_t*>(addr);
  if (!p) {
    return 0;
  }
  std::memset(p, 0, kNuiD3DTextureObjectSize);
  xe::store_and_swap<uint32_t>(
      p + 0x00, kNuiD3DCommonTypeTexture | kNuiD3DCommonD3DCreated |
                    kNuiD3DCommonCpuCachedMemory);
  xe::store_and_swap<uint32_t>(p + 0x04, 1);
  xe::store_and_swap<uint32_t>(p + 0x10, kNuiD3DTextureIdentifier);
  xe::store_and_swap<uint32_t>(p + 0x14, kNuiD3DFlushInitialValue);
  xe::store_and_swap<uint32_t>(p + 0x18, kNuiD3DFlushInitialValue);
  WriteNuiColorTextureFetch(p, color_yuv, true);
  return addr;
}

bool IsNuiImageTextureGuestAddress(uint32_t texture) {
  const uint32_t depth_texture =
      g_nui_image_texture_guest_addr.load(std::memory_order_relaxed);
  const uint32_t depth_only_texture =
      g_nui_depth_only_image_texture_guest_addr.load(
          std::memory_order_relaxed);
  const uint32_t depth_color_space_texture =
      g_nui_depth_color_space_image_texture_guest_addr.load(
          std::memory_order_relaxed);
  const uint32_t depth_color_space_only_texture =
      g_nui_depth_color_space_only_image_texture_guest_addr.load(
          std::memory_order_relaxed);
  const uint32_t color_texture =
      g_nui_color_image_texture_guest_addr.load(std::memory_order_relaxed);
  const uint32_t color_yuv_texture =
      g_nui_color_yuv_image_texture_guest_addr.load(std::memory_order_relaxed);
  const uint32_t mini_depth_texture =
      g_nui_mini_depth_image_texture_guest_addr.load(std::memory_order_relaxed);
  const uint32_t mini_depth_only_texture =
      g_nui_mini_depth_only_image_texture_guest_addr.load(
          std::memory_order_relaxed);
  return texture &&
         (texture == depth_texture || texture == depth_only_texture ||
          texture == depth_color_space_texture ||
          texture == depth_color_space_only_texture ||
          texture == color_texture || texture == color_yuv_texture ||
          texture == mini_depth_texture || texture == mini_depth_only_texture);
}

bool LockNuiImageTextureToGuest(uint32_t texture, uint32_t level,
                                uint32_t locked_rect, uint32_t rect,
                                uint32_t flags) {
  if (!IsNuiImageTextureGuestAddress(texture)) {
    return false;
  }

  auto* memory = kernel_state() ? kernel_state()->memory() : nullptr;
  auto* locked = memory && locked_rect
                     ? memory->TranslateVirtual<X_D3DLOCKED_RECT*>(locked_rect)
                     : nullptr;
  if (!locked || level != 0) {
    XELOGW("Kinect v1 NUI depth texture LockRect rejected: tex={:08X} "
           "level={} out={:08X}",
           texture, level, locked_rect);
    return true;
  }

  const bool is_color =
      texture ==
      g_nui_color_image_texture_guest_addr.load(std::memory_order_relaxed);
  const bool is_color_yuv =
      texture == g_nui_color_yuv_image_texture_guest_addr.load(
                     std::memory_order_relaxed);
  const bool is_mini_depth =
      texture == g_nui_mini_depth_image_texture_guest_addr.load(
                     std::memory_order_relaxed);
  const bool is_depth_only =
      texture == g_nui_depth_only_image_texture_guest_addr.load(
                     std::memory_order_relaxed);
  const bool is_depth_color_space =
      texture == g_nui_depth_color_space_image_texture_guest_addr.load(
                     std::memory_order_relaxed);
  const bool is_depth_color_space_only =
      texture == g_nui_depth_color_space_only_image_texture_guest_addr.load(
                     std::memory_order_relaxed);
  const bool is_mini_depth_only =
      texture == g_nui_mini_depth_only_image_texture_guest_addr.load(
                     std::memory_order_relaxed);
  const bool is_mini = is_mini_depth || is_mini_depth_only;
  const uint32_t width = (is_color || is_color_yuv)
                             ? kNuiD3DColorWidth
                             : is_mini ? kNuiD3DMiniDepthWidth
                                       : kNuiD3DDepthWidth;
  const uint32_t height = (is_color || is_color_yuv)
                              ? kNuiD3DColorHeight
                              : is_mini ? kNuiD3DMiniDepthHeight
                                        : kNuiD3DDepthHeight;
  const uint32_t pitch = is_color        ? kNuiD3DColorPitch
                         : is_color_yuv  ? kNuiD3DColorYuvPitch
                         : is_mini       ? kNuiD3DMiniDepthPitch
                                         : kNuiD3DDepthPitch;
  const uint32_t bytes_per_pixel = is_color ? 4u : 2u;
  uint32_t left = 0;
  uint32_t top = 0;
  auto* r = memory && rect ? memory->TranslateVirtual<X_RECT*>(rect) : nullptr;
  if (r) {
    left = static_cast<uint32_t>(
        std::clamp<int32_t>(int32_t(r->left), 0,
                            int32_t(width - 1)));
    top = static_cast<uint32_t>(
        std::clamp<int32_t>(int32_t(r->top), 0,
                            int32_t(height - 1)));
  }

  const uint32_t image =
      is_color             ? GetNuiColorGuestAddress()
      : is_color_yuv       ? GetNuiColorYuvGuestAddress()
      : is_mini_depth_only ? GetNuiMini128DepthOnlyGuestAddress()
      : is_mini_depth      ? GetNuiMini128DepthGuestAddress()
      : is_depth_color_space_only
          ? GetNuiDepthColorSpaceOnly384GuestAddress()
      : is_depth_color_space ? GetNuiDepthColorSpace384GuestAddress()
      : is_depth_only      ? GetNuiDepthOnly384GuestAddress()
                           : GetNuiDepth384GuestAddress();
  if (!image) {
    locked->pitch = 0;
    locked->bits = 0;
    XELOGW("Kinect v1 NUI {} texture LockRect has no frame yet",
           is_color             ? "color"
           : is_color_yuv       ? "color-yuv"
           : is_mini_depth_only ? "mini-depth-only"
           : is_mini_depth      ? "mini-depth+player"
           : is_depth_color_space_only
               ? "depth-only color-space"
           : is_depth_color_space ? "depth+player color-space"
           : is_depth_only      ? "depth-only"
                                : "depth+player");
    return true;
  }

  const uint32_t bits = image + top * pitch + left * bytes_per_pixel;
  locked->pitch = static_cast<int32_t>(pitch);
  locked->bits = bits;

  static std::atomic<uint32_t> nui_lock_calls{0};
  const uint32_t n =
      nui_lock_calls.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n <= 16 || (n & (n - 1)) == 0) {
    XELOGI("Kinect v1 NUI {} texture LockRect #{}: tex={:08X} "
           "level={} out={:08X} rect={:08X} flags={:08X} pitch={} bits={:08X}",
           is_color             ? "color"
           : is_color_yuv       ? "color-yuv"
           : is_mini_depth_only ? "mini-depth-only"
           : is_mini_depth      ? "mini-depth+player"
           : is_depth_color_space_only
               ? "depth-only color-space"
           : is_depth_color_space ? "depth+player color-space"
           : is_depth_only      ? "depth-only"
                                : "depth+player",
           n,
           texture, level, locked_rect, rect,
           flags, pitch, bits);
  }
  return true;
}

bool IsNuiColorImageType(uint32_t image_type) {
  return image_type == kNuiImageTypeColor ||
         image_type == kNuiImageTypeColorYuv ||
         image_type == kNuiImageTypeColorInDepthSpace;
}

uint32_t GetNuiTextureForImageType(uint32_t image_type) {
  switch (image_type) {
    case kNuiImageTypeColor:
    case kNuiImageTypeColorInDepthSpace:
      return GetNuiColorImageTextureGuestAddress();
    case kNuiImageTypeColorYuv:
      return GetNuiColorYuvImageTextureGuestAddress();
    case kNuiImageTypeDepthAndPlayerIndex80x60:
      return GetNuiMiniDepthImageTextureGuestAddress();
    case kNuiImageTypeDepth80x60:
      return GetNuiMiniDepthOnlyImageTextureGuestAddress();
    case kNuiImageTypeDepthAndPlayerIndex:
      return GetNuiImageTextureGuestAddress();
    case kNuiImageTypeDepthAndPlayerIndexInColorSpace:
      return GetNuiDepthColorSpaceImageTextureGuestAddress();
    case kNuiImageTypeDepth:
      return GetNuiDepthOnlyImageTextureGuestAddress();
    case kNuiImageTypeDepthInColorSpace:
      return GetNuiDepthColorSpaceOnlyImageTextureGuestAddress();
    default:
      return 0;
  }
}

int64_t GetNuiImageTimestamp(uint32_t image_type) {
  return IsNuiColorImageType(image_type) ? GetNuiColorTimestamp()
                                         : GetNuiDepthTimestamp();
}

uint32_t GetNuiImageFrameNumberForType(uint32_t image_type) {
  return IsNuiColorImageType(image_type) ? GetNuiColorFrameNumber()
                                         : GetNuiDepthFrameNumber();
}

bool GetNuiColorViewArea(uint32_t* digital_zoom, int32_t* center_x,
                         int32_t* center_y) {
  if (!digital_zoom || !center_x || !center_y) {
    return false;
  }
  *digital_zoom = kNuiImageDigitalZoom1x;
  *center_x = 0;
  *center_y = 0;
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    return runtime.CopyColorViewArea(digital_zoom, center_x, center_y);
  }
#endif
  return false;
}

uint32_t GetNuiImageFrameNodeGuestAddress(uint32_t image_type,
                                          uint32_t resolution) {
  if (image_type >= kNuiImageTypeCount || !kernel_state()) {
    return 0;
  }
  const uint32_t texture = GetNuiTextureForImageType(image_type);
  if (!texture) {
    return 0;
  }
  auto* memory = kernel_state()->memory();
  auto& slot = g_nui_image_frame_node_guest_addr_by_type[image_type];
  uint32_t addr = slot.load(std::memory_order_relaxed);
  if (!addr) {
    addr = memory->SystemHeapAlloc(64);
    if (!addr) {
      return 0;
    }
    slot.store(addr, std::memory_order_relaxed);
    XELOGI("Kinect v1 NUI image frame node type={} allocated at {:08X}",
           image_type, addr);
  }
  auto* p = memory->TranslateVirtual<uint8_t*>(addr);
  if (!p) {
    return 0;
  }

  // Xbox 360 NUI_IMAGE_FRAME layout from NuiImageCamera.h:
  // +0x00 LARGE_INTEGER, +0x08 frame number, +0x0C image type,
  // +0x10 resolution, +0x14 IDirect3DTexture9*, +0x18 flags,
  // +0x1C NUI_IMAGE_VIEW_AREA.
  std::memset(p, 0, 64);
  const int64_t timestamp = GetNuiImageTimestamp(image_type);
  const uint32_t frame_number = GetNuiImageFrameNumberForType(image_type);
  xe::store_and_swap<int64_t>(p + 0x00, timestamp);
  xe::store_and_swap<uint32_t>(p + 0x08, frame_number);
  xe::store_and_swap<uint32_t>(p + 0x0C, image_type);
  xe::store_and_swap<uint32_t>(p + 0x10, resolution);
  xe::store_and_swap<uint32_t>(p + 0x14, texture);
  xe::store_and_swap<uint32_t>(p + 0x18, 0);
  uint32_t digital_zoom = kNuiImageDigitalZoom1x;
  int32_t view_center_x = 0;
  int32_t view_center_y = 0;
  const bool view_area_valid =
      GetNuiColorViewArea(&digital_zoom, &view_center_x, &view_center_y);
  xe::store_and_swap<uint32_t>(p + 0x1C, digital_zoom);
  xe::store_and_swap<int32_t>(p + 0x20, view_center_x);
  xe::store_and_swap<int32_t>(p + 0x24, view_center_y);

  static std::atomic<uint32_t> log_count{0};
  const uint32_t n = log_count.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n <= 16 || (n & (n - 1)) == 0) {
    XELOGI(
        "Kinect v1 NUI image frame #{}: type={} res={} timestamp={} frame={} "
        "texture={:08X} view_area={} zoom={} center=({}, {})",
        n, image_type, resolution, timestamp, frame_number, texture,
        view_area_valid ? 1 : 0, digital_zoom, view_center_x, view_center_y);
  }
  return addr;
}

uint32_t GetNuiImageFrameNodeGuestAddress() {
  return GetNuiImageFrameNodeGuestAddress(
      kNuiImageTypeDepthAndPlayerIndex, kNuiImageResolution320x240);
}

uint32_t GetNuiImageFrameNodeRawAddress() {
  const auto& slot =
      g_nui_image_frame_node_guest_addr_by_type[kNuiImageTypeDepthAndPlayerIndex];
  return slot.load(std::memory_order_relaxed);
}

uint32_t GetNuiMiniDepthImageFrameNodeGuestAddress() {
  return GetNuiImageFrameNodeGuestAddress(
      kNuiImageTypeDepthAndPlayerIndex80x60, kNuiImageResolution80x60);
}

uint32_t GetNuiMiniDepthImageFrameNodeRawAddress() {
  const auto& slot = g_nui_image_frame_node_guest_addr_by_type
      [kNuiImageTypeDepthAndPlayerIndex80x60];
  return slot.load(std::memory_order_relaxed);
}

uint32_t GetNuiDepthColorSpaceImageFrameNodeGuestAddress(uint32_t image_type) {
  return GetNuiImageFrameNodeGuestAddress(image_type,
                                          kNuiImageResolution320x240);
}

uint32_t GetNuiDepthColorSpaceImageFrameNodeRawAddress() {
  const auto& slot = g_nui_image_frame_node_guest_addr_by_type
      [kNuiImageTypeDepthAndPlayerIndexInColorSpace];
  return slot.load(std::memory_order_relaxed);
}

uint32_t GetNuiColorImageFrameNodeGuestAddress() {
  return GetNuiImageFrameNodeGuestAddress(kNuiImageTypeColor,
                                          kNuiImageResolution640x480);
}

uint32_t GetNuiColorImageFrameNodeRawAddress() {
  return g_nui_image_frame_node_guest_addr_by_type[kNuiImageTypeColor].load(
      std::memory_order_relaxed);
}

bool IsNuiImageFrameNodeGuestAddressForType(uint32_t frame,
                                            uint32_t image_type) {
  return frame && image_type < kNuiImageTypeCount &&
         g_nui_image_frame_node_guest_addr_by_type[image_type].load(
             std::memory_order_relaxed) == frame;
}

uint32_t GetNuiInjectionState() {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (runtime.IsReady()) {
    return runtime.TrackedSkeletonCount() ? 2 : 1;
  }
#endif
  return ShouldSynthesizeNuiPlayer() ? 2 : 0;
}

uint32_t ConfigureNuiSkeletonTracking(uint32_t event_handle, uint32_t flags) {
#if XE_PLATFORM_WIN32
  return static_cast<uint32_t>(
      GetKinectV1SkeletonRuntime().ConfigureSkeletonTracking(event_handle,
                                                             flags));
#else
  (void)event_handle;
  (void)flags;
  return ShouldSynthesizeNuiPlayer() ? 0u : 0x80004005u;
#endif
}

uint32_t ConfigureNuiTrackedSkeletons(uint32_t tracking_ids_ptr) {
  if (!tracking_ids_ptr || tracking_ids_ptr > UINT32_MAX - 7 ||
      !kernel_state()) {
    return 0x80070057u;
  }
  auto* memory = kernel_state()->memory();
  auto* heap = memory->LookupHeap(tracking_ids_ptr);
  if (!heap || heap != memory->LookupHeap(tracking_ids_ptr + 7)) {
    return 0x80070057u;
  }
  const auto access =
      heap->QueryRangeAccess(tracking_ids_ptr, tracking_ids_ptr + 7);
  if (access != xe::memory::PageAccess::kReadOnly &&
      access != xe::memory::PageAccess::kReadWrite &&
      access != xe::memory::PageAccess::kExecuteReadOnly &&
      access != xe::memory::PageAccess::kExecuteReadWrite) {
    return 0x80070057u;
  }
  auto* ids =
      memory->TranslateVirtual<xe::be<uint32_t>*>(tracking_ids_ptr);
  if (!ids) {
    return 0x80070057u;
  }
#if XE_PLATFORM_WIN32
  return static_cast<uint32_t>(
      GetKinectV1SkeletonRuntime().ConfigureTrackedSkeletons(ids[0], ids[1]));
#else
  return ShouldSynthesizeNuiPlayer() ? 0u : 0x80004005u;
#endif
}

uint32_t ConfigureNuiFrameEndEvent(uint32_t event_handle, uint32_t flags) {
  if (!event_handle || (flags & ~1u)) {
    return 0x80070057u;
  }
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  if (!runtime.EnsureStarted()) {
    return 0x80004005u;
  }
  runtime.ConfigureFrameEndEvent(event_handle, flags);
  return 0;
#else
  (void)flags;
  return ShouldSynthesizeNuiPlayer() ? 0u : 0x80004005u;
#endif
}

void SetNuiSkeletonFrameEvent(uint32_t event_handle) {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  runtime.EnsureStarted();
  runtime.SetSkeletonFrameEventHandle(event_handle);
#else
  (void)event_handle;
#endif
}

void ResetNuiSkeletonFrameEvent() {
#if XE_PLATFORM_WIN32
  GetKinectV1SkeletonRuntime().ResetSkeletonFrameEvent();
#endif
}

void SetNuiImageFrameEvent(uint32_t event_handle) {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  runtime.EnsureStarted();
  runtime.RegisterImageFrameEventHandle(event_handle);
#else
  (void)event_handle;
#endif
}

void SetNuiColorFrameEvent(uint32_t event_handle) {
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
  runtime.EnsureStarted();
  runtime.RegisterColorFrameEventHandle(event_handle);
#else
  (void)event_handle;
#endif
}

void ResetNuiFrameEvent(uint32_t event_handle) {
#if XE_PLATFORM_WIN32
  GetKinectV1SkeletonRuntime().ResetFrameEvent(event_handle);
#else
  (void)event_handle;
#endif
}

void ResetNuiGuestEventRegistrations() {
#if XE_PLATFORM_WIN32
  GetKinectV1SkeletonRuntime().ResetGuestEventRegistrations();
#endif
}

bool BuildSyntheticNuiSkeletonFrame(X_NUI_SKELETON_FRAME* frame) {
  if (!frame || !ShouldSynthesizeNuiPlayer()) {
    return false;
  }

  struct Joint {
    float x;
    float y;
    float z;
  };
  static constexpr Joint kStandingPose[kNuiSkeletonPositionCount] = {
      {0.00f, 1.00f, 2.20f},   // HIP_CENTER
      {0.00f, 1.25f, 2.20f},   // SPINE
      {0.00f, 1.55f, 2.20f},   // SHOULDER_CENTER
      {0.00f, 1.78f, 2.20f},   // HEAD
      {-0.22f, 1.50f, 2.20f},  // SHOULDER_LEFT
      {-0.42f, 1.25f, 2.20f},  // ELBOW_LEFT
      {-0.55f, 1.05f, 2.20f},  // WRIST_LEFT
      {-0.60f, 0.95f, 2.20f},  // HAND_LEFT
      {0.22f, 1.50f, 2.20f},   // SHOULDER_RIGHT
      {0.42f, 1.25f, 2.20f},   // ELBOW_RIGHT
      {0.55f, 1.05f, 2.20f},   // WRIST_RIGHT
      {0.60f, 0.95f, 2.20f},   // HAND_RIGHT
      {-0.14f, 0.92f, 2.20f},  // HIP_LEFT
      {-0.16f, 0.52f, 2.20f},  // KNEE_LEFT
      {-0.16f, 0.12f, 2.20f},  // ANKLE_LEFT
      {-0.18f, 0.05f, 2.05f},  // FOOT_LEFT
      {0.14f, 0.92f, 2.20f},   // HIP_RIGHT
      {0.16f, 0.52f, 2.20f},   // KNEE_RIGHT
      {0.16f, 0.12f, 2.20f},   // ANKLE_RIGHT
      {0.18f, 0.05f, 2.05f},   // FOOT_RIGHT
  };

  std::memset(frame, 0, sizeof(*frame));
  static std::atomic<uint32_t> synthetic_frame_number{1};
  frame->timestamp = static_cast<int64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
  frame->frame_number =
      synthetic_frame_number.fetch_add(1, std::memory_order_relaxed);
  frame->normal_to_gravity.y = 1.0f;

  auto& skeleton = frame->skeleton_data[0];
  skeleton.tracking_state = kNuiSkeletonTracked;
  skeleton.tracking_id = kNuiSyntheticTrackingId;
  skeleton.enrollment_index = 0;
  skeleton.user_index = 0;
  skeleton.position.x = 0.0f;
  skeleton.position.y = 1.0f;
  skeleton.position.z = 2.2f;
  skeleton.position.w = 1.0f;
  for (uint32_t i = 0; i < kNuiSkeletonPositionCount; ++i) {
    skeleton.skeleton_positions[i].x = kStandingPose[i].x;
    skeleton.skeleton_positions[i].y = kStandingPose[i].y;
    skeleton.skeleton_positions[i].z = kStandingPose[i].z;
    skeleton.skeleton_positions[i].w = 1.0f;
    skeleton.skeleton_position_tracking_state[i] = kNuiSkeletonTracked;
  }
  return true;
}

bool WriteNuiSkeletonFrameToGuest(uint32_t frame_ptr, uint32_t wait_ms) {
  if (!frame_ptr || !kernel_state()) {
    return false;
  }
#if XE_PLATFORM_WIN32
  auto& runtime = GetKinectV1SkeletonRuntime();
#endif
  auto* memory = kernel_state()->memory();
  constexpr uint32_t frame_size =
      static_cast<uint32_t>(sizeof(X_NUI_SKELETON_FRAME));
  if (frame_ptr > UINT32_MAX - (frame_size - 1)) {
    return false;
  }
  auto* heap = memory->LookupHeap(frame_ptr);
  if (!heap || heap != memory->LookupHeap(frame_ptr + frame_size - 1)) {
    return false;
  }
  const auto access = heap->QueryRangeAccess(frame_ptr, frame_ptr + frame_size - 1);
  if (access != xe::memory::PageAccess::kReadWrite &&
      access != xe::memory::PageAccess::kExecuteReadWrite) {
    return false;
  }

  X_NUI_SKELETON_FRAME frame = {};
#if XE_PLATFORM_WIN32
  const bool copied = runtime.CopyNextSkeletonFrame(&frame, wait_ms);
  if (!copied && !BuildSyntheticNuiSkeletonFrame(&frame)) {
    return false;
  }
#else
  if (!BuildSyntheticNuiSkeletonFrame(&frame)) {
    return false;
  }
#endif
  auto* dst = memory->TranslateVirtual<uint8_t*>(frame_ptr);
  if (!dst) {
    return false;
  }
  std::memcpy(dst, &frame, sizeof(frame));
  return true;
}

bool WriteNuiSkeletonFrameToGuest(uint32_t frame_ptr) {
  return WriteNuiSkeletonFrameToGuest(frame_ptr, 0);
}

uint32_t GetEngagedNuiTrackingId() {
  const uint32_t best_tracking_id = GetNuiBestTrackingId();
  if (!IsNuiReady() || GetNuiTrackedSkeletonCount() == 0 ||
      best_tracking_id == kNuiInvalidTrackingId) {
    g_engaged_nui_tracking_id = 0;
  } else if (!g_engaged_nui_tracking_id ||
             g_engaged_nui_tracking_id != best_tracking_id) {
    g_engaged_nui_tracking_id = best_tracking_id;
  }
  return g_engaged_nui_tracking_id;
}

void SetEngagedNuiTrackingId(uint32_t tracking_id) {
  g_engaged_nui_tracking_id = IsNuiReady() ? tracking_id : 0;
}

bool ReadGuestDwords(uint32_t guest_address, uint32_t* words,
                     size_t word_count) {
  if (!kernel_state() || !guest_address || !words) {
    return false;
  }
  auto* source =
      kernel_state()->memory()->TranslateVirtual<const uint8_t*>(guest_address);
  if (!source) {
    return false;
  }
  for (size_t i = 0; i < word_count; ++i) {
    words[i] = xe::load_and_swap<uint32_t>(source + i * sizeof(uint32_t));
  }
  return true;
}

void WriteAnsiString(lpvoid_t buffer, dword_t buffer_size,
                     std::string_view value) {
  if (!buffer || !buffer_size || !kernel_state()) {
    return;
  }
  const uint32_t max_bytes = buffer_size.value();
  const uint32_t copy_bytes =
      std::min<uint32_t>(static_cast<uint32_t>(value.size()), max_bytes - 1);
  auto* dst = kernel_state()->memory()->TranslateVirtual<uint8_t*>(
      buffer.guest_address());
  if (!dst) {
    return;
  }
  if (copy_bytes) {
    std::memcpy(dst, value.data(), copy_bytes);
  }
  dst[copy_bytes] = 0;
}

bool IsKnownXStudioNuiCommand(uint32_t command) {
  switch (command) {
    case 0x00000001:
    case 0x00000006:
    case 0x00000007:
    case 0x00000251:
    case 0x00000252:
    case 0x00000301:
    case 0x00000302:
    case 0x00001003:
    case 0x00001004:
    case 0x00001005:
    case 0x00001401:
    case 0x00001402:
    case 0x00001410:
    case 0x00001502:
      return true;
    default:
      return false;
  }
}

dword_result_t XamXStudioRequest_entry(dword_t command, lpvoid_t in_out) {
  const uint32_t command_value = command.value();
  const uint32_t request_address = in_out.guest_address();
  uint32_t request_words[4] = {};
  const bool request_readable =
      ReadGuestDwords(request_address, request_words,
                      sizeof(request_words) / sizeof(request_words[0]));
  static std::atomic<uint32_t> log_count{0};
  const uint32_t call = log_count.fetch_add(1, std::memory_order_relaxed) + 1;
  const bool high_frequency_command =
      command_value == 0x00000006 || command_value == 0x00001003 ||
      command_value == 0x00001004 || command_value == 0x00001005 ||
      command_value == 0x00000252 || command_value == 0x00001401 ||
      command_value == 0x00001402;
  const bool should_log =
      call <= 16 || !high_frequency_command || (call & (call - 1)) == 0;
  if (should_log) {
    XELOGI(
        "KINECT: XamXStudioRequest #{} command={:08X} in_out={:08X} "
        "readable={} ready={} words=[{:08X} {:08X} {:08X} {:08X}]",
        call, command_value, request_address, request_readable ? 1 : 0,
        IsNuiReady() ? 1 : 0, request_words[0], request_words[1],
        request_words[2], request_words[3]);
  }

  // PROBE: DC3 polls cmd 0x1410 ~1264x (its per-frame stream read; caller
  // 0x829CCF94 builds a request with key 0x52B5 + stride 0x300). Dump the full
  // request so we can map the depth-delivery protocol (and later fill the reply
  // with our buffer). Also re-dump AFTER, to see which fields the title reads.
  if (command_value == 0x00001410 && request_address) {
    uint32_t r[16] = {};
    if (ReadGuestDwords(request_address, r, 16)) {
      static std::atomic<uint32_t> n1410{0};
      const uint32_t c = n1410.fetch_add(1, std::memory_order_relaxed) + 1;
      if (c <= 6 || c % 600 == 0) {
        XELOGI(
            "DC3 0x1410 #{} req[0..7]={:08X} {:08X} {:08X} {:08X} {:08X} "
            "{:08X} {:08X} {:08X}  [8..15]={:08X} {:08X} {:08X} {:08X} {:08X} "
            "{:08X} {:08X} {:08X}",
            c, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9],
            r[10], r[11], r[12], r[13], r[14], r[15]);
      }
    }
  }

  if (command_value == 0x00000006) {
    auto* out = request_address && kernel_state()
                    ? kernel_state()
                          ->memory()
                          ->TranslateVirtual<xe::be<uint32_t>*>(
                              request_address)
                    : nullptr;
    if (out) {
      out[0] = IsNuiReady() ? 0u : 0x80000000u;
    }
    return X_ERROR_SUCCESS;
  }

  if (command_value == 0x00001402) {
    const uint32_t out_frame_ptr = request_words[0];
    const uint32_t out_frame_size = request_words[1];
    // Repeated polls consume only newly published skeleton frames.
    const bool wrote =
        request_readable && out_frame_ptr &&
        out_frame_size >= static_cast<uint32_t>(sizeof(X_NUI_SKELETON_FRAME)) &&
        WriteNuiSkeletonFrameToGuest(out_frame_ptr);
    auto* out = request_address && kernel_state()
                    ? kernel_state()
                          ->memory()
                          ->TranslateVirtual<xe::be<uint32_t>*>(
                              request_address)
                    : nullptr;
    if (out) {
      out[2] = wrote ? static_cast<uint32_t>(sizeof(X_NUI_SKELETON_FRAME)) : 0u;
    }
    static std::atomic<uint32_t> skeleton_log_count{0};
    const uint32_t skeleton_call =
        skeleton_log_count.fetch_add(1, std::memory_order_relaxed) + 1;
    if (skeleton_call <= 16 ||
        (skeleton_call & (skeleton_call - 1)) == 0) {
      XELOGI(
          "KINECT: XamXStudioRequest(00001402) frame_ptr={:08X} size={:08X} "
          "wrote={} state={} tracked={}",
          out_frame_ptr, out_frame_size, wrote ? 1 : 0,
          GetNuiInjectionState(), GetNuiTrackedSkeletonCount());
    }
    return wrote ? X_ERROR_SUCCESS : kXamNuiFrameNoData;
  }

  if (command_value == 0x00001003 && request_address && kernel_state()) {
    auto* out = kernel_state()->memory()->TranslateVirtual<xe::be<uint32_t>*>(
        request_address);
    if (out) {
      out[1] = 0;  // wrapper HRESULT-ish return (>=0 = OK)
    }
    // cmd 0x1003 = the Kinect camera-device passthrough. request_words[0] points
    // to a _NUICAM_REQUEST whose first dword is the camera OPCODE. The guest's
    // nuiGfxInit issues a sequence of these; each expects specific output fields
    // written back (on hardware the kernel camera driver fills them). Writing
    // nothing made nuiGfxInit bail and the depth/silhouette pipeline never
    // initialise (the title then loops here forever -> black screen). Emulate the
    // driver replies used by the statically linked nuiapi camera init path.
    if (request_readable && request_words[0] && IsNuiReady()) {
      auto* req = kernel_state()->memory()->TranslateVirtual<xe::be<uint32_t>*>(
          request_words[0]);
      if (req) {
        const uint32_t opcode = req[0];
        static std::atomic<uint32_t> camera_request_log_count{0};
        const uint32_t camera_log =
            camera_request_log_count.fetch_add(1, std::memory_order_relaxed) +
            1;
        if (camera_log <= 32 || (camera_log & (camera_log - 1)) == 0) {
          XELOGI(
              "KINECT: XamXStudioRequest(00001003) camera #{} opcode={:08X} "
              "req={:08X} args=[{:08X} {:08X} {:08X}]",
              camera_log, opcode, request_words[0], uint32_t(req[1]),
              uint32_t(req[2]), uint32_t(req[3]));
        }
        // Push/DMA camera-model diagnostic for JD4 (555308B5) AND DC3
        // (373307D9). These titles register frame buffers via the _NUICAM
        // opcodes (0x05/0x18/0x21) and expect the "driver" to fill them, instead
        // of (only) the pull-fetch path (0x1004/0x1402) we emulate. The args
        // logged above are just req[1..3]; the buffer pointer/size/type live
        // deeper (e.g. req[6..8]). Dump req[0..19] so the exact registered
        // depth-buffer address+size can be mapped before filling it -- without
        // this the silhouette panel colorizes an all-zero buffer (renders empty).
        // Log-only, so it cannot affect title state.
        const uint32_t nuicam_tid = kernel_state()->title_id();
        if ((nuicam_tid == 0x555308B5u || nuicam_tid == 0x373307D9u) &&
            (camera_log <= 64 || (camera_log & (camera_log - 1)) == 0)) {
          uint32_t r[20] = {};
          if (ReadGuestDwords(request_words[0], r, 20)) {
            XELOGI(
                "_NUICAM #{} op={:08X} req={:08X} [0..9]={:08X} {:08X} "
                "{:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X}",
                camera_log, opcode, request_words[0], r[0], r[1], r[2], r[3],
                r[4], r[5], r[6], r[7], r[8], r[9]);
            XELOGI(
                "_NUICAM #{} op={:08X} [10..19]={:08X} {:08X} {:08X} {:08X} "
                "{:08X} {:08X} {:08X} {:08X} {:08X} {:08X}",
                camera_log, opcode, r[10], r[11], r[12], r[13], r[14], r[15],
                r[16], r[17], r[18], r[19]);
          }
        }
        switch (opcode) {
          case 0x00000000:
            // config-count query (nuiGfxInit's first call): report config 4
            // (full depth+colour camera) -> nuiGfxInit runs the full init path.
            req[1] = 4u;
            break;
          case 0x00000019:
            // depth geometry/timing query. req[9] (device+0x450) MUST be non-zero
            // or nuiGfxInit bails before the depth-resource create; req[13]/[14]
            // feed depth-buffer sizing. Kinect v1 depth = 320x240.
            for (int i = 1; i <= 14; ++i) {
              req[i] = 1u;
            }
            req[9] = 320u;
            req[13] = 320u;
            req[14] = 240u;
            break;
          case 0x0000000D:
            // firmware-version query: report 5.1 so the guest takes the full-
            // calibration path.
            req[1] = 0x05010000u;
            break;
          case 0x00000005:
            // JD2019: per-poll device/stream STATUS. JD4 (555308B5) AND DC3
            // (373307D9): this is also the camera frame-BUFFER REGISTRATION -
            // req[6]=type, req[7]=physical buffer address, req[8]=size. Both use
            // a PUSH model: they register their own 640x480 buffers (type 4 =
            // depth+player silhouette, type 2 = colour) and expect the driver to
            // DMA-fill them, instead of (only) pull-fetching. PROVEN from the
            // _NUICAM dump: DC3 registers 4 type-4 depth buffers (0xBDF00000,
            // 0xBDF96000, 0xBE02C000, 0xBE0C2000) + 3 type-2 colour buffers, all
            // 0x96000 bytes. Capture the type-4 buffers so the host depth pump
            // fills them each frame -> the silhouette panel gets real depth
            // instead of an all-zero buffer (which colorized to an empty panel).
            if (kernel_state()->title_id() == 0x555308B5u ||
                kernel_state()->title_id() == 0x373307D9u) {
              RegisterJd4CameraBuffer(uint32_t(req[6]), uint32_t(req[7]),
                                      uint32_t(req[8]));
            }
            req[1] = 1u;  // connected
            req[2] = 1u;  // ready / streams available
            break;
          case 0x00000024:
            // EXPERIMENTAL (JD4 555308B5 only). Frame-ready poll: JD4 registers
            // push buffers via 0x05 then polls 0x24 for a ready frame. Report a
            // frame ready in a rotating buffer so JD4 starts consuming the
            // host-filled silhouette. Only touches the request buffer (no other
            // guest memory) so it cannot corrupt JD4 state; the reply layout is
            // a best guess - toggle xam_nui_jd4_camera_experiment off if JD4
            // misbehaves.
            if (kernel_state()->title_id() == 0x555308B5u &&
                cvars::xam_nui_jd4_camera_experiment) {
              static std::atomic<uint32_t> jd4_frame_seq{0};
              const uint32_t seq =
                  jd4_frame_seq.fetch_add(1, std::memory_order_relaxed);
              req[1] = 1u;        // a frame is available
              req[2] = seq & 3u;  // rotating buffer index 0..3
              req[3] = seq + 1u;  // frame number
            }
            break;
          default:
            // Other opcodes acknowledged as success (out[1]=0 -> wrapper >=0).
            break;
        }
      }
    }
  }

  if (command_value == 0x00001004 && request_address && kernel_state()) {
    auto* out = kernel_state()->memory()->TranslateVirtual<xe::be<uint32_t>*>(
        request_address);
    if (out) {
      const bool have_depth = GetNuiDepth384GuestAddress() != 0;
      out[1] = have_depth ? 0x00000005u : 0x00000000u;
      static std::atomic<uint32_t> depth_fetch_log_count{0};
      const uint32_t n =
          depth_fetch_log_count.fetch_add(1, std::memory_order_relaxed) + 1;
      if (n <= 16 || (n & (n - 1)) == 0) {
        XELOGI(
            "KINECT: XamXStudioRequest(00001004) depth_fetch #{} "
            "stream={:08X} have_depth={} out1={:08X}",
            n, request_words[0], have_depth ? 1 : 0, uint32_t(out[1]));
      }
    }
  }

  if (IsKnownXStudioNuiCommand(command_value)) {
    return IsNuiReady() ? X_ERROR_SUCCESS : X_E_FAIL;
  }
  return X_E_FAIL;
}
DECLARE_XAM_EXPORT2(XamXStudioRequest, kNone, kImplemented, kHighFrequency);

struct X_NUI_DEVICE_STATUS {
  /* Notes:
     - for one side func of XamNuiGetDeviceStatus
       - if some data addressis less than zero then unk1 = it
       - else another func is called and its return can set unk1 = c0051200 or
     some value involving DetroitDeviceRequest
       - next PsCamDeviceRequest is called and if its return is less than zero
     then X_NUI_DEVICE_STATUS = return of PsCamDeviceRequest
       - else it equals an unknown local_1c
       - finally McaDeviceRequest is called and if its return is less than zero
     then unk2 = return of McaDeviceRequest
       - else it equals an unknown local_14
     - status can be set to X_NUI_DEVICE_STATUS[3] | 0x44 or | 0x40
  */
  xe::be<uint32_t> unk0;
  xe::be<uint32_t> unk1;
  xe::be<uint32_t> unk2;
  xe::be<uint32_t> status;
  xe::be<uint32_t> unk4;
  xe::be<uint32_t> unk5;
};
static_assert(sizeof(X_NUI_DEVICE_STATUS) == 24, "Size matters");

// Get
dword_result_t XamNuiGetDeviceStatus_entry(
    pointer_t<X_NUI_DEVICE_STATUS> status_ptr) {
  static std::atomic<uint32_t> call_count{0};
  const uint32_t call =
      call_count.fetch_add(1, std::memory_order_relaxed) + 1;
  if (call <= 16 || (call & (call - 1)) == 0) {
    XELOGI("KINECT: XamNuiGetDeviceStatus #{} status_ptr={:08X}", call,
           status_ptr.guest_address());
  }
  /* Notes:
     - it does return a value that is not always used
     - returns values are X_ERROR_SUCCESS, 0xC0050006, and others
     - 1) On func start *status_ptr = 0, status_ptr->unk1 = 0, status_ptr->unk2
     = 0, and status_ptr->status = 0
     - 2) calls XamXStudioRequest(6,&var <- = 0);
     - if return is greater than -1 && var & 0x80000000 != 0 then set
     status_ptr->unk1 = 0xC000009D, status_ptr->unk2 = 0xC000009D, and
     status_ptr->status = status_ptr[3] = 0x20
     - lots of branching functions after
  */

  status_ptr.Zero();

  const bool kinect_initialized = IsNuiReady();
  const uint32_t status =
      kinect_initialized ? kNuiHardwareStatusConnected | kNuiHardwareStatusReady
                         : 0;
  status_ptr->status = status;
  if (call <= 16 || (call & (call - 1)) == 0) {
    XELOGI("KINECT: XamNuiGetDeviceStatus #{} wrote status={:08X} result={:08X}",
           call, status,
           kinect_initialized ? X_ERROR_SUCCESS : 0xC0050006);
  }
  return kinect_initialized ? X_ERROR_SUCCESS : 0xC0050006;
}
DECLARE_XAM_EXPORT1(XamNuiGetDeviceStatus, kNone, kImplemented);

struct X_NUI_ENROLLMENT_INFORMATION {
  xe::be<uint32_t> enrollment_index;
  xe::be<uint32_t> user_index;
  xe::be<uint32_t> flags;
  xe::be<uint32_t> reserved;
};
static_assert(sizeof(X_NUI_ENROLLMENT_INFORMATION) == 16,
              "NUI enrollment information size must match XDK");

dword_result_t XamNuiIdentityGetEnrollmentInfo_entry(
    dword_t enrollment_index,
    pointer_t<X_NUI_ENROLLMENT_INFORMATION> information) {
  XELOGI("KINECT: XamNuiIdentityGetEnrollmentInfo enrollment={} out={:08X}",
         enrollment_index.value(), information.guest_address());
  if (!information || enrollment_index >= 8) {
    return X_E_INVALIDARG;
  }
  information.Zero();
  if (!IsNuiReady() || enrollment_index != 0) {
    return X_E_NO_SUCH_USER;
  }
  information->enrollment_index = 0;
  information->user_index = 0;
  information->flags = kNuiIdentityEnrollSkeletonFlag;
  return X_E_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityGetEnrollmentInfo, kNone, kImplemented);

dword_result_t XamNuiIdentityUnenroll_entry(dword_t enrollment_index) {
  XELOGI("KINECT: XamNuiIdentityUnenroll enrollment={}",
         enrollment_index.value());
  if (enrollment_index >= 8) {
    return X_E_INVALIDARG;
  }
  return IsNuiReady() ? X_E_SUCCESS : X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityUnenroll, kNone, kImplemented);

dword_result_t XamNuiIdentityGetColorTexture_entry(dword_t tracking_id,
                                                   lpdword_t texture) {
  XELOGI("KINECT: XamNuiIdentityGetColorTexture tracking_id={:08X} out={:08X}",
         tracking_id.value(), texture.guest_address());
  if (!texture) {
    return X_E_INVALIDARG;
  }
  *texture = 0;
  if (!HasNuiIdentityPlayer(tracking_id.value())) {
    return X_E_NO_SUCH_USER;
  }
  *texture = GetNuiColorImageTextureGuestAddress();
  return *texture ? X_E_SUCCESS : kXamNuiFrameNoData;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityGetColorTexture, kNone, kImplemented);

dword_result_t XamNuiEnableChatMic_entry(int_t enable) {
  g_nui_chat_mic_enabled.store(enable != 0, std::memory_order_relaxed);
  XELOGI("KINECT: XamNuiEnableChatMic enable={}", enable.value());
  return X_E_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiEnableChatMic, kNone, kImplemented);

dword_result_t XamNuiStoreDepthCalibration_entry(lpvoid_t calibration,
                                                 dword_t size) {
  XELOGI("KINECT: XamNuiStoreDepthCalibration data={:08X} size={}",
         calibration.guest_address(), size.value());
  return calibration && size ? X_E_SUCCESS : X_E_INVALIDARG;
}
DECLARE_XAM_EXPORT1(XamNuiStoreDepthCalibration, kNone, kImplemented);

dword_result_t XamUserNuiIsBiometricEnabled_entry(dword_t user_index) {
  if (user_index >= XUserMaxUserCount) {
    return false;
  }
  return g_nui_biometric_enabled[user_index.value()].load(
      std::memory_order_relaxed);
}
DECLARE_XAM_EXPORT1(XamUserNuiIsBiometricEnabled, kNone, kImplemented);

dword_result_t XamUserNuiBind_entry(dword_t user_index,
                                    dword_t enrollment_index) {
  XELOGI("KINECT: XamUserNuiBind user={} enrollment={}", user_index.value(),
         enrollment_index.value());
  if (user_index >= XUserMaxUserCount || enrollment_index >= 8) {
    return X_E_INVALIDARG;
  }
  if (!IsNuiReady()) {
    return X_E_FAIL;
  }
  g_nui_biometric_enabled[user_index.value()].store(true,
                                                    std::memory_order_relaxed);
  return X_E_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamUserNuiBind, kNone, kImplemented);

dword_result_t XamUserNuiUnbind_entry(dword_t user_index) {
  XELOGI("KINECT: XamUserNuiUnbind user={}", user_index.value());
  if (user_index >= XUserMaxUserCount) {
    return X_E_INVALIDARG;
  }
  g_nui_biometric_enabled[user_index.value()].store(false,
                                                    std::memory_order_relaxed);
  return X_E_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamUserNuiUnbind, kNone, kImplemented);

dword_result_t XamUserNuiGetUserIndex_entry(dword_t enrollment_index,
                                            lpdword_t index) {
  XELOGI("KINECT: XamUserNuiGetUserIndex enrollment={} index_ptr={:08X}",
         enrollment_index.value(), index.guest_address());
  // Per nuiIdentity.h: there are NUI_IDENTITY_MAX_ENROLLMENT_COUNT (8) enrollment
  // slots, each bound to at most ONE user (NUI_ENROLLMENT_INFORMATION.dwUserIndex).
  // The title enumerates all 8 to build the enrollment->user map. The old stub
  // returned user 0 for EVERY slot, telling the title 8 enrollments all belong to
  // user 0 and breaking identity resolution. Model a single enrolled synthetic
  // player at slot 0 (user 0); all other slots are empty.
  if (IsNuiReady() && index && enrollment_index == 0) {
    *index = 0;
    return X_E_SUCCESS;
  }
  return X_E_NO_SUCH_USER;
}
DECLARE_XAM_EXPORT1(XamUserNuiGetUserIndex, kNone, kImplemented);

dword_result_t XamUserNuiGetUserIndexForSignin_entry(lpdword_t index) {
  XELOGI("KINECT: XamUserNuiGetUserIndexForSignin index_ptr={:08X}",
         index.guest_address());
  for (uint32_t i = 0; i < XUserMaxUserCount; i++) {
    auto profile = kernel_state()->xam_state()->GetUserProfile(i);
    if (profile) {
      *index = i;
      return X_E_SUCCESS;
    }
  }

  return X_E_ACCESS_DENIED;
}
DECLARE_XAM_EXPORT1(XamUserNuiGetUserIndexForSignin, kNone, kImplemented);

dword_result_t XamUserNuiGetUserIndexForBind_entry(lpdword_t index) {
  XELOGI("KINECT: XamUserNuiGetUserIndexForBind index_ptr={:08X}",
         index.guest_address());
  if (IsNuiReady() && index) {
    *index = 0;
    return X_E_SUCCESS;
  }
  return X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamUserNuiGetUserIndexForBind, kNone, kImplemented);

dword_result_t XamNuiGetDepthCalibration_entry(lpdword_t unk1) {
  XELOGI("KINECT: XamNuiGetDepthCalibration out_ptr={:08X}",
         unk1.guest_address());
  /* Notes:
     - Possible returns X_STATUS_NO_SUCH_FILE, and 0x10000000
  */
  return X_STATUS_NO_SUCH_FILE;
}
DECLARE_XAM_EXPORT1(XamNuiGetDepthCalibration, kNone, kImplemented);

dword_result_t XamNuiGetLoadedDepthCalibration_entry(lpdword_t calibration) {
  XELOGI("KINECT: XamNuiGetLoadedDepthCalibration out={:08X}",
         calibration.guest_address());
  return XamNuiGetDepthCalibration_entry(calibration);
}
DECLARE_XAM_EXPORT1(XamNuiGetLoadedDepthCalibration, kNone, kImplemented);

dword_result_t XamNuiGetCameraIntrinsics_entry(lpvoid_t intrinsics,
                                               dword_t intrinsics_size) {
  XELOGI("KINECT: XamNuiGetCameraIntrinsics out={:08X} size={}",
         intrinsics.guest_address(), intrinsics_size.value());
  if (!intrinsics || intrinsics.guest_address() < 0x10000 ||
      intrinsics_size < 16 || intrinsics_size > 256) {
    return X_E_INVALIDARG;
  }
  std::memset(const_cast<void*>(static_cast<const void*>(intrinsics)), 0,
              intrinsics_size.value());
  auto* values = static_cast<xe::be<float>*>(
      const_cast<void*>(static_cast<const void*>(intrinsics)));
  values[0] = kNuiDepthFocalLength320x240;
  values[1] = kNuiDepthFocalLength320x240;
  values[2] = 160.0f;
  values[3] = 120.0f;
  return IsNuiReady() ? X_E_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiGetCameraIntrinsics, kNone, kImplemented);

dword_result_t XamNuiGetDeviceSerialNumber_entry(lpvoid_t buffer,
                                                 dword_t buffer_size) {
  XELOGI("KINECT: XamNuiGetDeviceSerialNumber buffer={:08X} size={}",
         buffer.guest_address(), buffer_size.value());
  WriteAnsiString(buffer, buffer_size, "XENIA-KINECT-00000001");
  return X_STATUS_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiGetDeviceSerialNumber, kNone, kImplemented);

dword_result_t XamNuiGetSupportString_entry(lpvoid_t buffer,
                                            dword_t buffer_size) {
  XELOGI("KINECT: XamNuiGetSupportString buffer={:08X} size={}",
         buffer.guest_address(), buffer_size.value());
  if (!buffer || !buffer_size) {
    return X_E_INVALIDARG;
  }
  WriteAnsiString(buffer, buffer_size, "Xenia Kinect compatibility runtime");
  return X_E_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiGetSupportString, kNone, kImplemented);

dword_result_t XamNuiGetFanRate_entry() {
  return 0;
}
DECLARE_XAM_EXPORT1(XamNuiGetFanRate, kNone, kImplemented);

dword_result_t XamFitnessGetCurrentBodyProfileRecord_entry(
    unknown_t arg0, unknown_t arg1, unknown_t arg2, unknown_t arg3) {
  XELOGI(
      "KINECT: XamFitnessGetCurrentBodyProfileRecord args=[{:08X} {:08X} "
      "{:08X} {:08X}]",
      arg0.value(), arg1.value(), arg2.value(), arg3.value());
  if (!kernel_state()) {
    return X_E_INVALIDARG;
  }
  const uint32_t candidates[] = {arg0.value(), arg1.value(), arg2.value(),
                                 arg3.value()};
  X_NUI_FITNESS_BODY_PROFILE_RECORD* body_profile = nullptr;
  for (const uint32_t candidate : candidates) {
    if (candidate < 0x10000) {
      continue;
    }
    body_profile =
        kernel_state()->memory()->TranslateVirtual<
            X_NUI_FITNESS_BODY_PROFILE_RECORD*>(candidate);
    if (body_profile) {
      break;
    }
  }
  if (!body_profile) {
    return X_E_INVALIDARG;
  }
  std::memset(body_profile, 0, sizeof(*body_profile));
  return X_E_NOTFOUND;
}
DECLARE_XAM_EXPORT1(XamFitnessGetCurrentBodyProfileRecord, kNone, kImplemented);

dword_result_t XamFitnessCompatibilitySuccess(unknown_t r3, unknown_t r4,
                                              unknown_t r5, unknown_t r6,
                                              unknown_t r7, unknown_t r8) {
  return X_E_SUCCESS;
}

dword_result_t XamFitnessCompatibilityNoData(unknown_t r3, unknown_t r4,
                                             unknown_t r5, unknown_t r6,
                                             unknown_t r7, unknown_t r8) {
  return X_E_NOTFOUND;
}

dword_result_t XamFitnessAddBodyProfileRecord_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilitySuccess(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessAddBodyProfileRecord, kNone, kSketchy);

dword_result_t XamFitnessClearBodyProfileRecords_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilitySuccess(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessClearBodyProfileRecords, kNone, kSketchy);

dword_result_t XamFitnessGetAllBodyProfileRecords_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilityNoData(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessGetAllBodyProfileRecords, kNone, kSketchy);

dword_result_t XamFitnessGetAllTitleSummaries_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilityNoData(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessGetAllTitleSummaries, kNone, kSketchy);

dword_result_t XamFitnessGetOverallSummary_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilityNoData(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessGetOverallSummary, kNone, kSketchy);

dword_result_t XamFitnessAddFitnessEvent_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilitySuccess(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessAddFitnessEvent, kNone, kSketchy);

dword_result_t XamFitnessCreateFitnessEventEnumerator_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilityNoData(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessCreateFitnessEventEnumerator, kNone, kSketchy);

dword_result_t XamFitnessInitialize_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilitySuccess(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessInitialize, kNone, kSketchy);

dword_result_t XamFitnessUnInitialize_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilitySuccess(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessUnInitialize, kNone, kSketchy);

dword_result_t XamFitnessClearAll_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilitySuccess(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessClearAll, kNone, kSketchy);

dword_result_t XamFitnessGetPrivacySettings_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilityNoData(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessGetPrivacySettings, kNone, kSketchy);

dword_result_t XamFitnessGetSyncStatus_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilitySuccess(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessGetSyncStatus, kNone, kSketchy);

dword_result_t XamFitnessInitializeForOneUser_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilitySuccess(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessInitializeForOneUser, kNone, kSketchy);

dword_result_t XampFitnessLetFireAndForgetsCatchUpInternal_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilitySuccess(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XampFitnessLetFireAndForgetsCatchUpInternal, kNone,
                    kSketchy);

dword_result_t XamFitnessGetSuggestedPrivacySettings_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilityNoData(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessGetSuggestedPrivacySettings, kNone, kSketchy);

dword_result_t XamFitnessGetTitleSummaries_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilityNoData(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessGetTitleSummaries, kNone, kSketchy);

dword_result_t XamFitnessGetTitleSummary_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilityNoData(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessGetTitleSummary, kNone, kSketchy);

dword_result_t XamFitnessContainsFitnessData_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return false;
}
DECLARE_XAM_EXPORT1(XamFitnessContainsFitnessData, kNone, kSketchy);

dword_result_t XamFitnessConvertByteMetToFloatMet_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilitySuccess(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessConvertByteMetToFloatMet, kNone, kSketchy);

dword_result_t XamFitnessConvertFloatMetToByteMet_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilitySuccess(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessConvertFloatMetToByteMet, kNone, kSketchy);

dword_result_t XamFitnessMsgTimeToSystemTime_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilitySuccess(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessMsgTimeToSystemTime, kNone, kSketchy);

dword_result_t XamFitnessSystemTimeToMsgTime_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilitySuccess(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamFitnessSystemTimeToMsgTime, kNone, kSketchy);

dword_result_t XamShowFitnessBodyProfileUI_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilitySuccess(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamShowFitnessBodyProfileUI, kNone, kSketchy);

dword_result_t XamShowFitnessWarnAboutPrivacyUI_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilitySuccess(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamShowFitnessWarnAboutPrivacyUI, kNone, kSketchy);

dword_result_t XamShowFitnessWarnAboutTimeUI_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilitySuccess(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamShowFitnessWarnAboutTimeUI, kNone, kSketchy);

dword_result_t XamShowFitnessClearUI_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamFitnessCompatibilitySuccess(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamShowFitnessClearUI, kNone, kSketchy);

// XamNuiGetTrueColorInfo (xam ord 0x8DF).
// Internal XAM export (not in the public xam.h), resolved dynamically by titles
// that use the live true-color camera path.
//
// Signature unknown from public headers -> we log the raw guest args so the
// real layout can be confirmed from a trace, and report "true color available"
// (640x480, the resolution Xenia already feeds for the color stream).
dword_result_t XamNuiGetTrueColorInfo_entry(dword_t arg0, dword_t arg1,
                                            dword_t out_ptr, dword_t out_size) {
  // Observed call: arg0=1 (a type/flag), arg1=0, out_ptr=guest buffer,
  // out_size=4 (a single dword). arg0 is NOT a pointer.
  XELOGI(
      "KINECT: XamNuiGetTrueColorInfo arg0={:08X} arg1={:08X} out_ptr={:08X} "
      "out_size={:08X} ready={}",
      arg0.value(), arg1.value(), out_ptr.value(), out_size.value(),
      IsNuiReady());
  // Write a zeroed info word into the caller's output buffer (out_size bytes,
  // observed as 4) and report success so the true-color setup path proceeds.
  // Guard the pointer so a non-pointer/garbage value can never fault the host.
  const uint32_t guest_addr = out_ptr.value();
  if (guest_addr >= 0x10000 && out_size.value() >= sizeof(uint32_t)) {
    auto out =
        kernel_state()->memory()->TranslateVirtual<xe::be<uint32_t>*>(guest_addr);
    if (out) {
      *out = 0u;
    }
  }
  return X_E_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiGetTrueColorInfo, kNone, kImplemented);

dword_result_t XamNatalDeviceAudioCalibrate_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  XELOGI("KINECT: XamNatalDeviceAudioCalibrate {:08X} {:08X} {:08X} {:08X} "
         "{:08X} {:08X}",
         r3.value(), r4.value(), r5.value(), r6.value(), r7.value(),
         r8.value());
  return X_STATUS_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNatalDeviceAudioCalibrate, kNone, kImplemented);

// Skeleton
qword_result_t XamNuiSkeletonGetBestSkeletonIndex_entry(int_t unk) {
  XELOGI("KINECT: XamNuiSkeletonGetBestSkeletonIndex unk={:08X}",
         unk.value());
  const uint32_t best_index = GetNuiBestSkeletonIndex();
  return best_index != 0xFFFFFFFFu ? best_index : 0xffffffffffffffff;
}
DECLARE_XAM_EXPORT1(XamNuiSkeletonGetBestSkeletonIndex, kNone, kImplemented);

/* XamNuiCamera Notes
   - most require message calls to xam in 0x0002Bxxx area
*/

dword_result_t XamNuiCameraTiltGetStatus_entry(lpvoid_t unk) {
  XELOGI("KINECT: XamNuiCameraTiltGetStatus status_ptr={:08X}",
         unk.guest_address());
  /* Notes:
     - Used by XamNuiCameraElevationGetAngle, and XamNuiCameraSetFlags
     - if it returns anything greater than -1 then both above functions continue
     - Both funcs send in a param of *unk = 0x50 bytes to copy
     - unk2
     - Ghidra decompile fails
  */
  return IsNuiReady() ? X_STATUS_SUCCESS : X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamNuiCameraTiltGetStatus, kNone, kImplemented);

dword_result_t XamNuiCameraTiltReportStatus_entry(unknown_t unk1,
                                                  unknown_t unk2,
                                                  unknown_t unk3) {
  XELOGI("KINECT: XamNuiCameraTiltReportStatus {:08X} {:08X} {:08X}",
         unk1.value(), unk2.value(), unk3.value());
  return IsNuiReady() ? X_STATUS_SUCCESS : X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamNuiCameraTiltReportStatus, kNone, kImplemented);

dword_result_t XamNuiCameraElevationGetAngle_entry(lpqword_t unk1,
                                                   lpdword_t unk2) {
  XELOGI("KINECT: XamNuiCameraElevationGetAngle angle_ptr={:08X} "
         "status_ptr={:08X}",
         unk1.guest_address(), unk2.guest_address());
  if (!unk1 || !unk2) {
    return X_E_INVALIDARG;
  }
  int32_t angle = 0;
  const uint32_t result = GetNuiCameraElevationAngle(&angle);
  if (XSUCCEEDED(result)) {
    *unk1 = static_cast<uint64_t>(static_cast<int64_t>(angle));
    *unk2 = 0;  // movement flags: the host call completed synchronously
  }
  return result;
}
DECLARE_XAM_EXPORT1(XamNuiCameraElevationGetAngle, kNone, kImplemented);

dword_result_t XamNuiCameraElevationSetAngle_entry(int_t angle) {
  XELOGI("KINECT: XamNuiCameraElevationSetAngle {}", angle.value());
  return SetNuiCameraElevationAngle(angle.value());
}
DECLARE_XAM_EXPORT1(XamNuiCameraElevationSetAngle, kNone, kImplemented);

dword_result_t XamNuiCameraElevationStopMovement_entry(unknown_t unk1) {
  XELOGI("KINECT: XamNuiCameraElevationStopMovement {:08X}", unk1.value());
  return IsNuiReady() ? X_STATUS_SUCCESS : X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamNuiCameraElevationStopMovement, kNone, kImplemented);

dword_result_t XamNuiCameraGetTiltControllerType_entry() {
  XELOGI("KINECT: XamNuiCameraGetTiltControllerType");
  /* Notes:
     - undefined unk[8]
     - undefined8 local_28;
     - undefined8 local_20;
     - undefined8 local_18;
     - undefined4 local_10;
     - local_20 = 0;
     - local_18 = 0;
     - local_10 = 0;
     - local_28 = 0xf030000000000;
     - calls DetroitDeviceRequest(unk) -> result
     - returns (ulonglong)(LZCOUNT(result) << 0x20) >> 0x25
  */
  return IsNuiReady() ? 1 : X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamNuiCameraGetTiltControllerType, kNone, kImplemented);

dword_result_t XamNuiCameraSetFlags_entry(qword_t unk1, dword_t unk2) {
  XELOGI("KINECT: XamNuiCameraSetFlags {:016X} {:08X}", unk1.value(),
         unk2.value());
  /* Notes:
     - if XamNuiCameraGetTiltControllerType returns 1 then operation is done
     - else 0xffffffff8007048f
  */
  X_STATUS result = X_E_DEVICE_NOT_CONNECTED;
  int Controller_Type = XamNuiCameraGetTiltControllerType_entry();

  if (Controller_Type == 1) {
    uint32_t tilt_status[] = {0x58745373, 0x50};  // (XtSs)? & bytes to copy
    result = XamNuiCameraTiltGetStatus_entry(tilt_status);
    if (XSUCCEEDED(result)) {
      // op here
      // result =
    }
  }
  return result;
}
DECLARE_XAM_EXPORT1(XamNuiCameraSetFlags, kNone, kImplemented);

dword_result_t XamNuiCameraRememberFloor_entry(unknown_t unk1,
                                               unknown_t unk2) {
  XELOGI("KINECT: XamNuiCameraRememberFloor {:08X} {:08X}", unk1.value(),
         unk2.value());
  return IsNuiReady() ? X_STATUS_SUCCESS : X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamNuiCameraRememberFloor, kNone, kImplemented);

dword_result_t XamNuiCameraTiltSetCallback_entry(unknown_t callback,
                                                 unknown_t context) {
  XELOGI("KINECT: XamNuiCameraTiltSetCallback callback={:08X} context={:08X}",
         callback.value(), context.value());
#if XE_PLATFORM_WIN32
  GetKinectV1SkeletonRuntime().SetTiltCallback(callback.value(),
                                              context.value());
#endif
  return IsNuiReady() ? X_STATUS_SUCCESS : X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamNuiCameraTiltSetCallback, kNone, kImplemented);

dword_result_t XamNuiCameraElevationAutoTilt_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  XELOGI("KINECT: XamNuiCameraElevationAutoTilt");
  return IsNuiReady() ? X_STATUS_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraElevationAutoTilt, kNone, kImplemented);

dword_result_t XamNuiCameraElevationReverseAutoTilt_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  XELOGI("KINECT: XamNuiCameraElevationReverseAutoTilt");
  return IsNuiReady() ? X_STATUS_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraElevationReverseAutoTilt, kNone, kImplemented);

dword_result_t XamNuiCameraElevationSetCallback_entry(unknown_t callback,
                                                      unknown_t context) {
  XELOGI(
      "KINECT: XamNuiCameraElevationSetCallback callback={:08X} context={:08X}",
      callback.value(), context.value());
  return IsNuiReady() ? X_STATUS_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraElevationSetCallback, kNone, kImplemented);

dword_result_t XamNuiCameraAdjustTilt_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  XELOGI("KINECT: XamNuiCameraAdjustTilt");
  return IsNuiReady() ? X_ERROR_SUCCESS : X_ERROR_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraAdjustTilt, kNone, kImplemented);

dword_result_t XamIsNuiUIActive_entry() {
  return kernel_state()->xam_state()->xam_nui_dialogs_shown_ > 0;
}
DECLARE_XAM_EXPORT1(XamIsNuiUIActive, kNone, kImplemented);

dword_result_t XamNuiIsDeviceReady_entry() {
  XELOGI("KINECT: XamNuiIsDeviceReady");
  /* device_state Notes:
   - used with XNotifyBroadcast(kXNotificationSystemNUIHardwareStatusChanged,
   device_state)
   - known values:
     - 0x0001
     - 0x0004
     - 0x0040
  */
  return IsNuiReady() ? 1 : 0;
}
DECLARE_XAM_EXPORT1(XamNuiIsDeviceReady, kNone, kImplemented);

dword_result_t XamNuiSetForceDeviceOff_entry(int_t force_off) {
  g_nui_force_device_off.store(force_off != 0, std::memory_order_relaxed);
  if (force_off) {
    g_engaged_nui_tracking_id = 0;
  }
  XELOGI("KINECT: XamNuiSetForceDeviceOff force_off={}", force_off.value());
  return X_STATUS_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiSetForceDeviceOff, kNone, kImplemented);

dword_result_t XamNuiNatalCameraUpdateStarting_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  XELOGI("KINECT: XamNuiNatalCameraUpdateStarting");
  return IsNuiReady() ? X_STATUS_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiNatalCameraUpdateStarting, kNone, kImplemented);

dword_result_t XamNuiNatalCameraUpdateComplete_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  XELOGI("KINECT: XamNuiNatalCameraUpdateComplete");
  return IsNuiReady() ? X_STATUS_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiNatalCameraUpdateComplete, kNone, kImplemented);

dword_result_t XamNuiGetSystemGestureControl_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  XELOGI("KINECT: XamNuiGetSystemGestureControl");
  return 0;
}
DECLARE_XAM_EXPORT1(XamNuiGetSystemGestureControl, kNone, kImplemented);

dword_result_t XamNuiHudInterpretFrame_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  const uint32_t tracking_id = GetNuiBestTrackingId();
  if (tracking_id) {
    SetEngagedNuiTrackingId(tracking_id);
  }
  return IsNuiReady() ? X_STATUS_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiHudInterpretFrame, kNone, kImplemented);

dword_result_t XamNuiHudEnableInputFilter_entry(int_t enable) {
  XELOGI("KINECT: XamNuiHudEnableInputFilter enable={}", enable.value());
  return X_STATUS_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiHudEnableInputFilter, kNone, kImplemented);

dword_result_t XamEnableNuiAutomation_entry(int_t enable) {
  g_nui_automation_enabled.store(enable != 0, std::memory_order_relaxed);
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamEnableNuiAutomation, kNone, kImplemented);

dword_result_t XamEnableNatalPlayback_entry(int_t enable) {
  g_natal_playback_enabled.store(enable != 0, std::memory_order_relaxed);
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamEnableNatalPlayback, kNone, kImplemented);

dword_result_t XamIsNuiAutomationEnabled_entry(unknown_t unk1, unknown_t unk2) {
  /* Notes:
     - XamIsNuiAutomationEnabled = XamIsNatalPlaybackEnabled
     - Always returns X_E_SUCCESS? Maybe check later versions
     - Recieves param but never interacts with them
     - No Operations
  */
  return g_nui_automation_enabled.load(std::memory_order_relaxed);
}
DECLARE_XAM_EXPORT2(XamIsNuiAutomationEnabled, kNone, kImplemented,
                    kHighFrequency);

dword_result_t XamIsNatalPlaybackEnabled_entry(unknown_t unk1, unknown_t unk2) {
  /* Notes:
     - XamIsNuiAutomationEnabled = XamIsNatalPlaybackEnabled
     - Always returns X_E_SUCCESS? Maybe check later versions
     - Recieves param but never interacts with them
     - No Operations
  */
  return g_natal_playback_enabled.load(std::memory_order_relaxed);
}
DECLARE_XAM_EXPORT2(XamIsNatalPlaybackEnabled, kNone, kImplemented,
                    kHighFrequency);

dword_result_t XamNuiIsChatMicEnabled_entry() {
  /* Notes:
     - calls a second function with a param of uint local_20 [4];
     - Second function calls ExGetXConfigSetting(7,9,local_30,0x1c,local_40);
     - Result is sent to *local_20[0] = ^
     - Once sent back to XamNuiIsChatMicEnabled it looks for byte that
     correlates to NUI mic setting
     - return uVar2 = (~(ulonglong)local_20[0] << 0x20) >> 0x23 & 1;
     - unless the second function returns something -1 or less then
     XamNuiIsChatMicEnabled 1
  */
  return g_nui_chat_mic_enabled.load(std::memory_order_relaxed);
}
DECLARE_XAM_EXPORT1(XamNuiIsChatMicEnabled, kNone, kImplemented);

/* HUD Notes:
   - XamNuiHudGetEngagedTrackingID, XamNuiHudIsEnabled,
   XamNuiHudSetEngagedTrackingID, XamNuiHudInterpretFrame, and
   XamNuiHudGetEngagedEnrollmentIndex all utilize the same data address
   - engaged_tracking_id set second param of XamShowNuiTroubleshooterUI
*/
uint32_t nui_unknown_1 = 0;
char nui_unknown_2 = '\0';

dword_result_t XamNuiHudSetEngagedTrackingID_entry(dword_t id) {
  XELOGI("KINECT: XamNuiHudSetEngagedTrackingID id={:08X}", id.value());
  if (IsNuiReady()) {
    const uint32_t tracking_id = id;
    SetEngagedNuiTrackingId(tracking_id ? tracking_id
                                        : GetSyntheticTrackingId());
    nui_unknown_1 = 1;
    return X_STATUS_SUCCESS;
  }

  if (!id) {
    return X_STATUS_SUCCESS;
  }

  if (nui_unknown_1 != 0) {
    SetEngagedNuiTrackingId(id);
    return X_STATUS_SUCCESS;
  }

  return X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamNuiHudSetEngagedTrackingID, kNone, kImplemented);

qword_result_t XamNuiHudGetEngagedTrackingID_entry() {
  XELOGI("KINECT: XamNuiHudGetEngagedTrackingID");
  if (IsNuiReady()) {
    const uint32_t tracking_id = GetEngagedNuiTrackingId();
    return tracking_id ? tracking_id : GetSyntheticTrackingId();
  }

  if (nui_unknown_1 != 0) {
    return GetEngagedNuiTrackingId();
  }

  return X_STATUS_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiHudGetEngagedTrackingID, kNone, kImplemented);

dword_result_t XamNuiHudGetEngagedEnrollmentIndex_entry() {
  const uint32_t tracking_id = GetEngagedNuiTrackingId();
  const uint32_t enrollment_index = GetNuiIdentityEnrollmentIndex(tracking_id);
  XELOGI("KINECT: XamNuiHudGetEngagedEnrollmentIndex -> {:08X}; "
         "tracking_id={:08X}, user={:08X}, flags={:08X}",
         enrollment_index, tracking_id, GetNuiIdentityUserIndex(tracking_id),
         GetNuiIdentityEnrollmentFlags(tracking_id));
  return enrollment_index;
}
DECLARE_XAM_EXPORT1(XamNuiHudGetEngagedEnrollmentIndex, kNone, kImplemented);

dword_result_t XamNuiHudIsEnabled_entry() {
  XELOGI("KINECT: XamNuiHudIsEnabled");
  /* Notes:
     - checks if XamNuiIsDeviceReady false, if nui_unknown_1 exists, and
     nui_unknown_2 is equal to null terminated string
     - only returns true if one check fails and allows for other NUI functions
     to progress
  */
  if (IsNuiReady()) {
    return true;
  }

  bool result = XamNuiIsDeviceReady_entry();
  if (nui_unknown_1 != 0 && nui_unknown_2 != '\0' && result) {
    return true;
  }
  return false;
}
DECLARE_XAM_EXPORT1(XamNuiHudIsEnabled, kNone, kImplemented);

uint32_t XeXamNuiHudCheck(dword_t unk1) {
  uint32_t check = XamNuiHudIsEnabled_entry();
  if (check == 0) {
    return X_ERROR_ACCESS_DENIED;
  }

  check = XamNuiHudSetEngagedTrackingID_entry(unk1);
  if (check != 0) {
    return X_ERROR_FUNCTION_FAILED;
  }
  return X_STATUS_SUCCESS;
}

dword_result_t XamNuiHudGetInitializeFlags_entry() {
  /* HUD_Flags Notes:
     - set by 0x2B003
     - set to 0 by unnamed func alongside version_id
     - known values:
       - 0x40000000
       - 0x200
  */
  return 0;
}
DECLARE_XAM_EXPORT1(XamNuiHudGetInitializeFlags, kNone, kImplemented);

void XamNuiHudGetVersions_entry(lpqword_t unk1, lpqword_t unk2) {
  /* version_id Notes:
     - set by 0x2B003
     - set to 0 by unnamed func alongside HUD_Flags
  */
  if (unk1) {
    *unk1 = 0;
  }
  if (unk2) {
    *unk2 = 0;
  }
}
DECLARE_XAM_EXPORT1(XamNuiHudGetVersions, kNone, kImplemented);

// UI
dword_result_t XamShowNuiTroubleshooterUI_entry(dword_t user_index,
                                                dword_t tracking_id,
                                                dword_t flags) {
  XELOGI("KINECT: XamShowNuiTroubleshooterUI user={} tracking_id={:08X} "
         "flags={:08X}",
         user_index.value(), tracking_id.value(), flags.value());
  /* Notes:
     - calls XamPackageManagerGetExperienceMode(&var) with var = 1
     - If returns less than zero or (var & 1) == 0 then get error message:
       - if XamPackageManagerGetExperienceMode = 0 then call XamShowMessageBoxUI
         - if XamShowMessageBoxUI returns 0x3e5 then XamShowNuiTroubleshooterUI
     returns 0
       - else XamShowNuiTroubleshooterUI returns 0x65b and call another func
     - else:
       - call XamNuiHudSetEngagedTrackingID(tracking_id) and doesn't care aboot
     return and set var2 = 2
       - checks if (flag & 0x800000) == 0
         - if true call XamNuiGetDeviceStatus.
           - if XamNuiGetDeviceStatus != 0 set var2 = 3
       - else var2 = 4
       - XamAppRequestLoadEx(var2);
       - if return = 0 then XamShowNuiTroubleshooterUI returns 5
       - else set buffer[8] and call
     XMsgSystemProcessCall(0xfe,0x21028,buffer,0xc);
     - XamNuiNatalCameraUpdateComplete calls
     XamShowNuiTroubleshooterUI(0xff,0,0) if param = -0x7ff8fffe
  */

  if (IsNuiReady()) {
    return X_ERROR_SUCCESS;
  }

  if (cvars::headless) {
    return 0;
  }

  const Emulator* emulator = kernel_state()->emulator();
  ui::Window* display_window = emulator->display_window();
  ui::ImGuiDrawer* imgui_drawer = emulator->imgui_drawer();
  if (display_window && imgui_drawer) {
    xe::threading::Fence fence;
    if (display_window->app_context().CallInUIThreadSynchronous([&]() {
          xe::ui::ImGuiDialog::ShowMessageBox(
              imgui_drawer, "NUI Troubleshooter",
              "The game has indicated there is a problem with NUI (Kinect).")
              ->Then(&fence);
        })) {
      kernel_state()->xam_state()->xam_dialogs_shown_++;
      fence.Wait();
      kernel_state()->xam_state()->xam_dialogs_shown_--;
    }
  }

  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamShowNuiTroubleshooterUI, kNone, kImplemented);

dword_result_t XamShowNuiHardwareRequiredUI_entry(unknown_t unk1) {
  XELOGI("KINECT: XamShowNuiHardwareRequiredUI {:08X}", unk1.value());
  if (unk1 != 0) {
    return X_ERROR_INVALID_PARAMETER;
  }

  return XamShowNuiTroubleshooterUI_entry(0xff, 0, 0x400000);
}
DECLARE_XAM_EXPORT1(XamShowNuiHardwareRequiredUI, kNone, kImplemented);

dword_result_t XamShowNuiGuideUI_entry(unknown_t unk1, unknown_t unk2) {
  XELOGI("KINECT: XamShowNuiGuideUI {:08X} {:08X}", unk1.value(),
         unk2.value());
  /* Notes:
   - calls an unnamed function that checks XamNuiHudIsEnabled and
   XamNuiHudSetEngagedTrackingID
     - if XamNuiHudIsEnabled returns false then fuctions fails return
   X_ERROR_ACCESS_DENIED
     - else calls XamNuiHudSetEngagedTrackingID and if returns less than 0 then
   returns X_ERROR_FUNCTION_FAILED
     - else return X_ERROR_SUCCESS
   - if return offunc is X_ERROR_SUCCESS then call up ui screen
   - else return value of func
  */

  // decompiler error stops me from knowing which param gets used here
  uint32_t result = XeXamNuiHudCheck(0);
  if (!result) {
    // operations here
    // XMsgSystemProcessCall(0xfe,0x21030, undefined local_30[8] ,0xc);
  }
  return result;
}
DECLARE_XAM_EXPORT1(XamShowNuiGuideUI, kNone, kSketchy);

dword_result_t XamShowNuiGamerCardUIForXUID_entry(unknown_t unk1,
                                                  unknown_t unk2,
                                                  unknown_t unk3) {
  XELOGI("KINECT: XamShowNuiGamerCardUIForXUID {:08X} {:08X} {:08X}",
         unk1.value(), unk2.value(), unk3.value());
  return IsNuiReady() ? X_ERROR_SUCCESS : X_ERROR_FUNCTION_FAILED;
}
DECLARE_XAM_EXPORT1(XamShowNuiGamerCardUIForXUID, kNone, kSketchy);

dword_result_t XamShowNuiAchievementsUI_entry(unknown_t unk1, unknown_t unk2,
                                              unknown_t unk3) {
  XELOGI("KINECT: XamShowNuiAchievementsUI {:08X} {:08X} {:08X}",
         unk1.value(), unk2.value(), unk3.value());
  return IsNuiReady() ? X_ERROR_SUCCESS : X_ERROR_FUNCTION_FAILED;
}
DECLARE_XAM_EXPORT1(XamShowNuiAchievementsUI, kNone, kSketchy);

dword_result_t XamShowNuiMarketplaceUI_entry(unknown_t unk1, unknown_t unk2,
                                             unknown_t unk3, unknown_t unk4,
                                             unknown_t unk5, unknown_t unk6) {
  XELOGI("KINECT: XamShowNuiMarketplaceUI {:08X} {:08X} {:08X} {:08X} "
         "{:08X} {:08X}",
         unk1.value(), unk2.value(), unk3.value(), unk4.value(), unk5.value(),
         unk6.value());
  return IsNuiReady() ? X_ERROR_SUCCESS : X_ERROR_FUNCTION_FAILED;
}
DECLARE_XAM_EXPORT1(XamShowNuiMarketplaceUI, kNone, kSketchy);

dword_result_t XamShowNuiDeviceSelectorUI_entry(unknown_t unk1,
                                                unknown_t unk2,
                                                unknown_t unk3,
                                                unknown_t unk4) {
  XELOGI("KINECT: XamShowNuiDeviceSelectorUI {:08X} {:08X} {:08X} {:08X}",
         unk1.value(), unk2.value(), unk3.value(), unk4.value());
  return IsNuiReady() ? X_ERROR_SUCCESS : X_ERROR_FUNCTION_FAILED;
}
DECLARE_XAM_EXPORT1(XamShowNuiDeviceSelectorUI, kNone, kSketchy);

void XamShowNuiDirtyDiscErrorUI_entry(unknown_t unk1) {
  XELOGI("KINECT: XamShowNuiDirtyDiscErrorUI {:08X}", unk1.value());
}
DECLARE_XAM_EXPORT1(XamShowNuiDirtyDiscErrorUI, kNone, kSketchy);

dword_result_t XamShowNuiCompatibilityUI(unknown_t r3, unknown_t r4,
                                         unknown_t r5, unknown_t r6,
                                         unknown_t r7, unknown_t r8) {
  return IsNuiReady() ? X_ERROR_SUCCESS : X_ERROR_FUNCTION_FAILED;
}

dword_result_t XamShowNuiCommunitySessionsUI_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamShowNuiCompatibilityUI(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamShowNuiCommunitySessionsUI, kNone, kSketchy);

dword_result_t XamShowNuiControllerRequiredUI_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamShowNuiCompatibilityUI(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamShowNuiControllerRequiredUI, kNone, kSketchy);

dword_result_t XamShowNuiFriendRequestUI_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamShowNuiCompatibilityUI(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamShowNuiFriendRequestUI, kNone, kSketchy);

dword_result_t XamShowNuiFriendsUI_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamShowNuiCompatibilityUI(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamShowNuiFriendsUI, kNone, kSketchy);

dword_result_t XamShowNuiGameInviteUI_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamShowNuiCompatibilityUI(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamShowNuiGameInviteUI, kNone, kSketchy);

dword_result_t XamShowNuiGamesUI_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamShowNuiCompatibilityUI(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamShowNuiGamesUI, kNone, kSketchy);

dword_result_t XamShowNuiJoinSessionInProgressUI_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamShowNuiCompatibilityUI(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamShowNuiJoinSessionInProgressUI, kNone, kSketchy);

dword_result_t XamShowNuiMarketplaceDownloadItemsUI_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamShowNuiCompatibilityUI(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamShowNuiMarketplaceDownloadItemsUI, kNone, kSketchy);

dword_result_t XamShowNuiMessageBoxUI_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamShowNuiCompatibilityUI(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamShowNuiMessageBoxUI, kNone, kSketchy);

dword_result_t XamShowNuiMessagesUI_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamShowNuiCompatibilityUI(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamShowNuiMessagesUI, kNone, kSketchy);

dword_result_t XamShowNuiPartyUI_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamShowNuiCompatibilityUI(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamShowNuiPartyUI, kNone, kSketchy);

dword_result_t XamShowNuiVideoRichPresenceUI_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamShowNuiCompatibilityUI(r3, r4, r5, r6, r7, r8);
}
DECLARE_XAM_EXPORT1(XamShowNuiVideoRichPresenceUI, kNone, kSketchy);

dword_result_t XamTestShowNuiTroubleshooterUI_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return XamShowNuiTroubleshooterUI_entry(0xff, 0, 0);
}
DECLARE_XAM_EXPORT1(XamTestShowNuiTroubleshooterUI, kNone, kImplemented);

dword_result_t XamReportKinectSettingsChangedEvent_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  XELOGI("KINECT: XamReportKinectSettingsChangedEvent");
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamReportKinectSettingsChangedEvent, kNone, kImplemented);

dword_result_t XamKinectGetHardwareType_entry() {
  return IsNuiReady() ? 1 : 0;
}
DECLARE_XAM_EXPORT1(XamKinectGetHardwareType, kNone, kImplemented);

dword_result_t XamLoaderIsKinectUIPreferredForLogonTitle_entry() {
  return IsNuiReady();
}
DECLARE_XAM_EXPORT1(XamLoaderIsKinectUIPreferredForLogonTitle, kNone,
                    kImplemented);

dword_result_t ControlpackNuiCursorSetTrackingId_entry(dword_t tracking_id) {
  SetEngagedNuiTrackingId(tracking_id.value());
  return IsNuiReady() ? X_ERROR_SUCCESS : X_ERROR_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(ControlpackNuiCursorSetTrackingId, kNone, kImplemented);

pointer_result_t ControlPackSideNavControlGetNuiHandle_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  return 0;
}
DECLARE_XAM_EXPORT1(ControlPackSideNavControlGetNuiHandle, kNone, kSketchy);

/* XamNuiIdentity Notes:
   - most require message calls to xam in 0x0002Cxxx area
*/
uint64_t NUI_Session_Id = 0;

qword_result_t XamNuiIdentityGetSessionId_entry() {
  XELOGI("KINECT: XamNuiIdentityGetSessionId");
  if (NUI_Session_Id == 0) {
    // xboxkrnl::XeCryptRandom_entry(NUI_Session_Id, 8);
    NUI_Session_Id = 0xDEADF00DDEADF00D;
  }
  return NUI_Session_Id;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityGetSessionId, kNone, kImplemented);

dword_result_t XamNuiIdentityEnrollForSignIn_entry(dword_t unk1, qword_t unk2,
                                                   qword_t unk3, dword_t unk4) {
  XELOGI("KINECT: XamNuiIdentityEnrollForSignIn {:08X} {:016X} {:016X} "
         "{:08X}",
         unk1.value(), unk2.value(), unk3.value(), unk4.value());
  /* Notes:
     - Decompiler issues so double check
  */
  if (IsNuiReady()) {
    return X_STATUS_SUCCESS;
  }

  if (XamNuiHudIsEnabled_entry() == false) {
    return X_E_FAIL;
  }
  // buffer [2]
  // buffer[0] = unk
  // var = unk4
  // return func(0xfe,0x2c010,buffer,0xc,unk3);
  return X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityEnrollForSignIn, kNone, kImplemented);

dword_result_t XamNuiIdentityIdentifyWithBiometric_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  const uint32_t tracking_id =
      uint32_t(r3) ? uint32_t(r3) : GetEngagedNuiTrackingId();
  if (tracking_id) {
    SetEngagedNuiTrackingId(tracking_id);
  }
  XELOGI("KINECT: XamNuiIdentityIdentifyWithBiometric {:08X} {:08X} {:08X} "
         "{:08X} {:08X} {:08X}; tracking_id={:08X}",
         uint32_t(r3), uint32_t(r4), uint32_t(r5), uint32_t(r6), uint32_t(r7),
         uint32_t(r8), GetEngagedNuiTrackingId());
  return IsNuiReady() ? X_STATUS_SUCCESS : X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityIdentifyWithBiometric, kNone, kImplemented);

dword_result_t XamNuiIdentityGetQualityFlags_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  XELOGI("KINECT: XamNuiIdentityGetQualityFlags {:08X} {:08X} {:08X} "
         "{:08X} {:08X} {:08X}",
         uint32_t(r3), uint32_t(r4), uint32_t(r5), uint32_t(r6), uint32_t(r7),
         uint32_t(r8));
  return 0;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityGetQualityFlags, kNone, kImplemented);

dword_result_t XamNuiIdentityGetQualityFlagsMessage_entry(
    lpvoid_t message, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  const uint32_t tracking_id = GetEngagedNuiTrackingId();
  const bool has_player = HasNuiIdentityPlayer(tracking_id);
  if (message && has_player) {
    auto* p = static_cast<uint8_t*>(message);
    std::memset(p, 0, 0x1C);
    xe::store_and_swap<uint32_t>(p + 0x00, 1);
    xe::store_and_swap<uint32_t>(p + 0x04, 1);
    xe::store_and_swap<uint32_t>(p + 0x08, tracking_id);
    xe::store_and_swap<uint32_t>(p + 0x0C, GetNuiSkeletonFrameNumber());
    xe::store_and_swap<uint32_t>(p + 0x10, X_STATUS_SUCCESS);
    xe::store_and_swap<uint32_t>(p + 0x14,
                                 GetNuiIdentityEnrollmentIndex(tracking_id));
    xe::store_and_swap<uint32_t>(p + 0x18, 1);
  }
  const uint32_t result = has_player ? X_STATUS_SUCCESS : X_E_NO_MORE_FILES;
  XELOGI("KINECT: XamNuiIdentityGetQualityFlagsMessage msg={:08X} "
         "{:08X} {:08X} {:08X} {:08X} {:08X} -> {:08X}; tracking_id={:08X}",
         message.guest_address(), uint32_t(r4), uint32_t(r5), uint32_t(r6),
         uint32_t(r7), uint32_t(r8), result, tracking_id);
  return result;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityGetQualityFlagsMessage, kNone, kImplemented);

dword_result_t XamNuiIdentityAbort_entry(dword_t unk) {
  XELOGI("KINECT: XamNuiIdentityAbort {:08X}", unk.value());
  if (IsNuiReady()) {
    return X_STATUS_SUCCESS;
  }

  if (XamNuiHudIsEnabled_entry() == false) {
    return X_E_FAIL;
  }
  // buffer [4]
  // buffer[0] = unk
  // return func(0xfe,0x2c00e,buffer,4,0)
  return X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityAbort, kNone, kImplemented);

// Other
dword_result_t XamUserNuiEnableBiometric_entry(dword_t user_index,
                                               int_t enable) {
  XELOGI("KINECT: XamUserNuiEnableBiometric user={} enable={}",
         user_index.value(), enable.value());
  if (user_index >= XUserMaxUserCount) {
    return X_E_INVALIDARG;
  }
  if (IsNuiReady()) {
    g_nui_biometric_enabled[user_index.value()].store(
        enable != 0, std::memory_order_relaxed);
    return X_STATUS_SUCCESS;
  }
  return X_E_INVALIDARG;
}
DECLARE_XAM_EXPORT1(XamUserNuiEnableBiometric, kNone, kImplemented);

dword_result_t XamUserNuiGetEnrollmentIndex_entry(dword_t user_index,
                                                  lpdword_t index) {
  XELOGI("KINECT: XamUserNuiGetEnrollmentIndex user={} index_ptr={:08X}",
         user_index.value(), index.guest_address());
  if (IsNuiReady() && index) {
    *index = 0;
    return X_E_SUCCESS;
  }
  return X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamUserNuiGetEnrollmentIndex, kNone, kImplemented);

dword_result_t XamReadBiometricData_entry(unknown_t unk1, unknown_t unk2,
                                          unknown_t unk3, unknown_t unk4,
                                          unknown_t unk5) {
  XELOGI("KINECT: XamReadBiometricData {:08X} {:08X} {:08X} {:08X} {:08X}",
         unk1.value(), unk2.value(), unk3.value(), unk4.value(), unk5.value());
  return IsNuiReady() ? X_ERROR_SUCCESS : X_ERROR_FUNCTION_FAILED;
}
DECLARE_XAM_EXPORT1(XamReadBiometricData, kNone, kImplemented);

dword_result_t XamWriteBiometricData_entry(unknown_t unk1, unknown_t unk2,
                                           unknown_t unk3, unknown_t unk4,
                                           unknown_t unk5) {
  XELOGI("KINECT: XamWriteBiometricData {:08X} {:08X} {:08X} {:08X} {:08X}",
         unk1.value(), unk2.value(), unk3.value(), unk4.value(), unk5.value());
  return IsNuiReady() ? X_ERROR_SUCCESS : X_ERROR_FUNCTION_FAILED;
}
DECLARE_XAM_EXPORT1(XamWriteBiometricData, kNone, kImplemented);

dword_result_t XamNuiSkeletonScoreUpdate_entry(
    unknown_t r3, unknown_t r4, unknown_t r5, unknown_t r6, unknown_t r7,
    unknown_t r8) {
  XELOGI("KINECT: XamNuiSkeletonScoreUpdate {:08X} {:08X} {:08X} {:08X} "
         "{:08X} {:08X}",
         uint32_t(r3), uint32_t(r4), uint32_t(r5), uint32_t(r6), uint32_t(r7),
         uint32_t(r8));
  return X_STATUS_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiSkeletonScoreUpdate, kNone, kImplemented);

dword_result_t XamNuiPlayerEngagementUpdate_entry(
    qword_t unk1, unknown_t unk2, lpvoid_t engagement_state, unknown_t r6,
    unknown_t r7, unknown_t r8) {
  const uint32_t tracking_id = GetEngagedNuiTrackingId();
  const uint32_t tracked_count = GetNuiTrackedSkeletonCount();
  const uint32_t best_index = GetNuiBestSkeletonIndex();
  const uint32_t frame_number = GetNuiSkeletonFrameNumber();
  const bool has_player =
      tracked_count != 0 && tracking_id != kNuiInvalidTrackingId;

  if (engagement_state) {
    auto* state = static_cast<uint8_t*>(engagement_state);
    std::memset(state, 0, 0x1C);
    if (has_player) {
      constexpr uint32_t kKinectDetectionResultNo = 1;
      constexpr uint32_t kKinectDetectionResultYes = 3;
      xe::store_and_swap<uint32_t>(state + 0x00, 1);
      xe::store_and_swap<uint32_t>(state + 0x04, tracking_id);
      xe::store_and_swap<uint32_t>(
          state + 0x08, best_index != 0xFFFFFFFFu ? best_index : 0);
      xe::store_and_swap<uint32_t>(state + 0x0C, kKinectDetectionResultYes);
      xe::store_and_swap<uint32_t>(state + 0x10, kKinectDetectionResultNo);
      xe::store_and_swap<uint32_t>(state + 0x14, kKinectDetectionResultYes);
      xe::store_and_swap<uint32_t>(state + 0x18, frame_number);
    }
  }

  XELOGI("KINECT: XamNuiPlayerEngagementUpdate {:016X} {:08X} ptr={:08X} "
         "{:08X} {:08X} {:08X}; tracked_count={} tracking_id={:08X} "
         "best_index={:08X} engaged={}",
         unk1.value(), uint32_t(unk2), engagement_state.guest_address(),
         uint32_t(r6), uint32_t(r7), uint32_t(r8), tracked_count, tracking_id,
         best_index, has_player ? 1 : 0);
  return X_STATUS_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiPlayerEngagementUpdate, kNone, kImplemented);

void D3DTexture_LockRect_entry(lpvoid_t texture, dword_t level,
                               lpvoid_t locked_rect, lpvoid_t rect,
                               dword_t flags) {
  const uint32_t texture_addr = texture.guest_address();
  if (!LockNuiImageTextureToGuest(
          texture_addr, level.value(), locked_rect.guest_address(),
          rect.guest_address(), flags.value())) {
    static std::atomic<uint32_t> non_nui_calls{0};
    const uint32_t n =
        non_nui_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n <= 8 || (n & (n - 1)) == 0) {
      XELOGW("D3DTexture_LockRect non-NUI texture #{} tex={:08X} level={} "
             "out={:08X} rect={:08X} flags={:08X}",
             n, texture_addr, level.value(), locked_rect.guest_address(),
             rect.guest_address(), flags.value());
    }
    return;
  }
}
DECLARE_XAM_EXPORT2(D3DTexture_LockRect, kVideo, kImplemented,
                    kHighFrequency);

void D3DTexture_UnlockRect_entry(lpvoid_t texture, dword_t level) {
  if (!IsNuiImageTextureGuestAddress(texture.guest_address())) {
    return;
  }
  static std::atomic<uint32_t> nui_unlock_calls{0};
  const uint32_t n =
      nui_unlock_calls.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n <= 16 || (n & (n - 1)) == 0) {
    XELOGI("Kinect v1 NUI depth texture UnlockRect #{}: tex={:08X} level={}",
           n, texture.guest_address(), level.value());
  }
}
DECLARE_XAM_EXPORT2(D3DTexture_UnlockRect, kVideo, kImplemented,
                    kHighFrequency);

}  // namespace xam
}  // namespace kernel
}  // namespace xe

DECLARE_XAM_EMPTY_REGISTER_EXPORTS(NUI);
