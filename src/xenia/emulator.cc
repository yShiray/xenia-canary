/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2023 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <ranges>
#include <thread>

#include "xenia/emulator.h"

#include "config.h"
#include "third_party/fmt/include/fmt/format.h"
#include "third_party/tabulate/single_include/tabulate/tabulate.hpp"
#include "third_party/zarchive/include/zarchive/zarchivecommon.h"
#include "third_party/zarchive/include/zarchive/zarchivewriter.h"
#include "third_party/zarchive/src/sha_256.h"
#include "xenia/apu/audio_system.h"
#include "xenia/base/assert.h"
#include "xenia/base/byte_stream.h"
#include "xenia/base/clock.h"
#include "xenia/base/cvar.h"
#include "xenia/base/debugging.h"
#include "xenia/base/exception_handler.h"
#include "xenia/base/literals.h"
#include "xenia/base/logging.h"
#include "xenia/base/mapped_memory.h"
#include "xenia/base/platform.h"
#include "xenia/base/string.h"
#include "xenia/base/system.h"
#include "xenia/cpu/backend/code_cache.h"
#include "xenia/cpu/backend/null_backend.h"
#include "xenia/cpu/function.h"
#include "xenia/cpu/cpu_flags.h"
#include "xenia/cpu/ppc/ppc_context.h"
#include "xenia/cpu/thread_state.h"
#include "xenia/gpu/command_processor.h"
#include "xenia/gpu/graphics_system.h"
#include "xenia/hid/input_driver.h"
#include "xenia/hid/input_system.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/title_id_utils.h"
#include "xenia/kernel/user_module.h"
#include "xenia/kernel/xam/achievement_manager.h"
#include "xenia/kernel/xam/xam_module.h"
#include "xenia/kernel/xam/xdbf/spa_info.h"
#include "xenia/kernel/xbdm/xbdm_module.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_module.h"
#include "xenia/memory.h"
#include "xenia/ui/file_picker.h"
#include "xenia/ui/imgui_dialog.h"
#include "xenia/ui/imgui_drawer.h"
#include "xenia/ui/imgui_host_notification.h"
#include "xenia/ui/window.h"
#include "xenia/ui/windowed_app_context.h"
#include "xenia/vfs/device.h"
#include "xenia/vfs/devices/disc_image_device.h"
#include "xenia/vfs/devices/disc_zarchive_device.h"
#include "xenia/vfs/devices/host_path_device.h"
#include "xenia/vfs/devices/null_device.h"
#include "xenia/vfs/devices/xcontent_container_device.h"
#include "xenia/vfs/virtual_file_system.h"

#if XE_ARCH_AMD64
#include "xenia/cpu/backend/x64/x64_backend.h"
#elif XE_ARCH_ARM64
#include "xenia/cpu/backend/a64/a64_backend.h"
#endif  // XE_ARCH

DEFINE_double(time_scalar, 1.0,
              "Scalar used to speed or slow time (1x, 2x, 1/2x, etc).",
              "General");

DEFINE_string(
    launch_module, "",
    "Executable to launch from the .iso or the package instead of default.xex "
    "or the module specified by the game. Leave blank to launch the default "
    "module.",
    "General");

DEFINE_bool(allow_game_relative_writes, false,
            "Not useful to non-developers. Allows code to write to paths "
            "relative to game://. Used for "
            "generating test data to compare with original hardware. ",
            "General");

DECLARE_bool(allow_plugins);

DEFINE_int32(priority_class, 0,
             "Forces Xenia to use different process priority than default one. "
             "It might affect performance and cause unexpected bugs. Possible "
             "values: 0 - Normal, 1 - Above normal, 2 - High",
             "General");

namespace xe {
using namespace xe::literals;

namespace kernel {
namespace xam {
uint32_t GetEngagedNuiTrackingId();
uint32_t GetNuiTrackedSkeletonCount();
uint32_t GetNuiInjectionState();
bool WriteNuiSkeletonFrameToGuest(uint32_t frame_ptr);
bool WriteNuiSkeletonFrameToGuest(uint32_t frame_ptr, uint32_t wait_ms);
uint32_t ConfigureNuiSkeletonTracking(uint32_t event_handle, uint32_t flags);
uint32_t ConfigureNuiTrackedSkeletons(uint32_t tracking_ids_ptr);
uint32_t ConfigureNuiFrameEndEvent(uint32_t event_handle, uint32_t flags);
void SetNuiSkeletonFrameEvent(uint32_t event_handle);
void ResetNuiSkeletonFrameEvent();
void SetNuiImageFrameEvent(uint32_t event_handle);
void SetNuiColorFrameEvent(uint32_t event_handle);
void ResetNuiFrameEvent(uint32_t event_handle);
void ResetNuiGuestEventRegistrations();
uint32_t GetNuiImageFrameNodeGuestAddress();
uint32_t GetNuiImageFrameNodeGuestAddress(uint32_t image_type,
                                          uint32_t resolution);
uint32_t GetNuiImageFrameNumberForType(uint32_t image_type);
bool IsNuiImageFrameNodeGuestAddressForType(uint32_t frame,
                                            uint32_t image_type);
bool IsNuiImageTextureGuestAddress(uint32_t texture);
bool LockNuiImageTextureToGuest(uint32_t texture, uint32_t level,
                                uint32_t locked_rect, uint32_t rect,
                                uint32_t flags);
}  // namespace xam
}  // namespace kernel

namespace {
constexpr uint32_t kNuiFaultingLoadWord = 0x80000000u;
constexpr uint32_t kNuiFrameNoData = 0x83010001u;
constexpr uint32_t kNuiEmulatedStreamHandleBase = 0xD0010000u;
constexpr uint32_t kNuiEmulatedStreamCapacity = 32;
constexpr uint32_t kNuiImageTypeDepthAndPlayerIndex = 0;
constexpr uint32_t kNuiImageTypeColor = 1;
constexpr uint32_t kNuiImageTypeColorYuv = 2;
constexpr uint32_t kNuiImageTypeDepth = 3;
constexpr uint32_t kNuiImageTypeDepthAndPlayerIndexInColorSpace = 4;
constexpr uint32_t kNuiImageTypeDepthInColorSpace = 5;
constexpr uint32_t kNuiImageTypeColorInDepthSpace = 6;
constexpr uint32_t kNuiImageTypeDepthAndPlayerIndex80x60 = 7;
constexpr uint32_t kNuiImageTypeDepth80x60 = 8;
constexpr uint32_t kNuiImageResolution80x60 = 0;
constexpr uint32_t kNuiImageResolution320x240 = 1;
constexpr uint32_t kNuiImageResolution640x480 = 2;
constexpr uint32_t kNuiImageStreamFlagSuppressNoFrameData = 0x00010000u;
constexpr uint32_t kNuiImageStreamFrameLimitMaximum = 48;
constexpr uint32_t kNuiMaximumWaitMilliseconds = 8000;
constexpr uint32_t kNuiSkeletonFrameSize = 2736;

struct NuiCodeRange {
  uint32_t begin = 0;
  uint32_t end = 0;

  bool valid() const { return begin < end; }

  bool Contains(uint32_t address, uint32_t size = sizeof(uint32_t)) const {
    return valid() && size && address >= begin && address < end &&
           size <= end - address;
  }
};

NuiCodeRange MakeNuiCodeRange(uint32_t low_address, uint32_t high_address) {
  if (low_address >= high_address) {
    return {};
  }
  return {low_address, high_address};
}

std::atomic<uint32_t> g_nui_next_stream_slot{0};
std::array<std::atomic<uint32_t>, kNuiEmulatedStreamCapacity>
    g_nui_stream_handles{};
std::array<std::atomic<uint32_t>, kNuiEmulatedStreamCapacity>
    g_nui_stream_image_types{};
std::array<std::atomic<uint32_t>, kNuiEmulatedStreamCapacity>
    g_nui_stream_resolutions{};
std::array<std::atomic<uint32_t>, kNuiEmulatedStreamCapacity>
    g_nui_stream_frame_flags{};
std::array<std::atomic<uint32_t>, kNuiEmulatedStreamCapacity>
    g_nui_stream_frame_limits{};
std::array<std::atomic<uint32_t>, kNuiEmulatedStreamCapacity>
    g_nui_stream_events{};
std::array<std::atomic<uint32_t>, kNuiEmulatedStreamCapacity>
    g_nui_stream_last_frame_numbers{};

std::atomic<uint32_t> g_nui_skeleton_get_next_frame_hook{0};
std::atomic<uint32_t> g_nui_skeleton_tracking_enable_hook{0};
std::atomic<uint32_t> g_nui_skeleton_set_tracked_hook{0};
std::atomic<uint32_t> g_nui_set_frame_end_event_hook{0};
std::atomic<uint32_t> g_nui_image_stream_open_hook{0};
std::atomic<uint32_t> g_nui_image_stream_get_next_frame_hook{0};
std::atomic<uint32_t> g_nui_image_stream_release_frame_hook{0};
constexpr size_t kNuiTextureRouteCapacity = 64;
std::array<std::atomic<uint32_t>, kNuiTextureRouteCapacity>
    g_nui_d3d_texture_lock_calls{};
std::array<std::atomic<uint32_t>, kNuiTextureRouteCapacity>
    g_nui_d3d_texture_lock_targets{};
std::array<std::atomic<uint32_t>, kNuiTextureRouteCapacity>
    g_nui_d3d_texture_unlock_calls{};
std::array<std::atomic<uint32_t>, kNuiTextureRouteCapacity>
    g_nui_d3d_texture_unlock_targets{};

void NuiSkeletonGetNextFrameTrampoline(cpu::ppc::PPCContext* context,
                                       kernel::KernelState* kernel_state);
void NuiSkeletonTrackingEnableTrampoline(cpu::ppc::PPCContext* context,
                                         kernel::KernelState* kernel_state);
void NuiSkeletonSetTrackedSkeletonsTrampoline(
    cpu::ppc::PPCContext* context, kernel::KernelState* kernel_state);
void NuiSetFrameEndEventTrampoline(cpu::ppc::PPCContext* context,
                                   kernel::KernelState* kernel_state);
void NuiImageStreamOpenTrampoline(cpu::ppc::PPCContext* context,
                                  kernel::KernelState* kernel_state);
void NuiImageStreamGetNextFrameTrampoline(cpu::ppc::PPCContext* context,
                                          kernel::KernelState* kernel_state);
void NuiImageStreamReleaseFrameTrampoline(cpu::ppc::PPCContext* context,
                                          kernel::KernelState* kernel_state);

bool NuiScanRangeReadable(Memory* memory, uint32_t address, uint32_t size) {
  if (!memory || !size || address > UINT32_MAX - (size - 1)) {
    return false;
  }
  auto* heap = memory->LookupHeap(address);
  if (!heap || heap != memory->LookupHeap(address + size - 1)) {
    return false;
  }
  const auto access = heap->QueryRangeAccess(address, address + size - 1);
  return access == xe::memory::PageAccess::kReadOnly ||
         access == xe::memory::PageAccess::kReadWrite ||
         access == xe::memory::PageAccess::kExecuteReadOnly ||
         access == xe::memory::PageAccess::kExecuteReadWrite;
}

bool NuiScanRangeWritable(Memory* memory, uint32_t address, uint32_t size) {
  if (!memory || !address || !size || address > UINT32_MAX - (size - 1)) {
    return false;
  }
  auto* heap = memory->LookupHeap(address);
  if (!heap || heap != memory->LookupHeap(address + size - 1)) {
    return false;
  }
  const auto access = heap->QueryRangeAccess(address, address + size - 1);
  return access == xe::memory::PageAccess::kReadWrite ||
         access == xe::memory::PageAccess::kExecuteReadWrite;
}

bool NuiBytesMatch(Memory* memory, uint32_t address, const uint8_t* expect,
                   uint32_t len) {
  if (!NuiScanRangeReadable(memory, address, len)) {
    return false;
  }
  const auto* p = memory->TranslateVirtual<const uint8_t*>(address);
  return p && std::memcmp(p, expect, len) == 0;
}

// Reads a single big-endian PowerPC instruction word from guest memory.
bool NuiReadInstr(Memory* memory, uint32_t address, uint32_t* out) {
  if (!NuiScanRangeReadable(memory, address, sizeof(uint32_t))) {
    return false;
  }
  const auto* p = memory->TranslateVirtual<const uint8_t*>(address);
  if (!p) {
    return false;
  }
  *out = xe::load_and_swap<uint32_t>(p);
  return true;
}

// Locates NuiSkeletonGetNextFrame inside the statically-linked nuiapi.lib,
// independent of title and XDK version. Stack-frame sizes and prologue layouts
// vary between XDK builds, so fixed-offset byte signatures are insufficient.
//
// Instead of matching exact offsets we anchor on the function's parameter
// null-check, whose shape is stable across every XDK because it comes straight
// from the SDK source `if (!pSkeletonFrame) return E_INVALIDARG;`:
//
//   mflr   r12                 <- function entry (what we hook)
//   [bl    <profiler>]         <- present in some builds, absent in others
//   stwu   r1,-FRAME(r1)       <- establishes the stack frame
//   ...
//   cmplwi/cmpwi  rX,0         <- null-check the out param
//   bne    <body>
//   lis    r3,0x8007           <- E_INVALIDARG (the anchor)
//   ori    r3,r3,0x0057
//   addi   r1,r1,FRAME         <- early return UNWINDS the same frame
//   b      <epilogue>
//
// The "E_INVALIDARG immediately followed by `addi r1,r1,FRAME; b`, with a
// matching `stwu r1,-FRAME(r1)` and `mflr r12` above it" pattern is specific
// enough to avoid the many other E_INVALIDARG constants in a title image.
uint32_t FindNuiSkeletonGetNextFrame(Memory* memory,
                                     const NuiCodeRange& scan_range) {
  static const uint8_t kAnchor[8] = {0x3C, 0x60, 0x80, 0x07,
                                     0x60, 0x63, 0x00, 0x57};
  constexpr uint32_t kMflrR12 = 0x7D8802A6u;
  for (uint32_t page = scan_range.begin; page < scan_range.end;
       page += 0x1000u) {
    if (!NuiScanRangeReadable(memory, page, 0x1000u)) {
      continue;
    }
    const auto* base = memory->TranslateVirtual<const uint8_t*>(page);
    if (!base) {
      continue;
    }
    for (uint32_t off = 0; off + sizeof(kAnchor) <= 0x1000u; off += 4) {
      if (std::memcmp(base + off, kAnchor, sizeof(kAnchor)) != 0) {
        continue;
      }
      const uint32_t anchor = page + off;

      // The two instructions after the E_INVALIDARG load must be the
      // frame-unwinding early return: `addi r1,r1,FRAME` then unconditional `b`.
      uint32_t unwind = 0;
      uint32_t branch = 0;
      if (!NuiReadInstr(memory, anchor + 0x08u, &unwind) ||
          !NuiReadInstr(memory, anchor + 0x0Cu, &branch)) {
        continue;
      }
      if ((unwind >> 16) != 0x3821u) {  // addi r1,r1,imm
        continue;
      }
      const int32_t frame = static_cast<int16_t>(unwind & 0xFFFFu);
      if (frame <= 0 || frame > 0x4000) {
        continue;
      }
      if ((branch >> 26) != 18u || (branch & 1u) != 0u) {  // unconditional b
        continue;
      }

      // The two instructions before the anchor must be a pointer null-check:
      // `cmplwi/cmpwi rX,0` then a conditional branch (`bne`).
      uint32_t cmp = 0;
      uint32_t cond = 0;
      if (!NuiReadInstr(memory, anchor - 0x08u, &cmp) ||
          !NuiReadInstr(memory, anchor - 0x04u, &cond)) {
        continue;
      }
      const uint32_t cmp_op = cmp >> 26;
      if ((cmp_op != 10u && cmp_op != 11u) || (cmp & 0xFFFFu) != 0u) {
        continue;  // not a compare-immediate against 0
      }
      if ((cond >> 26) != 16u) {  // bc (the bne to the body)
        continue;
      }

      // Walk back to the matching `stwu r1,-FRAME(r1)` and the `mflr r12`
      // entry right above it (allowing an optional profiler `bl` in between).
      const uint16_t stwu_imm = static_cast<uint16_t>(-frame);
      uint32_t entry = 0;
      for (uint32_t back = 1; back <= 40 && entry == 0; ++back) {
        uint32_t insn = 0;
        if (!NuiReadInstr(memory, anchor - back * 4u, &insn)) {
          break;
        }
        if ((insn >> 16) != 0x9421u ||
            static_cast<uint16_t>(insn & 0xFFFFu) != stwu_imm) {
          continue;
        }
        const uint32_t stwu_addr = anchor - back * 4u;
        for (uint32_t up = 1; up <= 3; ++up) {
          uint32_t maybe_mflr = 0;
          if (NuiReadInstr(memory, stwu_addr - up * 4u, &maybe_mflr) &&
              maybe_mflr == kMflrR12) {
            entry = stwu_addr - up * 4u;
            break;
          }
        }
        break;
      }
      if (entry == 0) {
        continue;
      }

      // Many functions validate a pointer parameter and early-return
      // E_INVALIDARG, so the prologue match above is not enough on its own.
      // Confirm this is really NuiSkeletonGetNextFrame by the unique
      // dwMillisecondsToWait clamp `cmplwi r3,0x1F40` (8000ms) that only this
      // function's timeout logic contains - verified to appear exactly once in
      // the title image, and present across XDK versions.
      //
      // The scan MUST stop at the next function's entry (the following
      // `mflr r12`) so it cannot spill past this function's epilogue into a
      // neighbour's body and match its constant.
      bool confirmed = false;
      for (uint32_t scan = 4; scan <= 0x400u; scan += 4u) {
        uint32_t word = 0;
        if (!NuiReadInstr(memory, entry + scan, &word)) {
          break;
        }
        if (word == kMflrR12) {
          break;  // reached the next function - fingerprint not in this one
        }
        if (word == 0x2B031F40u) {
          confirmed = true;
          break;
        }
      }
      if (!confirmed) {
        continue;
      }

      XELOGI(
          "NUI skeleton injector: NuiSkeletonGetNextFrame found at {:08X} "
          "(frame=0x{:X}, anchor={:08X})",
          entry, frame, anchor);
      return entry;
    }
  }
  return 0;
}

// Locates NuiSkeletonTrackingEnable independent of XDK version. We hook it only
// to capture the title's hNextFrameEvent (r3) so Xenia can signal it when a host
// skeleton frame is ready - otherwise an event-driven title would block in
// WaitForSingleObject and never call the getter.
//
// The stable, near-unique fingerprint is its two back-to-back early returns,
// both unwinding the same stack frame straight from the SDK source:
//
//   ... device-not-ready check ...
//   lis  r3,0x8301           <- 0x83010005 (E_NUI_DEVICE_NOT_GENUINE/not ready)
//   ori  r3,r3,0x0005        (the anchor)
//   addi r1,r1,FRAME
//   b    <epilogue>
//   ... dwFlags validation ...
//   lis  r3,0x8007           <- E_INVALIDARG, within a few instructions
//   ori  r3,r3,0x0057
//   addi r1,r1,FRAME
//   b    <epilogue>
uint32_t FindNuiSkeletonTrackingEnable(Memory* memory,
                                       const NuiCodeRange& scan_range) {
  static const uint8_t kAnchor[8] = {0x3C, 0x60, 0x83, 0x01,
                                     0x60, 0x63, 0x00, 0x05};
  static const uint8_t kInvalidArg[8] = {0x3C, 0x60, 0x80, 0x07,
                                         0x60, 0x63, 0x00, 0x57};
  constexpr uint32_t kMflrR12 = 0x7D8802A6u;
  for (uint32_t page = scan_range.begin; page < scan_range.end;
       page += 0x1000u) {
    if (!NuiScanRangeReadable(memory, page, 0x1000u)) {
      continue;
    }
    const auto* base = memory->TranslateVirtual<const uint8_t*>(page);
    if (!base) {
      continue;
    }
    for (uint32_t off = 0; off + sizeof(kAnchor) <= 0x1000u; off += 4) {
      if (std::memcmp(base + off, kAnchor, sizeof(kAnchor)) != 0) {
        continue;
      }
      const uint32_t anchor = page + off;

      // 0x83010005 must be a frame-unwinding early return.
      uint32_t unwind = 0;
      uint32_t branch = 0;
      if (!NuiReadInstr(memory, anchor + 0x08u, &unwind) ||
          !NuiReadInstr(memory, anchor + 0x0Cu, &branch)) {
        continue;
      }
      if ((unwind >> 16) != 0x3821u) {  // addi r1,r1,imm
        continue;
      }
      const int32_t frame = static_cast<int16_t>(unwind & 0xFFFFu);
      if (frame <= 0 || frame > 0x4000) {
        continue;
      }
      if ((branch >> 26) != 18u || (branch & 1u) != 0u) {  // unconditional b
        continue;
      }

      // Walk back to the matching stwu/mflr prologue (the function entry).
      const uint16_t stwu_imm = static_cast<uint16_t>(-frame);
      uint32_t entry = 0;
      for (uint32_t back = 1; back <= 64 && entry == 0; ++back) {
        uint32_t insn = 0;
        if (!NuiReadInstr(memory, anchor - back * 4u, &insn)) {
          break;
        }
        if ((insn >> 16) != 0x9421u ||
            static_cast<uint16_t>(insn & 0xFFFFu) != stwu_imm) {
          continue;
        }
        const uint32_t stwu_addr = anchor - back * 4u;
        for (uint32_t up = 1; up <= 3; ++up) {
          uint32_t maybe_mflr = 0;
          if (NuiReadInstr(memory, stwu_addr - up * 4u, &maybe_mflr) &&
              maybe_mflr == kMflrR12) {
            entry = stwu_addr - up * 4u;
            break;
          }
        }
        break;
      }
      if (entry == 0) {
        continue;
      }

      // The function must ALSO carry an E_INVALIDARG early return (the dwFlags
      // validation) - this tells TrackingEnable apart from other NUI functions
      // that early-return 0x83010005. The two error returns can appear in either
      // order depending on the compiler, so scan the whole prologue body between
      // the entry and just past the 0x83010005 anchor.
      bool has_invalid_arg = false;
      for (uint32_t addr = entry; addr <= anchor + 0x20u; addr += 4u) {
        if (NuiBytesMatch(memory, addr, kInvalidArg, sizeof(kInvalidArg))) {
          has_invalid_arg = true;
          break;
        }
      }
      if (!has_invalid_arg) {
        continue;
      }

      XELOGI(
          "NUI skeleton injector: NuiSkeletonTrackingEnable found at "
          "{:08X} (frame=0x{:X}, anchor={:08X})",
          entry, frame, anchor);
      return entry;
    }
  }
  return 0;
}

bool NuiLooksLikeFunctionEntry(Memory* memory, uint32_t entry) {
  static const uint8_t kEntry[4] = {0x7D, 0x88, 0x02, 0xA6};
  return NuiBytesMatch(memory, entry, kEntry, sizeof(kEntry));
}

bool NuiFunctionContainsSequence(Memory* memory, uint32_t entry,
                                 const uint32_t* sequence,
                                 size_t sequence_length,
                                 uint32_t maximum_size = 0x500u) {
  constexpr uint32_t kMflrR12 = 0x7D8802A6u;
  if (!memory || !entry || !sequence || !sequence_length) {
    return false;
  }
  for (uint32_t offset = 4; offset < maximum_size; offset += 4) {
    uint32_t word = 0;
    if (!NuiReadInstr(memory, entry + offset, &word)) {
      return false;
    }
    if (word == kMflrR12) {
      return false;
    }
    if (word != sequence[0]) {
      continue;
    }
    bool matches = true;
    for (size_t i = 1; i < sequence_length; ++i) {
      uint32_t next = 0;
      if (!NuiReadInstr(memory,
                        entry + offset + static_cast<uint32_t>(i * 4),
                        &next) ||
          next != sequence[i]) {
        matches = false;
        break;
      }
    }
    if (matches) {
      return true;
    }
  }
  return false;
}

uint32_t FindNuiSkeletonSetTrackedSkeletons(Memory* memory,
                                            const NuiCodeRange& scan_range) {
  static constexpr uint32_t kInvalidArg[] = {0x3C608007u, 0x60630057u};
  static constexpr uint32_t kStreamNotEnabled[] = {0x3C608301u, 0x60630002u};
  static constexpr uint32_t kFeatureNotInitialized[] = {0x3C608301u,
                                                         0x60630005u};
  constexpr uint32_t kMflrR12 = 0x7D8802A6u;
  for (uint32_t page = scan_range.begin; page < scan_range.end;
       page += 0x1000u) {
    if (!NuiScanRangeReadable(memory, page, 0x1000u)) {
      continue;
    }
    const auto* base = memory->TranslateVirtual<const uint8_t*>(page);
    if (!base) {
      continue;
    }
    for (uint32_t offset = 0; offset < 0x1000u; offset += 4) {
      if (xe::load_and_swap<uint32_t>(base + offset) != kMflrR12) {
        continue;
      }
      const uint32_t address = page + offset;
      if (NuiFunctionContainsSequence(memory, address, kInvalidArg,
                                      std::size(kInvalidArg)) &&
          NuiFunctionContainsSequence(memory, address, kStreamNotEnabled,
                                      std::size(kStreamNotEnabled)) &&
          NuiFunctionContainsSequence(memory, address, kFeatureNotInitialized,
                                      std::size(kFeatureNotInitialized))) {
        XELOGI("NUI NuiSkeletonSetTrackedSkeletons found at {:08X}", address);
        return address;
      }
    }
  }
  return 0;
}

uint32_t FindNuiSetFrameEndEvent(Memory* memory,
                                 const NuiCodeRange& scan_range) {
  static constexpr uint32_t kInvalidArg[] = {0x3C608007u, 0x60630057u};
  static constexpr uint32_t kFlagsMask = 0x57CB07FEu;
  constexpr uint32_t kMflrR12 = 0x7D8802A6u;
  constexpr uint32_t kSync = 0x7C0004ACu;
  for (uint32_t page = scan_range.begin; page < scan_range.end;
       page += 0x1000u) {
    if (!NuiScanRangeReadable(memory, page, 0x1000u)) {
      continue;
    }
    const auto* base = memory->TranslateVirtual<const uint8_t*>(page);
    if (!base) {
      continue;
    }
    for (uint32_t page_offset = 0; page_offset < 0x1000u; page_offset += 4) {
      if (xe::load_and_swap<uint32_t>(base + page_offset) != kMflrR12) {
        continue;
      }
      const uint32_t address = page + page_offset;
      if (!NuiFunctionContainsSequence(memory, address, kInvalidArg,
                                       std::size(kInvalidArg), 0x200u) ||
          !NuiFunctionContainsSequence(memory, address, &kFlagsMask, 1,
                                       0x200u)) {
        continue;
      }
      uint32_t sync_count = 0;
      for (uint32_t offset = 4; offset < 0x200u; offset += 4) {
        uint32_t word = 0;
        if (!NuiReadInstr(memory, address + offset, &word) ||
            word == kMflrR12) {
          break;
        }
        sync_count += word == kSync ? 1u : 0u;
      }
      if (sync_count >= 2) {
        XELOGI("NUI NuiSetFrameEndEvent found at {:08X}", address);
        return address;
      }
    }
  }
  return 0;
}

bool FindNuiImageStreamLayout(Memory* memory, uint32_t skeleton_getter,
                              const NuiCodeRange& scan_range,
                              uint32_t* open_out, uint32_t* get_out,
                              uint32_t* release_out) {
  if (!memory || !skeleton_getter || !open_out || !get_out || !release_out) {
    return false;
  }

  struct NuiImageLayout {
    int32_t open_offset;
    int32_t get_offset;
    int32_t release_offset;
  };
  // Known Xbox 360 SDK nuiapi.lib layouts, anchored from the signature-matched
  // NuiSkeletonGetNextFrame entry. Every candidate is validated by the expected
  // PPC function entry before use, so this remains independent of title IDs and
  // absolute addresses.
  static constexpr NuiImageLayout kLayouts[] = {
      {0x00010450, 0x0000F8B0, 0x0000FBD8},
      {-0x00002680, -0x000032C0, -0x00002F98},
      // Additional debug nuiapi.lib layout.
      {0x00006BA0, 0x00005F60, 0x00006288},
  };
  for (const auto& layout : kLayouts) {
    const uint32_t open = skeleton_getter + layout.open_offset;
    const uint32_t get = skeleton_getter + layout.get_offset;
    const uint32_t release = skeleton_getter + layout.release_offset;
    if (!scan_range.Contains(open) || !scan_range.Contains(get) ||
        !scan_range.Contains(release) ||
        !NuiLooksLikeFunctionEntry(memory, open) ||
        !NuiLooksLikeFunctionEntry(memory, get) ||
        !NuiLooksLikeFunctionEntry(memory, release)) {
      continue;
    }
    *open_out = open;
    *get_out = get;
    *release_out = release;
    XELOGI(
        "NUI image stream SDK layout matched: skeleton={:08X} open={:08X} "
        "get={:08X} release={:08X}",
        skeleton_getter, open, get, release);
    return true;
  }
  return false;
}

bool PatchNuiFaultingHookEntry(Memory* memory, uint32_t entry,
                               const char* what) {
  auto* heap = memory->LookupHeap(entry);
  if (!heap) {
    return false;
  }
  const uint32_t page_size = heap->page_size();
  const uint32_t page = entry & ~(page_size - 1);
  uint32_t old_protect = kMemoryProtectRead;
  if (!heap->Protect(page, page_size,
                     kMemoryProtectRead | kMemoryProtectWrite, &old_protect)) {
    XELOGW("NUI {}: cannot make {:08X} writable; skipped", what, entry);
    return false;
  }
  auto* code = memory->TranslateVirtual<uint8_t*>(entry);
  const uint32_t original = xe::load_and_swap<uint32_t>(code);
  if (original != 0x7D8802A6u) {
    heap->Protect(page, page_size, old_protect);
    XELOGW("NUI {}: unexpected first instr {:08X} @ {:08X}; skipped", what,
           original, entry);
    return false;
  }
  xe::store_and_swap<uint32_t>(code, kNuiFaultingLoadWord);
  heap->Protect(page, page_size, old_protect);
  XELOGW("NUI {} fault hook installed @ {:08X}", what, entry);
  return true;
}

uint32_t NuiPpcLis(uint32_t reg, uint32_t value) {
  return 0x3C000000u | ((reg & 31u) << 21) | ((value >> 16) & 0xFFFFu);
}

uint32_t NuiPpcOri(uint32_t ra, uint32_t rs, uint32_t value) {
  return 0x60000000u | ((rs & 31u) << 21) | ((ra & 31u) << 16) |
         (value & 0xFFFFu);
}

bool PatchNuiHookEntry(Memory* memory, uint32_t entry, uint32_t trampoline,
                       const char* what) {
  auto* heap = memory->LookupHeap(entry);
  if (!heap || !trampoline) {
    return false;
  }
  const uint32_t page_size = heap->page_size();
  constexpr uint32_t kPatchSize = 4 * sizeof(uint32_t);
  if ((entry & (page_size - 1)) > page_size - kPatchSize) {
    XELOGW("NUI {}: {:08X} crosses a page; skipped", what, entry);
    return false;
  }
  const uint32_t page = entry & ~(page_size - 1);
  uint32_t old_protect = kMemoryProtectRead;
  if (!heap->Protect(page, page_size,
                     kMemoryProtectRead | kMemoryProtectWrite, &old_protect)) {
    XELOGW("NUI {}: cannot make {:08X} writable; skipped", what, entry);
    return false;
  }

  auto* code = memory->TranslateVirtual<uint8_t*>(entry);
  const uint32_t original = xe::load_and_swap<uint32_t>(code);
  if (original != 0x7D8802A6u) {
    heap->Protect(page, page_size, old_protect);
    XELOGW("NUI {}: unexpected first instr {:08X} @ {:08X}; skipped", what,
           original, entry);
    return false;
  }

  xe::store_and_swap<uint32_t>(code + 0x00, NuiPpcLis(12, trampoline));
  xe::store_and_swap<uint32_t>(code + 0x04, NuiPpcOri(12, 12, trampoline));
  xe::store_and_swap<uint32_t>(code + 0x08, 0x7D8903A6u);  // mtctr r12
  xe::store_and_swap<uint32_t>(code + 0x0C, 0x4E800420u);  // bctr
  heap->Protect(page, page_size, old_protect);
  XELOGW("NUI {} trampoline installed @ {:08X} -> {:08X}", what, entry,
         trampoline);
  return true;
}

uint32_t GenerateNuiStaticTrampoline(
    Emulator* emulator, const char* name,
    cpu::GuestFunction::ExternHandler handler) {
  if (!emulator || !emulator->kernel_state()) {
    return 0;
  }
  auto xam =
      emulator->kernel_state()->GetKernelModule<kernel::xam::XamModule>(
          "xam.xex");
  if (!xam) {
    XELOGW("NUI {}: xam.xex unavailable for trampoline generation", name);
    return 0;
  }
  const uint32_t trampoline =
      xam->GenerateStaticTrampoline(fmt::format("NUI_{}", name), handler);
  if (!trampoline) {
    XELOGW("NUI {}: trampoline generation failed", name);
  }
  return trampoline;
}

bool PatchNuiSdkFunction(Emulator* emulator, Memory* memory, uint32_t entry,
                         const char* name,
                         cpu::GuestFunction::ExternHandler handler,
                         const char* what) {
  const uint32_t trampoline =
      GenerateNuiStaticTrampoline(emulator, name, handler);
  if (trampoline && PatchNuiHookEntry(memory, entry, trampoline, what)) {
    return true;
  }
  return PatchNuiFaultingHookEntry(memory, entry, what);
}

uint32_t NuiDirectBranchTarget(uint32_t address, uint32_t instruction) {
  if ((instruction >> 26) != 18 || !(instruction & 1)) {
    return 0;
  }
  int32_t displacement = static_cast<int32_t>(instruction & 0x03FFFFFC);
  if (displacement & 0x02000000) {
    displacement |= static_cast<int32_t>(0xFC000000);
  }
  return (instruction & 2) ? static_cast<uint32_t>(displacement)
                           : address + static_cast<uint32_t>(displacement);
}

bool PatchNuiCallSite(Memory* memory, uint32_t call_site, uint32_t target,
                      const char* what);

void FindAndPatchNuiD3DTextureRoutes(Memory* memory, uint32_t consumer,
                                     const NuiCodeRange& scan_range) {
  // Match Xbox 360 SDK texture calls prepared from
  // NUI_IMAGE_FRAME::pFrameTexture at offset 0x14. A prepared r5 identifies
  // LockRect; level=0 without r5 identifies UnlockRect.
  constexpr uint32_t kLwzR3FrameTextureMask = 0xFFE0FFFFu;
  constexpr uint32_t kLwzR3FrameTexture = 0x80600014u;
  constexpr uint32_t kLiR4Zero = 0x38800000u;
  constexpr uint32_t kSetR5Mask = 0xFFE00000u;
  constexpr uint32_t kAddiR5 = 0x38A00000u;
  size_t lock_count = 0;
  size_t unlock_count = 0;
  for (uint32_t i = 1; i < 256; ++i) {
    const uint32_t call_address = consumer + i * sizeof(uint32_t);
    if (!scan_range.Contains(call_address)) {
      break;
    }
    const auto* call_p = memory->TranslateVirtual<const uint8_t*>(call_address);
    if (!call_p) {
      break;
    }
    const uint32_t target = NuiDirectBranchTarget(
        call_address, xe::load_and_swap<uint32_t>(call_p));
    if (!target || !NuiScanRangeReadable(memory, target, sizeof(uint32_t))) {
      continue;
    }

    bool loads_frame_texture = false;
    bool sets_level_zero = false;
    bool sets_locked_rect = false;
    for (uint32_t j = 1; j <= 8 && j <= i; ++j) {
      const auto* setup_p = memory->TranslateVirtual<const uint8_t*>(
          call_address - j * sizeof(uint32_t));
      if (!setup_p) {
        break;
      }
      const uint32_t setup = xe::load_and_swap<uint32_t>(setup_p);
      loads_frame_texture |=
          (setup & kLwzR3FrameTextureMask) == kLwzR3FrameTexture;
      sets_level_zero |= setup == kLiR4Zero;
      sets_locked_rect |= (setup & kSetR5Mask) == kAddiR5;
    }
    if (!loads_frame_texture || !sets_level_zero) {
      continue;
    }

    auto& calls = sets_locked_rect ? g_nui_d3d_texture_lock_calls
                                   : g_nui_d3d_texture_unlock_calls;
    auto& targets = sets_locked_rect ? g_nui_d3d_texture_lock_targets
                                     : g_nui_d3d_texture_unlock_targets;
    size_t& count = sets_locked_rect ? lock_count : unlock_count;
    size_t route_index = kNuiTextureRouteCapacity;
    for (size_t route = 0; route < calls.size(); ++route) {
      const uint32_t existing = calls[route].load(std::memory_order_relaxed);
      if (existing == call_address) {
        route_index = kNuiTextureRouteCapacity;
        break;
      }
      if (!existing && route_index == kNuiTextureRouteCapacity) {
        route_index = route;
      }
    }
    if (route_index == kNuiTextureRouteCapacity) {
      continue;
    }
    const char* what = sets_locked_rect ? "static D3DTexture_LockRect call"
                                        : "static D3DTexture_UnlockRect call";
    if (PatchNuiCallSite(memory, call_address, target, what)) {
      targets[route_index].store(target, std::memory_order_relaxed);
      calls[route_index].store(call_address, std::memory_order_relaxed);
      ++count;
    }
  }
  XELOGI("NUI texture routes installed: lock={} unlock={}", lock_count,
         unlock_count);
}

void FindAndPatchStaticNuiD3DTextureRoutes(Memory* memory, uint32_t image_getter,
                                           const NuiCodeRange& scan_range) {
  if (!memory || !image_getter || !scan_range.valid()) {
    return;
  }
  std::array<uint32_t, kNuiTextureRouteCapacity> consumers{};
  size_t consumer_count = 0;
  for (uint32_t page = scan_range.begin; page < scan_range.end;
       page += 0x1000u) {
    if (!NuiScanRangeReadable(memory, page, 0x1000u)) {
      continue;
    }
    for (uint32_t offset = 0; offset < 0x1000u; offset += 4) {
      const uint32_t call_address = page + offset;
      uint32_t instruction = 0;
      if (!NuiReadInstr(memory, call_address, &instruction) ||
          NuiDirectBranchTarget(call_address, instruction) != image_getter) {
        continue;
      }
      for (uint32_t next = 1; next <= 32; ++next) {
        const uint32_t next_address = call_address + next * sizeof(uint32_t);
        if (!scan_range.Contains(next_address)) {
          break;
        }
        uint32_t next_instruction = 0;
        if (!NuiReadInstr(memory, next_address, &next_instruction)) {
          break;
        }
        const uint32_t consumer =
            NuiDirectBranchTarget(next_address, next_instruction);
        if (!consumer || !scan_range.Contains(consumer)) {
          continue;
        }
        if (std::find(consumers.begin(), consumers.begin() + consumer_count,
                      consumer) == consumers.begin() + consumer_count &&
            consumer_count < consumers.size()) {
          consumers[consumer_count++] = consumer;
          FindAndPatchNuiD3DTextureRoutes(memory, consumer, scan_range);
        }
        break;
      }
    }
  }
  XELOGI("NUI static image consumers discovered: {}", consumer_count);
}

bool FindNuiTextureRoute(
    uint32_t guest_pc,
    const std::array<std::atomic<uint32_t>, kNuiTextureRouteCapacity>& calls,
    const std::array<std::atomic<uint32_t>, kNuiTextureRouteCapacity>& targets,
    uint32_t* target_out) {
  for (size_t i = 0; i < calls.size(); ++i) {
    if (calls[i].load(std::memory_order_relaxed) == guest_pc) {
      if (target_out) {
        *target_out = targets[i].load(std::memory_order_relaxed);
      }
      return true;
    }
  }
  return false;
}

bool PatchNuiCallSite(Memory* memory, uint32_t call_site, uint32_t target,
                      const char* what) {
  auto* heap = memory->LookupHeap(call_site);
  if (!heap || !target) {
    return false;
  }
  const uint32_t page_size = heap->page_size();
  const uint32_t page = call_site & ~(page_size - 1);
  if (!heap->Protect(page, page_size,
                     kMemoryProtectRead | kMemoryProtectWrite)) {
    XELOGW("NUI {}: cannot make {:08X} writable; skipped", what, call_site);
    return false;
  }
  auto* code = memory->TranslateVirtual<uint8_t*>(call_site);
  const uint32_t original = xe::load_and_swap<uint32_t>(code);
  if (NuiDirectBranchTarget(call_site, original) != target) {
    heap->Protect(page, page_size, kMemoryProtectRead);
    XELOGW("NUI {}: unexpected call {:08X} @ {:08X}; skipped", what, original,
           call_site);
    return false;
  }
  xe::store_and_swap<uint32_t>(code, kNuiFaultingLoadWord);
  heap->Protect(page, page_size, kMemoryProtectRead);
  XELOGW("NUI {} installed @ {:08X} -> {:08X}", what, call_site, target);
  return true;
}

void ResetNuiStaticInjectorState() {
  g_nui_skeleton_get_next_frame_hook.store(0, std::memory_order_relaxed);
  g_nui_skeleton_tracking_enable_hook.store(0, std::memory_order_relaxed);
  g_nui_skeleton_set_tracked_hook.store(0, std::memory_order_relaxed);
  g_nui_set_frame_end_event_hook.store(0, std::memory_order_relaxed);
  g_nui_image_stream_open_hook.store(0, std::memory_order_relaxed);
  g_nui_image_stream_get_next_frame_hook.store(0, std::memory_order_relaxed);
  g_nui_image_stream_release_frame_hook.store(0, std::memory_order_relaxed);
  g_nui_next_stream_slot.store(0, std::memory_order_relaxed);
  kernel::xam::ResetNuiGuestEventRegistrations();
  for (auto& handle : g_nui_stream_handles) {
    handle.store(0, std::memory_order_relaxed);
  }
  for (auto& frame_number : g_nui_stream_last_frame_numbers) {
    frame_number.store(0, std::memory_order_relaxed);
  }
  for (auto& call : g_nui_d3d_texture_lock_calls) {
    call.store(0, std::memory_order_relaxed);
  }
  for (auto& target : g_nui_d3d_texture_lock_targets) {
    target.store(0, std::memory_order_relaxed);
  }
  for (auto& call : g_nui_d3d_texture_unlock_calls) {
    call.store(0, std::memory_order_relaxed);
  }
  for (auto& target : g_nui_d3d_texture_unlock_targets) {
    target.store(0, std::memory_order_relaxed);
  }
}

void InstallNuiStaticSdkHooks(Emulator* emulator) {
  Memory* memory = emulator ? emulator->memory() : nullptr;
  if (!memory) {
    return;
  }
  ResetNuiStaticInjectorState();

  const NuiCodeRange scan_range = MakeNuiCodeRange(0x82000000u, 0x83000000u);
  XELOGI("NUI static SDK scan: code={:08X}-{:08X}", scan_range.begin,
         scan_range.end);

  const uint32_t getter = FindNuiSkeletonGetNextFrame(memory, scan_range);
  if (!getter) {
    XELOGW("NUI skeleton injector: NuiSkeletonGetNextFrame signature not found");
    return;
  }
  if (PatchNuiSdkFunction(emulator, memory, getter, "NuiSkeletonGetNextFrame",
                          NuiSkeletonGetNextFrameTrampoline,
                          "skeleton injector")) {
    g_nui_skeleton_get_next_frame_hook.store(getter,
                                             std::memory_order_relaxed);
  }

  uint32_t image_open = 0;
  uint32_t image_get = 0;
  uint32_t image_release = 0;
  if (FindNuiImageStreamLayout(memory, getter, scan_range, &image_open,
                               &image_get, &image_release)) {
    if (PatchNuiSdkFunction(emulator, memory, image_open,
                            "NuiImageStreamOpen",
                            NuiImageStreamOpenTrampoline,
                            "image stream open (NuiImageStreamOpen)")) {
      g_nui_image_stream_open_hook.store(image_open,
                                         std::memory_order_relaxed);
    }
    if (PatchNuiSdkFunction(
            emulator, memory, image_get, "NuiImageStreamGetNextFrame",
            NuiImageStreamGetNextFrameTrampoline,
            "image stream getter (NuiImageStreamGetNextFrame)")) {
      g_nui_image_stream_get_next_frame_hook.store(
          image_get, std::memory_order_relaxed);
      FindAndPatchStaticNuiD3DTextureRoutes(memory, image_get, scan_range);
    }
    if (PatchNuiSdkFunction(
            emulator, memory, image_release, "NuiImageStreamReleaseFrame",
            NuiImageStreamReleaseFrameTrampoline,
            "image stream release (NuiImageStreamReleaseFrame)")) {
      g_nui_image_stream_release_frame_hook.store(
          image_release, std::memory_order_relaxed);
    }
  } else {
    XELOGW("NUI image stream injector: SDK layout not found");
  }

  const uint32_t tracking_enable =
      FindNuiSkeletonTrackingEnable(memory, scan_range);
  if (!tracking_enable) {
    XELOGW(
        "NUI skeleton injector: NuiSkeletonTrackingEnable signature not found "
        "(event-driven titles may fall back to the getter's wait timeout)");
  } else if (PatchNuiSdkFunction(emulator, memory, tracking_enable,
                                 "NuiSkeletonTrackingEnable",
                                 NuiSkeletonTrackingEnableTrampoline,
                                 "skeleton event capture")) {
    g_nui_skeleton_tracking_enable_hook.store(tracking_enable,
                                              std::memory_order_relaxed);
  }

  const uint32_t set_tracked =
      FindNuiSkeletonSetTrackedSkeletons(memory, scan_range);
  if (!set_tracked) {
    XELOGW("NUI NuiSkeletonSetTrackedSkeletons signature not found");
  } else if (PatchNuiSdkFunction(emulator, memory, set_tracked,
                                 "NuiSkeletonSetTrackedSkeletons",
                                 NuiSkeletonSetTrackedSkeletonsTrampoline,
                                 "title skeleton selection")) {
    g_nui_skeleton_set_tracked_hook.store(set_tracked,
                                          std::memory_order_relaxed);
  }

  const uint32_t frame_end_event =
      FindNuiSetFrameEndEvent(memory, scan_range);
  if (!frame_end_event) {
    XELOGW("NUI NuiSetFrameEndEvent signature not found");
  } else if (PatchNuiSdkFunction(emulator, memory, frame_end_event,
                                 "NuiSetFrameEndEvent",
                                 NuiSetFrameEndEventTrampoline,
                                 "frame-end event capture")) {
    g_nui_set_frame_end_event_hook.store(frame_end_event,
                                         std::memory_order_relaxed);
  }
}

bool NuiIsColorImageType(uint32_t image_type) {
  return image_type == kNuiImageTypeColor ||
         image_type == kNuiImageTypeColorYuv ||
         image_type == kNuiImageTypeColorInDepthSpace;
}

bool NuiIsSupportedImageStream(uint32_t image_type, uint32_t resolution) {
  switch (image_type) {
    case kNuiImageTypeDepthAndPlayerIndex:
    case kNuiImageTypeDepth:
    case kNuiImageTypeDepthAndPlayerIndexInColorSpace:
    case kNuiImageTypeDepthInColorSpace:
      return resolution == kNuiImageResolution320x240;
    case kNuiImageTypeColor:
    case kNuiImageTypeColorYuv:
    case kNuiImageTypeColorInDepthSpace:
      return resolution == kNuiImageResolution640x480;
    case kNuiImageTypeDepthAndPlayerIndex80x60:
    case kNuiImageTypeDepth80x60:
      return resolution == kNuiImageResolution80x60;
    default:
      return false;
  }
}

uint32_t NuiRegisterEmulatedImageStream(uint32_t image_type,
                                        uint32_t resolution,
                                        uint32_t frame_flags,
                                        uint32_t frame_limit,
                                        uint32_t event_handle) {
  const uint32_t sequence =
      g_nui_next_stream_slot.fetch_add(1, std::memory_order_relaxed);
  const uint32_t slot = sequence % kNuiEmulatedStreamCapacity;
  const uint32_t generation = (sequence % 0xFFFFu) + 1;
  const uint32_t handle = kNuiEmulatedStreamHandleBase | generation;
  g_nui_stream_image_types[slot].store(image_type, std::memory_order_relaxed);
  g_nui_stream_resolutions[slot].store(resolution, std::memory_order_relaxed);
  g_nui_stream_frame_flags[slot].store(frame_flags, std::memory_order_relaxed);
  g_nui_stream_frame_limits[slot].store(frame_limit, std::memory_order_relaxed);
  g_nui_stream_events[slot].store(event_handle, std::memory_order_relaxed);
  g_nui_stream_last_frame_numbers[slot].store(0, std::memory_order_relaxed);
  g_nui_stream_handles[slot].store(handle, std::memory_order_release);
  return handle;
}

bool NuiLookupEmulatedImageStream(uint32_t handle, uint32_t* image_type,
                                  uint32_t* resolution,
                                  uint32_t* event_handle) {
  if ((handle & 0xFFFF0000u) != kNuiEmulatedStreamHandleBase) {
    return false;
  }
  for (uint32_t i = 0; i < kNuiEmulatedStreamCapacity; ++i) {
    if (g_nui_stream_handles[i].load(std::memory_order_acquire) != handle) {
      continue;
    }
    if (image_type) {
      *image_type =
          g_nui_stream_image_types[i].load(std::memory_order_relaxed);
    }
    if (resolution) {
      *resolution =
          g_nui_stream_resolutions[i].load(std::memory_order_relaxed);
    }
    if (event_handle) {
      *event_handle = g_nui_stream_events[i].load(std::memory_order_relaxed);
    }
    return true;
  }
  return false;
}

bool NuiConsumeEmulatedImageFrame(uint32_t handle, uint32_t frame_number) {
  if (!frame_number) {
    return false;
  }
  for (uint32_t i = 0; i < kNuiEmulatedStreamCapacity; ++i) {
    if (g_nui_stream_handles[i].load(std::memory_order_acquire) != handle) {
      continue;
    }
    return g_nui_stream_last_frame_numbers[i].exchange(
               frame_number, std::memory_order_acq_rel) != frame_number;
  }
  return false;
}

bool NuiWaitForEmulatedImageFrame(uint32_t handle, uint32_t image_type,
                                  uint32_t wait_ms) {
  const auto try_consume = [handle, image_type]() {
    return NuiConsumeEmulatedImageFrame(
        handle, kernel::xam::GetNuiImageFrameNumberForType(image_type));
  };
  if (try_consume()) {
    return true;
  }
  if (wait_ms == 0) {
    return false;
  }

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
  do {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (try_consume()) {
      return true;
    }
  } while (std::chrono::steady_clock::now() < deadline);
  return false;
}

Memory* NuiKernelMemory(kernel::KernelState* kernel_state) {
  return kernel_state ? kernel_state->memory() : nullptr;
}

void NuiBackoffHotSkeletonNoDataPoll(cpu::ppc::PPCContext* context,
                                     uint32_t wait_ms, uint32_t frame_ptr,
                                     uint32_t result) {
  thread_local uint32_t last_lr = 0;
  thread_local uint32_t last_frame_ptr = 0;
  thread_local uint32_t no_data_streak = 0;

  if (!context || wait_ms != 0 || result != kNuiFrameNoData) {
    last_lr = 0;
    last_frame_ptr = 0;
    no_data_streak = 0;
    return;
  }

  const uint32_t lr = static_cast<uint32_t>(context->lr);
  if (lr != last_lr || frame_ptr != last_frame_ptr) {
    last_lr = lr;
    last_frame_ptr = frame_ptr;
    no_data_streak = 1;
    return;
  }

  ++no_data_streak;
  if (no_data_streak >= 1024) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

uint32_t HandleNuiSkeletonGetNextFrame(cpu::ppc::PPCContext* context,
                                       Memory* memory) {
  if (!context || !memory) {
    return 0x80070057u;
  }
  const uint32_t wait_ms = static_cast<uint32_t>(context->r[3]);
  const uint32_t frame_ptr = static_cast<uint32_t>(context->r[4]);
  const bool valid_arguments =
      wait_ms <= kNuiMaximumWaitMilliseconds &&
      NuiScanRangeWritable(memory, frame_ptr, kNuiSkeletonFrameSize);
  const bool injected =
      valid_arguments &&
      kernel::xam::WriteNuiSkeletonFrameToGuest(frame_ptr, wait_ms);

  static std::atomic<uint32_t> getter_calls{0};
  const uint32_t n = getter_calls.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n <= 16 || (n & (n - 1)) == 0) {
    XELOGW(
        "NUI skeleton getter #{} @ {:08X}: wait={} frame={:08X} "
        "injected={} state={} tracked={} engaged_id={} lr={:08X}",
        n, g_nui_skeleton_get_next_frame_hook.load(std::memory_order_relaxed),
        wait_ms, frame_ptr, injected ? 1 : 0,
        kernel::xam::GetNuiInjectionState(),
        kernel::xam::GetNuiTrackedSkeletonCount(),
        kernel::xam::GetEngagedNuiTrackingId(),
        static_cast<uint32_t>(context->lr));
  }
  const uint32_t result =
      valid_arguments ? (injected ? 0u : kNuiFrameNoData) : 0x80070057u;
  if (injected) {
    kernel::xam::ResetNuiSkeletonFrameEvent();
  }
  NuiBackoffHotSkeletonNoDataPoll(context, wait_ms, frame_ptr, result);
  return result;
}

uint32_t HandleNuiSkeletonTrackingEnable(cpu::ppc::PPCContext* context) {
  if (!context) {
    return 0x80070057u;
  }
  const uint32_t event_handle = static_cast<uint32_t>(context->r[3]);
  const uint32_t flags = static_cast<uint32_t>(context->r[4]);
  const uint32_t result =
      kernel::xam::ConfigureNuiSkeletonTracking(event_handle, flags);
  static std::atomic<uint32_t> enable_calls{0};
  const uint32_t n = enable_calls.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n <= 8 || (n & (n - 1)) == 0) {
    XELOGW(
        "NUI skeleton tracking enable #{} @ {:08X}: hEvent={:08X} "
        "flags={:08X}",
        n, g_nui_skeleton_tracking_enable_hook.load(std::memory_order_relaxed),
        event_handle, flags);
  }
  return result;
}

uint32_t HandleNuiSkeletonSetTrackedSkeletons(cpu::ppc::PPCContext* context) {
  if (!context) {
    return 0x80070057u;
  }
  const uint32_t tracking_ids_ptr = static_cast<uint32_t>(context->r[3]);
  return kernel::xam::ConfigureNuiTrackedSkeletons(tracking_ids_ptr);
}

uint32_t HandleNuiSetFrameEndEvent(cpu::ppc::PPCContext* context) {
  if (!context) {
    return 0x80070057u;
  }
  const uint32_t event_handle = static_cast<uint32_t>(context->r[3]);
  const uint32_t flags = static_cast<uint32_t>(context->r[4]);
  return kernel::xam::ConfigureNuiFrameEndEvent(event_handle, flags);
}

uint32_t HandleNuiImageStreamOpen(cpu::ppc::PPCContext* context,
                                  Memory* memory) {
  if (!context || !memory) {
    return 0x80070057u;
  }
  const uint32_t image_type = static_cast<uint32_t>(context->r[3]);
  const uint32_t resolution = static_cast<uint32_t>(context->r[4]);
  const uint32_t frame_flags = static_cast<uint32_t>(context->r[5]);
  const uint32_t frame_limit = static_cast<uint32_t>(context->r[6]);
  const uint32_t event_handle = static_cast<uint32_t>(context->r[7]);
  const uint32_t stream_out = static_cast<uint32_t>(context->r[8]);
  const bool valid_arguments =
      NuiIsSupportedImageStream(image_type, resolution) && frame_limit &&
      frame_limit <= kNuiImageStreamFrameLimitMaximum &&
      !(frame_flags & ~kNuiImageStreamFlagSuppressNoFrameData) &&
      NuiScanRangeWritable(memory, stream_out, sizeof(uint32_t));

  static std::atomic<uint32_t> open_calls{0};
  const uint32_t n = open_calls.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n <= 16 || (n & (n - 1)) == 0 || valid_arguments) {
    XELOGW(
        "NUI image open #{} @ {:08X}: type={} res={} flags={:08X} "
        "limit={} event={:08X} out={:08X} emulated={}",
        n, g_nui_image_stream_open_hook.load(std::memory_order_relaxed),
        image_type, resolution, frame_flags, frame_limit, event_handle,
        stream_out, valid_arguments ? 1 : 0);
  }

  if (!valid_arguments) {
    return 0x80070057u;
  }
  auto* p = memory->TranslateVirtual<uint8_t*>(stream_out);
  if (!p) {
    return 0x80070057u;
  }
  const uint32_t stream_handle = NuiRegisterEmulatedImageStream(
      image_type, resolution, frame_flags, frame_limit, event_handle);
  xe::store_and_swap<uint32_t>(p, stream_handle);
  if (NuiIsColorImageType(image_type)) {
    kernel::xam::SetNuiColorFrameEvent(event_handle);
  } else {
    kernel::xam::SetNuiImageFrameEvent(event_handle);
  }
  return 0;
}

uint32_t HandleNuiImageStreamGetNextFrame(cpu::ppc::PPCContext* context,
                                          Memory* memory) {
  if (!context || !memory) {
    return 0x80070057u;
  }
  const uint32_t stream = static_cast<uint32_t>(context->r[3]);
  const uint32_t wait_ms = static_cast<uint32_t>(context->r[4]);
  const uint32_t frame_out = static_cast<uint32_t>(context->r[5]);
  uint32_t image_type = 0;
  uint32_t resolution = 0;
  uint32_t event_handle = 0;
  const bool is_emulated_stream =
      NuiLookupEmulatedImageStream(stream, &image_type, &resolution,
                                   &event_handle);

  uint32_t node = 0;
  uint32_t result = 0x80070057u;
  if (is_emulated_stream &&
      NuiScanRangeWritable(memory, frame_out, sizeof(uint32_t)) &&
      wait_ms <= kNuiMaximumWaitMilliseconds) {
    kernel::xam::ResetNuiFrameEvent(event_handle);
    node = NuiWaitForEmulatedImageFrame(stream, image_type, wait_ms)
               ? kernel::xam::GetNuiImageFrameNodeGuestAddress(image_type,
                                                               resolution)
               : 0;
    if (auto* p = memory->TranslateVirtual<uint8_t*>(frame_out)) {
      xe::store_and_swap<uint32_t>(p, node);
      result = node ? 0u : kNuiFrameNoData;
    }
  }

  static std::atomic<uint32_t> get_calls{0};
  const uint32_t n = get_calls.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n <= 16 || (n & (n - 1)) == 0) {
    XELOGW(
        "NUI image getter #{} @ {:08X}: stream={:08X} wait={} "
        "type={} res={} event={:08X} ppFrame={:08X} node={:08X} "
        "result={:08X} lr={:08X}",
        n,
        g_nui_image_stream_get_next_frame_hook.load(std::memory_order_relaxed),
        stream, wait_ms, image_type, resolution, event_handle, frame_out, node,
        result, static_cast<uint32_t>(context->lr));
  }
  return result;
}

uint32_t HandleNuiImageStreamReleaseFrame(cpu::ppc::PPCContext* context) {
  if (!context) {
    return 0x80070057u;
  }
  const uint32_t stream = static_cast<uint32_t>(context->r[3]);
  const uint32_t frame = static_cast<uint32_t>(context->r[4]);
  uint32_t image_type = 0;
  uint32_t resolution = 0;
  const bool is_emulated_stream =
      NuiLookupEmulatedImageStream(stream, &image_type, &resolution, nullptr);
  if (!is_emulated_stream ||
      !kernel::xam::IsNuiImageFrameNodeGuestAddressForType(frame,
                                                           image_type)) {
    return 0x80070057u;
  }

  static std::atomic<uint32_t> release_calls{0};
  const uint32_t n = release_calls.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n <= 16 || (n & (n - 1)) == 0) {
    XELOGW(
        "NUI image release #{} @ {:08X}: stream={:08X} frame={:08X} "
        "type={} res={} skipped lr={:08X}",
        n, g_nui_image_stream_release_frame_hook.load(std::memory_order_relaxed),
        stream, frame, image_type, resolution, static_cast<uint32_t>(context->lr));
  }
  return 0;
}

void NuiSkeletonGetNextFrameTrampoline(cpu::ppc::PPCContext* context,
                                       kernel::KernelState* kernel_state) {
  context->r[3] = HandleNuiSkeletonGetNextFrame(
      context, NuiKernelMemory(kernel_state));
}

void NuiSkeletonTrackingEnableTrampoline(cpu::ppc::PPCContext* context,
                                         kernel::KernelState*) {
  context->r[3] = HandleNuiSkeletonTrackingEnable(context);
}

void NuiSkeletonSetTrackedSkeletonsTrampoline(
    cpu::ppc::PPCContext* context, kernel::KernelState*) {
  context->r[3] = HandleNuiSkeletonSetTrackedSkeletons(context);
}

void NuiSetFrameEndEventTrampoline(cpu::ppc::PPCContext* context,
                                   kernel::KernelState*) {
  context->r[3] = HandleNuiSetFrameEndEvent(context);
}

void NuiImageStreamOpenTrampoline(cpu::ppc::PPCContext* context,
                                  kernel::KernelState* kernel_state) {
  context->r[3] =
      HandleNuiImageStreamOpen(context, NuiKernelMemory(kernel_state));
}

void NuiImageStreamGetNextFrameTrampoline(cpu::ppc::PPCContext* context,
                                          kernel::KernelState* kernel_state) {
  context->r[3] = HandleNuiImageStreamGetNextFrame(
      context, NuiKernelMemory(kernel_state));
}

void NuiImageStreamReleaseFrameTrampoline(
    cpu::ppc::PPCContext* context, kernel::KernelState*) {
  context->r[3] = HandleNuiImageStreamReleaseFrame(context);
}

}  // namespace

void Emulator::InstallNuiSkeletonInjector() {
  InstallNuiStaticSdkHooks(this);
}

Emulator::GameConfigLoadCallback::GameConfigLoadCallback(Emulator& emulator)
    : emulator_(emulator) {
  emulator_.AddGameConfigLoadCallback(this);
}

Emulator::GameConfigLoadCallback::~GameConfigLoadCallback() {
  emulator_.RemoveGameConfigLoadCallback(this);
}

Emulator::Emulator(const std::filesystem::path& command_line,
                   const std::filesystem::path& storage_root,
                   const std::filesystem::path& content_root,
                   const std::filesystem::path& cache_root)
    : on_launch(),
      on_terminate(),
      on_exit(),
      command_line_(command_line),
      storage_root_(storage_root),
      content_root_(content_root),
      cache_root_(cache_root),
      title_name_(),
      title_version_(),
      display_window_(nullptr),
      memory_(),
      audio_system_(),
      audio_media_player_(),
      graphics_system_(),
      input_system_(),
      export_resolver_(),
      file_system_(),
      kernel_state_(),
      main_thread_(),
      title_id_(std::nullopt),
      game_info_database_(),
      paused_(false),
      restoring_(false),
      restore_fence_() {
  if (cvars::priority_class != 0) {
    if (SetProcessPriorityClass(cvars::priority_class)) {
      XELOGI("Higher priority class request: Successful. New priority: {}",
             cvars::priority_class);
    }
  }

#if XE_PLATFORM_WIN32 == 1
  // Show a disclaimer that links to the quickstart
  // guide the first time they ever open the emulator
  uint64_t persistent_flags = GetPersistentEmulatorFlags();
  if (!(persistent_flags & EmulatorFlagDisclaimerAcknowledged)) {
    if ((MessageBoxW(
             nullptr,
             L"DISCLAIMER: Xenia is not for enabling illegal activity, and "
             "support is unavailable for illegally obtained software.\n\n"
             "Please respect this policy as no further reminders will be "
             "given.\n\nThe quickstart guide explains how to use digital or "
             "physical games from your Xbox 360 console.\n\nWould you like "
             "to open it?",
             L"Xenia", MB_YESNO | MB_ICONQUESTION) == IDYES)) {
      LaunchWebBrowser(
          "https://github.com/xenia-canary/xenia-canary/wiki/"
          "Quickstart#how-to-rip-games");
    }
    SetPersistentEmulatorFlags(persistent_flags |
                               EmulatorFlagDisclaimerAcknowledged);
  }
#endif
}

Emulator::~Emulator() {
  // Note that we delete things in the reverse order they were initialized.

  // Give the systems time to shutdown before we delete them.
  if (graphics_system_) {
    graphics_system_->Shutdown();
  }
  if (audio_system_) {
    audio_system_->Shutdown();
  }

  input_system_.reset();
  graphics_system_.reset();
  audio_system_.reset();
  audio_media_player_.reset();

  kernel_state_.reset();
  file_system_.reset();

  processor_.reset();

  export_resolver_.reset();

  ExceptionHandler::Uninstall(Emulator::ExceptionCallbackThunk, this);
}

X_STATUS Emulator::Setup(
    ui::Window* display_window, ui::ImGuiDrawer* imgui_drawer,
    bool require_cpu_backend,
    std::function<std::unique_ptr<apu::AudioSystem>(cpu::Processor*)>
        audio_system_factory,
    std::function<std::unique_ptr<gpu::GraphicsSystem>()>
        graphics_system_factory,
    std::function<std::vector<std::unique_ptr<hid::InputDriver>>(ui::Window*)>
        input_driver_factory) {
  X_STATUS result = X_STATUS_UNSUCCESSFUL;

  display_window_ = display_window;
  imgui_drawer_ = imgui_drawer;

  // Initialize clock.
  // 360 uses a 50MHz clock.
  Clock::set_guest_tick_frequency(50000000);
  // We could reset this with save state data/constant value to help replays.
  Clock::set_guest_system_time_base(Clock::QueryHostSystemTime());
  // This can be adjusted dynamically, as well.
  Clock::set_guest_time_scalar(cvars::time_scalar);

  // Before we can set thread affinity we must enable the process to use all
  // logical processors.
  xe::threading::EnableAffinityConfiguration();

  XELOGI("{}: Initializing Memory...", __func__);
  // Create memory system first, as it is required for other systems.
  memory_ = std::make_unique<Memory>();
  if (!memory_->Initialize()) {
    XELOGE("{}: Cannot initalize memory!", __func__);
    return result;
  }

  XELOGI("{}: Initializing Exports...", __func__);
  // Shared export resolver used to attach and query for HLE exports.
  export_resolver_ = std::make_unique<xe::cpu::ExportResolver>();

  std::unique_ptr<xe::cpu::backend::Backend> backend;
#if XE_ARCH_AMD64
  if (cvars::cpu == "x64") {
    backend.reset(new xe::cpu::backend::x64::X64Backend());
  }
#elif XE_ARCH_ARM64
  if (cvars::cpu == "a64") {
    backend.reset(new xe::cpu::backend::a64::A64Backend());
  }
#endif  // XE_ARCH
  if (cvars::cpu == "any") {
    if (!backend) {
#if XE_ARCH_AMD64
      backend.reset(new xe::cpu::backend::x64::X64Backend());
#elif XE_ARCH_ARM64
      backend.reset(new xe::cpu::backend::a64::A64Backend());
#endif  // XE_ARCH
    }
  }
  if (!backend && !require_cpu_backend) {
    backend.reset(new xe::cpu::backend::NullBackend());
  }

  XELOGI("{}: Initializing Processor...", __func__);
  // Initialize the CPU.
  processor_ = std::make_unique<xe::cpu::Processor>(memory_.get(),
                                                    export_resolver_.get());
  if (!processor_->Setup(std::move(backend))) {
    XELOGE("{}: Cannot initalize processor!", __func__);
    return X_STATUS_UNSUCCESSFUL;
  }

  XELOGI("{}: Initializing Audio...", __func__);
  // Initialize the APU.
  if (audio_system_factory) {
    audio_system_ = audio_system_factory(processor_.get());
    if (!audio_system_) {
      XELOGE("{}: Cannot initalize audio_system!", __func__);
      return X_STATUS_NOT_IMPLEMENTED;
    }
  }

  XELOGI("{}: Initializing Graphics...", __func__);
  // Initialize the GPU.
  graphics_system_ = graphics_system_factory();
  if (!graphics_system_) {
    XELOGE("{}: Cannot initalize graphics_system!", __func__);
    return X_STATUS_NOT_IMPLEMENTED;
  }

  XELOGI("{}: Initializing HID...", __func__);
  // Initialize the HID.
  input_system_ = std::make_unique<xe::hid::InputSystem>(display_window_);
  if (!input_system_) {
    XELOGE("{}: Cannot initalize input_system!", __func__);
    return X_STATUS_NOT_IMPLEMENTED;
  }
  if (input_driver_factory) {
    auto input_drivers = input_driver_factory(display_window_);
    for (size_t i = 0; i < input_drivers.size(); ++i) {
      input_system_->AddDriver(std::move(input_drivers[i]));
    }
  }

  result = input_system_->Setup();
  if (result) {
    return result;
  }

  // Add inputSystem to UI
  imgui_drawer_->LoadInputSystem(input_system_.get());

  XELOGI("{}: Initializing VFS...", __func__);
  // Bring up the virtual filesystem used by the kernel.
  file_system_ = std::make_unique<xe::vfs::VirtualFileSystem>();

  patcher_ = std::make_unique<xe::patcher::Patcher>(storage_root_);

  XELOGI("{}: Initializing Kernel...", __func__);
  // Shared kernel state.
  kernel_state_ = std::make_unique<xe::kernel::KernelState>(this);
#define LOAD_KERNEL_MODULE(t) \
  static_cast<void>(kernel_state_->LoadKernelModule<kernel::t>())
  // HLE kernel modules.
  LOAD_KERNEL_MODULE(xboxkrnl::XboxkrnlModule);
  LOAD_KERNEL_MODULE(xam::XamModule);
  LOAD_KERNEL_MODULE(xbdm::XbdmModule);
#undef LOAD_KERNEL_MODULE
  plugin_loader_ = std::make_unique<xe::patcher::PluginLoader>(
      kernel_state_.get(), storage_root() / "plugins");

  XELOGI("{}: Starting graphics_system...", __func__);
  // Setup the core components.
  result = graphics_system_->Setup(
      processor_.get(), kernel_state_.get(),
      display_window_ ? &display_window_->app_context() : nullptr,
      display_window_ != nullptr);
  if (result) {
    XELOGE("{}: Failed to setup graphics_system!", __func__);
    return result;
  }

  if (audio_system_) {
    XELOGI("{}: Starting audio_system...", __func__);
    result = audio_system_->Setup(kernel_state_.get());
    if (result) {
      XELOGE("{}: Failed to setup audio_system!", __func__);
      return result;
    }
    audio_media_player_ = std::make_unique<apu::AudioMediaPlayer>(
        audio_system_.get(), kernel_state_.get());
    audio_media_player_->Setup();
  }

  // Initialize emulator fallback exception handling last.
  ExceptionHandler::Install(Emulator::ExceptionCallbackThunk, this);

  return result;
}

X_STATUS Emulator::TerminateTitle() {
  if (!is_title_open()) {
    return X_STATUS_UNSUCCESSFUL;
  }

  kernel_state_->TerminateTitle();
  title_id_ = std::nullopt;
  title_name_ = "";
  title_version_ = "";
  on_terminate();
  return X_STATUS_SUCCESS;
}

const std::unique_ptr<vfs::Device> Emulator::CreateVfsDevice(
    const std::filesystem::path& path, const std::string_view mount_path) {
  // Must check if the type has changed e.g. XamSwapDisc
  switch (GetFileSignature(path)) {
    case FileSignatureType::XEX1:
    case FileSignatureType::XEX2:
    case FileSignatureType::ELF: {
      auto parent_path = path.parent_path();
      return std::make_unique<vfs::HostPathDevice>(
          mount_path, parent_path, !cvars::allow_game_relative_writes);
    } break;
    case FileSignatureType::LIVE:
    case FileSignatureType::CON:
    case FileSignatureType::PIRS: {
      return vfs::XContentContainerDevice::CreateContentDevice(mount_path,
                                                               path);
    } break;
    case FileSignatureType::XISO: {
      return std::make_unique<vfs::DiscImageDevice>(mount_path, path);
    } break;
    case FileSignatureType::ZAR: {
      return std::make_unique<vfs::DiscZarchiveDevice>(mount_path, path);
    } break;
    case FileSignatureType::EXE:
    case FileSignatureType::Unknown:
    default:
      return nullptr;
      break;
  }
}

uint64_t Emulator::GetPersistentEmulatorFlags() {
#if XE_PLATFORM_WIN32 == 1
  uint64_t value = 0;
  DWORD value_size = sizeof(value);
  HKEY xenia_hkey = nullptr;
  LSTATUS lstat =
      RegOpenKeyA(HKEY_CURRENT_USER, "SOFTWARE\\Xenia", &xenia_hkey);
  if (!xenia_hkey) {
    // let the Set function create the key and initialize it to 0
    SetPersistentEmulatorFlags(0ULL);
    return 0ULL;
  }

  lstat = RegQueryValueExA(xenia_hkey, "XEFLAGS", 0, NULL,
                           reinterpret_cast<LPBYTE>(&value), &value_size);
  RegCloseKey(xenia_hkey);
  if (lstat) {
    return 0ULL;
  }
  return value;
#else
  return EmulatorFlagDisclaimerAcknowledged;
#endif
}
void Emulator::SetPersistentEmulatorFlags(uint64_t new_flags) {
#if XE_PLATFORM_WIN32 == 1
  uint64_t value = new_flags;
  DWORD value_size = sizeof(value);
  HKEY xenia_hkey = nullptr;
  LSTATUS lstat =
      RegOpenKeyA(HKEY_CURRENT_USER, "SOFTWARE\\Xenia", &xenia_hkey);
  if (!xenia_hkey) {
    lstat = RegCreateKeyA(HKEY_CURRENT_USER, "SOFTWARE\\Xenia", &xenia_hkey);
  }

  lstat = RegSetValueExA(xenia_hkey, "XEFLAGS", 0, REG_QWORD,
                         reinterpret_cast<const BYTE*>(&value), 8);
  RegFlushKey(xenia_hkey);
  RegCloseKey(xenia_hkey);
#endif
}

X_STATUS Emulator::MountPath(const std::filesystem::path& path,
                             const std::string_view mount_path) {
  auto device = CreateVfsDevice(path, mount_path);
  if (!device || !device->Initialize()) {
    XELOGE(
        "Unable to mount the selected file, it is an unsupported format or "
        "corrupted.");
    return X_STATUS_NO_SUCH_FILE;
  }
  if (!file_system_->RegisterDevice(std::move(device))) {
    XELOGE("Unable to register the input file to {}.", mount_path);
    return X_STATUS_NO_SUCH_FILE;
  }

  file_system_->UnregisterSymbolicLink(kDefaultPartitionSymbolicLink);
  file_system_->UnregisterSymbolicLink(kDefaultGameSymbolicLink);
  file_system_->UnregisterSymbolicLink("plugins:");

  // Create symlinks to the device.
  file_system_->RegisterSymbolicLink(kDefaultGameSymbolicLink, mount_path);
  file_system_->RegisterSymbolicLink(kDefaultPartitionSymbolicLink, mount_path);

  return X_STATUS_SUCCESS;
}

Emulator::FileSignatureType Emulator::GetFileSignature(
    const std::filesystem::path& path) {
  FILE* file = xe::filesystem::OpenFile(path, "rb");

  if (!file) {
    return FileSignatureType::Unknown;
  }

  const uint64_t file_size = std::filesystem::file_size(path);
  constexpr int64_t header_size = 4;

  if (file_size < header_size) {
    return FileSignatureType::Unknown;
  }

  char file_magic[header_size];
  fread(file_magic, sizeof(file_magic), 1, file);

  fourcc_t magic_value =
      make_fourcc(file_magic[0], file_magic[1], file_magic[2], file_magic[3]);

  fclose(file);

  switch (magic_value) {
    case xe::cpu::kXEX1Signature:
      return FileSignatureType::XEX1;
    case xe::cpu::kXEX2Signature:
      return FileSignatureType::XEX2;
    case xe::vfs::kCONSignature:
      return FileSignatureType::CON;
    case xe::vfs::kLIVESignature:
      return FileSignatureType::LIVE;
    case xe::vfs::kPIRSSignature:
      return FileSignatureType::PIRS;
    case xe::vfs::kXSFSignature:
      return FileSignatureType::XISO;
    case xe::cpu::kElfSignature:
      return FileSignatureType::ELF;
    default:
      break;
  }

  magic_value = make_fourcc(file_magic[0], file_magic[1], 0, 0);

  if (xe::kernel::kEXESignature == magic_value) {
    return FileSignatureType::EXE;
  }

  file = xe::filesystem::OpenFile(path, "rb");
  xe::filesystem::Seek(file, -header_size, SEEK_END);
  fread(file_magic, 1, header_size, file);
  fclose(file);

  magic_value =
      make_fourcc(file_magic[0], file_magic[1], file_magic[2], file_magic[3]);

  if (xe::vfs::kZarMagic == magic_value) {
    return FileSignatureType::ZAR;
  }

  // Check if XISO
  std::unique_ptr<vfs::Device> device =
      std::make_unique<vfs::DiscImageDevice>("", path);

  XELOGI("Checking for XISO");

  if (device->Initialize()) {
    return FileSignatureType::XISO;
  }

  XELOGE("{}: {} ({:08X})", __func__, path.extension(), magic_value);
  return FileSignatureType::Unknown;
}

X_STATUS Emulator::LaunchPath(const std::filesystem::path& path) {
  X_STATUS mount_result = X_STATUS_SUCCESS;

  switch (GetFileSignature(path)) {
    case FileSignatureType::XEX1:
    case FileSignatureType::XEX2:
    case FileSignatureType::ELF: {
      mount_result = MountPath(path, "\\Device\\Harddisk0\\Partition1");
      return mount_result ? mount_result : LaunchXexFile(path);
    } break;
    case FileSignatureType::LIVE:
    case FileSignatureType::CON:
    case FileSignatureType::PIRS: {
      mount_result = MountPath(path, "\\Device\\Package_0");
      return mount_result ? mount_result : LaunchStfsContainer(path);
    } break;
    case FileSignatureType::XISO: {
      mount_result = MountPath(path, "\\Device\\Cdrom0");
      return mount_result ? mount_result : LaunchDiscImage(path);
    } break;
    case FileSignatureType::ZAR: {
      mount_result = MountPath(path, "\\Device\\Cdrom0");
      return mount_result ? mount_result : LaunchDiscArchive(path);
    } break;
    case FileSignatureType::EXE:
    case FileSignatureType::Unknown:
    default:
      return X_STATUS_NOT_SUPPORTED;
      break;
  }
}

X_STATUS Emulator::LaunchXexFile(const std::filesystem::path& path) {
  // We create a virtual filesystem pointing to its directory and symlink
  // that to the game filesystem.
  // e.g., /my/files/foo.xex will get a local fs at:
  // \\Device\\Harddisk0\\Partition1
  // and then get that symlinked to game:\, so
  // -> game:\foo.xex
  // Get just the filename (foo.xex).
  auto file_name = path.filename();

  // Launch the game.
  auto fs_path = fmt::format("{}\\", kDefaultGameSymbolicLink) +
                 xe::path_to_utf8(file_name);
  X_STATUS result = CompleteLaunch(path, fs_path);

  if (XFAILED(result)) {
    return result;
  }

  kernel_state_->deployment_type_ = XDeploymentType::kInstalledToHDD;

  if (!kernel::IsSystemTitle(kernel_state_->title_id())) {
    return result;
  }

  const std::string mount_path =
      utf8::find_base_guest_path(kernel_state_->GetExecutableModule()->path());

  // System related symlinks. This should point to dashboard location in the
  // future.
  file_system_->RegisterSymbolicLink("\\SystemRoot", mount_path);

  auto module = kernel_state_->LoadUserModule("xam.xex");

  if (!module) {
    module = kernel_state_->LoadUserModule("$flash_xam.xex");
  }

  if (module) {
    result = kernel_state_->FinishLoadingUserModule(module, false);
  }

  return result;
}

X_STATUS Emulator::LaunchDiscImage(const std::filesystem::path& path) {
  std::string module_path = FindLaunchModule();
  X_STATUS result = CompleteLaunch(path, module_path);

  if (result == X_STATUS_NOT_FOUND && !cvars::launch_module.empty()) {
    return LaunchDefaultModule(path);
  }
  kernel_state_->deployment_type_ = XDeploymentType::kOpticalDisc;
  return result;
}

X_STATUS Emulator::LaunchDiscArchive(const std::filesystem::path& path) {
  std::string module_path = FindLaunchModule();
  X_STATUS result = CompleteLaunch(path, module_path);

  if (result == X_STATUS_NOT_FOUND && !cvars::launch_module.empty()) {
    return LaunchDefaultModule(path);
  }
  kernel_state_->deployment_type_ = XDeploymentType::kOpticalDisc;
  return result;
}

X_STATUS Emulator::LaunchStfsContainer(const std::filesystem::path& path) {
  std::string module_path = FindLaunchModule();
  X_STATUS result = CompleteLaunch(path, module_path);

  if (result == X_STATUS_NOT_FOUND && !cvars::launch_module.empty()) {
    return LaunchDefaultModule(path);
  }
  kernel_state_->deployment_type_ = XDeploymentType::kDownload;
  return result;
}

X_STATUS Emulator::LaunchDefaultModule(const std::filesystem::path& path) {
  cvars::launch_module = "";
  std::string module_path = FindLaunchModule();
  X_STATUS result = CompleteLaunch(path, module_path);

  if (XSUCCEEDED(result)) {
    kernel_state_->deployment_type_ = XDeploymentType::kInstalledToHDD;
    auto title_id = kernel_state_->title_id();
    if (!kernel::IsSystemTitle(title_id)) {
      // Assumption that any loaded game is loaded as a disc.
      kernel_state_->deployment_type_ = XDeploymentType::kOpticalDisc;
    }
  }
  return result;
}

X_STATUS Emulator::DataMigration(const uint64_t xuid) {
  uint32_t failure_count = 0;
  const std::string xuid_string = fmt::format("{:016X}", xuid);
  const std::string common_xuid_string = fmt::format("{:016X}", 0);
  const std::filesystem::path path_to_profile_data =
      content_root_ / xuid_string / "FFFE07D1" / "00010000" / xuid_string;
  // Filter directories inside. First we need to find any content type
  // directories.
  // Savefiles must go to user specific directory
  // Everything else goes to common
  const auto titles_to_move = xe::filesystem::FilterByName(
      xe::filesystem::ListDirectories(content_root_),
      std::regex("[A-F0-9]{8}"));

  for (const auto& title : titles_to_move) {
    if (xe::path_to_utf8(title.name) == "FFFE07D1" ||
        xe::path_to_utf8(title.name) == "00000000") {
      // SKip any dashboard/profile related data that was previously installed
      continue;
    }

    const auto content_type_dirs = xe::filesystem::FilterByName(
        xe::filesystem::ListDirectories(title.path / title.name),
        std::regex("[A-F0-9]{8}"));

    for (const auto& content_type : content_type_dirs) {
      const std::string used_xuid =
          xe::path_to_utf8(content_type.name) == "00000001"
              ? xuid_string
              : common_xuid_string;

      const auto previous_path = content_root_ / title.name / content_type.name;
      const auto path = content_root_ / used_xuid / title.name;

      if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path);
      }

      std::error_code ec;
      std::filesystem::rename(previous_path, path / content_type.name, ec);

      if (ec) {
        failure_count++;
        XELOGW("{}: Moving from: {} to: {} failed! Error message: {} ({:08X})",
               __func__, previous_path, path / content_type.name, ec.message(),
               ec.value());
      }
    }
    // Other directories:
    // Headers - Just copy everything to both common and xuid locations
    // profile - ?
    if (std::filesystem::exists(title.path / title.name / "Headers")) {
      const auto xuid_path =
          content_root_ / xuid_string / title.name / "Headers";

      std::filesystem::create_directories(xuid_path);

      std::error_code ec;
      // Copy to specific user
      std::filesystem::copy(title.path / title.name / "Headers", xuid_path,
                            std::filesystem::copy_options::recursive |
                                std::filesystem::copy_options::skip_existing,
                            ec);
      if (ec) {
        failure_count++;
        XELOGW("{}: Copying from: {} to: {} failed! Error message: {} ({:08X})",
               __func__, title.path / title.name / "Headers", xuid_path,
               ec.message(), ec.value());
      }

      const auto header_types =
          xe::filesystem::ListDirectories(title.path / title.name / "Headers");

      if (!(header_types.size() == 1 &&
            header_types.at(0).name == "00000001")) {
        const auto common_path =
            content_root_ / common_xuid_string / title.name / "Headers";

        std::filesystem::create_directories(common_path);

        // Copy to common, skip cases where only savefile header is available
        std::filesystem::copy(title.path / title.name / "Headers", common_path,
                              std::filesystem::copy_options::recursive |
                                  std::filesystem::copy_options::skip_existing,
                              ec);
        if (ec) {
          failure_count++;
          XELOGW(
              "{}: Copying from: {} to: {} failed! Error message: {} ({:08X})",
              __func__, title.path / title.name / "Headers", common_path,
              ec.message(), ec.value());
        }
      }

      if (!ec) {
        // Remove previous directory
        std::error_code ec;
        std::filesystem::remove_all(title.path / title.name / "Headers", ec);
      }
    }

    if (std::filesystem::exists(title.path / title.name / "profile")) {
      // Find directory with previous username. There should be only one!
      const auto old_profile_data =
          xe::filesystem::ListDirectories(title.path / title.name / "profile");

      xe::filesystem::FileInfo entry_to_copy = xe::filesystem::FileInfo();
      if (old_profile_data.size() != 1) {
        for (const auto& entry : old_profile_data) {
          if (entry.name == "User") {
            entry_to_copy = entry;
          }
        }
      } else {
        entry_to_copy = old_profile_data.front();
      }

      const auto path_from =
          title.path / title.name / "profile" / entry_to_copy.name;
      std::error_code ec;
      // Move files from inside to outside for convenience
      std::filesystem::rename(path_from, path_to_profile_data / title.name, ec);
      if (ec) {
        failure_count++;
        XELOGW("{}: Moving from: {} to: {} failed! Error message: {} ({:08X})",
               __func__, path_from, path_to_profile_data / title.name,
               ec.message(), ec.value());
      } else {
        std::error_code ec;
        std::filesystem::remove_all(title.path / title.name / "profile", ec);
      }
    }

    const auto remaining_file_list =
        xe::filesystem::ListDirectories(title.path / title.name);

    if (remaining_file_list.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(title.path / title.name, ec);
    }
  }

  std::string migration_status_message =
      fmt::format("Migration finished with {} {}.", failure_count,
                  failure_count == 1 ? "error" : "errors");

  if (failure_count) {
    migration_status_message.append(
        " For more information check xenia.log file.");
  }
  new xe::ui::HostNotificationWindow(imgui_drawer_, "Migration Status",
                                     migration_status_message, 0);
  return X_STATUS_SUCCESS;
}

X_STATUS Emulator::ProcessContentPackageHeader(
    const std::filesystem::path& path, ContentInstallEntry& installation_info) {
  installation_info.name_ = "Invalid Content Package!";
  installation_info.content_type_ = XContentType::kInvalid;
  installation_info.data_installation_path_ = xe::path_to_utf8(path.filename());

  const auto header = vfs::XContentContainerDevice::ReadContainerHeader(path);

  if (!header || !header->content_header.is_magic_valid()) {
    installation_info.installation_state_ = InstallState::failed;
    installation_info.installation_result_ = X_STATUS_INVALID_PARAMETER;
    installation_info.installation_error_message_ = "Invalid Package Type!";
    XELOGE("Failed to initialize device");
    return X_STATUS_INVALID_PARAMETER;
  }

  // Always install savefiles to user signed to slot 0.
  const auto profile =
      kernel_state_->xam_state()->profile_manager()->GetProfile(
          static_cast<uint8_t>(0));

  uint64_t xuid = header->content_metadata.profile_id;
  if (header->content_metadata.content_type == XContentType::kSavedGame &&
      profile) {
    xuid = profile->xuid();
  }

  installation_info.data_installation_path_ = fmt::format(
      "{:016X}/{:08X}/{:08X}/{}", xuid,
      header->content_metadata.execution_info.title_id.get(),
      static_cast<uint32_t>(header->content_metadata.content_type.get()),
      path.filename());

  installation_info.header_installation_path_ = fmt::format(
      "{:016X}/{:08X}/Headers/{:08X}/{}", xuid,
      header->content_metadata.execution_info.title_id.get(),
      static_cast<uint32_t>(header->content_metadata.content_type.get()),
      path.filename());

  installation_info.name_ =
      xe::to_utf8(header->content_metadata.display_name(XLanguage::kEnglish));
  installation_info.content_type_ =
      static_cast<XContentType>(header->content_metadata.content_type);
  installation_info.content_size_ = header->content_metadata.content_size;
  installation_info.installation_state_ = InstallState::pending;

  installation_info.icon_ = imgui_drawer_->LoadImGuiIcon(
      std::span<const uint8_t>(header->content_metadata.title_thumbnail,
                               header->content_metadata.title_thumbnail_size));
  return X_STATUS_SUCCESS;
}

X_STATUS Emulator::InstallContentPackage(
    const std::filesystem::path& path, ContentInstallEntry& installation_info) {
  installation_info.installation_state_ = InstallState::preparing;

  std::unique_ptr<vfs::XContentContainerDevice> device =
      vfs::XContentContainerDevice::CreateContentDevice("", path);

  if (!device || !device->Initialize()) {
    installation_info.installation_state_ = InstallState::failed;
    installation_info.installation_error_message_ =
        "Device initialization failed!";
    installation_info.installation_result_ = X_STATUS_ACCESS_DENIED;
    XELOGE("Failed to initialize device");
    return X_STATUS_INVALID_PARAMETER;
  }

  const std::filesystem::path installation_path =
      content_root() / installation_info.data_installation_path_;

  const std::filesystem::path header_path =
      content_root() / installation_info.header_installation_path_;

  if (!std::filesystem::exists(content_root())) {
    const std::error_code ec = xe::filesystem::CreateFolder(content_root());
    if (ec) {
      installation_info.installation_state_ = InstallState::failed;
      installation_info.installation_error_message_ = ec.message();
      installation_info.installation_result_ = X_STATUS_ACCESS_DENIED;
      return X_STATUS_ACCESS_DENIED;
    }
  }

  const auto disk_space = std::filesystem::space(content_root());
  if (disk_space.available < installation_info.content_size_ * 1.1f) {
    installation_info.installation_state_ = InstallState::failed;
    installation_info.installation_error_message_ = "Insufficient disk space!";
    installation_info.installation_result_ = X_STATUS_DISK_FULL;
    return X_STATUS_DISK_FULL;
  }

  if (std::filesystem::exists(installation_path)) {
    // TODO(Gliniak): Popup
    // Do you want to overwrite already existing data?
  } else {
    std::error_code error_code;
    std::filesystem::create_directories(installation_path, error_code);
    if (error_code) {
      installation_info.installation_state_ = InstallState::failed;
      installation_info.installation_error_message_ =
          "Cannot Create Content Directory!";
      installation_info.installation_result_ = error_code.value();
      return error_code.value();
    }
  }

  installation_info.content_size_ = device->data_size();
  installation_info.installation_state_ = InstallState::installing;

  vfs::VirtualFileSystem::ExtractContentHeader(device.get(), header_path);

  X_STATUS error_code = vfs::VirtualFileSystem::ExtractContentFiles(
      device.get(), installation_path,
      installation_info.currently_installed_size_);
  if (error_code != X_ERROR_SUCCESS) {
    installation_info.installation_state_ = InstallState::failed;
    return error_code;
  }

  installation_info.installation_state_ = InstallState::installed;
  installation_info.currently_installed_size_ = installation_info.content_size_;
  kernel_state()->BroadcastNotification(kXNotificationLiveContentInstalled, 0);

  if (installation_info.content_type_ == XContentType::kProfile) {
    kernel_state_->xam_state()->profile_manager()->ReloadProfiles();
  }

  return error_code;
}

X_STATUS Emulator::ExtractZarchivePackage(
    const std::filesystem::path& path,
    const std::filesystem::path& extract_dir) {
  std::unique_ptr<vfs::Device> device =
      std::make_unique<vfs::DiscZarchiveDevice>("", path);
  if (!device->Initialize()) {
    XELOGE("Failed to initialize device");
    return X_STATUS_INVALID_PARAMETER;
  }

  if (std::filesystem::exists(extract_dir)) {
    // TODO(Gliniak): Popup
    // Do you want to overwrite already existing data?
  } else {
    std::error_code error_code;
    std::filesystem::create_directories(extract_dir, error_code);
    if (error_code) {
      return error_code.value();
    }
  }

  uint64_t progress = 0;
  return vfs::VirtualFileSystem::ExtractContentFiles(device.get(), extract_dir,
                                                     progress);
}

X_STATUS Emulator::CreateZarchivePackage(
    const std::filesystem::path& inputDirectory,
    const std::filesystem::path& outputFile) {
  std::vector<uint8_t> buffer;
  buffer.resize(64 * 1024);

  std::error_code ec;
  PackContext packContext;
  packContext.outputFilePath = outputFile;

  ZArchiveWriter zWriter(
      [](int32_t partIndex, void* ctx) {
        PackContext* packContext = reinterpret_cast<PackContext*>(ctx);
        packContext->currentOutputFile =
            std::ofstream(packContext->outputFilePath, std::ios::binary);

        if (!packContext->currentOutputFile.is_open()) {
          XELOGI("Failed to create output file: {}\n",
                 packContext->outputFilePath.string());
          packContext->hasError = true;
        }
      },
      [](const void* data, size_t length, void* ctx) {
        PackContext* packContext = reinterpret_cast<PackContext*>(ctx);
        packContext->currentOutputFile.write(
            reinterpret_cast<const char*>(data), length);
      },
      &packContext);

  if (packContext.hasError) {
    return X_STATUS_UNSUCCESSFUL;
  }

  for (auto const& dirEntry :
       std::filesystem::recursive_directory_iterator(inputDirectory)) {
    std::filesystem::path pathEntry =
        std::filesystem::relative(dirEntry.path(), inputDirectory, ec);

    if (ec) {
      XELOGI("Failed to get relative path {}\n", pathEntry.string());
      return X_STATUS_UNSUCCESSFUL;
    }

    if (dirEntry.is_directory()) {
      if (!zWriter.MakeDir(pathEntry.generic_string().c_str(), false)) {
        XELOGI("Failed to create directory {}\n", pathEntry.string());
        return X_STATUS_UNSUCCESSFUL;
      }
    } else if (dirEntry.is_regular_file()) {
      // Don't pack itself to prevent infinite packing.
      if (dirEntry == outputFile) {
        continue;
      }

      XELOGI("Adding file: {}\n", pathEntry.string());

      if (!zWriter.StartNewFile(pathEntry.generic_string().c_str())) {
        XELOGI("Failed to create archive file {}\n", pathEntry.string());
        return X_STATUS_UNSUCCESSFUL;
      }

      std::filesystem::path file_to_pack_path = inputDirectory / pathEntry;
      FILE* file = xe::filesystem::OpenFile(file_to_pack_path, "rb");

      if (!file) {
        XELOGI("Failed to open input file {}\n", pathEntry.string());
        return X_STATUS_UNSUCCESSFUL;
      }

      const uint64_t file_size = std::filesystem::file_size(file_to_pack_path);
      uint64_t total_bytes_read = 0;

      while (total_bytes_read < file_size) {
        uint64_t bytes_read = fread(buffer.data(), 1, buffer.size(), file);

        total_bytes_read += bytes_read;

        zWriter.AppendData(buffer.data(), bytes_read);
      }

      fclose(file);
    }

    if (packContext.hasError) {
      return X_STATUS_UNSUCCESSFUL;
    }
  }

  zWriter.Finalize();

  return X_STATUS_SUCCESS;
}

void Emulator::Pause() {
  if (paused_) {
    return;
  }
  paused_ = true;

  // Don't hold the lock on this (so any waits follow through)
  graphics_system_->Pause();
  audio_system_->Pause();

  auto lock = global_critical_region::AcquireDirect();
  auto threads =
      kernel_state()->object_table()->GetObjectsByType<kernel::XThread>(
          kernel::XObject::Type::Thread);
  auto current_thread = kernel::XThread::IsInThread()
                            ? kernel::XThread::GetCurrentThread()
                            : nullptr;
  for (auto thread : threads) {
    // Don't pause ourself or host threads.
    if (thread == current_thread || !thread->can_debugger_suspend()) {
      continue;
    }

    if (thread->is_running()) {
      thread->thread()->Suspend(nullptr);
    }
  }

  XELOGD("! EMULATOR PAUSED !");
}

void Emulator::Resume() {
  if (!paused_) {
    return;
  }
  paused_ = false;
  XELOGD("! EMULATOR RESUMED !");

  graphics_system_->Resume();
  audio_system_->Resume();

  auto threads =
      kernel_state()->object_table()->GetObjectsByType<kernel::XThread>(
          kernel::XObject::Type::Thread);
  for (auto thread : threads) {
    if (!thread->can_debugger_suspend()) {
      // Don't pause host threads.
      continue;
    }

    if (!thread->is_running()) {
      thread->thread()->Resume(nullptr);
    }
  }
}

bool Emulator::SaveToFile(const std::filesystem::path& path) {
  Pause();

  filesystem::CreateEmptyFile(path);
  auto map = MappedMemory::Open(path, MappedMemory::Mode::kReadWrite, 0, 2_GiB);
  if (!map) {
    return false;
  }

  // Save the emulator state to a file
  ByteStream stream(map->data(), map->size());
  stream.Write(kEmulatorSaveSignature);
  stream.Write(title_id_.has_value());
  if (title_id_.has_value()) {
    stream.Write(title_id_.value());
  }

  // It's important we don't hold the global lock here! XThreads need to step
  // forward (possibly through guarded regions) without worry!
  processor_->Save(&stream);
  graphics_system_->Save(&stream);
  audio_system_->Save(&stream);
  kernel_state_->Save(&stream);
  memory_->Save(&stream);
  map->Close(stream.offset());

  Resume();
  return true;
}

bool Emulator::RestoreFromFile(const std::filesystem::path& path) {
  // Restore the emulator state from a file
  auto map = MappedMemory::Open(path, MappedMemory::Mode::kReadWrite);
  if (!map) {
    return false;
  }

  restoring_ = true;

  // Terminate any loaded titles.
  Pause();
  kernel_state_->TerminateTitle();

  auto lock = global_critical_region::AcquireDirect();
  ByteStream stream(map->data(), map->size());
  if (stream.Read<uint32_t>() != kEmulatorSaveSignature) {
    return false;
  }

  auto has_title_id = stream.Read<bool>();
  std::optional<uint32_t> title_id;
  if (!has_title_id) {
    title_id = {};
  } else {
    title_id = stream.Read<uint32_t>();
  }
  if (title_id_.has_value() != title_id.has_value() ||
      title_id_.value() != title_id.value()) {
    // Swapping between titles is unsupported at the moment.
    assert_always();
    return false;
  }

  if (!processor_->Restore(&stream)) {
    XELOGE("Could not restore processor!");
    return false;
  }
  if (!graphics_system_->Restore(&stream)) {
    XELOGE("Could not restore graphics system!");
    return false;
  }
  if (!audio_system_->Restore(&stream)) {
    XELOGE("Could not restore audio system!");
    return false;
  }
  if (!kernel_state_->Restore(&stream)) {
    XELOGE("Could not restore kernel state!");
    return false;
  }
  if (!memory_->Restore(&stream)) {
    XELOGE("Could not restore memory!");
    return false;
  }

  // Update the main thread.
  auto threads =
      kernel_state_->object_table()->GetObjectsByType<kernel::XThread>();
  for (auto thread : threads) {
    if (thread->main_thread()) {
      main_thread_ = thread;
      break;
    }
  }

  Resume();

  restore_fence_.Signal();
  restoring_ = false;

  return true;
}

const std::filesystem::path Emulator::GetNewDiscPath(
    std::string window_message) {
  std::filesystem::path path = "";

  auto file_picker = xe::ui::FilePicker::Create();
  file_picker->set_mode(ui::FilePicker::Mode::kOpen);
  file_picker->set_type(ui::FilePicker::Type::kFile);
  file_picker->set_multi_selection(false);
  file_picker->set_title(!window_message.empty() ? window_message
                                                 : "Select Content Package");
  file_picker->set_extensions({
      {"Supported Files", "*.iso;*.xex;*.xcp;*.*"},
      {"Disc Image (*.iso)", "*.iso"},
      {"Xbox Executable (*.xex)", "*.xex"},
      {"All Files (*.*)", "*.*"},
  });

  if (file_picker->Show()) {
    auto selected_files = file_picker->selected_files();
    if (!selected_files.empty()) {
      path = selected_files[0];
    }
  }
  return path;
}

bool Emulator::ExceptionCallbackThunk(Exception* ex, void* data) {
  return reinterpret_cast<Emulator*>(data)->ExceptionCallback(ex);
}

bool Emulator::ExceptionCallback(Exception* ex) {
  // Check to see if the exception occurred in guest code.
  auto code_cache = processor()->backend()->code_cache();
  auto code_base = code_cache->execute_base_address();
  auto code_end = code_base + code_cache->total_size();

  if (!processor()->is_debugger_attached() && debugging::IsDebuggerAttached()) {
    // If Xenia's debugger isn't attached but another one is, pass it to that
    // debugger.
    return false;
  } else if (processor()->is_debugger_attached()) {
    // Let the debugger handle this exception. It may decide to continue past
    // it (if it was a stepping breakpoint, etc).
    return processor()->OnUnhandledException(ex);
  }

  if (!(ex->pc() >= code_base && ex->pc() < code_end)) {
    // Didn't occur in guest code. Let it pass.
    return false;
  }

  auto current_thread = kernel::XThread::GetCurrentThread();
  assert_not_null(current_thread);

  auto guest_function = code_cache->LookupFunction(ex->pc());
  assert_not_null(guest_function);

  auto context = current_thread->thread_state()->context();
  auto memory = processor()->memory();
  auto guest_pc = guest_function->MapMachineCodeToGuestAddress(ex->pc());

  auto return_to_guest_lr = [&]() -> bool {
    const uint32_t lr = static_cast<uint32_t>(context->lr);
    for (auto* fn : processor()->FindFunctionsWithAddress(lr)) {
      if (fn && fn->is_guest()) {
        auto* caller_fn = static_cast<cpu::GuestFunction*>(fn);
        ex->set_resume_pc(caller_fn->MapGuestAddressToMachineCode(lr));
        return true;
      }
    }
    return false;
  };
  auto resume_after_hook_entry = [&]() -> bool {
    context->r[12] = context->lr;
    ex->set_resume_pc(guest_function->MapGuestAddressToMachineCode(
        guest_pc + sizeof(uint32_t)));
    return true;
  };

  const bool nui_hook_fault =
      ex->code() == Exception::Code::kAccessViolation &&
      ex->access_violation_operation() ==
          Exception::AccessViolationOperation::kRead;

  {
    const uint32_t getter =
        g_nui_skeleton_get_next_frame_hook.load(std::memory_order_relaxed);
    if (nui_hook_fault && getter && guest_pc == getter) {
      context->r[3] = HandleNuiSkeletonGetNextFrame(context, memory);
      if (return_to_guest_lr()) {
        return true;
      }

      return resume_after_hook_entry();
    }
  }

  {
    const uint32_t tracking_enable =
        g_nui_skeleton_tracking_enable_hook.load(std::memory_order_relaxed);
    if (nui_hook_fault && tracking_enable && guest_pc == tracking_enable) {
      context->r[3] = HandleNuiSkeletonTrackingEnable(context);
      if (return_to_guest_lr()) {
        return true;
      }
      return resume_after_hook_entry();
    }
  }

  {
    const uint32_t set_tracked =
        g_nui_skeleton_set_tracked_hook.load(std::memory_order_relaxed);
    if (nui_hook_fault && set_tracked && guest_pc == set_tracked) {
      context->r[3] = HandleNuiSkeletonSetTrackedSkeletons(context);
      if (return_to_guest_lr()) {
        return true;
      }
      return resume_after_hook_entry();
    }
  }

  {
    const uint32_t frame_end =
        g_nui_set_frame_end_event_hook.load(std::memory_order_relaxed);
    if (nui_hook_fault && frame_end && guest_pc == frame_end) {
      context->r[3] = HandleNuiSetFrameEndEvent(context);
      if (return_to_guest_lr()) {
        return true;
      }
      return resume_after_hook_entry();
    }
  }

  {
    uint32_t lock_rect_target = 0;
    if (nui_hook_fault &&
        FindNuiTextureRoute(guest_pc, g_nui_d3d_texture_lock_calls,
                            g_nui_d3d_texture_lock_targets,
                            &lock_rect_target)) {
      const uint32_t texture = static_cast<uint32_t>(context->r[3]);
      const bool handled = kernel::xam::LockNuiImageTextureToGuest(
          texture, static_cast<uint32_t>(context->r[4]),
          static_cast<uint32_t>(context->r[5]),
          static_cast<uint32_t>(context->r[6]),
          static_cast<uint32_t>(context->r[7]));
      if (handled) {
        context->lr = guest_pc + sizeof(uint32_t);
        ex->set_resume_pc(guest_function->MapGuestAddressToMachineCode(
            guest_pc + sizeof(uint32_t)));
        return true;
      }

      // Forward non-NUI textures to the original direct branch target.
      context->lr = guest_pc + sizeof(uint32_t);
      for (auto* fn : processor()->FindFunctionsWithAddress(lock_rect_target)) {
        if (fn && fn->is_guest()) {
          auto* target_fn = static_cast<cpu::GuestFunction*>(fn);
          ex->set_resume_pc(
              target_fn->MapGuestAddressToMachineCode(lock_rect_target));
          return true;
        }
      }
      return false;
    }
  }

  {
    uint32_t unlock_rect_target = 0;
    if (nui_hook_fault &&
        FindNuiTextureRoute(guest_pc, g_nui_d3d_texture_unlock_calls,
                            g_nui_d3d_texture_unlock_targets,
                            &unlock_rect_target)) {
      const uint32_t texture = static_cast<uint32_t>(context->r[3]);
      if (kernel::xam::IsNuiImageTextureGuestAddress(texture)) {
        context->lr = guest_pc + sizeof(uint32_t);
        ex->set_resume_pc(guest_function->MapGuestAddressToMachineCode(
            guest_pc + sizeof(uint32_t)));
        return true;
      }

      context->lr = guest_pc + sizeof(uint32_t);
      for (auto* fn :
           processor()->FindFunctionsWithAddress(unlock_rect_target)) {
        if (fn && fn->is_guest()) {
          auto* target_fn = static_cast<cpu::GuestFunction*>(fn);
          ex->set_resume_pc(
              target_fn->MapGuestAddressToMachineCode(unlock_rect_target));
          return true;
        }
      }
      return false;
    }
  }

  {
    const uint32_t image_open =
        g_nui_image_stream_open_hook.load(std::memory_order_relaxed);
    if (nui_hook_fault && image_open && guest_pc == image_open) {
      context->r[3] = HandleNuiImageStreamOpen(context, memory);
      if (return_to_guest_lr()) {
        return true;
      }
      return resume_after_hook_entry();
    }
  }

  {
    const uint32_t image_get =
        g_nui_image_stream_get_next_frame_hook.load(std::memory_order_relaxed);
    if (nui_hook_fault && image_get && guest_pc == image_get) {
      context->r[3] = HandleNuiImageStreamGetNextFrame(context, memory);
      if (return_to_guest_lr()) {
        return true;
      }
      return resume_after_hook_entry();
    }
  }

  {
    const uint32_t image_release =
        g_nui_image_stream_release_frame_hook.load(std::memory_order_relaxed);
    if (nui_hook_fault && image_release && guest_pc == image_release) {
      context->r[3] = HandleNuiImageStreamReleaseFrame(context);
      if (return_to_guest_lr()) {
        return true;
      }
      return resume_after_hook_entry();
    }
  }

  // Within range. Pause the emulator and eat the exception.
  Pause();

  // Dump information into the log.

  std::string crash_msg;
  crash_msg.append("==== CRASH DUMP ====\n");
  crash_msg.append(fmt::format("Thread ID (Host: 0x{:08X} / Guest: 0x{:08X})\n",
                               current_thread->thread()->system_id(),
                               current_thread->thread_id()));
  crash_msg.append(
      fmt::format("Thread Handle: 0x{:08X}\n", current_thread->handle()));
  crash_msg.append(
      fmt::format("PC: 0x{:08X}\n",
                  guest_function->MapMachineCodeToGuestAddress(ex->pc())));
  if (ex->code() == Exception::Code::kAccessViolation) {
    const char* op_str = "unknown";
    if (ex->access_violation_operation() ==
        Exception::AccessViolationOperation::kRead) {
      op_str = "read";
    } else if (ex->access_violation_operation() ==
               Exception::AccessViolationOperation::kWrite) {
      op_str = "write";
    }
    crash_msg.append(fmt::format("Access Violation: {} at 0x{:016X}\n", op_str,
                                 ex->fault_address()));
  } else if (ex->code() == Exception::Code::kIllegalInstruction) {
    crash_msg.append("Illegal Instruction\n");
  }
  crash_msg.append("Registers:\n");
  for (int i = 0; i < 32; i++) {
    crash_msg.append(fmt::format(" r{:<3} = {:016X}\n", i, context->r[i]));
  }
  for (int i = 0; i < 32; i++) {
    crash_msg.append(fmt::format(" f{:<3} = {:016X} = (double){} = (float){}\n",
                                 i,
                                 *reinterpret_cast<uint64_t*>(&context->f[i]),
                                 context->f[i], *(float*)&context->f[i]));
  }
  for (int i = 0; i < 128; i++) {
    crash_msg.append(
        fmt::format(" v{:<3} = [0x{:08X}, 0x{:08X}, 0x{:08X}, 0x{:08X}]\n", i,
                    context->v[i].u32[0], context->v[i].u32[1],
                    context->v[i].u32[2], context->v[i].u32[3]));
  }
  XELOGE("{}", crash_msg);
  std::string crash_dlg = fmt::format(
      "The guest has crashed.\n\n"
      "Xenia has now paused itself.\n\n"
      "{}",
      crash_msg);
  // Display a dialog telling the user the guest has crashed.
  if (display_window_ && imgui_drawer_) {
    display_window_->app_context().CallInUIThreadSynchronous([this,
                                                              &crash_dlg]() {
      xe::ui::ImGuiDialog::ShowMessageBox(imgui_drawer_, "Uh-oh!", crash_dlg);
    });
  }

  // Now suspend ourself (we should be a guest thread).
  current_thread->Suspend(nullptr);

  // We should not arrive here!
  assert_always();
  return false;
}

void Emulator::WaitUntilExit() {
  while (true) {
    if (main_thread_) {
      xe::threading::Wait(main_thread_->thread(), false);
    }

    if (restoring_) {
      restore_fence_.Wait();
    } else {
      // Not restoring and the thread exited. We're finished.
      break;
    }
  }

  on_exit();
}

void Emulator::AddGameConfigLoadCallback(GameConfigLoadCallback* callback) {
  assert_not_null(callback);
  // Game config load callbacks handling is entirely in the UI thread.
  assert_true(!display_window_ ||
              display_window_->app_context().IsInUIThread());
  // Check if already added.
  if (std::find(game_config_load_callbacks_.cbegin(),
                game_config_load_callbacks_.cend(),
                callback) != game_config_load_callbacks_.cend()) {
    return;
  }
  game_config_load_callbacks_.push_back(callback);
}

void Emulator::RemoveGameConfigLoadCallback(GameConfigLoadCallback* callback) {
  assert_not_null(callback);
  // Game config load callbacks handling is entirely in the UI thread.
  assert_true(!display_window_ ||
              display_window_->app_context().IsInUIThread());
  auto it = std::find(game_config_load_callbacks_.cbegin(),
                      game_config_load_callbacks_.cend(), callback);
  if (it == game_config_load_callbacks_.cend()) {
    return;
  }
  if (game_config_load_callback_loop_next_index_ != SIZE_MAX) {
    // Actualize the next callback index after the erasure from the vector.
    size_t existing_index =
        size_t(std::distance(game_config_load_callbacks_.cbegin(), it));
    if (game_config_load_callback_loop_next_index_ > existing_index) {
      --game_config_load_callback_loop_next_index_;
    }
  }
  game_config_load_callbacks_.erase(it);
}

std::string Emulator::FindLaunchModule() {
  std::string path(fmt::format("{}\\", kDefaultGameSymbolicLink));

  auto xam = kernel_state()->GetKernelModule<kernel::xam::XamModule>("xam.xex");

  if (!xam->loader_data().launch_path.empty()) {
    std::string symbolic_link_path;
    if (kernel_state_->file_system()->FindSymbolicLink(kDefaultGameSymbolicLink,
                                                       symbolic_link_path)) {
      std::filesystem::path file_path = symbolic_link_path;
      // Remove previous symbolic links.
      // Some titles can provide root within specific directory.
      kernel_state_->file_system()->UnregisterSymbolicLink(
          kDefaultPartitionSymbolicLink);
      kernel_state_->file_system()->UnregisterSymbolicLink(
          kDefaultGameSymbolicLink);

      file_path /= std::filesystem::path(xam->loader_data().launch_path);

      kernel_state_->file_system()->RegisterSymbolicLink(
          kDefaultPartitionSymbolicLink,
          xe::path_to_utf8(file_path.parent_path()));
      kernel_state_->file_system()->RegisterSymbolicLink(
          kDefaultGameSymbolicLink, xe::path_to_utf8(file_path.parent_path()));

      return xe::path_to_utf8(file_path);
    }
  }

  if (!cvars::launch_module.empty()) {
    return path + cvars::launch_module;
  }

  return path + "default.xex";
}

static std::string format_version(xex2_version version) {
  // fmt::format doesn't like bit fields we use + to bypass it
  return fmt::format("{}.{}.{}.{}", +version.major, +version.minor,
                     +version.build, +version.qfe);
}

X_STATUS Emulator::CompleteLaunch(const std::filesystem::path& path,
                                  const std::string_view module_path) {
  // Making changes to the UI (setting the icon) and executing game config
  // load callbacks which expect to be called from the UI thread.
  // If not on UI thread, dispatch to it synchronously.
  if (!display_window_->app_context().IsInUIThread()) {
    X_STATUS result = X_STATUS_UNSUCCESSFUL;
    display_window_->app_context().CallInUIThreadSynchronous(
        [this, &path, &module_path, &result]() {
          result = CompleteLaunch(path, module_path);
        });
    return result;
  }

  // Setup NullDevices for raw HDD partition accesses
  // Cache/STFC code baked into games tries reading/writing to these
  // By using a NullDevice that just returns success to all IO requests it
  // should allow games to believe cache/raw disk was accessed successfully

  // NOTE: this should probably be moved to xenia_main.cc, but right now we
  // need to register the \Device\Harddisk0\ NullDevice _after_ the
  // \Device\Harddisk0\Partition1 HostPathDevice, otherwise requests to
  // Partition1 will go to this. Registering during CompleteLaunch allows us
  // to make sure any HostPathDevices are ready beforehand. (see comment above
  // cache:\ device registration for more info about why)
  auto null_paths = {std::string("\\Partition0"), std::string("\\Cache0"),
                     std::string("\\Cache1")};
  auto null_device =
      std::make_unique<vfs::NullDevice>("\\Device\\Harddisk0", null_paths);
  if (null_device->Initialize()) {
    file_system_->RegisterDevice(std::move(null_device));
  }

  // Reset state.
  title_id_ = std::nullopt;
  title_name_ = "";
  title_version_ = "";
  display_window_->SetIcon(nullptr, 0);
  ResetNuiStaticInjectorState();

  // Allow xam to request module loads.
  auto xam = kernel_state()->GetKernelModule<kernel::xam::XamModule>("xam.xex");

  XELOGI("Loading module {}", module_path);
  auto module = kernel_state_->LoadUserModule(module_path);
  if (!module) {
    XELOGE("Failed to load user module {}", path);
    return X_STATUS_NOT_FOUND;
  }

  if (!module->is_executable()) {
    kernel_state_->UnloadUserModule(module, false);
    XELOGE("Failed to load user module {}", path);
    return X_STATUS_NOT_SUPPORTED;
  }

  X_RESULT result = kernel_state_->ApplyTitleUpdate(module);
  if (XFAILED(result)) {
    XELOGE("Failed to apply title update! Cannot run module {}", path);
    return result;
  }

  result = kernel_state_->FinishLoadingUserModule(module);
  if (XFAILED(result)) {
    XELOGE("Failed to initialize user module {}", path);
    return result;
  }
  // Grab the current title ID.
  xex2_opt_execution_info* info = nullptr;
  uint32_t workspace_address = 0;
  module->GetOptHeader(XEX_HEADER_EXECUTION_INFO, &info);

  kernel_state_->memory()
      ->LookupHeapByType(false, 0x1000)
      ->Alloc(module->workspace_size(), 0x1000,
              kMemoryAllocationReserve | kMemoryAllocationCommit,
              kMemoryProtectRead | kMemoryProtectWrite, false,
              &workspace_address);

  if (!info) {
    title_id_ = 0;
  } else {
    title_id_ = info->title_id;
    auto title_version = info->version();
    if (title_version.value != 0) {
      title_version_ = format_version(title_version);
    }
  }

  // Try and load the resource database (xex only).
  if (module->title_id()) {
    auto title_id = fmt::format("{:08X}", module->title_id());

    // Load the per-game configuration file and make sure updates are handled
    // by the callbacks.
    config::LoadGameConfig(title_id);
    assert_true(game_config_load_callback_loop_next_index_ == SIZE_MAX);
    game_config_load_callback_loop_next_index_ = 0;
    while (game_config_load_callback_loop_next_index_ <
           game_config_load_callbacks_.size()) {
      game_config_load_callbacks_[game_config_load_callback_loop_next_index_++]
          ->PostGameConfigLoad();
    }
    game_config_load_callback_loop_next_index_ = SIZE_MAX;

    const auto db = kernel_state_->module_xdbf(module);

    game_info_database_ =
        std::make_unique<kernel::util::GameInfoDatabase>(db.get());
    kernel_state_->xam_state()->LoadSpaInfo(db.get());

    kernel_state_->xam_state()->user_tracker()->AddTitleToPlayedList();

    if (game_info_database_->IsValid()) {
      title_name_ = game_info_database_->GetTitleName(static_cast<XLanguage>(
          kernel_state_->xconfig()->ReadSetting<uint32_t>(
              kernel::XCONFIG_USER_CATEGORY, kernel::XCONFIG_USER_LANGUAGE)));
      XELOGI("Title name: {}", title_name_);

      // Show achievments data
      tabulate::Table table;
      table.format().multi_byte_characters(true);
      table.add_row({"ID", "Title", "Description", "Type", "Gamerscore"});

      const std::vector<kernel::util::GameInfoDatabase::Achievement>
          achievement_list = game_info_database_->GetAchievements();
      for (const kernel::util::GameInfoDatabase::Achievement& entry :
           achievement_list) {
        const std::string type = GetAchievementTypeName(
            kernel::xam::GetAchievementType(entry.flags));

        table.add_row({fmt::format("{}", entry.id), entry.label,
                       entry.description, type,
                       fmt::format("{}", entry.gamerscore)});
      }
      XELOGI("\n-------------------- ACHIEVEMENTS --------------------\n{}",
             table.str());

      const std::vector<kernel::util::GameInfoDatabase::Property>
          properties_list = game_info_database_->GetProperties();

      // 4D5307DC SPA contains a lot of properties, limit properties to log.
      const auto properties_list_limit =
          properties_list | std::views::take(150);

      table = tabulate::Table();
      table.format().multi_byte_characters(true);
      table.add_row({"ID", "Name", "Matchmaking", "Data Size"});

      for (const kernel::util::GameInfoDatabase::Property& entry :
           properties_list_limit) {
        std::string label =
            string_util::remove_eol(string_util::trim(entry.description));

        table.add_row({fmt::format("{:08X}", entry.id), label,
                       entry.is_matchmaking ? "True" : "False",
                       fmt::format("{}", entry.data_size)});
      }

      std::string properties_totals;

      if (properties_list.size() > properties_list_limit.size()) {
        properties_totals =
            fmt::format("\nProperties: {}/{}", properties_list_limit.size(),
                        properties_list.size());
      }

      XELOGI("\n-------------------- PROPERTIES --------------------{}\n{}",
             properties_totals.c_str(), table.str());

      const std::vector<kernel::util::GameInfoDatabase::Context> contexts_list =
          game_info_database_->GetContexts();

      table = tabulate::Table();
      table.format().multi_byte_characters(true);
      table.add_row(
          {"ID", "Name", "Matchmaking", "Default Value", "Max Value"});

      for (const kernel::util::GameInfoDatabase::Context& entry :
           contexts_list) {
        std::string label =
            string_util::remove_eol(string_util::trim(entry.description));

        table.add_row({fmt::format("{:08X}", entry.id), label,
                       entry.is_matchmaking ? "True" : "False",
                       fmt::format("{}", entry.default_value),
                       fmt::format("{}", entry.max_value)});
      }
      XELOGI("\n-------------------- CONTEXTS --------------------\n{}",
             table.str());

      const std::vector<kernel::util::GameInfoDatabase::StatsView> stats_views =
          game_info_database_->GetStatsViews();

      // 4D5307EA SPA contains a lot of stats, limit views to log.
      const auto stats_views_limit = stats_views | std::views::take(100);

      table = tabulate::Table();
      table.format().multi_byte_characters(true);
      table.add_row({"ID", "View Type", "Name", "Skilled", "Arbitrated",
                     "Hidden", "Team View", "Online Only"});

      for (const kernel::util::GameInfoDatabase::StatsView& entry :
           stats_views_limit) {
        const std::string name =
            string_util::remove_eol(string_util::trim(entry.view.name));

        const std::string view_type =
            kernel::xam::GetViewTypeName(entry.view.view_type);

        table.add_row({fmt::format("{:08X}", entry.view.id), view_type, name,
                       entry.view.skilled ? "True" : "False",
                       entry.view.arbitrated ? "True" : "False",
                       entry.view.hidden ? "True" : "False",
                       entry.view.team_view ? "True" : "False",
                       entry.view.online_only ? "True" : "False"});
      }

      std::string stats_view_totals;

      if (stats_views.size() > stats_views_limit.size()) {
        stats_view_totals = fmt::format(
            "\nViews: {}/{}", stats_views_limit.size(), stats_views.size());
      }
      XELOGI("\n-------------------- STATS VIEWS --------------------{}\n{}",
             stats_view_totals.c_str(), table.str());

      const std::vector<kernel::util::GameInfoDatabase::PresenceMode>
          presence_modes = game_info_database_->GetPresenceModes();

      table = tabulate::Table();
      table.format().multi_byte_characters(true);
      table.add_row({"Context Value", "Contexts Count", "Properties Count"});

      for (const kernel::util::GameInfoDatabase::PresenceMode& entry :
           presence_modes) {
        table.add_row(
            {fmt::format("{}", entry.context_value),
             fmt::format("{}", entry.property_bag.contexts.size()),
             fmt::format("{}", entry.property_bag.properties.size())});
      }
      XELOGI("\n-------------------- PRESENCE MODES --------------------\n{}",
             table.str());

      auto icon_block = game_info_database_->GetIcon();
      if (!icon_block.empty()) {
        display_window_->SetIcon(icon_block.data(), icon_block.size());
      }
    }
  }

  // Initialize shader storage asynchronously - pipeline compilation happens in
  // background while the game goes through its normal startup (loading screens,
  // intro videos, etc.). With async_shader_compilation enabled, draws are
  // skipped until pipelines are ready, so this is safe. By the time actual
  // gameplay starts, most cached pipelines should be compiled.
  if (graphics_system_) {
    on_shader_storage_initialization(true);
    graphics_system_->InitializeShaderStorage(
        cache_root_, title_id_.value(), false,
        [this]() { on_shader_storage_initialization(false); });
  }

  auto main_thread = kernel_state_->LaunchModule(module);
  if (!main_thread) {
    return X_STATUS_UNSUCCESSFUL;
  }
  main_thread_ = main_thread;
  on_launch(title_id_.value(), title_name_);

  // Plugins must be loaded after calling LaunchModule() and
  // FinishLoadingUserModule() which will apply TUs and patching to the main
  // xex.
  if (cvars::allow_plugins) {
    if (plugin_loader_->IsAnyPluginForTitleAvailable(title_id_.value(),
                                                     module->hash().value())) {
      plugin_loader_->LoadTitlePlugins(title_id_.value(),
                                       module->hash().value());
    }
  }

  // Resume the main thread now.
  // If the debugger has requested a suspend this will just decrement the
  // suspend count without resuming it until the debugger wants.
  main_thread_->Resume();

  return X_STATUS_SUCCESS;
}

}  // namespace xe
