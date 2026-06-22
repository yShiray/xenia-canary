/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <vector>

#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/string_util.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/smc.h"
#include "xenia/kernel/user_module.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xam/xam_content_device.h"
#include "xenia/kernel/xam/xam_module.h"
#include "xenia/kernel/xam/xam_private.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_module.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_threading.h"
#include "xenia/kernel/xenumerator.h"
#include "xenia/ui/imgui_dialog.h"
#include "xenia/ui/imgui_drawer.h"
#include "xenia/vfs/devices/host_path_entry.h"
#include "xenia/vfs/devices/stfs_xbox.h"
#include "xenia/vfs/devices/xcontent_devices/stfs_container_device.h"
#include "xenia/vfs/virtual_file_system.h"
#include "xenia/xbox.h"

DEFINE_int32(
    license_mask, 0,
    "Set license mask for activated content.\n"
    " 0 = No licenses enabled.\n"
    " 1 = First license enabled. Generally the full version license in\n"
    "     Xbox Live Arcade titles.\n"
    " -1 or 0xFFFFFFFF = All possible licenses enabled. Generally a\n"
    "                    bad idea, could lead to undefined behavior.",
    "Content");

namespace xe {
namespace kernel {
namespace xam {

dword_result_t XamContentGetLicenseMask_entry(lpdword_t mask_ptr,
                                              lpvoid_t overlapped_ptr) {
  if (!mask_ptr) {
    return X_ERROR_INVALID_PARAMETER;
  }

  auto run = [mask_ptr](uint32_t& extended_error, uint32_t& length) {
    X_RESULT result = X_ERROR_FUNCTION_FAILED;

    // Remark: This cannot be reflected as on console. Xenia can boot games
    // directly and XBLA games can be repacked to ZAR. For these titles we must
    // provide some license. Normally it should fail for OpticalDisc type.
    if (kernel_state()->deployment_type_ != XDeploymentType::kOther) {
      // Each bit in the mask represents a granted license. Available licenses
      // seems to vary from game to game, but most appear to use bit 0 to
      // indicate if the game is purchased or not.
      *mask_ptr = static_cast<uint32_t>(cvars::license_mask);
      result = X_ERROR_SUCCESS;
    }

    extended_error = X_HRESULT_FROM_WIN32(result);
    length = 0;
    return result;
  };

  if (!overlapped_ptr) {
    uint32_t extended_error, length;
    return run(extended_error, length);
  } else {
    kernel_state()->CompleteOverlappedDeferredEx(run, overlapped_ptr);
    return X_ERROR_IO_PENDING;
  }
}
DECLARE_XAM_EXPORT2(XamContentGetLicenseMask, kContent, kStub, kHighFrequency);

dword_result_t xeXamContentResolve(
    dword_t user_index, lpvoid_t content_data_ptr, dword_t content_data_size,
    lpstring_t path_ptr, dword_t path_size, dword_t create_directory,
    lpstring_t root_name_ptr, pointer_t<XAM_OVERLAPPED> overlapped_ptr) {
  auto run = [user_index, content_data_ptr, content_data_size, path_ptr,
              path_size, create_directory, root_name_ptr](
                 uint32_t& extended_error, uint32_t& length) -> X_RESULT {
    XCONTENT_AGGREGATE_DATA content_data;
    if (content_data_size == sizeof(XCONTENT_DATA)) {
      content_data = *content_data_ptr.as<XCONTENT_DATA*>();
    } else if (content_data_size == sizeof(XCONTENT_DATA_INTERNAL)) {
      // Due to the current implementation of content data we can't use
      // XCONTENT_DATA_INTERNAL
      content_data = *content_data_ptr.as<XCONTENT_AGGREGATE_DATA*>();
    } else {
      assert_always();
      return X_ERROR_INVALID_PARAMETER;
    }
    uint64_t xuid = 0;
    if (user_index < XUserMaxUserCount) {
      const auto profile =
          kernel_state()->xam_state()->profile_manager()->GetProfile(
              static_cast<uint8_t>(user_index));
      if (profile && content_data.content_type == XContentType::kSavedGame) {
        xuid = profile->xuid();
      }
    }

    std::string root_device_path = "";

    if (root_name_ptr) {
      // Check if root_name is valid.
      // root_device_path = std::string(root_name_ptr);
      // Unsupported for now.
      return X_ERROR_INVALID_PARAMETER;
    } else {
      if (content_data.device_id == static_cast<uint32_t>(DummyDeviceId::HDD)) {
        root_device_path = "\\Device\\Harddisk0\\Partition1\\Content\\";
      } else if (content_data.device_id ==
                 static_cast<uint32_t>(DummyDeviceId::ODD)) {
        // Or GAME, but D: usually means DVD drive meanwhile GAME always
        // pinpoints to game, even if it is running from HDD
        root_device_path = "D:\\content\\";
      } else {
        return X_ERROR_INVALID_PARAMETER;
      }
    }

    const std::string relative_path = fmt::format(
        "{:016X}\\{:08X}\\{:08X}\\{}", xuid, kernel_state()->title_id(),
        static_cast<uint32_t>(content_data.content_type.get()),
        content_data.file_name());

    string_util::copy_truncating(path_ptr, root_device_path + relative_path,
                                 path_size);

    // Check if it exists and try to mount that package
    // Result of buffer_ptr is sent to RtlInitAnsiString.
    // buffer_size is usually 260 (max path).
    return X_ERROR_SUCCESS;
  };

  if (!overlapped_ptr) {
    uint32_t extended_error, length;
    return run(extended_error, length);
  } else {
    kernel_state()->CompleteOverlappedDeferredEx(run, overlapped_ptr);
    return X_ERROR_IO_PENDING;
  }
}

dword_result_t XamContentResolve_entry(
    dword_t user_index, lpvoid_t content_data_ptr, lpstring_t path_ptr,
    dword_t path_size, dword_t create_directory, lpstring_t root_name_ptr,
    pointer_t<XAM_OVERLAPPED> overlapped_ptr) {
  return xeXamContentResolve(user_index, content_data_ptr,
                             sizeof(XCONTENT_DATA), path_ptr, path_size,
                             create_directory, root_name_ptr, overlapped_ptr);
}
DECLARE_XAM_EXPORT1(XamContentResolve, kContent, kSketchy);

dword_result_t XamContentResolveInternal_entry(
    lpvoid_t content_data_ptr, lpstring_t path_ptr, dword_t path_size,
    dword_t create_directory, lpstring_t root_name_ptr,
    pointer_t<XAM_OVERLAPPED> overlapped_ptr) {
  return xeXamContentResolve(
      XUserIndexNone, content_data_ptr, sizeof(XCONTENT_DATA_INTERNAL),
      path_ptr, path_size, create_directory, root_name_ptr, overlapped_ptr);
}
DECLARE_XAM_EXPORT1(XamContentResolveInternal, kContent, kSketchy);

// https://github.com/MrColdbird/gameservice/blob/master/ContentManager.cpp
dword_result_t XamContentCreateEnumeratorInternal_entry(
    qword_t xuid, dword_t device_id, dword_t content_type, dword_t title_id,
    dword_t content_flags, dword_t items_per_enumerate,
    lpdword_t buffer_size_ptr, lpdword_t handle_out) {
  assert_not_null(handle_out);

  auto device_info = device_id == 0 ? nullptr : GetDummyDeviceInfo(device_id);
  if ((device_id && device_info == nullptr) || !handle_out) {
    if (buffer_size_ptr) {
      *buffer_size_ptr = 0;
    }

    // TODO(benvanik): memset 0 the data?
    return X_E_INVALIDARG;
  }

  if (buffer_size_ptr) {
    *buffer_size_ptr = sizeof(XCONTENT_DATA) * items_per_enumerate;
  }

  auto e = object_ref<ContentEnumerator>(
      new ContentEnumerator(kernel_state(), items_per_enumerate));

  auto result =
      e->Initialize(XUserIndexAny, 0xFE, 0x20005, 0x20007, 0, 0x98, nullptr);
  if (XFAILED(result)) {
    return result;
  }

  uint32_t title = title_id;
  if (!title) {
    title = kernel_state()->title_id();
  }

  std::vector<XCONTENT_AGGREGATE_DATA> enumerated_content =
      {};  // XCONTENT_DATA_INTERNAL

  if (!device_info || device_info->device_id == DummyDeviceId::HDD) {
    std::vector<uint64_t> xuids_to_enumerate = {};
    if (xuid) {
      xuids_to_enumerate.push_back(xuid);
    }

    if (!xuid || !(content_flags & vfs::XContentFlag::kExcludeCommon)) {
      xuids_to_enumerate.push_back(0);  // Common content
    }

    for (const auto& xuid : xuids_to_enumerate) {
      auto user_enumerated_data =
          kernel_state()->content_manager()->ListContent(
              static_cast<uint32_t>(DummyDeviceId::HDD), xuid, title,
              static_cast<XContentType>(content_type.value()));

      enumerated_content.insert(enumerated_content.end(),
                                user_enumerated_data.cbegin(),
                                user_enumerated_data.cend());
    }

    // Remove duplicates
    enumerated_content.erase(
        std::unique(enumerated_content.begin(), enumerated_content.end()),
        enumerated_content.end());
  }

  if (!device_info || device_info->device_id == DummyDeviceId::ODD) {
    auto disc_enumerated_data =
        kernel_state()->content_manager()->ListContentODD(
            static_cast<uint32_t>(DummyDeviceId::ODD), 0, title,
            XContentType(uint32_t(content_type)));

    enumerated_content.insert(enumerated_content.end(),
                              disc_enumerated_data.cbegin(),
                              disc_enumerated_data.cend());
  }

  for (const auto& content_data : enumerated_content) {
    e->AppendItem(content_data);
    XELOGI("{}: Adding: {} (Filename: {}) to enumerator result", __func__,
           xe::to_utf8(content_data.display_name()), content_data.file_name());
  }

  XELOGD("XamContentCreateEnumeratorInternal: added {} items to enumerator",
         e->item_count());

  *handle_out = e->handle();
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamContentCreateEnumeratorInternal, kContent, kImplemented);

dword_result_t XamContentCreateEnumerator_entry(
    dword_t user_index, dword_t device_id, dword_t content_type,
    dword_t content_flags, dword_t items_per_enumerate,
    lpdword_t buffer_size_ptr, lpdword_t handle_out) {
  uint64_t xuid = 0;
  if (user_index < XUserMaxUserCount) {
    const auto& user = kernel_state()->xam_state()->GetUserProfile(user_index);

    if (!user) {
      return X_ERROR_NO_SUCH_USER;
    }

    xuid = user->xuid();
  }

  return XamContentCreateEnumeratorInternal_entry(
      xuid, device_id, content_type, 0, content_flags, items_per_enumerate,
      buffer_size_ptr, handle_out);
}
DECLARE_XAM_EXPORT1(XamContentCreateEnumerator, kContent, kImplemented);

enum class kDispositionState : uint32_t { Unknown = 0, Create = 1, Open = 2 };

dword_result_t xeXamContentCreate(dword_t user_index, lpstring_t root_name,
                                  lpvoid_t content_data_ptr,
                                  dword_t content_data_size, dword_t flags,
                                  lpdword_t disposition_ptr,
                                  lpdword_t license_mask_ptr,
                                  dword_t cache_size, qword_t content_size,
                                  pointer_t<XAM_OVERLAPPED> overlapped_ptr) {
  uint64_t xuid = 0;
  if (user_index != XUserIndexNone && user_index != XUserIndexAny) {
    const auto& user = kernel_state()->xam_state()->GetUserProfile(user_index);

    if (!user) {
      return X_ERROR_NO_SUCH_USER;
    }

    xuid = user->xuid();
  }

  if (!root_name || *root_name == '\0') {
    return X_ERROR_INVALID_NAME;
  }

  // Check if we have something under provided symlink.
  std::string symlink_path = root_name.value();
  if (!symlink_path.ends_with(':')) {
    symlink_path += ':';
  }

  if (kernel_state()->file_system()->IsSymbolicLinkRegistered(symlink_path)) {
    return X_ERROR_INVALID_PARAMETER;
  }

  XCONTENT_AGGREGATE_DATA content_data;
  if (content_data_size == sizeof(XCONTENT_DATA)) {
    content_data = *content_data_ptr.as<XCONTENT_DATA*>();
  } else if (content_data_size == sizeof(XCONTENT_AGGREGATE_DATA)) {
    content_data = *content_data_ptr.as<XCONTENT_AGGREGATE_DATA*>();
  } else {
    assert_always();
    return X_ERROR_INVALID_PARAMETER;
  }

  if (content_data.content_type == XContentType::kMarketplaceContent) {
    xuid = 0;
  }

  auto content_manager = kernel_state()->content_manager();

  if (overlapped_ptr && disposition_ptr) {
    *disposition_ptr = 0;
  }

  auto run = [content_manager, xuid, root_name = root_name.value(), flags,
              content_data, disposition_ptr, license_mask_ptr, overlapped_ptr](
                 uint32_t& extended_error, uint32_t& length) -> X_RESULT {
    X_RESULT result = X_ERROR_INVALID_PARAMETER;
    kDispositionState disposition = kDispositionState::Unknown;
    switch (flags & 0xF) {
      case vfs::XContentFlag::kCreateNew:
        // Fail if exists.
        if (content_manager->ContentExists(xuid, content_data)) {
          result = X_ERROR_ALREADY_EXISTS;
        } else {
          disposition = kDispositionState::Create;
        }
        break;
      case vfs::XContentFlag::kCreateAlways:
        // Overwrite existing, if any.
        if (content_manager->ContentExists(xuid, content_data)) {
          content_manager->DeleteContent(xuid, content_data);
        }
        disposition = kDispositionState::Create;
        break;
      case vfs::XContentFlag::kOpenExisting:
        // Open only if exists.
        if (!content_manager->ContentExists(xuid, content_data)) {
          result = X_ERROR_PATH_NOT_FOUND;
        } else {
          disposition = kDispositionState::Open;
        }
        break;
      case vfs::XContentFlag::kOpenAlways:
        // Create if needed.
        if (!content_manager->ContentExists(xuid, content_data)) {
          disposition = kDispositionState::Create;
        } else {
          disposition = kDispositionState::Open;
        }
        break;
      case vfs::XContentFlag::kTruncateExisting:
        // Fail if doesn't exist, if does exist delete and recreate.
        if (!content_manager->ContentExists(xuid, content_data)) {
          result = X_ERROR_PATH_NOT_FOUND;
        } else {
          content_manager->DeleteContent(xuid, content_data);
          disposition = kDispositionState::Create;
        }
        break;
      default:
        assert_unhandled_case(flags & 0xF);
        break;
    }

    uint32_t content_license = 0;
    if (disposition == kDispositionState::Create) {
      result = content_manager->CreateContent(root_name, xuid, content_data);
      if (XSUCCEEDED(result)) {
        content_manager->WriteContentHeaderFile(xuid, content_data);
      }
    } else if (disposition == kDispositionState::Open) {
      result = content_manager->OpenContent(root_name, xuid, content_data,
                                            content_license);
    }

    if (license_mask_ptr && XSUCCEEDED(result)) {
      *license_mask_ptr = content_license;
    }

    extended_error = X_HRESULT_FROM_WIN32(result);
    length = static_cast<uint32_t>(disposition);

    if (disposition_ptr) {
      *disposition_ptr = static_cast<uint32_t>(disposition);
    }

    if (result && overlapped_ptr) {
      result = X_ERROR_FUNCTION_FAILED;
    }
    return result;
  };

  if (!overlapped_ptr) {
    uint32_t extended_error, length;
    return run(extended_error, length);
  } else {
    kernel_state()->CompleteOverlappedDeferredEx(run, overlapped_ptr);
    return X_ERROR_IO_PENDING;
  }
}

dword_result_t XamContentCreateEx_entry(
    dword_t user_index, lpstring_t root_name, lpvoid_t content_data_ptr,
    dword_t flags, lpdword_t disposition_ptr, lpdword_t license_mask_ptr,
    dword_t cache_size, qword_t content_size,
    pointer_t<XAM_OVERLAPPED> overlapped_ptr) {
  auto content_data = *content_data_ptr.as<XCONTENT_DATA*>();
  if (content_data.file_name_raw[0] == '\0') {
    return X_ERROR_INVALID_NAME;
  }
  return xeXamContentCreate(user_index, root_name, content_data_ptr,
                            sizeof(XCONTENT_DATA), flags, disposition_ptr,
                            license_mask_ptr, cache_size, content_size,
                            overlapped_ptr);
}
DECLARE_XAM_EXPORT1(XamContentCreateEx, kContent, kImplemented);

dword_result_t XamContentCreate_entry(
    dword_t user_index, lpstring_t root_name, lpvoid_t content_data_ptr,
    dword_t flags, lpdword_t disposition_ptr, lpdword_t license_mask_ptr,
    pointer_t<XAM_OVERLAPPED> overlapped_ptr) {
  return XamContentCreateEx_entry(user_index, root_name, content_data_ptr,
                                  flags, disposition_ptr, license_mask_ptr, 0,
                                  0, overlapped_ptr);
}
DECLARE_XAM_EXPORT1(XamContentCreate, kContent, kImplemented);

dword_result_t XamContentCreateInternal_entry(
    lpstring_t root_name, lpvoid_t content_data_ptr, dword_t flags,
    lpdword_t disposition_ptr, lpdword_t license_mask_ptr, dword_t cache_size,
    qword_t content_size, pointer_t<XAM_OVERLAPPED> overlapped_ptr) {
  auto content_data_internal = *content_data_ptr.as<XCONTENT_AGGREGATE_DATA*>();
  if (content_data_internal.file_name_raw[0] == '\0') {
    return X_ERROR_INVALID_NAME;
  }
  return xeXamContentCreate(XUserIndexNone, root_name, content_data_ptr,
                            sizeof(XCONTENT_AGGREGATE_DATA), flags,
                            disposition_ptr, license_mask_ptr, cache_size,
                            content_size, overlapped_ptr);
}
DECLARE_XAM_EXPORT1(XamContentCreateInternal, kContent, kImplemented);

static bool IsStfsContainerFile(const std::filesystem::path& host_path) {
  std::ifstream file(host_path, std::ios::binary);
  if (!file) {
    return false;
  }
  char magic[4] = {0};
  file.read(magic, sizeof(magic));
  if (file.gcount() != sizeof(magic)) {
    return false;
  }
  // STFS package magics: console signed (CON ), retail signed (LIVE),
  // and system signed (PIRS).
  return std::memcmp(magic, "CON ", 4) == 0 ||
         std::memcmp(magic, "LIVE", 4) == 0 ||
         std::memcmp(magic, "PIRS", 4) == 0;
}

// When the content file is NOT host-backed (e.g. it lives inside a disc image),
// it can't be opened by host path to be mounted as an STFS container. Read it
// through the VFS and, if it is an STFS package (CON/LIVE/PIRS), copy it to a
// temp host file so the host-path mount path below can use it. This is what lets
// a DISC-based Kinect title (e.g. Dance Central 3 from an ISO) mount its NUI
// skeleton database (Database.xmplr -> stexemplar:\database.gmsodf); host-folder
// installs (like the Just Dance 2019 folder) mount the file directly.
static std::filesystem::path ExtractStfsContainerToTempFile(vfs::Entry* entry) {
  if (!entry) {
    return {};
  }
  size_t size = entry->size();
  if (size < 4) {
    return {};
  }
  vfs::File* in_file = nullptr;
  if (entry->Open(vfs::FileAccess::kFileReadData, &in_file) !=
          X_STATUS_SUCCESS ||
      !in_file) {
    return {};
  }
  std::vector<uint8_t> data(size);
  size_t bytes_read = 0;
  X_STATUS st = in_file->ReadSync(std::span<uint8_t>(data.data(), data.size()),
                                  0, &bytes_read);
  in_file->Destroy();
  if (st != X_STATUS_SUCCESS || bytes_read < 4) {
    return {};
  }
  // STFS magic: 'CON ' / 'LIVE' / 'PIRS'.
  if (!((data[0] == 'C' && data[1] == 'O' && data[2] == 'N' &&
         data[3] == ' ') ||
        (data[0] == 'L' && data[1] == 'I' && data[2] == 'V' &&
         data[3] == 'E') ||
        (data[0] == 'P' && data[1] == 'I' && data[2] == 'R' &&
         data[3] == 'S'))) {
    return {};
  }
  std::error_code ec;
  std::filesystem::path dir =
      std::filesystem::temp_directory_path(ec) / "xenia_nui_stfs";
  std::filesystem::create_directories(dir, ec);
  static std::atomic<uint32_t> ctr{0};
  std::filesystem::path out_path =
      dir / ("stfs_" + std::to_string(ctr.fetch_add(1)) + ".bin");
  std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return {};
  }
  out.write(reinterpret_cast<const char*>(data.data()), bytes_read);
  out.close();
  XELOGI(
      "XamContentOpenFile: extracted disc-backed STFS container ({} bytes) to a "
      "temp host file for mounting",
      bytes_read);
  return out_path;
}

dword_result_t XamContentOpenFile_entry(
    dword_t user_index, lpstring_t root_name, lpstring_t path, dword_t flags,
    lpdword_t disposition_ptr, lpdword_t license_mask_ptr,
    pointer_t<XAM_OVERLAPPED> overlapped_ptr) {
  // Opens an existing on-disc/content file and maps it to root_name:.
  // This is what JD2019's NuiInitialize uses to open the NUI databases
  // (game:\Database.xmplr, NuiIdentity.bin.be, nuisp<locale>). Those files are
  // STFS content packages (PIRS), and the title then reads the data files that
  // live *inside* them (e.g. stexemplar:\database.gmsodf,
  // nuispeech:\speech\en-us\l1033.ini). So we MOUNT the package as a device
  // under root_name: rather than just symlinking the root to the package file;
  // otherwise the inner reads fail and NuiInitialize bails before starting the
  // NUI skeleton processor. The original Xenia stub always returned
  // FILE_NOT_FOUND.
  X_RESULT result = X_ERROR_FILE_NOT_FOUND;
  uint32_t disposition = 0;  // 0 = not found / not opened.
  uint32_t license_mask = 0;

  std::string path_str = path ? std::string(path.value()) : std::string();
  std::string root_str =
      root_name ? std::string(root_name.value()) : std::string();

  auto fs = kernel_state()->file_system();
  if (fs && !path_str.empty()) {
    auto entry = fs->ResolvePath(path_str);
    if (entry) {
      bool mapped = false;
      std::string link_name = root_str.empty() ? std::string() : root_str + ":";

      // If we've already mapped this root (the title re-opens), reuse it.
      std::string existing_target;
      if (!link_name.empty() &&
          fs->FindSymbolicLink(link_name, existing_target)) {
        mapped = true;
      }

      // Host-backed files mount directly by host path; disc-image (or other
      // non-host) backed files are extracted to a temp host file first.
      std::filesystem::path host_path;
      if (auto* host_entry = dynamic_cast<vfs::HostPathEntry*>(entry)) {
        host_path = host_entry->host_path();
      } else {
        host_path = ExtractStfsContainerToTempFile(entry);
      }

      if (!mapped && !link_name.empty() && !host_path.empty() &&
          IsStfsContainerFile(host_path)) {
        static std::atomic<uint32_t> mount_counter{0};
        std::string mount_path =
            "\\Device\\XamContent" + std::to_string(mount_counter++);
        auto device =
            std::make_unique<vfs::StfsContainerDevice>(mount_path, host_path);
        if (device->Initialize() && fs->RegisterDevice(std::move(device)) &&
            fs->RegisterSymbolicLink(link_name, mount_path)) {
          mapped = true;
          XELOGI("XamContentOpenFile mounted STFS '{}' as {} ({})", path_str,
                 link_name, mount_path);
        } else {
          XELOGW("XamContentOpenFile failed to mount STFS '{}'", path_str);
        }
      }

      // Fall back for non-package files: map the root straight to the file, in
      // case the title reads it back through the supplied root. Harmless if
      // unused (unique prefix, never shadows game:/update:).
      if (!mapped && !link_name.empty()) {
        fs->RegisterSymbolicLink(link_name, path_str);
      }

      disposition = 1;  // Existing file opened.
      license_mask = cvars::license_mask ? cvars::license_mask : 0xFFFFFFFFu;
      result = X_ERROR_SUCCESS;
      XELOGI("XamContentOpenFile('{}', '{}') -> SUCCESS", root_str, path_str);
    } else {
      XELOGW("XamContentOpenFile('{}', '{}') -> file not found", root_str,
             path_str);
    }
  }

  if (disposition_ptr) {
    *disposition_ptr = disposition;
  }
  if (license_mask_ptr) {
    *license_mask_ptr = license_mask;
  }

  if (overlapped_ptr) {
    kernel_state()->CompleteOverlappedImmediate(overlapped_ptr, result);
    return X_ERROR_IO_PENDING;
  }
  return result;
}
DECLARE_XAM_EXPORT1(XamContentOpenFile, kContent, kImplemented);

dword_result_t XamContentFlush_entry(lpstring_t root_name,
                                     pointer_t<XAM_OVERLAPPED> overlapped_ptr) {
  X_RESULT result = X_ERROR_SUCCESS;
  if (overlapped_ptr) {
    kernel_state()->CompleteOverlappedImmediate(overlapped_ptr, result);
    return X_ERROR_IO_PENDING;
  } else {
    return result;
  }
}
DECLARE_XAM_EXPORT1(XamContentFlush, kContent, kStub);

dword_result_t XamContentClose_entry(lpstring_t root_name,
                                     pointer_t<XAM_OVERLAPPED> overlapped_ptr) {
  // Closes a previously opened root from XamContentCreate*.
  auto result =
      kernel_state()->content_manager()->CloseContent(root_name.value());

  if (overlapped_ptr) {
    kernel_state()->CompleteOverlappedImmediate(overlapped_ptr, result);
    return X_ERROR_IO_PENDING;
  } else {
    return result;
  }
}
DECLARE_XAM_EXPORT1(XamContentClose, kContent, kImplemented);

dword_result_t XamContentGetCreator_entry(
    dword_t user_index, lpvoid_t content_data_ptr, lpdword_t is_creator_ptr,
    lpqword_t creator_xuid_ptr, pointer_t<XAM_OVERLAPPED> overlapped_ptr) {
  if (!is_creator_ptr) {
    return X_ERROR_INVALID_PARAMETER;
  }

  const auto& user = kernel_state()->xam_state()->GetUserProfile(user_index);

  if (!user) {
    return X_ERROR_NO_SUCH_USER;
  }

  XCONTENT_AGGREGATE_DATA content_data = *content_data_ptr.as<XCONTENT_DATA*>();

  auto run = [content_data, xuid = user->xuid(), user_index, is_creator_ptr,
              creator_xuid_ptr, overlapped_ptr](uint32_t& extended_error,
                                                uint32_t& length) -> X_RESULT {
    X_RESULT result = X_ERROR_SUCCESS;

    bool content_exists =
        kernel_state()->content_manager()->ContentExists(xuid, content_data);

    if (content_exists) {
      if (content_data.content_type == XContentType::kSavedGame) {
        // User always creates saves.
        *is_creator_ptr = 1;
        if (creator_xuid_ptr) {
          if (kernel_state()->xam_state()->IsUserSignedIn(user_index)) {
            const auto& user_profile =
                kernel_state()->xam_state()->GetUserProfile(user_index);
            *creator_xuid_ptr = user_profile->xuid();
          } else {
            result = X_ERROR_NO_SUCH_USER;
          }
        }
      } else {
        *is_creator_ptr = 0;
        if (creator_xuid_ptr) {
          *creator_xuid_ptr = 0;
        }
      }
    } else {
      result = X_ERROR_PATH_NOT_FOUND;
    }

    extended_error = X_HRESULT_FROM_WIN32(result);
    length = 0;

    if (result && overlapped_ptr) {
      result = X_ERROR_FUNCTION_FAILED;
    }

    return result;
  };

  if (!overlapped_ptr) {
    uint32_t extended_error, length;
    return run(extended_error, length);
  } else {
    kernel_state()->CompleteOverlappedDeferredEx(run, overlapped_ptr);
    return X_ERROR_IO_PENDING;
  }
}
DECLARE_XAM_EXPORT1(XamContentGetCreator, kContent, kImplemented);

dword_result_t XamContentGetThumbnail_entry(
    dword_t user_index, lpvoid_t content_data_ptr, lpvoid_t buffer_ptr,
    lpdword_t buffer_size_ptr, pointer_t<XAM_OVERLAPPED> overlapped_ptr) {
  const auto& user = kernel_state()->xam_state()->GetUserProfile(user_index);

  if (!user) {
    return X_ERROR_NO_SUCH_USER;
  }

  assert_not_null(buffer_size_ptr);
  uint32_t buffer_size = *buffer_size_ptr;
  XCONTENT_AGGREGATE_DATA content_data = *content_data_ptr.as<XCONTENT_DATA*>();

  // Get thumbnail (if it exists).
  std::vector<uint8_t> buffer;
  auto result = kernel_state()->content_manager()->GetContentThumbnail(
      user->xuid(), content_data, &buffer);

  *buffer_size_ptr = uint32_t(buffer.size());

  if (XSUCCEEDED(result)) {
    // Write data, if we were given a pointer.
    // This may have just been a size query.
    if (buffer_ptr) {
      if (buffer_size < buffer.size()) {
        // Dest buffer too small.
        result = X_ERROR_INSUFFICIENT_BUFFER;
      } else {
        // Copy data.
        std::memcpy((uint8_t*)buffer_ptr, buffer.data(), buffer.size());
      }
    }
  }

  if (overlapped_ptr) {
    kernel_state()->CompleteOverlappedImmediate(overlapped_ptr, result);
    return X_ERROR_IO_PENDING;
  } else {
    return result;
  }
}
DECLARE_XAM_EXPORT1(XamContentGetThumbnail, kContent, kImplemented);

dword_result_t XamContentSetThumbnail_entry(
    dword_t user_index, lpvoid_t content_data_ptr, lpvoid_t buffer_ptr,
    dword_t buffer_size, pointer_t<XAM_OVERLAPPED> overlapped_ptr) {
  const auto& user = kernel_state()->xam_state()->GetUserProfile(user_index);

  if (!user) {
    return X_ERROR_NO_SUCH_USER;
  }

  XCONTENT_AGGREGATE_DATA content_data = *content_data_ptr.as<XCONTENT_DATA*>();

  // Buffer is PNG data.
  auto buffer = std::vector<uint8_t>((uint8_t*)buffer_ptr,
                                     (uint8_t*)buffer_ptr + buffer_size);
  auto result = kernel_state()->content_manager()->SetContentThumbnail(
      user->xuid(), content_data, std::move(buffer));

  if (overlapped_ptr) {
    kernel_state()->CompleteOverlappedImmediate(overlapped_ptr, result);
    return X_ERROR_IO_PENDING;
  } else {
    return result;
  }
}
DECLARE_XAM_EXPORT1(XamContentSetThumbnail, kContent, kImplemented);

dword_result_t xeXamContentDelete(dword_t user_index, lpvoid_t content_data_ptr,
                                  dword_t content_data_size,
                                  pointer_t<XAM_OVERLAPPED> overlapped_ptr) {
  uint64_t xuid = 0;
  XCONTENT_AGGREGATE_DATA content_data = *content_data_ptr.as<XCONTENT_DATA*>();
  if (content_data_size == sizeof(XCONTENT_AGGREGATE_DATA)) {
    content_data = *content_data_ptr.as<XCONTENT_AGGREGATE_DATA*>();
  }

  if (user_index != XUserIndexNone) {
    const auto& user = kernel_state()->xam_state()->GetUserProfile(user_index);

    if (!user) {
      return X_ERROR_NO_SUCH_USER;
    }

    xuid = user->xuid();
  }

  auto result =
      kernel_state()->content_manager()->DeleteContent(xuid, content_data);

  if (overlapped_ptr) {
    kernel_state()->CompleteOverlappedImmediate(overlapped_ptr, result);
    return X_ERROR_IO_PENDING;
  } else {
    return result;
  }
}

dword_result_t XamContentDelete_entry(
    dword_t user_index, lpvoid_t content_data_ptr,
    pointer_t<XAM_OVERLAPPED> overlapped_ptr) {
  return xeXamContentDelete(user_index, content_data_ptr, sizeof(XCONTENT_DATA),
                            overlapped_ptr);
}
DECLARE_XAM_EXPORT1(XamContentDelete, kContent, kImplemented);

dword_result_t XamContentDeleteInternal_entry(
    lpvoid_t content_data_ptr, pointer_t<XAM_OVERLAPPED> overlapped_ptr) {
  // INFO: Analysis of xam.xex shows that "internal" functions are wrappers with
  // 0xFE as user_index.
  // In XAM content size is set to 0x200.
  return xeXamContentDelete(XUserIndexNone, content_data_ptr,
                            sizeof(XCONTENT_AGGREGATE_DATA), overlapped_ptr);
}
DECLARE_XAM_EXPORT1(XamContentDeleteInternal, kContent, kImplemented);

typedef struct {
  xe::be<uint32_t> stringTitlePtr;
  xe::be<uint32_t> stringTextPtr;
  xe::be<uint32_t> stringBtnMsgPtr;
} X_SWAPDISC_ERROR_MESSAGE;
static_assert_size(X_SWAPDISC_ERROR_MESSAGE, 12);

dword_result_t XamSwapDisc_entry(
    dword_t disc_number, pointer_t<X_KEVENT> completion_handle,
    pointer_t<X_SWAPDISC_ERROR_MESSAGE> error_message) {
  xex2_opt_execution_info* info = nullptr;
  kernel_state()->GetExecutableModule()->GetOptHeader(XEX_HEADER_EXECUTION_INFO,
                                                      &info);

  if (info->disc_number > info->disc_count) {
    return X_ERROR_INVALID_PARAMETER;
  }

  auto completion_event = [completion_handle]() -> void {
    auto kevent = xboxkrnl::xeKeSetEvent(completion_handle, 1, 0);

    // Release the completion handle
    auto object =
        XObject::GetNativeObject<XObject>(kernel_state(), completion_handle);
    if (object) {
      object->Retain();
    }
  };

  if (info->disc_number == disc_number) {
    completion_event();
    return X_ERROR_SUCCESS;
  }

  auto filesystem = kernel_state()->file_system();
  auto mount_path = "\\Device\\LauncherData";

  if (filesystem->ResolvePath(mount_path) != NULL) {
    filesystem->UnregisterDevice(mount_path);
  }

  std::u16string text_message = xe::load_and_swap<std::u16string>(
      kernel_state()->memory()->TranslateVirtual(error_message->stringTextPtr));

  const std::filesystem::path new_disc_path =
      kernel_state()->emulator()->GetNewDiscPath(xe::to_utf8(text_message));
  XELOGI("GetNewDiscPath returned path {}.", new_disc_path.string().c_str());

  // TODO(Gliniak): Implement checking if inserted file is requested one
  kernel_state()->emulator()->MountPath(new_disc_path, mount_path);
  completion_event();

  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamSwapDisc, kContent, kSketchy);

dword_result_t XamLoaderGetDvdTrayState_entry() {
  return static_cast<uint8_t>(kernel_state()->smc()->GetTrayState());
}
DECLARE_XAM_EXPORT1(XamLoaderGetDvdTrayState, kNone, kImplemented);

void XamLoaderGetMediaInfoEx_entry(lpdword_t media_type, lpdword_t unk2,
                                   lpdword_t unk3) {
  if (media_type) {
    *media_type = X_DVD_DISC_STATE::XBOX_360_GAME_DISC;
  }
  if (unk2) {
    *unk2 = 0;
  }
  if (unk3) {
    *unk3 = 0;
  }
}
DECLARE_XAM_EXPORT1(XamLoaderGetMediaInfoEx, kNone, kStub);

void XamLoaderGetMediaInfo_entry(lpdword_t media_type, lpdword_t unk2) {
  XamLoaderGetMediaInfoEx_entry(media_type, unk2, 0);
}
DECLARE_XAM_EXPORT1(XamLoaderGetMediaInfo, kNone, kStub);

dword_result_t xeXamContentLaunchImage(dword_t user_index,
                                       lpstring_t image_location,
                                       lpvoid_t content_data_ptr,
                                       dword_t content_data_size,
                                       lpstring_t xex_path, dword_t flag) {
  /* Notes:
      - In code this subfunction is used by all XamContentLaunchImage
     functions
      - Due to the current implementation of content data we can't use
     XCONTENT_DATA_INTERNAL
      - flags used by XamLoaderLaunchTitleEx
      - user_index used by xeXamContentOpenFile
      - if image_location null use xeXamContentOpenFile else use
     exXamContentCreate
      - root_name is "XSYSLAUNCH" while XamLoaderLaunchTitleEx uses
     "XSYSLAUNCH:\\""
      - title_id is usually written into first 8 characters of filename
  */
  vfs::Entry* entry;
  if (!image_location) {
    XCONTENT_AGGREGATE_DATA content_data =
        *content_data_ptr.as<XCONTENT_DATA*>();
    const uint32_t title_id = xe::string_util::from_string<uint32_t>(
        content_data.file_name().substr(0, 8), true);

    // This should be done via content_manager, however as it isn't capable of
    // such action we need to improvise.
    const std::string package_path =
        fmt::format("GAME:/Content/0000000000000000/{:08X}/{:08X}/{}", title_id,
                    static_cast<uint32_t>(content_data.content_type.get()),
                    content_data.file_name());

    entry = kernel_state()->file_system()->ResolvePath(package_path);
  } else {
    entry = kernel_state()->file_system()->ResolvePath(image_location.value());
  }

  if (!entry) {
    return X_STATUS_NO_SUCH_FILE;
  }

  const std::filesystem::path host_path =
      kernel_state()->emulator()->content_root() / entry->name();

  if (!std::filesystem::exists(host_path)) {
    uint64_t progress = 0;
    vfs::VirtualFileSystem::ExtractContentFile(
        entry, kernel_state()->emulator()->content_root(), progress, true);
  }

  auto xam = kernel_state()->GetKernelModule<XamModule>("xam.xex");

  auto& loader_data = xam->loader_data();
  loader_data.host_path = xe::path_to_utf8(host_path);
  loader_data.launch_path = xex_path.value();

  xam->SaveLoaderData();

  auto display_window = kernel_state()->emulator()->display_window();
  auto imgui_drawer = kernel_state()->emulator()->imgui_drawer();

  if (display_window && imgui_drawer) {
    display_window->app_context().CallInUIThreadSynchronous([imgui_drawer]() {
      xe::ui::ImGuiDialog::ShowMessageBox(
          imgui_drawer, "Launching new title!",
          "Launching new title. \nPlease close Xenia and launch it again. Game "
          "should load automatically.");
    });
  }

  kernel_state()->TerminateTitle();
  return X_ERROR_SUCCESS;
}

dword_result_t XamContentLaunchImageFromFileInternal_entry(
    lpstring_t image_location, lpstring_t xex_name) {
  return xeXamContentLaunchImage(XUserIndexNone, image_location, nullptr, NULL,
                                 xex_name, NULL);
}
DECLARE_XAM_EXPORT1(XamContentLaunchImageFromFileInternal, kContent, kStub);

dword_result_t XamContentLaunchImage_entry(dword_t user_index,
                                           lpvoid_t content_data_ptr,
                                           lpstring_t xex_path) {
  return xeXamContentLaunchImage(user_index, nullptr, content_data_ptr,
                                 sizeof(XCONTENT_DATA), xex_path, NULL);
}
DECLARE_XAM_EXPORT1(XamContentLaunchImage, kContent, kStub);

dword_result_t XamContentLaunchImageInternal_entry(lpvoid_t content_data_ptr,
                                                   lpstring_t xex_path) {
  return xeXamContentLaunchImage(XUserIndexNone, nullptr, content_data_ptr,
                                 sizeof(XCONTENT_DATA_INTERNAL), xex_path,
                                 NULL);
}
DECLARE_XAM_EXPORT1(XamContentLaunchImageInternal, kContent, kStub);

dword_result_t XamContentLaunchImageInternalEx_entry(lpvoid_t content_data_ptr,
                                                     lpstring_t xex_path,
                                                     dword_t flags) {
  return xeXamContentLaunchImage(XUserIndexNone, nullptr, content_data_ptr,
                                 sizeof(XCONTENT_DATA_INTERNAL), xex_path,
                                 flags);
}
DECLARE_XAM_EXPORT1(XamContentLaunchImageInternalEx, kContent, kStub);

void XamContentRegisterChangeCallback_entry(dword_t callback) {
  kernel_state()->xam_state()->SetContentRegisterCallback(callback);
}
DECLARE_XAM_EXPORT1(XamContentRegisterChangeCallback, kContent, kImplemented);

dword_result_t XamContentGetDeviceVolumePath_entry(dword_t device_id,
                                                   lpvoid_t path_ptr,
                                                   dword_t path_size,
                                                   dword_t append_backslash) {
  std::string volume_path = "hdd0\\";
  if (device_id != static_cast<uint32_t>(DummyDeviceId::HDD)) {
    return X_ERROR_FUNCTION_FAILED;
  }

  char* path =
      kernel_memory()->TranslateVirtual<char*>(path_ptr.guest_address());

  string_util::copy_truncating(path, volume_path, path_size);

  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamContentGetDeviceVolumePath, kContent, kStub);
}  // namespace xam
}  // namespace kernel
}  // namespace xe

DECLARE_XAM_EMPTY_REGISTER_EXPORTS(Content);
