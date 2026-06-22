/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2021 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_VFS_DEVICES_NULL_DEVICE_H_
#define XENIA_VFS_DEVICES_NULL_DEVICE_H_

#include <string>

#include "xenia/vfs/device.h"

namespace xe {
namespace vfs {

class NullEntry;

class NullDevice : public Device {
 public:
  NullDevice(const std::string& mount_path,
             const std::initializer_list<std::string>& null_paths);
  ~NullDevice() override;

  bool Initialize() override;
  void Dump(StringBuffer* string_buffer) override;
  Entry* ResolvePath(const std::string_view path) override;

  bool is_read_only() const override { return false; }

  const std::string& name() const override { return name_; }
  uint32_t attributes() const override { return 0; }
  uint32_t component_name_max_length() const override { return 40; }

  // 0x2000 units * 0x80 sectors * 0x200 bytes = 512 MiB. The HDD title cache
  // partition (\Device\Harddisk0\Cache*) is backed by this NullDevice; JD2019
  // formats it as FATX for its Autodance video buffer and disables Autodance
  // (TRC_AUTODANCEREQUIRESHDD) if it looks too small. MUST stay consistent with
  // cache_size in NtDeviceIoControlFile (xboxkrnl_io.cc).
  uint32_t total_allocation_units() const override { return 0x2000; }
  uint32_t available_allocation_units() const override { return 0x2000; }

  // STFC/cache code seems to require the product of the next two to equal
  // 0x10000
  uint32_t sectors_per_allocation_unit() const override { return 0x80; }

  // STFC requires <= 0x1000
  uint32_t bytes_per_sector() const override { return 0x200; }

 private:
  std::string name_;
  std::unique_ptr<Entry> root_entry_;
  std::vector<std::string> null_paths_;
};

}  // namespace vfs
}  // namespace xe

#endif  // XENIA_VFS_DEVICES_NULL_DEVICE_H_
