/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/lzx.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <vector>

#include "xenia/base/byte_order.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/memory.h"
#include "xenia/kernel/util/xex2_info.h"

#include "third_party/mspack/lzx.h"
#include "third_party/mspack/mspack.h"

typedef struct mspack_memory_file_t {
  mspack_system sys;
  void* buffer;
  off_t buffer_size;
  off_t offset;
} mspack_memory_file;

mspack_memory_file* mspack_memory_open(mspack_system* sys, void* buffer,
                                       const size_t buffer_size) {
  assert_true(buffer_size < INT_MAX);
  if (buffer_size >= INT_MAX) {
    return NULL;
  }
  auto memfile =
      (mspack_memory_file*)std::calloc(1, sizeof(mspack_memory_file));
  if (!memfile) {
    return NULL;
  }
  memfile->buffer = buffer;
  memfile->buffer_size = (off_t)buffer_size;
  memfile->offset = 0;
  return memfile;
}

void mspack_memory_close(mspack_memory_file* file) {
  auto memfile = (mspack_memory_file*)file;
  std::free(memfile);
}

int mspack_memory_read(mspack_file* file, void* buffer, int chars) {
  auto memfile = (mspack_memory_file*)file;
  const off_t remaining = memfile->buffer_size - memfile->offset;
  const off_t total = std::min(static_cast<off_t>(chars), remaining);
  std::memcpy(buffer, (uint8_t*)memfile->buffer + memfile->offset, total);
  memfile->offset += total;
  return (int)total;
}

int mspack_memory_write(mspack_file* file, void* buffer, int chars) {
  auto memfile = (mspack_memory_file*)file;
  const off_t remaining = memfile->buffer_size - memfile->offset;
  const off_t total = std::min(static_cast<off_t>(chars), remaining);
  std::memcpy((uint8_t*)memfile->buffer + memfile->offset, buffer, total);
  memfile->offset += total;
  return (int)total;
}

void* mspack_memory_alloc(mspack_system* sys, size_t chars) {
  return std::calloc(chars, 1);
}

void mspack_memory_free(void* ptr) { std::free(ptr); }

void mspack_memory_copy(void* src, void* dest, size_t chars) {
  std::memcpy(dest, src, chars);
}

mspack_system* mspack_memory_sys_create() {
  auto sys = (mspack_system*)std::calloc(1, sizeof(mspack_system));
  if (!sys) {
    return NULL;
  }
  sys->read = mspack_memory_read;
  sys->write = mspack_memory_write;
  sys->alloc = mspack_memory_alloc;
  sys->free = mspack_memory_free;
  sys->copy = mspack_memory_copy;
  return sys;
}

void mspack_memory_sys_destroy(struct mspack_system* sys) { free(sys); }

int lzx_decompress(const void* lzx_data, size_t lzx_len, void* dest,
                   size_t dest_len, uint32_t window_size, void* window_data,
                   size_t window_data_len) {
  int result_code = 1;

  uint32_t window_bits;
  if (!xe::bit_scan_forward(window_size, &window_bits)) {
    return result_code;
  }

  mspack_system* sys = mspack_memory_sys_create();
  mspack_memory_file* lzxsrc =
      mspack_memory_open(sys, (void*)lzx_data, lzx_len);
  mspack_memory_file* lzxdst = mspack_memory_open(sys, dest, dest_len);
  lzxd_stream* lzxd = lzxd_init(sys, (mspack_file*)lzxsrc, (mspack_file*)lzxdst,
                                window_bits, 0, 0x8000, (off_t)dest_len, 0);

  if (lzxd) {
    if (window_data) {
      // zero the window and then copy window_data to the end of it
      auto padding_len = window_size - window_data_len;
      std::memset(&lzxd->window[0], 0, padding_len);
      std::memcpy(&lzxd->window[padding_len], window_data, window_data_len);
      // TODO(gibbed): should this be set regardless if source window data is
      // available or not?
      lzxd->ref_data_size = window_size;
    }

    result_code = lzxd_decompress(lzxd, (off_t)dest_len);

    lzxd_free(lzxd);
    lzxd = NULL;
  }

  if (lzxsrc) {
    mspack_memory_close(lzxsrc);
    lzxsrc = NULL;
  }

  if (lzxdst) {
    mspack_memory_close(lzxdst);
    lzxdst = NULL;
  }

  if (sys) {
    mspack_memory_sys_destroy(sys);
    sys = NULL;
  }

  return result_code;
}

// Streaming LZX decompressor (xboxkrnl LDI / XMemDecompress).
//
// The Xbox 360 XCompress "LZXNATIVE" format feeds a series of compressed blocks
// (XCOMPRESS_BLOCK_HEADER_LZXNATIVE) through ONE decompression context. The
// canonical XMemDecompress loop creates the context once and decompresses each
// block with a separate call, never resetting the context between blocks. The
// blocks are byte aligned in the file, but the LZX *coding* state PERSISTS
// across them: a single LZX block (block_type + 24-bit block_length, with its
// huffman trees) routinely spans several fed blocks (observed block_length =
// 70394 for JD2019's database.gmsodf, ~2.5 of the title's 32KB blocks). So the
// huffman trees, R0/R1/R2 LRU offsets, the once-per-stream intel-E8 header, the
// current LZX block's remaining length and the sliding window all carry over.
//
// We therefore drive one persistent lzxd_stream with reset_interval = 0 and, at
// each LDIDecompress, only REALIGN the bit reader to the new block's first byte
// (each block starts byte aligned; the previous block's trailing padding bits
// must be discarded). Everything else - huffman/R0R1R2/header_read/
// block_remaining/window/window_posn/frame_posn - is left untouched. Input is
// served from a queue drained by the mspack read callback; output is re-pointed
// at the caller's buffer each call.
struct lzx_block_decompressor {
  mspack_system* sys = nullptr;
  mspack_memory_file* out = nullptr;
  lzxd_stream* lzxd = nullptr;
  std::vector<uint8_t> in_queue;
  size_t in_read_pos = 0;
};

// mspack read callback for the input side: drains the pending input queue.
// The mspack_file* handed in is the lzx_block_decompressor itself.
static int lzx_block_in_read(mspack_file* file, void* buffer, int chars) {
  auto* d = reinterpret_cast<lzx_block_decompressor*>(file);
  if (chars < 0) {
    return -1;
  }
  size_t remaining = d->in_queue.size() - d->in_read_pos;
  size_t n = std::min(static_cast<size_t>(chars), remaining);
  if (n) {
    std::memcpy(buffer, d->in_queue.data() + d->in_read_pos, n);
    d->in_read_pos += n;
  }
  return static_cast<int>(n);
}

void* lzx_create_block_decompressor(uint32_t window_size,
                                    uint32_t uncompressed_block_size) {
  (void)uncompressed_block_size;
  uint32_t window_bits;
  if (!xe::bit_scan_forward(window_size, &window_bits)) {
    return nullptr;
  }
  // Regular (non-delta) LZX windows are 15..21 bits (32KB..2MB).
  window_bits = std::min<uint32_t>(std::max<uint32_t>(window_bits, 15), 21);

  auto* d = new lzx_block_decompressor();
  d->sys = mspack_memory_sys_create();
  if (d->sys) {
    // Input is served from the queue, not a fixed memory buffer.
    d->sys->read = lzx_block_in_read;
    // Output starts as an empty placeholder; re-pointed each block.
    static uint8_t kEmpty[1] = {0};
    d->out = mspack_memory_open(d->sys, kEmpty, 0);
  }
  if (d->sys && d->out) {
    d->lzxd = lzxd_init(d->sys, reinterpret_cast<mspack_file*>(d),
                        reinterpret_cast<mspack_file*>(d->out),
                        static_cast<int>(window_bits), /*reset_interval=*/0,
                        /*input_buffer_size=*/0x8000, /*output_length=*/0,
                        /*is_delta=*/0);
  }
  if (!d->lzxd) {
    lzx_free_block_decompressor(d);
    return nullptr;
  }
  return d;
}

int lzx_decompress_block(void* handle, const void* src, size_t src_len,
                         void* dst, size_t dst_capacity, size_t* out_produced) {
  auto* d = reinterpret_cast<lzx_block_decompressor*>(handle);
  if (out_produced) {
    *out_produced = 0;
  }
  if (!d || !d->lzxd) {
    return 1;
  }

  // Serve this block's compressed bytes from the queue.
  d->in_queue.clear();
  d->in_read_pos = 0;
  if (src && src_len) {
    const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
    d->in_queue.assign(s, s + src_len);
  }

  // Each XCOMPRESS block is byte aligned in the file, but the LZX *coding* state
  // (huffman trees, R0/R1/R2, the once-per-stream intel-E8 header, the current
  // LZX block's remaining length, and the window) PERSISTS across blocks - a
  // single LZX block routinely spans several fed blocks. The canonical
  // XMemDecompress loop never resets the context between blocks. So we ONLY
  // realign the bit reader to this block's first byte (discard the previous
  // block's byte-padding bits) and leave all coding state intact.
  lzxd_stream* lzxd = d->lzxd;
  lzxd->i_ptr = lzxd->i_end;
  lzxd->bit_buffer = 0;
  lzxd->bits_left = 0;
  lzxd->input_end = 0;

  // Point the output at the caller's destination buffer for this call.
  d->out->buffer = dst;
  d->out->buffer_size = static_cast<off_t>(dst_capacity);
  d->out->offset = 0;

  // Bound the stream length to exactly what this call requests. mspack derives
  // end_frame as (offset + out_bytes)/32KB + 1, so without a length it tries to
  // decode one extra 32KB look-ahead frame - reading input this block doesn't
  // contain. Setting length makes the per-frame size clamp turn the look-ahead
  // into a no-op and clamp a final partial frame to its real size.
  lzxd->length = lzxd->offset + static_cast<off_t>(dst_capacity);

  // Resync the frame counter with the actual output position. mspack's very
  // first decompress over-iterates by one frame, and the clamped look-ahead
  // no-op still does frame++ - so frame drifts one ahead of offset/32KB.
  // Re-deriving frame from offset before each block keeps the loop bound
  // correct.
  lzxd->frame = static_cast<unsigned int>(lzxd->offset / LZX_FRAME_SIZE);

  int rc = lzxd_decompress(lzxd, static_cast<off_t>(dst_capacity));
  if (out_produced) {
    *out_produced = static_cast<size_t>(d->out->offset);
  }
  return rc;
}

void lzx_free_block_decompressor(void* handle) {
  auto* d = reinterpret_cast<lzx_block_decompressor*>(handle);
  if (!d) {
    return;
  }
  if (d->lzxd) {
    lzxd_free(d->lzxd);
  }
  if (d->out) {
    mspack_memory_close(d->out);
  }
  if (d->sys) {
    mspack_memory_sys_destroy(d->sys);
  }
  delete d;
}

int lzxdelta_apply_patch(xe::xex2_delta_patch* patch, size_t patch_len,
                         uint32_t window_size, void* dest) {
  void* patch_end = (char*)patch + patch_len;
  auto* cur_patch = patch;

  while (patch_end > cur_patch) {
    int patch_sz = -4;  // 0 byte patches need us to remove 4 byte from next
                        // patch addr because of patch_data field
    if (cur_patch->compressed_len == 0 && cur_patch->uncompressed_len == 0 &&
        cur_patch->new_addr == 0 && cur_patch->old_addr == 0) {
      break;
    }
    switch (cur_patch->compressed_len) {
      case 0:  // fill with 0
        std::memset((char*)dest + cur_patch->new_addr, 0,
                    cur_patch->uncompressed_len);
        break;
      case 1:  // copy from old -> new
        std::memcpy((char*)dest + cur_patch->new_addr,
                    (char*)dest + cur_patch->old_addr,
                    cur_patch->uncompressed_len);
        break;
      default:  // delta patch
        patch_sz =
            cur_patch->compressed_len - 4;  // -4 because of patch_data field

        int result = lzx_decompress(
            cur_patch->patch_data, cur_patch->compressed_len,
            (char*)dest + cur_patch->new_addr, cur_patch->uncompressed_len,
            window_size, (char*)dest + cur_patch->old_addr,
            cur_patch->uncompressed_len);

        if (result) {
          return result;
        }
        break;
    }

    cur_patch++;
    cur_patch = (xe::xex2_delta_patch*)((char*)cur_patch +
                                        patch_sz);  // TODO: make this less ugly
  }

  return 0;
}
