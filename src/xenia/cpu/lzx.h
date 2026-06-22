/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_CPU_LZX_H_
#define XENIA_CPU_LZX_H_

#include <string>
#include <vector>

#include "xenia/cpu/module.h"

namespace xe {
struct xex2_delta_patch;
}  // namespace xe

int lzx_decompress(const void* lzx_data, size_t lzx_len, void* dest,
                   size_t dest_len, uint32_t window_size, void* window_data,
                   size_t window_data_len);

int lzxdelta_apply_patch(xe::xex2_delta_patch* patch, size_t patch_len,
                         uint32_t window_size, void* dest);

// Streaming, block-based LZX decompressor used by the xboxkrnl LZX block
// decompression interface (LDICreateDecompression / LDIDecompress /
// LDIDestroyDecompression, the layer XMemDecompress is built on). The Xbox 360
// XCompress block format feeds a series of independently byte-aligned
// compressed blocks through a single context; the LZX window/history persists
// across blocks while the huffman trees + Intel-E8 header reset at each block.
//
// Returns an opaque handle (or nullptr on failure). uncompressed_block_size is
// the producer's block granularity (e.g. 0x8000), used to derive the LZX reset
// interval.
void* lzx_create_block_decompressor(uint32_t window_size,
                                    uint32_t uncompressed_block_size);

// Decompresses one byte-aligned block. dst_capacity is the expected
// uncompressed size of this block; *out_produced receives the bytes written.
// Returns 0 (MSPACK_ERR_OK) on success.
int lzx_decompress_block(void* handle, const void* src, size_t src_len,
                         void* dst, size_t dst_capacity, size_t* out_produced);

void lzx_free_block_decompressor(void* handle);

#endif  // XENIA_CPU_LZX_H_
