#pragma once

#include <ATen/ATen.h>

#include <array>
#include <cstdint>

namespace flagquantum_native {

using LinearLookup = std::array<std::array<uint64_t, 256>, 8>;

inline LinearLookup build_linear_lookup(const at::Tensor& images) {
  const int64_t n_wires = images.numel();
  const int64_t* image_data = images.const_data_ptr<int64_t>();
  LinearLookup lookup{};
  const int64_t chunk_count = (n_wires + 7) / 8;
  for (int64_t chunk = 0; chunk < chunk_count; ++chunk) {
    for (int64_t value = 1; value < 256; ++value) {
      const int64_t lowest = value & -value;
      const int64_t bit = __builtin_ctz(static_cast<unsigned int>(lowest));
      const int64_t position = chunk * 8 + bit;
      lookup[chunk][value] = lookup[chunk][value ^ lowest];
      if (position < n_wires) {
        lookup[chunk][value] ^=
            static_cast<uint64_t>(image_data[n_wires - position - 1]);
      }
    }
  }
  return lookup;
}

inline int64_t linear_shift_mode(const at::Tensor& images) {
  const int64_t n_wires = images.numel();
  const int64_t* image_data = images.const_data_ptr<int64_t>();
  const uint64_t state_mask = (uint64_t{1} << n_wires) - 1;
  bool shift_right = true;
  bool shift_left = true;
  bool prefix_right = true;
  bool prefix_left = true;
  for (int64_t wire = 0; wire < n_wires; ++wire) {
    const uint64_t basis = uint64_t{1} << (n_wires - wire - 1);
    const uint64_t image = static_cast<uint64_t>(image_data[wire]);
    shift_right &= image == (basis ^ (basis >> 1));
    shift_left &= image == (basis ^ ((basis << 1) & state_mask));
    uint64_t right_image = basis;
    uint64_t left_image = basis;
    for (int64_t shift = 1; shift < n_wires; shift <<= 1) {
      right_image ^= right_image >> shift;
      left_image ^= (left_image << shift) & state_mask;
    }
    prefix_right &= image == right_image;
    prefix_left &= image == left_image;
  }
  return shift_right ? 1
      : shift_left    ? 2
      : prefix_right ? 3
      : prefix_left  ? 4
                     : 0;
}

inline int64_t apply_linear_lookup(
    uint64_t destination,
    const LinearLookup& lookup,
    int64_t chunk_count,
    int64_t shift_mode,
    uint64_t state_mask) {
  if (shift_mode == 1) {
    return static_cast<int64_t>(destination ^ (destination >> 1));
  }
  if (shift_mode == 2) {
    return static_cast<int64_t>(
        destination ^ ((destination << 1) & state_mask));
  }
  if (shift_mode == 3) {
    for (int64_t shift = 1; shift < 64; shift <<= 1) {
      destination ^= destination >> shift;
    }
    return static_cast<int64_t>(destination);
  }
  if (shift_mode == 4) {
    for (int64_t shift = 1; shift < 64; shift <<= 1) {
      destination ^= (destination << shift) & state_mask;
    }
    return static_cast<int64_t>(destination);
  }
  uint64_t source = 0;
  for (int64_t chunk = 0; chunk < chunk_count; ++chunk) {
    source ^= lookup[chunk][(destination >> (chunk * 8)) & 255];
  }
  return static_cast<int64_t>(source);
}

}  // namespace flagquantum_native
