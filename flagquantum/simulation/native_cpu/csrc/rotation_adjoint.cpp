#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/util/complex.h>
#include <torch/library.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <vector>

#include "linear_permutation.h"

// GCC 11 rejects OpenMP SIMD directives nested in the ATen dispatch and
// parallel lambdas below, even though newer GCC and Clang accept them. Keep
// those loops scalar on GCC 11 and older so the native CPU extension remains
// buildable on Ubuntu 22.04 without changing their numerical semantics.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 12
#define FQ_OPENMP_SIMD_PRAGMA(...)
#else
#define FQ_OPENMP_PRAGMA_IMPL(value) _Pragma(#value)
#define FQ_OPENMP_PRAGMA(value) FQ_OPENMP_PRAGMA_IMPL(value)
#define FQ_OPENMP_SIMD_PRAGMA(...) FQ_OPENMP_PRAGMA(omp simd __VA_ARGS__)
#endif

namespace {

using flagquantum_native::LinearLookup;
using flagquantum_native::apply_linear_lookup;
using flagquantum_native::build_linear_lookup;
using flagquantum_native::linear_shift_mode;

at::Tensor fused_observable_expectation_cpu(
    const at::Tensor& ket,
    const at::Tensor& weights) {
  TORCH_CHECK(ket.device().is_cpu(), "ket must be on CPU");
  TORCH_CHECK(weights.device().is_cpu(), "weights must be on CPU");
  TORCH_CHECK(ket.is_contiguous(), "ket must be contiguous");
  TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
  TORCH_CHECK(ket.dim() == 2, "ket must have shape [batch, amplitudes]");
  TORCH_CHECK(weights.dim() == 1, "weights must be one-dimensional");
  TORCH_CHECK(weights.numel() == ket.size(1), "weight count must match ket width");
  TORCH_CHECK(
      ket.scalar_type() == at::kComplexFloat || ket.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(
      (ket.scalar_type() == at::kComplexFloat && weights.scalar_type() == at::kFloat) ||
          (ket.scalar_type() == at::kComplexDouble && weights.scalar_type() == at::kDouble),
      "weight dtype must match the real component of ket");

  const int64_t width = ket.size(1);
  const int64_t item_count = ket.numel();
  const int64_t thread_count = std::max<int64_t>(1, at::get_num_threads());
  at::Tensor partials = at::zeros({thread_count}, weights.options());
  at::Tensor result = at::empty({}, weights.options());
  AT_DISPATCH_COMPLEX_TYPES(ket.scalar_type(), "fused_observable_expectation_cpu", [&] {
    using real_t = typename scalar_t::value_type;
    const scalar_t* ket_data = ket.const_data_ptr<scalar_t>();
    const real_t* weight_data = weights.const_data_ptr<real_t>();
    real_t* partial_data = partials.data_ptr<real_t>();
    at::parallel_for(int64_t{0}, item_count, int64_t{4096}, [&](int64_t begin, int64_t end) {
      real_t local = real_t{0};
      for (int64_t item = begin; item < end; ++item) {
        const scalar_t value = ket_data[item];
        local += (value.real() * value.real() + value.imag() * value.imag()) *
            weight_data[item % width];
      }
      partial_data[at::get_thread_num()] += local;
    });
    real_t total = real_t{0};
    for (int64_t thread = 0; thread < thread_count; ++thread) {
      total += partial_data[thread];
    }
    *result.data_ptr<real_t>() = total;
  });
  return result;
}

at::Tensor fused_observable_adjoint_seed_cpu(
    const at::Tensor& ket,
    const at::Tensor& weights) {
  TORCH_CHECK(ket.device().is_cpu(), "ket must be on CPU");
  TORCH_CHECK(weights.device().is_cpu(), "weights must be on CPU");
  TORCH_CHECK(ket.is_contiguous(), "ket must be contiguous");
  TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
  TORCH_CHECK(ket.dim() == 2, "ket must have shape [batch, amplitudes]");
  TORCH_CHECK(weights.dim() == 1, "weights must be one-dimensional");
  TORCH_CHECK(weights.numel() == ket.size(1), "weight count must match ket width");
  TORCH_CHECK(
      ket.scalar_type() == at::kComplexFloat ||
          ket.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(
      (ket.scalar_type() == at::kComplexFloat && weights.scalar_type() == at::kFloat) ||
          (ket.scalar_type() == at::kComplexDouble && weights.scalar_type() == at::kDouble),
      "weight dtype must match the real component of ket");

  at::Tensor adjoint = at::empty_like(ket);
  const int64_t width = ket.size(1);
  const int64_t item_count = ket.numel();
  AT_DISPATCH_COMPLEX_TYPES(ket.scalar_type(), "fused_observable_adjoint_seed_cpu", [&] {
    using real_t = typename scalar_t::value_type;
    const scalar_t* ket_data = ket.const_data_ptr<scalar_t>();
    const real_t* weight_data = weights.const_data_ptr<real_t>();
    scalar_t* adjoint_data = adjoint.data_ptr<scalar_t>();
    at::parallel_for(int64_t{0}, item_count, int64_t{4096}, [&](int64_t begin, int64_t end) {
      for (int64_t item = begin; item < end; ++item) {
        adjoint_data[item] = ket_data[item] * (real_t{2} * weight_data[item % width]);
      }
    });
  });
  return adjoint;
}

inline int64_t popcount(uint64_t value) {
  int64_t count = 0;
  while (value != 0) {
    value &= value - 1;
    ++count;
  }
  return count;
}

at::Tensor fused_static_product_state_initialization_cpu(
    at::Tensor state,
    const at::Tensor& gate_codes,
    const at::Tensor& wires,
    int64_t n_wires) {
  TORCH_CHECK(
      state.device().is_cpu() && gate_codes.device().is_cpu() &&
          wires.device().is_cpu(),
      "static product-state initialization inputs must be on CPU");
  TORCH_CHECK(
      state.is_contiguous() && gate_codes.is_contiguous() && wires.is_contiguous(),
      "static product-state initialization inputs must be contiguous");
  TORCH_CHECK(state.dim() == 2, "state must have shape [batch, amplitudes]");
  TORCH_CHECK(
      state.scalar_type() == at::kComplexFloat ||
          state.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(gate_codes.scalar_type() == at::kChar, "gate_codes must be int8");
  TORCH_CHECK(wires.scalar_type() == at::kLong, "wires must be int64");
  TORCH_CHECK(n_wires > 1 && n_wires < 63, "n_wires must be in [2, 62]");
  TORCH_CHECK(
      gate_codes.dim() == 1 && wires.dim() == 1 &&
          gate_codes.numel() == n_wires && wires.numel() == n_wires,
      "one static Clifford gate is required for every wire");
  const int64_t amplitudes = int64_t{1} << n_wires;
  TORCH_CHECK(state.size(1) == amplitudes, "state width does not match n_wires");

  const int8_t* code_data = gate_codes.const_data_ptr<int8_t>();
  const int64_t* wire_data = wires.const_data_ptr<int64_t>();
  std::array<int64_t, 62> hadamard_masks{};
  int64_t hadamard_count = 0;
  int64_t y_count = 0;
  uint64_t occupied_mask = 0;
  uint64_t required_one_mask = 0;
  for (int64_t gate = 0; gate < n_wires; ++gate) {
    TORCH_CHECK(
        wire_data[gate] >= 0 && wire_data[gate] < n_wires,
        "static product-state wire is out of range");
    TORCH_CHECK(
        code_data[gate] >= 1 && code_data[gate] <= 6,
        "static product-state gate code is invalid");
    const uint64_t mask = uint64_t{1} << (n_wires - wire_data[gate] - 1);
    TORCH_CHECK(
        (occupied_mask & mask) == 0,
        "static product-state wires must be unique");
    occupied_mask |= mask;
    if (code_data[gate] == 1) {
      hadamard_masks[hadamard_count++] = static_cast<int64_t>(mask);
    } else if (code_data[gate] == 4 || code_data[gate] == 5) {
      required_one_mask |= mask;
      y_count += code_data[gate] == 5;
    }
  }

  state.zero_();
  const int64_t support_size = int64_t{1} << hadamard_count;
  const int64_t item_count = state.size(0) * support_size;
  AT_DISPATCH_COMPLEX_TYPES(
      state.scalar_type(), "fused_static_product_state_initialization_cpu", [&] {
        using real_t = typename scalar_t::value_type;
        scalar_t value(
            real_t{1} / std::sqrt(static_cast<real_t>(support_size)), real_t{0});
        switch (y_count & 3) {
          case 1:
            value = scalar_t(real_t{0}, value.real());
            break;
          case 2:
            value = -value;
            break;
          case 3:
            value = scalar_t(real_t{0}, -value.real());
            break;
          default:
            break;
        }
        scalar_t* state_data = state.data_ptr<scalar_t>();
        at::parallel_for(
            int64_t{0}, item_count, int64_t{4096}, [&](int64_t begin, int64_t end) {
              for (int64_t item = begin; item < end; ++item) {
                const int64_t row = item / support_size;
                const int64_t local = item - row * support_size;
                int64_t basis = static_cast<int64_t>(required_one_mask);
                for (int64_t bit = 0; bit < hadamard_count; ++bit) {
                  if ((local & (int64_t{1} << bit)) != 0) {
                    basis |= hadamard_masks[bit];
                  }
                }
                state_data[row * amplitudes + basis] = value;
              }
            });
      });
  return state;
}

at::Tensor fused_static_clifford_layer_cpu(
    at::Tensor state,
    const at::Tensor& gate_codes,
    const at::Tensor& wires,
    int64_t n_wires) {
  TORCH_CHECK(state.device().is_cpu(), "state must be on CPU");
  TORCH_CHECK(gate_codes.device().is_cpu(), "gate_codes must be on CPU");
  TORCH_CHECK(wires.device().is_cpu(), "wires must be on CPU");
  TORCH_CHECK(
      state.is_contiguous() && gate_codes.is_contiguous() && wires.is_contiguous(),
      "Clifford layer inputs must be contiguous");
  TORCH_CHECK(state.dim() == 2, "state must have shape [batch, amplitudes]");
  TORCH_CHECK(
      state.scalar_type() == at::kComplexFloat ||
          state.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(gate_codes.scalar_type() == at::kChar, "gate_codes must be int8");
  TORCH_CHECK(wires.scalar_type() == at::kLong, "wires must be int64");
  TORCH_CHECK(
      gate_codes.dim() == 1 && wires.dim() == 1 &&
          gate_codes.numel() == wires.numel(),
      "gate_codes and wires must be matching vectors");
  TORCH_CHECK(n_wires > 1 && n_wires < 63, "n_wires must be in [2, 62]");

  const int64_t gate_count = wires.numel();
  TORCH_CHECK(
      gate_count >= 2 && gate_count <= n_wires,
      "a Clifford layer must contain between two and n_wires gates");
  const int64_t amplitudes = int64_t{1} << n_wires;
  TORCH_CHECK(state.size(1) == amplitudes, "state width does not match n_wires");

  const int8_t* code_data = gate_codes.const_data_ptr<int8_t>();
  const int64_t* wire_data = wires.const_data_ptr<int64_t>();
  std::array<int64_t, 11> hadamard_masks{};
  std::array<int64_t, 11> hadamard_positions{};
  int64_t hadamard_count = 0;
  uint64_t occupied_mask = 0;
  uint64_t flip_mask = 0;
  uint64_t s_mask = 0;
  uint64_t sdg_mask = 0;
  uint64_t z_mask = 0;
  uint64_t y_mask = 0;
  int64_t y_count = 0;
  for (int64_t gate = 0; gate < gate_count; ++gate) {
    TORCH_CHECK(
        wire_data[gate] >= 0 && wire_data[gate] < n_wires,
        "Clifford layer wire is out of range");
    TORCH_CHECK(
        code_data[gate] >= 1 && code_data[gate] <= 6,
        "Clifford gate code is invalid");
    const int64_t bit_position = n_wires - wire_data[gate] - 1;
    const uint64_t mask = uint64_t{1} << bit_position;
    TORCH_CHECK(
        (occupied_mask & mask) == 0,
        "Clifford layer wires must be unique");
    occupied_mask |= mask;
    if (code_data[gate] == 1) {
      TORCH_CHECK(
          hadamard_count < 11,
          "a Clifford layer supports at most 11 Hadamards");
      hadamard_masks[hadamard_count] = static_cast<int64_t>(mask);
      hadamard_positions[hadamard_count] = bit_position;
      ++hadamard_count;
    } else if (code_data[gate] == 2) {
      s_mask |= mask;
    } else if (code_data[gate] == 3) {
      sdg_mask |= mask;
    } else if (code_data[gate] == 4) {
      flip_mask |= mask;
    } else if (code_data[gate] == 5) {
      flip_mask |= mask;
      y_mask |= mask;
      ++y_count;
    } else {
      z_mask |= mask;
    }
  }
  std::sort(
      hadamard_positions.begin(),
      hadamard_positions.begin() + hadamard_count);

  const int64_t local_size = int64_t{1} << hadamard_count;
  std::array<int64_t, 2048> offsets{};
  for (int64_t local = 0; local < local_size; ++local) {
    int64_t offset = 0;
    for (int64_t wire = 0; wire < hadamard_count; ++wire) {
      if ((local & (int64_t{1} << wire)) != 0) {
        offset |= hadamard_masks[wire];
      }
    }
    offsets[local] = offset;
  }

  const int64_t blocks_per_row = amplitudes >> hadamard_count;
  const int64_t item_count = state.size(0) * blocks_per_row;
  // X and Y exchange whole Hadamard blocks. Processing only the block with
  // the anchor bit cleared lets one task load both sources before either is
  // overwritten, so the transform remains in-place and race-free.
  const uint64_t flip_anchor = flip_mask & (~flip_mask + 1);
  AT_DISPATCH_COMPLEX_TYPES(
      state.scalar_type(), "fused_static_clifford_layer_cpu", [&] {
        using real_t = typename scalar_t::value_type;
        scalar_t* state_data = state.data_ptr<scalar_t>();
        const real_t normalization =
            real_t{1} / std::sqrt(static_cast<real_t>(local_size));
        const auto phase = [&](uint64_t basis) -> scalar_t {
          // Encode every exact diagonal factor as a power of i. Y contributes
          // -i before its X permutation and another sign on destination |1>.
          const int64_t exponent =
              (3 * y_count + popcount(basis & s_mask) +
               3 * popcount(basis & sdg_mask) +
               2 * popcount(basis & z_mask) +
               2 * popcount(basis & y_mask)) &
              3;
          if (exponent == 0) {
            return scalar_t(real_t{1}, real_t{0});
          }
          if (exponent == 1) {
            return scalar_t(real_t{0}, real_t{1});
          }
          if (exponent == 2) {
            return scalar_t(real_t{-1}, real_t{0});
          }
          return scalar_t(real_t{0}, real_t{-1});
        };
        const auto hadamard = [&](std::array<scalar_t, 2048>& values) {
          for (int64_t stride = 1; stride < local_size; stride *= 2) {
            for (int64_t block = 0; block < local_size; block += 2 * stride) {
              for (int64_t offset = 0; offset < stride; ++offset) {
                const int64_t zero = block + offset;
                const int64_t one = zero + stride;
                const scalar_t zero_value = values[zero];
                const scalar_t one_value = values[one];
                values[zero] = zero_value + one_value;
                values[one] = zero_value - one_value;
              }
            }
          }
        };
        at::parallel_for(
            int64_t{0}, item_count, int64_t{64}, [&](int64_t begin, int64_t end) {
              std::array<scalar_t, 2048> first{};
              std::array<scalar_t, 2048> second{};
              for (int64_t item = begin; item < end; ++item) {
                const int64_t row = item / blocks_per_row;
                int64_t base = item - row * blocks_per_row;
                for (int64_t wire = 0; wire < hadamard_count; ++wire) {
                  const int64_t position = hadamard_positions[wire];
                  const int64_t lower_mask = (int64_t{1} << position) - 1;
                  base = (base & lower_mask) | ((base & ~lower_mask) << 1);
                }
                if (flip_mask != 0 &&
                    (static_cast<uint64_t>(base) & flip_anchor) != 0) {
                  continue;
                }
                const int64_t paired = base ^ static_cast<int64_t>(flip_mask);
                const int64_t row_offset = row * amplitudes;
                for (int64_t local = 0; local < local_size; ++local) {
                  first[local] = state_data[row_offset + base + offsets[local]];
                  if (flip_mask != 0) {
                    second[local] = state_data[row_offset + paired + offsets[local]];
                  }
                }
                hadamard(first);
                if (flip_mask != 0) {
                  hadamard(second);
                }
                const scalar_t first_phase = phase(static_cast<uint64_t>(base));
                const scalar_t paired_phase = phase(static_cast<uint64_t>(paired));
                for (int64_t local = 0; local < local_size; ++local) {
                  const int64_t first_position = row_offset + base + offsets[local];
                  if (flip_mask == 0) {
                    state_data[first_position] =
                        first_phase * first[local] * normalization;
                  } else {
                    const int64_t paired_position = row_offset + paired + offsets[local];
                    state_data[first_position] =
                        first_phase * second[local] * normalization;
                    state_data[paired_position] =
                        paired_phase * first[local] * normalization;
                  }
                }
              }
            });
      });
  return state;
}

template <int64_t gate_kind, typename scalar_t>
inline void apply_inverse_rotation_pair(
    typename scalar_t::value_type cosine,
    typename scalar_t::value_type sine,
    scalar_t& zero,
    scalar_t& one) {
  using real_t = typename scalar_t::value_type;
  const real_t zero_real = zero.real();
  const real_t zero_imag = zero.imag();
  const real_t one_real = one.real();
  const real_t one_imag = one.imag();
  if constexpr (gate_kind == 0) {
    zero = scalar_t(
        cosine * zero_real - sine * one_imag,
        cosine * zero_imag + sine * one_real);
    one = scalar_t(
        cosine * one_real - sine * zero_imag,
        cosine * one_imag + sine * zero_real);
  } else if constexpr (gate_kind == 1) {
    zero = scalar_t(
        cosine * zero_real + sine * one_real,
        cosine * zero_imag + sine * one_imag);
    one = scalar_t(
        cosine * one_real - sine * zero_real,
        cosine * one_imag - sine * zero_imag);
  } else {
    zero = scalar_t(
        cosine * zero_real - sine * zero_imag,
        sine * zero_real + cosine * zero_imag);
    one = scalar_t(
        cosine * one_real + sine * one_imag,
        -sine * one_real + cosine * one_imag);
  }
}

template <int64_t gate_kind, typename scalar_t>
inline typename scalar_t::value_type rotation_pair_gradient(
    const scalar_t& ket_zero,
    const scalar_t& ket_one,
    const scalar_t& adjoint_zero,
    const scalar_t& adjoint_one) {
  using real_t = typename scalar_t::value_type;
  if constexpr (gate_kind == 0) {
    return real_t{0.5} * (
        (adjoint_zero.real() * ket_one.imag() -
         adjoint_zero.imag() * ket_one.real()) +
        (adjoint_one.real() * ket_zero.imag() -
         adjoint_one.imag() * ket_zero.real()));
  } else if constexpr (gate_kind == 1) {
    return real_t{0.5} * (
        (adjoint_one.real() * ket_zero.real() +
         adjoint_one.imag() * ket_zero.imag()) -
        (adjoint_zero.real() * ket_one.real() +
         adjoint_zero.imag() * ket_one.imag()));
  } else {
    return real_t{0.5} * (
        (adjoint_zero.real() * ket_zero.imag() -
         adjoint_zero.imag() * ket_zero.real()) -
        (adjoint_one.real() * ket_one.imag() -
         adjoint_one.imag() * ket_one.real()));
  }
}

template <typename scalar_t>
inline void rotation_pair_gradients(
    const scalar_t& ket_zero,
    const scalar_t& ket_one,
    const scalar_t& adjoint_zero,
    const scalar_t& adjoint_one,
    typename scalar_t::value_type& gradient_x,
    typename scalar_t::value_type& gradient_y,
    typename scalar_t::value_type& gradient_z) {
  using real_t = typename scalar_t::value_type;
  gradient_x = real_t{0.5} * (
      (adjoint_zero.real() * ket_one.imag() -
       adjoint_zero.imag() * ket_one.real()) +
      (adjoint_one.real() * ket_zero.imag() -
       adjoint_one.imag() * ket_zero.real()));
  gradient_y = real_t{0.5} * (
      (adjoint_one.real() * ket_zero.real() +
       adjoint_one.imag() * ket_zero.imag()) -
      (adjoint_zero.real() * ket_one.real() +
       adjoint_zero.imag() * ket_one.imag()));
  gradient_z = real_t{0.5} * (
      (adjoint_zero.real() * ket_zero.imag() -
       adjoint_zero.imag() * ket_zero.real()) -
      (adjoint_one.real() * ket_one.imag() -
       adjoint_one.imag() * ket_one.real()));
}

at::Tensor fused_rotation_block_forward_cpu(
    at::Tensor state,
    const at::Tensor& matrices,
    const at::Tensor& wires,
    int64_t n_wires,
    const c10::optional<at::Tensor>& rzz_angles,
    const c10::optional<at::Tensor>& rzz_first_wires,
    const c10::optional<at::Tensor>& rzz_second_wires,
    bool specialized_rotations,
    const c10::optional<at::Tensor>& cx_controls,
    const c10::optional<at::Tensor>& cx_targets) {
  TORCH_CHECK(state.device().is_cpu(), "state must be on CPU");
  TORCH_CHECK(matrices.device().is_cpu(), "matrices must be on CPU");
  TORCH_CHECK(wires.device().is_cpu(), "wires must be on CPU");
  TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
  TORCH_CHECK(matrices.is_contiguous(), "matrices must be contiguous");
  TORCH_CHECK(wires.is_contiguous(), "wires must be contiguous");
  TORCH_CHECK(state.dim() == 2, "state must have shape [batch, amplitudes]");
  TORCH_CHECK(
      matrices.dim() == 3 || matrices.dim() == 4,
      "matrices must have shape [gates, 2, 2] or [batch, gates, 2, 2]");
  const bool batched_matrices = matrices.dim() == 4;
  TORCH_CHECK(
      matrices.size(-2) == 2 && matrices.size(-1) == 2,
      "every rotation matrix must be 2 by 2");
  TORCH_CHECK(
      !batched_matrices || matrices.size(0) == state.size(0),
      "batched matrix rows must match the state batch");
  TORCH_CHECK(wires.dim() == 1, "wires must be one-dimensional");
  TORCH_CHECK(wires.scalar_type() == at::kLong, "wires must be int64");
  TORCH_CHECK(
      matrices.scalar_type() == state.scalar_type(),
      "matrix dtype must match state");
  TORCH_CHECK(
      state.scalar_type() == at::kComplexFloat ||
          state.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(n_wires > 1 && n_wires < 63, "n_wires must be in [2, 62]");

  const int64_t gate_count = matrices.size(batched_matrices ? 1 : 0);
  TORCH_CHECK(
      gate_count >= 2 && gate_count <= 11,
      "a fused rotation block must contain between two and eleven wires");
  TORCH_CHECK(wires.numel() == gate_count, "wire and matrix counts must match");
  const int64_t amplitudes = int64_t{1} << n_wires;
  TORCH_CHECK(state.size(1) == amplitudes, "state width does not match n_wires");
  const bool fuse_rzz = rzz_angles.has_value();
  TORCH_CHECK(
      fuse_rzz == rzz_first_wires.has_value() && fuse_rzz == rzz_second_wires.has_value(),
      "all fused RZZ metadata must be provided together");
  int64_t rzz_count = 0;
  uint64_t rzz_transition_mask = 0;
  if (fuse_rzz) {
    const at::Tensor& angle_tensor = rzz_angles.value();
    const at::Tensor& first_tensor = rzz_first_wires.value();
    const at::Tensor& second_tensor = rzz_second_wires.value();
    TORCH_CHECK(angle_tensor.device().is_cpu() && first_tensor.device().is_cpu() &&
                    second_tensor.device().is_cpu(), "fused RZZ metadata must be on CPU");
    TORCH_CHECK(angle_tensor.is_contiguous() && first_tensor.is_contiguous() &&
                    second_tensor.is_contiguous(), "fused RZZ metadata must be contiguous");
    rzz_count = angle_tensor.numel();
    TORCH_CHECK(rzz_count > 1 && first_tensor.numel() == rzz_count &&
                    second_tensor.numel() == rzz_count, "fused RZZ metadata lengths must match");
    TORCH_CHECK(first_tensor.scalar_type() == at::kLong &&
                    second_tensor.scalar_type() == at::kLong, "fused RZZ wires must be int64");
    const int64_t* first_data = first_tensor.const_data_ptr<int64_t>();
    const int64_t* second_data = second_tensor.const_data_ptr<int64_t>();
    for (int64_t gate = 0; gate < rzz_count; ++gate) {
      TORCH_CHECK(first_data[gate] >= 0 && first_data[gate] < n_wires &&
                      second_data[gate] >= 0 && second_data[gate] < n_wires,
                  "fused RZZ wire is out of range");
      TORCH_CHECK(std::abs(first_data[gate] - second_data[gate]) == 1,
                  "fused RZZ gates must connect adjacent wires");
      const int64_t first_mask = int64_t{1} << (n_wires - first_data[gate] - 1);
      const int64_t second_mask = int64_t{1} << (n_wires - second_data[gate] - 1);
      const uint64_t transition_bit = std::min(first_mask, second_mask);
      TORCH_CHECK((rzz_transition_mask & transition_bit) == 0,
                  "fused RZZ gates must have unique adjacent pairs");
      rzz_transition_mask |= transition_bit;
    }
  }

  const bool fuse_cx = cx_controls.has_value();
  TORCH_CHECK(
      fuse_cx == cx_targets.has_value(),
      "both fused CX wire tensors must be provided together");

  const int64_t* wire_data = wires.const_data_ptr<int64_t>();
  std::array<int64_t, 11> masks{};
  std::array<int64_t, 11> bit_positions{};
  int64_t target_mask = 0;
  for (int64_t gate = 0; gate < gate_count; ++gate) {
    TORCH_CHECK(
        wire_data[gate] >= 0 && wire_data[gate] < n_wires,
        "rotation wire is out of range");
    const int64_t bit_position = n_wires - wire_data[gate] - 1;
    const int64_t mask = int64_t{1} << bit_position;
    TORCH_CHECK((target_mask & mask) == 0, "rotation wires must be unique");
    masks[gate] = mask;
    bit_positions[gate] = bit_position;
    target_mask |= mask;
  }
  std::sort(bit_positions.begin(), bit_positions.begin() + gate_count);

  std::array<int64_t, 5> cx_control_local_masks{};
  std::array<int64_t, 5> cx_target_local_masks{};
  int64_t cx_count = 0;
  if (fuse_cx) {
    const at::Tensor& control_tensor = cx_controls.value();
    const at::Tensor& target_tensor = cx_targets.value();
    TORCH_CHECK(
        control_tensor.device().is_cpu() && target_tensor.device().is_cpu(),
        "fused CX wires must be on CPU");
    TORCH_CHECK(
        control_tensor.is_contiguous() && target_tensor.is_contiguous(),
        "fused CX wires must be contiguous");
    TORCH_CHECK(
        control_tensor.scalar_type() == at::kLong &&
            target_tensor.scalar_type() == at::kLong,
        "fused CX wires must be int64");
    TORCH_CHECK(
        control_tensor.dim() == 1 && target_tensor.sizes() == control_tensor.sizes(),
        "fused CX controls and targets must have matching one-dimensional shapes");
    cx_count = control_tensor.numel();
    TORCH_CHECK(
        cx_count > 0 && cx_count <= 5,
        "a fused rotation tile supports between one and five CX gates");
    const int64_t* control_data = control_tensor.const_data_ptr<int64_t>();
    const int64_t* target_data = target_tensor.const_data_ptr<int64_t>();
    int64_t occupied_local_wires = 0;
    for (int64_t edge = 0; edge < cx_count; ++edge) {
      int64_t control_gate = -1;
      int64_t target_gate = -1;
      for (int64_t gate = 0; gate < gate_count; ++gate) {
        control_gate = wire_data[gate] == control_data[edge] ? gate : control_gate;
        target_gate = wire_data[gate] == target_data[edge] ? gate : target_gate;
      }
      TORCH_CHECK(
          control_gate >= 0 && target_gate >= 0 && control_gate != target_gate,
          "every fused CX wire must belong to its rotation tile");
      const int64_t control_mask = int64_t{1} << control_gate;
      const int64_t target_mask = int64_t{1} << target_gate;
      TORCH_CHECK(
          (occupied_local_wires & (control_mask | target_mask)) == 0,
          "fused CX gates must be wire-disjoint");
      occupied_local_wires |= control_mask | target_mask;
      cx_control_local_masks[edge] = control_mask;
      cx_target_local_masks[edge] = target_mask;
    }
  }

  const int64_t local_size = int64_t{1} << gate_count;
  std::array<int64_t, 2048> offsets{};
  std::array<int64_t, 2048> cx_sources{};
  for (int64_t local = 0; local < local_size; ++local) {
    int64_t offset = 0;
    for (int64_t gate = 0; gate < gate_count; ++gate) {
      if ((local & (int64_t{1} << gate)) != 0) {
        offset |= masks[gate];
      }
    }
    offsets[local] = offset;
    int64_t source = local;
    for (int64_t edge = 0; edge < cx_count; ++edge) {
      if ((local & cx_control_local_masks[edge]) != 0) {
        source ^= cx_target_local_masks[edge];
      }
    }
    cx_sources[local] = source;
  }

  const int64_t blocks_per_row = amplitudes >> gate_count;
  const int64_t item_count = state.size(0) * blocks_per_row;
  AT_DISPATCH_COMPLEX_TYPES(state.scalar_type(), "fused_rotation_block_forward_cpu", [&] {
    using real_t = typename scalar_t::value_type;
    scalar_t* state_data = state.data_ptr<scalar_t>();
    const scalar_t* matrix_data = matrices.const_data_ptr<scalar_t>();
    const int64_t matrix_rows = batched_matrices ? state.size(0) : 1;
    std::array<uint8_t, 11> shared_rotation_kinds{};
    std::vector<uint8_t> batch_rotation_kinds(
        batched_matrices ? matrix_rows * gate_count : 0);
    for (int64_t matrix_row = 0; matrix_row < matrix_rows; ++matrix_row) {
      for (int64_t gate = 0; gate < gate_count; ++gate) {
        const scalar_t* matrix =
            matrix_data + (matrix_row * gate_count + gate) * 4;
        const bool diagonal = matrix[1] == scalar_t{} && matrix[2] == scalar_t{};
        const bool rx = matrix[0].imag() == real_t{} &&
                        matrix[3].imag() == real_t{} &&
                        matrix[1].real() == real_t{} &&
                        matrix[2].real() == real_t{} &&
                        matrix[0].real() == matrix[3].real() &&
                        matrix[1].imag() == matrix[2].imag();
        const bool ry = matrix[0].imag() == real_t{} &&
                        matrix[1].imag() == real_t{} &&
                        matrix[2].imag() == real_t{} &&
                        matrix[3].imag() == real_t{} &&
                        matrix[0].real() == matrix[3].real() &&
                        matrix[1].real() == -matrix[2].real();
        const bool hadamard = matrix[0].imag() == real_t{} &&
                              matrix[1].imag() == real_t{} &&
                              matrix[2].imag() == real_t{} &&
                              matrix[3].imag() == real_t{} &&
                              matrix[0].real() != real_t{} &&
                              matrix[0].real() == matrix[1].real() &&
                              matrix[0].real() == matrix[2].real() &&
                              matrix[0].real() == -matrix[3].real();
        const bool bit_flip = matrix[0] == scalar_t{} &&
                              matrix[3] == scalar_t{} &&
                              matrix[1] == scalar_t{real_t{1}, real_t{0}} &&
                              matrix[2] == scalar_t{real_t{1}, real_t{0}};
        const uint8_t rotation_kind = specialized_rotations
            ? (diagonal ? 1
                        : (rx ? 2
                              : (ry ? 3
                                    : (hadamard ? 4 : (bit_flip ? 5 : 0)))))
            : 0;
        if (batched_matrices) {
          batch_rotation_kinds[matrix_row * gate_count + gate] = rotation_kind;
        } else {
          shared_rotation_kinds[gate] = rotation_kind;
        }
      }
    }
    std::vector<scalar_t> rzz_phases;
    if (fuse_rzz) {
      const at::Tensor& angle_tensor = rzz_angles.value();
      TORCH_CHECK(angle_tensor.scalar_type() == c10::CppTypeToScalarType<real_t>::value,
                  "fused RZZ angle dtype must match state precision");
      const real_t* angle_data = angle_tensor.const_data_ptr<real_t>();
      for (int64_t gate = 1; gate < rzz_count; ++gate) {
        TORCH_CHECK(angle_data[gate] == angle_data[0],
                    "fused RZZ gates must share one angle");
      }
      rzz_phases.resize(rzz_count + 1);
      for (int64_t transitions = 0; transitions <= rzz_count; ++transitions) {
        const real_t sign_sum = static_cast<real_t>(rzz_count - 2 * transitions);
        const real_t phase = real_t{-0.5} * sign_sum * angle_data[0];
        rzz_phases[transitions] = scalar_t(std::cos(phase), std::sin(phase));
      }
    }
    at::parallel_for(int64_t{0}, item_count, int64_t{128}, [&](int64_t begin, int64_t end) {
      std::array<scalar_t, 2048> values{};
      for (int64_t item = begin; item < end; ++item) {
        const int64_t row = item / blocks_per_row;
        int64_t base = item - row * blocks_per_row;
        for (int64_t gate = 0; gate < gate_count; ++gate) {
          const int64_t position = bit_positions[gate];
          const int64_t lower_mask = (int64_t{1} << position) - 1;
          base = (base & lower_mask) | ((base & ~lower_mask) << 1);
        }
        base += row * amplitudes;
        for (int64_t local = 0; local < local_size; ++local) {
          const int64_t position = base + offsets[local];
          values[local] = state_data[position];
          if (fuse_rzz) {
            const int64_t basis = position - row * amplitudes;
            const uint64_t transitions =
                (static_cast<uint64_t>(basis) ^
                 (static_cast<uint64_t>(basis) >> 1)) & rzz_transition_mask;
            values[local] *= rzz_phases[popcount(transitions)];
          }
        }
        for (int64_t gate = 0; gate < gate_count; ++gate) {
          const int64_t matrix_row = batched_matrices ? row : 0;
          const scalar_t* matrix =
              matrix_data + (matrix_row * gate_count + gate) * 4;
          const uint8_t rotation_kind = batched_matrices
              ? batch_rotation_kinds[matrix_row * gate_count + gate]
              : shared_rotation_kinds[gate];
          const int64_t local_mask = int64_t{1} << gate;
          if (rotation_kind == 1) {
            for (int64_t block = 0; block < local_size;
                 block += 2 * local_mask) {
              for (int64_t offset = 0; offset < local_mask; ++offset) {
                const int64_t local = block + offset;
                const int64_t one = local + local_mask;
                const scalar_t zero_value = values[local];
                const scalar_t one_value = values[one];
                values[local] = matrix[0] * zero_value;
                values[one] = matrix[3] * one_value;
              }
            }
          } else if (rotation_kind == 2) {
            const real_t cosine = matrix[0].real();
            const real_t sine = matrix[1].imag();
            for (int64_t block = 0; block < local_size;
                 block += 2 * local_mask) {
              for (int64_t offset = 0; offset < local_mask; ++offset) {
                const int64_t local = block + offset;
                const int64_t one = local + local_mask;
                const scalar_t zero_value = values[local];
                const scalar_t one_value = values[one];
                values[local] = scalar_t(
                    cosine * zero_value.real() - sine * one_value.imag(),
                    cosine * zero_value.imag() + sine * one_value.real());
                values[one] = scalar_t(
                    cosine * one_value.real() - sine * zero_value.imag(),
                    cosine * one_value.imag() + sine * zero_value.real());
              }
            }
          } else if (rotation_kind == 3) {
            const real_t cosine = matrix[0].real();
            const real_t upper_sine = matrix[1].real();
            const real_t lower_sine = matrix[2].real();
            for (int64_t block = 0; block < local_size;
                 block += 2 * local_mask) {
              for (int64_t offset = 0; offset < local_mask; ++offset) {
                const int64_t local = block + offset;
                const int64_t one = local + local_mask;
                const scalar_t zero_value = values[local];
                const scalar_t one_value = values[one];
                values[local] = scalar_t(
                    cosine * zero_value.real() + upper_sine * one_value.real(),
                    cosine * zero_value.imag() + upper_sine * one_value.imag());
                values[one] = scalar_t(
                    lower_sine * zero_value.real() + cosine * one_value.real(),
                    lower_sine * zero_value.imag() + cosine * one_value.imag());
              }
            }
          } else if (rotation_kind == 4) {
            const real_t scale = matrix[0].real();
            for (int64_t block = 0; block < local_size;
                 block += 2 * local_mask) {
              for (int64_t offset = 0; offset < local_mask; ++offset) {
                const int64_t local = block + offset;
                const int64_t one = local + local_mask;
                const scalar_t zero_value = values[local];
                const scalar_t one_value = values[one];
                values[local] = scalar_t(
                    (zero_value.real() + one_value.real()) * scale,
                    (zero_value.imag() + one_value.imag()) * scale);
                values[one] = scalar_t(
                    (zero_value.real() - one_value.real()) * scale,
                    (zero_value.imag() - one_value.imag()) * scale);
              }
            }
          } else if (rotation_kind == 5) {
            for (int64_t block = 0; block < local_size;
                 block += 2 * local_mask) {
              for (int64_t offset = 0; offset < local_mask; ++offset) {
                const int64_t local = block + offset;
                const int64_t one = local + local_mask;
                std::swap(values[local], values[one]);
              }
            }
          } else {
            if (!specialized_rotations) {
              for (int64_t local = 0; local < local_size; ++local) {
                if ((local & local_mask) != 0) {
                  continue;
                }
                const int64_t one = local | local_mask;
                const scalar_t zero_value = values[local];
                const scalar_t one_value = values[one];
                values[local] = matrix[0] * zero_value + matrix[1] * one_value;
                values[one] = matrix[2] * zero_value + matrix[3] * one_value;
              }
            } else {
              for (int64_t block = 0; block < local_size;
                   block += 2 * local_mask) {
                for (int64_t offset = 0; offset < local_mask; ++offset) {
                  const int64_t local = block + offset;
                  const int64_t one = local + local_mask;
                  const scalar_t zero_value = values[local];
                  const scalar_t one_value = values[one];
                  values[local] = matrix[0] * zero_value + matrix[1] * one_value;
                  values[one] = matrix[2] * zero_value + matrix[3] * one_value;
                }
              }
            }
          }
        }
        for (int64_t local = 0; local < local_size; ++local) {
          state_data[base + offsets[local]] = values[cx_sources[local]];
        }
      }
    });
  });
  return state;
}

at::Tensor fused_product_state_initialization_cpu(
    at::Tensor state,
    const at::Tensor& matrices,
    const at::Tensor& wires,
    int64_t n_wires,
    const at::Tensor& cx_controls,
    const at::Tensor& cx_targets) {
  TORCH_CHECK(
      state.device().is_cpu() && matrices.device().is_cpu() &&
          wires.device().is_cpu() && cx_controls.device().is_cpu() &&
          cx_targets.device().is_cpu(),
      "product-state initialization inputs must be on CPU");
  TORCH_CHECK(
      state.is_contiguous() && matrices.is_contiguous() &&
          wires.is_contiguous() && cx_controls.is_contiguous() &&
          cx_targets.is_contiguous(),
      "product-state initialization inputs must be contiguous");
  TORCH_CHECK(state.dim() == 2, "state must have shape [batch, amplitudes]");
  TORCH_CHECK(
      matrices.dim() == 3 || matrices.dim() == 4,
      "matrices must be shared or batched");
  TORCH_CHECK(
      state.scalar_type() == at::kComplexFloat ||
          state.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(
      matrices.scalar_type() == state.scalar_type(),
      "matrix dtype must match state");
  TORCH_CHECK(n_wires > 1 && n_wires < 63, "n_wires must be in [2, 62]");
  const bool batched_matrices = matrices.dim() == 4;
  TORCH_CHECK(
      matrices.size(-3) == n_wires && matrices.size(-2) == 2 &&
          matrices.size(-1) == 2,
      "one 2 by 2 matrix is required for every wire");
  TORCH_CHECK(
      !batched_matrices || matrices.size(0) == state.size(0),
      "batched matrix rows must match the state batch");
  TORCH_CHECK(
      wires.dim() == 1 && wires.numel() == n_wires &&
          wires.scalar_type() == at::kLong,
      "wires must be an int64 permutation of every wire");
  TORCH_CHECK(
      cx_controls.dim() == 1 && cx_targets.sizes() == cx_controls.sizes() &&
          cx_controls.scalar_type() == at::kLong &&
          cx_targets.scalar_type() == at::kLong,
      "CX controls and targets must be matching int64 vectors");
  const int64_t amplitudes = int64_t{1} << n_wires;
  TORCH_CHECK(state.size(1) == amplitudes, "state width does not match n_wires");

  const int64_t* wire_data = wires.const_data_ptr<int64_t>();
  std::array<int64_t, 62> gate_by_wire{};
  gate_by_wire.fill(-1);
  for (int64_t gate = 0; gate < n_wires; ++gate) {
    TORCH_CHECK(
        wire_data[gate] >= 0 && wire_data[gate] < n_wires,
        "rotation wire is out of range");
    TORCH_CHECK(gate_by_wire[wire_data[gate]] < 0, "rotation wires must be unique");
    gate_by_wire[wire_data[gate]] = gate;
  }

  std::array<int64_t, 62> first_gates{};
  std::array<int64_t, 62> second_gates{};
  std::array<int64_t, 62> first_masks{};
  std::array<int64_t, 62> second_masks{};
  std::array<bool, 62> paired{};
  std::array<bool, 62> consumed{};
  int64_t component_count = 0;
  const int64_t* control_data = cx_controls.const_data_ptr<int64_t>();
  const int64_t* target_data = cx_targets.const_data_ptr<int64_t>();
  for (int64_t edge = 0; edge < cx_controls.numel(); ++edge) {
    const int64_t control = control_data[edge];
    const int64_t target = target_data[edge];
    TORCH_CHECK(
        control >= 0 && control < n_wires && target >= 0 &&
            target < n_wires && control != target,
        "CX wire is out of range");
    TORCH_CHECK(!consumed[control] && !consumed[target], "CX gates must be disjoint");
    consumed[control] = true;
    consumed[target] = true;
    first_gates[component_count] = gate_by_wire[control];
    second_gates[component_count] = gate_by_wire[target];
    first_masks[component_count] = int64_t{1} << (n_wires - control - 1);
    second_masks[component_count] = int64_t{1} << (n_wires - target - 1);
    paired[component_count] = true;
    ++component_count;
  }
  for (int64_t wire = 0; wire < n_wires; ++wire) {
    if (consumed[wire]) {
      continue;
    }
    first_gates[component_count] = gate_by_wire[wire];
    first_masks[component_count] = int64_t{1} << (n_wires - wire - 1);
    paired[component_count] = false;
    ++component_count;
  }

  AT_DISPATCH_COMPLEX_TYPES(
      state.scalar_type(), "fused_product_state_initialization_cpu", [&] {
        scalar_t* state_data = state.data_ptr<scalar_t>();
        const scalar_t* matrix_data = matrices.const_data_ptr<scalar_t>();
        const int64_t matrix_rows = batched_matrices ? state.size(0) : 1;
        std::vector<scalar_t> factors(
            matrix_rows * component_count * int64_t{4}, scalar_t{});
        for (int64_t row = 0; row < matrix_rows; ++row) {
          for (int64_t component = 0; component < component_count; ++component) {
            const scalar_t* first = matrix_data +
                (row * n_wires + first_gates[component]) * 4;
            scalar_t* factor = factors.data() +
                (row * component_count + component) * 4;
            if (!paired[component]) {
              factor[0] = first[0];
              factor[1] = first[2];
              continue;
            }
            const scalar_t* second = matrix_data +
                (row * n_wires + second_gates[component]) * 4;
            factor[0] = first[0] * second[0];
            factor[1] = first[0] * second[2];
            factor[2] = first[2] * second[2];
            factor[3] = first[2] * second[0];
          }
        }
        const int64_t item_count = state.numel();
        at::parallel_for(
            int64_t{0}, item_count, int64_t{4096},
            [&](int64_t begin, int64_t end) {
              for (int64_t item = begin; item < end; ++item) {
                const int64_t row = item / amplitudes;
                const int64_t basis = item - row * amplitudes;
                const int64_t matrix_row = batched_matrices ? row : 0;
                const scalar_t* row_factors = factors.data() +
                    matrix_row * component_count * 4;
                scalar_t value{1, 0};
                for (int64_t component = 0; component < component_count; ++component) {
                  int64_t local = (basis & first_masks[component]) != 0 ? 2 : 0;
                  if (paired[component]) {
                    local += (basis & second_masks[component]) != 0 ? 1 : 0;
                  } else {
                    local >>= 1;
                  }
                  value *= row_factors[component * 4 + local];
                }
                state_data[item] = value;
              }
            });
      });
  return state;
}

at::Tensor fused_hadamard_block_adjoint_cpu(
    at::Tensor ket,
    at::Tensor adjoint,
    const at::Tensor& wires,
    int64_t n_wires) {
  TORCH_CHECK(ket.device().is_cpu(), "ket must be on CPU");
  TORCH_CHECK(adjoint.device().is_cpu(), "adjoint must be on CPU");
  TORCH_CHECK(wires.device().is_cpu(), "wires must be on CPU");
  TORCH_CHECK(ket.is_contiguous(), "ket must be contiguous");
  TORCH_CHECK(adjoint.is_contiguous(), "adjoint must be contiguous");
  TORCH_CHECK(wires.is_contiguous(), "wires must be contiguous");
  TORCH_CHECK(ket.dim() == 2, "ket must have shape [batch, amplitudes]");
  TORCH_CHECK(adjoint.sizes() == ket.sizes(), "adjoint shape must match ket");
  TORCH_CHECK(wires.dim() == 1, "wires must be one-dimensional");
  TORCH_CHECK(wires.scalar_type() == at::kLong, "wires must be int64");
  TORCH_CHECK(
      ket.scalar_type() == adjoint.scalar_type(),
      "ket and adjoint dtypes must match");
  TORCH_CHECK(
      ket.scalar_type() == at::kComplexFloat ||
          ket.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(n_wires > 1 && n_wires < 63, "n_wires must be in [2, 62]");

  const int64_t gate_count = wires.numel();
  TORCH_CHECK(
      gate_count >= 2 && gate_count <= 11,
      "a fused Hadamard block must contain between two and eleven wires");
  const int64_t amplitudes = int64_t{1} << n_wires;
  TORCH_CHECK(ket.size(1) == amplitudes, "state width does not match n_wires");

  const int64_t* wire_data = wires.const_data_ptr<int64_t>();
  std::array<int64_t, 11> masks{};
  std::array<int64_t, 11> bit_positions{};
  int64_t target_mask = 0;
  for (int64_t gate = 0; gate < gate_count; ++gate) {
    TORCH_CHECK(
        wire_data[gate] >= 0 && wire_data[gate] < n_wires,
        "fixed-block wire is out of range");
    const int64_t bit_position = n_wires - wire_data[gate] - 1;
    const int64_t mask = int64_t{1} << bit_position;
    TORCH_CHECK((target_mask & mask) == 0, "fixed-block wires must be unique");
    masks[gate] = mask;
    bit_positions[gate] = bit_position;
    target_mask |= mask;
  }
  std::sort(bit_positions.begin(), bit_positions.begin() + gate_count);

  const int64_t local_size = int64_t{1} << gate_count;
  std::array<int64_t, 2048> offsets{};
  for (int64_t local = 0; local < local_size; ++local) {
    int64_t offset = 0;
    for (int64_t gate = 0; gate < gate_count; ++gate) {
      if ((local & (int64_t{1} << gate)) != 0) {
        offset |= masks[gate];
      }
    }
    offsets[local] = offset;
  }

  const int64_t blocks_per_row = amplitudes >> gate_count;
  const int64_t item_count = ket.size(0) * blocks_per_row;
  AT_DISPATCH_COMPLEX_TYPES(ket.scalar_type(), "fused_hadamard_block_adjoint_cpu", [&] {
    using real_t = typename scalar_t::value_type;
    constexpr real_t inverse_sqrt_two = static_cast<real_t>(0.7071067811865475244);
    scalar_t* ket_data = ket.data_ptr<scalar_t>();
    scalar_t* adjoint_data = adjoint.data_ptr<scalar_t>();
    at::parallel_for(int64_t{0}, item_count, int64_t{32}, [&](int64_t begin, int64_t end) {
      std::array<scalar_t, 2048> ket_values{};
      std::array<scalar_t, 2048> adjoint_values{};
      for (int64_t item = begin; item < end; ++item) {
        const int64_t row = item / blocks_per_row;
        int64_t base = item - row * blocks_per_row;
        for (int64_t gate = 0; gate < gate_count; ++gate) {
          const int64_t position = bit_positions[gate];
          const int64_t lower_mask = (int64_t{1} << position) - 1;
          base = (base & lower_mask) | ((base & ~lower_mask) << 1);
        }
        base += row * amplitudes;
        for (int64_t local = 0; local < local_size; ++local) {
          const int64_t index = base + offsets[local];
          ket_values[local] = ket_data[index];
          adjoint_values[local] = adjoint_data[index];
        }
        for (int64_t gate = 0; gate < gate_count; ++gate) {
          const int64_t local_mask = int64_t{1} << gate;
          for (int64_t local = 0; local < local_size; ++local) {
            if ((local & local_mask) != 0) {
              continue;
            }
            const int64_t one = local | local_mask;
            const scalar_t ket_zero = ket_values[local];
            const scalar_t ket_one = ket_values[one];
            ket_values[local] = scalar_t(
                (ket_zero.real() + ket_one.real()) * inverse_sqrt_two,
                (ket_zero.imag() + ket_one.imag()) * inverse_sqrt_two);
            ket_values[one] = scalar_t(
                (ket_zero.real() - ket_one.real()) * inverse_sqrt_two,
                (ket_zero.imag() - ket_one.imag()) * inverse_sqrt_two);
            const scalar_t adjoint_zero = adjoint_values[local];
            const scalar_t adjoint_one = adjoint_values[one];
            adjoint_values[local] = scalar_t(
                (adjoint_zero.real() + adjoint_one.real()) * inverse_sqrt_two,
                (adjoint_zero.imag() + adjoint_one.imag()) * inverse_sqrt_two);
            adjoint_values[one] = scalar_t(
                (adjoint_zero.real() - adjoint_one.real()) * inverse_sqrt_two,
                (adjoint_zero.imag() - adjoint_one.imag()) * inverse_sqrt_two);
          }
        }
        for (int64_t local = 0; local < local_size; ++local) {
          const int64_t index = base + offsets[local];
          ket_data[index] = ket_values[local];
          adjoint_data[index] = adjoint_values[local];
        }
      }
    });
  });
  return ket;
}

at::Tensor fused_rotation_adjoint_cpu(
    at::Tensor ket,
    at::Tensor adjoint,
    const at::Tensor& matrix,
    int64_t wire,
    int64_t second_wire,
    int64_t n_wires,
    int64_t gate_kind) {
  TORCH_CHECK(ket.device().is_cpu(), "ket must be on CPU");
  TORCH_CHECK(adjoint.device().is_cpu(), "adjoint must be on CPU");
  TORCH_CHECK(matrix.device().is_cpu(), "matrix must be on CPU");
  TORCH_CHECK(ket.is_contiguous(), "ket must be contiguous");
  TORCH_CHECK(adjoint.is_contiguous(), "adjoint must be contiguous");
  TORCH_CHECK(matrix.is_contiguous(), "matrix must be contiguous");
  TORCH_CHECK(ket.dim() == 2, "ket must have shape [batch, amplitudes]");
  TORCH_CHECK(adjoint.sizes() == ket.sizes(), "adjoint shape must match ket");
  TORCH_CHECK(
      (gate_kind == 3 && matrix.sizes() == at::IntArrayRef({4, 4})) ||
          (gate_kind != 3 && matrix.sizes() == at::IntArrayRef({2, 2})),
      "matrix shape does not match the rotation kind");
  TORCH_CHECK(adjoint.scalar_type() == ket.scalar_type(), "adjoint dtype must match ket");
  TORCH_CHECK(matrix.scalar_type() == ket.scalar_type(), "matrix dtype must match ket");
  TORCH_CHECK(
      ket.scalar_type() == at::kComplexFloat || ket.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(n_wires > 0 && n_wires < 63, "n_wires must be in [1, 62]");
  TORCH_CHECK(wire >= 0 && wire < n_wires, "wire is out of range");
  TORCH_CHECK(gate_kind >= 0 && gate_kind <= 3, "unsupported rotation kind");
  TORCH_CHECK(
      gate_kind != 3 ||
          (second_wire >= 0 && second_wire < n_wires && second_wire != wire),
      "second wire is invalid for RZZ");

  const int64_t amplitudes = int64_t{1} << n_wires;
  TORCH_CHECK(ket.size(1) == amplitudes, "state width does not match n_wires");
  const int64_t stride = int64_t{1} << (n_wires - wire - 1);
  const int64_t pairs_per_row = amplitudes >> 1;
  const int64_t item_count =
      ket.size(0) * (gate_kind == 3 ? amplitudes : pairs_per_row);

  at::Tensor result;
  AT_DISPATCH_COMPLEX_TYPES(ket.scalar_type(), "fused_rotation_adjoint_cpu", [&] {
    using real_t = typename scalar_t::value_type;
    scalar_t* ket_data = ket.data_ptr<scalar_t>();
    scalar_t* adjoint_data = adjoint.data_ptr<scalar_t>();
    const scalar_t* matrix_data = matrix.const_data_ptr<scalar_t>();

    const scalar_t inverse_00 = std::conj(matrix_data[0]);
    const scalar_t inverse_01 =
        gate_kind == 3 ? scalar_t{0} : std::conj(matrix_data[2]);
    const scalar_t inverse_10 =
        gate_kind == 3 ? scalar_t{0} : std::conj(matrix_data[1]);
    const scalar_t inverse_11 =
        gate_kind == 3 ? std::conj(matrix_data[5]) : std::conj(matrix_data[3]);
    const int64_t first_mask = int64_t{1} << (n_wires - wire - 1);
    const int64_t second_mask = gate_kind == 3
        ? int64_t{1} << (n_wires - second_wire - 1)
        : int64_t{0};

    const real_t gradient = at::parallel_reduce(
        int64_t{0}, item_count, int64_t{4096}, real_t{0},
        [&](int64_t begin, int64_t end, real_t subtotal) -> real_t {
          if (gate_kind == 3) {
            for (int64_t item = begin; item < end; ++item) {
              const int64_t basis = item & (amplitudes - 1);
              const bool odd_parity = ((basis & first_mask) != 0) !=
                  ((basis & second_mask) != 0);
              const scalar_t ket_value = ket_data[item];
              const scalar_t adjoint_value = adjoint_data[item];
              const real_t cross = adjoint_value.real() * ket_value.imag() -
                  adjoint_value.imag() * ket_value.real();
              subtotal += (odd_parity ? real_t{-0.5} : real_t{0.5}) * cross;
              const scalar_t inverse = odd_parity ? inverse_11 : inverse_00;
              ket_data[item] = inverse * ket_value;
              adjoint_data[item] = inverse * adjoint_value;
            }
            return subtotal;
          }
          for (int64_t pair = begin; pair < end; ++pair) {
            const int64_t row = pair >> (n_wires - 1);
            const int64_t local_pair = pair & (pairs_per_row - 1);
            const int64_t offset_mask = stride - 1;
            const int64_t zero = row * amplitudes +
                ((local_pair & ~offset_mask) << 1) +
                (local_pair & offset_mask);
            const int64_t one = zero + stride;

            const scalar_t ket_zero = ket_data[zero];
            const scalar_t ket_one = ket_data[one];
            const scalar_t adjoint_zero = adjoint_data[zero];
            const scalar_t adjoint_one = adjoint_data[one];

            if (gate_kind == 0) {
              subtotal += real_t{0.5} * (
                  (adjoint_zero.real() * ket_one.imag() - adjoint_zero.imag() * ket_one.real()) +
                  (adjoint_one.real() * ket_zero.imag() - adjoint_one.imag() * ket_zero.real()));
            } else if (gate_kind == 1) {
              subtotal += real_t{0.5} * (
                  (adjoint_one.real() * ket_zero.real() + adjoint_one.imag() * ket_zero.imag()) -
                  (adjoint_zero.real() * ket_one.real() + adjoint_zero.imag() * ket_one.imag()));
            } else {
              subtotal += real_t{0.5} * (
                  (adjoint_zero.real() * ket_zero.imag() - adjoint_zero.imag() * ket_zero.real()) -
                  (adjoint_one.real() * ket_one.imag() - adjoint_one.imag() * ket_one.real()));
            }

            ket_data[zero] = inverse_00 * ket_zero + inverse_01 * ket_one;
            ket_data[one] = inverse_10 * ket_zero + inverse_11 * ket_one;
            adjoint_data[zero] = inverse_00 * adjoint_zero + inverse_01 * adjoint_one;
            adjoint_data[one] = inverse_10 * adjoint_zero + inverse_11 * adjoint_one;
          }
          return subtotal;
        },
        [](real_t left, real_t right) -> real_t { return left + right; });

    result = at::empty({}, ket.options().dtype(c10::CppTypeToScalarType<real_t>::value));
    *result.data_ptr<real_t>() = gradient;
  });
  return result;
}

at::Tensor fused_rzz_segment_forward_cpu(
    at::Tensor state,
    const at::Tensor& angles,
    const at::Tensor& first_wires,
    const at::Tensor& second_wires,
    int64_t n_wires,
    bool enable_shared_phase_lookup) {
  TORCH_CHECK(state.device().is_cpu(), "state must be on CPU");
  TORCH_CHECK(
      angles.device().is_cpu() && first_wires.device().is_cpu() &&
          second_wires.device().is_cpu(),
      "segment metadata must be on CPU");
  TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
  TORCH_CHECK(
      angles.is_contiguous() && first_wires.is_contiguous() &&
          second_wires.is_contiguous(),
      "segment metadata must be contiguous");
  TORCH_CHECK(state.dim() == 2, "state must have shape [batch, amplitudes]");
  TORCH_CHECK(
      state.scalar_type() == at::kComplexFloat ||
          state.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(first_wires.scalar_type() == at::kLong, "first_wires must be int64");
  TORCH_CHECK(second_wires.scalar_type() == at::kLong, "second_wires must be int64");
  TORCH_CHECK(
      angles.dim() == 1 && first_wires.dim() == 1 && second_wires.dim() == 1,
      "segment metadata must be one-dimensional");
  TORCH_CHECK(
      angles.numel() > 1 && first_wires.numel() == angles.numel() &&
          second_wires.numel() == angles.numel(),
      "segment metadata lengths must match and contain at least two gates");
  TORCH_CHECK(n_wires > 1 && n_wires < 63, "n_wires must be in [2, 62]");

  const int64_t amplitudes = int64_t{1} << n_wires;
  TORCH_CHECK(state.size(1) == amplitudes, "state width does not match n_wires");
  const int64_t gate_count = angles.numel();
  const int64_t* first_data = first_wires.const_data_ptr<int64_t>();
  const int64_t* second_data = second_wires.const_data_ptr<int64_t>();
  std::vector<int64_t> first_masks(gate_count);
  std::vector<int64_t> second_masks(gate_count);
  uint64_t transition_mask = 0;
  bool shared_path = true;
  for (int64_t gate = 0; gate < gate_count; ++gate) {
    TORCH_CHECK(
        first_data[gate] >= 0 && first_data[gate] < n_wires &&
            second_data[gate] >= 0 && second_data[gate] < n_wires &&
            first_data[gate] != second_data[gate],
        "RZZ segment contains an invalid wire pair");
    first_masks[gate] = int64_t{1} << (n_wires - first_data[gate] - 1);
    second_masks[gate] = int64_t{1} << (n_wires - second_data[gate] - 1);
    if (std::abs(first_data[gate] - second_data[gate]) != 1) {
      shared_path = false;
      continue;
    }
    const uint64_t transition_bit = static_cast<uint64_t>(
        std::min(first_masks[gate], second_masks[gate]));
    if ((transition_mask & transition_bit) != 0) {
      shared_path = false;
    }
    transition_mask |= transition_bit;
  }

  AT_DISPATCH_COMPLEX_TYPES(state.scalar_type(), "fused_rzz_segment_forward_cpu", [&] {
    using real_t = typename scalar_t::value_type;
    TORCH_CHECK(
        angles.scalar_type() == c10::CppTypeToScalarType<real_t>::value,
        "angle dtype must match the state precision");
    const real_t* angle_data = angles.const_data_ptr<real_t>();
    for (int64_t gate = 1; gate < gate_count; ++gate) {
      if (angle_data[gate] != angle_data[0]) {
        shared_path = false;
        break;
      }
    }
    scalar_t* state_data = state.data_ptr<scalar_t>();
    const int64_t item_count = state.numel();
    std::vector<scalar_t> shared_phases;
    if (shared_path && enable_shared_phase_lookup) {
      shared_phases.resize(gate_count + 1);
      for (int64_t transitions = 0; transitions <= gate_count; ++transitions) {
        const real_t sign_sum = static_cast<real_t>(gate_count - 2 * transitions);
        const real_t phase = real_t{-0.5} * sign_sum * angle_data[0];
        shared_phases[transitions] = scalar_t(std::cos(phase), std::sin(phase));
      }
    }
    at::parallel_for(int64_t{0}, item_count, int64_t{4096}, [&](int64_t begin, int64_t end) {
      for (int64_t item = begin; item < end; ++item) {
        const int64_t basis = item & (amplitudes - 1);
        real_t phase = real_t{0};
        if (shared_path) {
          const uint64_t transitions =
              (static_cast<uint64_t>(basis) ^
               (static_cast<uint64_t>(basis) >> 1)) &
              transition_mask;
          const real_t sign_sum = static_cast<real_t>(
              gate_count - 2 * popcount(transitions));
          phase = real_t{-0.5} * sign_sum * angle_data[0];
        } else {
          for (int64_t gate = 0; gate < gate_count; ++gate) {
            const bool odd_parity = ((basis & first_masks[gate]) != 0) !=
                ((basis & second_masks[gate]) != 0);
            const real_t sign = odd_parity ? real_t{-1} : real_t{1};
            phase -= real_t{0.5} * sign * angle_data[gate];
          }
        }
        state_data[item] *= shared_phases.empty()
            ? scalar_t(std::cos(phase), std::sin(phase))
            : shared_phases[popcount(
                  (static_cast<uint64_t>(basis) ^
                   (static_cast<uint64_t>(basis) >> 1)) & transition_mask)];
      }
    });
  });
  return state;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_rotation_segment_adjoint_cpu(
    at::Tensor ket,
    at::Tensor adjoint,
    const at::Tensor& angles,
    const at::Tensor& gate_kinds,
    const at::Tensor& wires,
    int64_t n_wires,
    bool aggregate_shared_parameter,
    int64_t max_tile_wires,
    int64_t parallel_grain,
    bool enable_pair_fast_path,
    bool enable_flat_pair_simd,
    bool enable_euler_post_reduction,
    bool restore_state,
    bool fuse_preceding_hadamards,
    const c10::optional<at::Tensor>& rzz_angles,
    const c10::optional<at::Tensor>& rzz_first_wires,
    const c10::optional<at::Tensor>& rzz_second_wires,
    const c10::optional<at::Tensor>& observable_weights,
    const c10::optional<at::Tensor>& cx_index,
    const c10::optional<at::Tensor>& cx_images) {
  TORCH_CHECK(ket.device().is_cpu(), "ket must be on CPU");
  TORCH_CHECK(adjoint.device().is_cpu(), "adjoint must be on CPU");
  TORCH_CHECK(angles.device().is_cpu(), "angles must be on CPU");
  TORCH_CHECK(gate_kinds.device().is_cpu(), "gate kinds must be on CPU");
  TORCH_CHECK(wires.device().is_cpu(), "wires must be on CPU");
  TORCH_CHECK(ket.is_contiguous(), "ket must be contiguous");
  TORCH_CHECK(adjoint.is_contiguous(), "adjoint must be contiguous");
  TORCH_CHECK(angles.is_contiguous(), "angles must be contiguous");
  TORCH_CHECK(gate_kinds.is_contiguous(), "gate kinds must be contiguous");
  TORCH_CHECK(wires.is_contiguous(), "wires must be contiguous");
  TORCH_CHECK(ket.dim() == 2, "ket must have shape [batch, amplitudes]");
  TORCH_CHECK(adjoint.sizes() == ket.sizes(), "adjoint shape must match ket");
  TORCH_CHECK(angles.dim() == 1, "angles must be one-dimensional");
  TORCH_CHECK(gate_kinds.dim() == 1, "gate kinds must be one-dimensional");
  TORCH_CHECK(wires.dim() == 1, "wires must be one-dimensional");
  TORCH_CHECK(
      parallel_grain >= 0,
      "parallel grain must be zero (adaptive) or positive");
  TORCH_CHECK(gate_kinds.scalar_type() == at::kLong, "gate kinds must be int64");
  TORCH_CHECK(wires.scalar_type() == at::kLong, "wires must be int64");
  TORCH_CHECK(adjoint.scalar_type() == ket.scalar_type(), "state dtypes must match");
  TORCH_CHECK(
      ket.scalar_type() == at::kComplexFloat || ket.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(n_wires > 1 && n_wires < 63, "n_wires must be in [2, 62]");
  TORCH_CHECK(
      max_tile_wires >= 2 && max_tile_wires <= 11,
      "max_tile_wires must be in [2, 11]");

  const int64_t gate_count = angles.size(0);
  TORCH_CHECK(
      gate_count >= 2 && gate_count <= 256,
      "a fused rotation segment must contain between two and two hundred fifty-six gates");
  TORCH_CHECK(gate_kinds.numel() == gate_count, "gate-kind count must match matrices");
  TORCH_CHECK(wires.numel() == gate_count, "wire count must match matrices");
  const int64_t amplitudes = int64_t{1} << n_wires;
  TORCH_CHECK(ket.size(1) == amplitudes, "state width does not match n_wires");
  const bool fuse_rzz = rzz_angles.has_value();
  const bool seed_observable = observable_weights.has_value();
  const bool fuse_cx_index = cx_index.has_value();
  const bool fuse_cx_images = cx_images.has_value();
  const bool fuse_cx = fuse_cx_index || fuse_cx_images;
  TORCH_CHECK(
      !(fuse_cx_index && fuse_cx_images),
      "CX index and compact images are mutually exclusive");
  TORCH_CHECK(
      !fuse_preceding_hadamards || fuse_rzz,
      "preceding Hadamards require a fused RZZ boundary");
  TORCH_CHECK(
      restore_state || (!fuse_rzz && !fuse_preceding_hadamards),
      "terminal no-restore cannot absorb an earlier RZZ or Hadamard boundary");
  TORCH_CHECK(
      fuse_rzz == rzz_first_wires.has_value() && fuse_rzz == rzz_second_wires.has_value(),
      "all fused RZZ metadata must be provided together");
  int64_t rzz_count = 0;
  uint64_t rzz_transition_mask = 0;
  if (fuse_rzz) {
    const at::Tensor& rzz_angle_tensor = rzz_angles.value();
    const at::Tensor& first_tensor = rzz_first_wires.value();
    const at::Tensor& second_tensor = rzz_second_wires.value();
    TORCH_CHECK(rzz_angle_tensor.device().is_cpu() && first_tensor.device().is_cpu() &&
                    second_tensor.device().is_cpu(), "fused RZZ metadata must be on CPU");
    TORCH_CHECK(rzz_angle_tensor.is_contiguous() && first_tensor.is_contiguous() &&
                    second_tensor.is_contiguous(), "fused RZZ metadata must be contiguous");
    rzz_count = rzz_angle_tensor.numel();
    TORCH_CHECK(rzz_count > 1 && first_tensor.numel() == rzz_count &&
                    second_tensor.numel() == rzz_count, "fused RZZ metadata lengths must match");
    TORCH_CHECK(first_tensor.scalar_type() == at::kLong &&
                    second_tensor.scalar_type() == at::kLong, "fused RZZ wires must be int64");
    const int64_t* first_data = first_tensor.const_data_ptr<int64_t>();
    const int64_t* second_data = second_tensor.const_data_ptr<int64_t>();
    for (int64_t gate = 0; gate < rzz_count; ++gate) {
      TORCH_CHECK(first_data[gate] >= 0 && first_data[gate] < n_wires &&
                      second_data[gate] >= 0 && second_data[gate] < n_wires,
                  "fused RZZ wire is out of range");
      TORCH_CHECK(std::abs(first_data[gate] - second_data[gate]) == 1,
                  "fused RZZ gates must connect adjacent wires");
      const int64_t first_mask = int64_t{1} << (n_wires - first_data[gate] - 1);
      const int64_t second_mask = int64_t{1} << (n_wires - second_data[gate] - 1);
      const uint64_t transition_bit = std::min(first_mask, second_mask);
      TORCH_CHECK((rzz_transition_mask & transition_bit) == 0,
                  "fused RZZ gates must have unique adjacent pairs");
      rzz_transition_mask |= transition_bit;
    }
  }
  if (seed_observable) {
    const at::Tensor& weights = observable_weights.value();
    TORCH_CHECK(weights.device().is_cpu(), "observable weights must be on CPU");
    TORCH_CHECK(weights.is_contiguous(), "observable weights must be contiguous");
    TORCH_CHECK(weights.dim() == 1 && weights.numel() == amplitudes,
                "observable weight count must match state width");
  }
  if (fuse_cx_index) {
    const at::Tensor& index = cx_index.value();
    TORCH_CHECK(index.device().is_cpu(), "CX index must be on CPU");
    TORCH_CHECK(index.is_contiguous(), "CX index must be contiguous");
    TORCH_CHECK(index.dim() == 1 && index.numel() == amplitudes,
                "CX index width must match state width");
    TORCH_CHECK(index.scalar_type() == at::kInt || index.scalar_type() == at::kLong,
                "CX index must be int32 or int64");
  }
  if (fuse_cx_images) {
    const at::Tensor& images = cx_images.value();
    TORCH_CHECK(images.device().is_cpu(), "CX images must be on CPU");
    TORCH_CHECK(images.is_contiguous(), "CX images must be contiguous");
    TORCH_CHECK(images.dim() == 1 && images.numel() == n_wires,
                "CX image count must match n_wires");
    TORCH_CHECK(images.scalar_type() == at::kLong, "CX images must be int64");
  }

  const int64_t* kind_data = gate_kinds.const_data_ptr<int64_t>();
  const int64_t* wire_data = wires.const_data_ptr<int64_t>();
  uint64_t completed_wires = 0;
  int64_t previous_wire = -1;
  int64_t unique_wire_count = 0;
  for (int64_t gate = 0; gate < gate_count; ++gate) {
    TORCH_CHECK(kind_data[gate] >= 0 && kind_data[gate] <= 2, "unsupported gate kind");
    TORCH_CHECK(
        wire_data[gate] >= 0 && wire_data[gate] < n_wires,
        "rotation wire is out of range");
    if (wire_data[gate] != previous_wire) {
      if (previous_wire >= 0) {
        completed_wires |= uint64_t{1} << previous_wire;
      }
      TORCH_CHECK(
          (completed_wires & (uint64_t{1} << wire_data[gate])) == 0,
          "rotation wires must form contiguous groups");
      previous_wire = wire_data[gate];
      ++unique_wire_count;
    }
  }
  TORCH_CHECK(unique_wire_count >= 2, "a fused rotation segment requires at least two wires");
  if (!restore_state) {
    TORCH_CHECK(enable_pair_fast_path && enable_flat_pair_simd,
                "terminal no-restore requires the flat Euler fast path");
    TORCH_CHECK(gate_count % 3 == 0,
                "terminal no-restore requires complete Euler triples");
    for (int64_t gate = 0; gate < gate_count; gate += 3) {
      TORCH_CHECK(
          kind_data[gate] == 2 && kind_data[gate + 1] == 1 &&
              kind_data[gate + 2] == 0 &&
              wire_data[gate] == wire_data[gate + 1] &&
              wire_data[gate] == wire_data[gate + 2],
          "terminal no-restore requires contiguous RZ-RY-RX triples");
    }
  }
  at::Tensor result;
  at::Tensor transformed_ket = fuse_cx && restore_state ? at::empty_like(ket) : ket;
  at::Tensor transformed_adjoint =
      fuse_cx && restore_state ? at::empty_like(adjoint) : adjoint;
  AT_DISPATCH_COMPLEX_TYPES(ket.scalar_type(), "fused_rotation_segment_adjoint_cpu", [&] {
    using real_t = typename scalar_t::value_type;
    constexpr real_t inverse_sqrt_two =
        static_cast<real_t>(0.7071067811865475244);
    TORCH_CHECK(
        angles.scalar_type() == c10::CppTypeToScalarType<real_t>::value,
        "angle dtype must match the state precision");
    const scalar_t* source_ket_data = ket.const_data_ptr<scalar_t>();
    const scalar_t* source_adjoint_data = adjoint.const_data_ptr<scalar_t>();
    scalar_t* ket_data = transformed_ket.data_ptr<scalar_t>();
    scalar_t* adjoint_data = transformed_adjoint.data_ptr<scalar_t>();
    const int32_t* cx_index32 = fuse_cx_index && cx_index.value().scalar_type() == at::kInt
        ? cx_index.value().const_data_ptr<int32_t>()
        : nullptr;
    const int64_t* cx_index64 = fuse_cx_index && cx_index.value().scalar_type() == at::kLong
        ? cx_index.value().const_data_ptr<int64_t>()
        : nullptr;
    const int64_t cx_chunk_count = fuse_cx_images ? (n_wires + 7) / 8 : 0;
    const LinearLookup cx_lookup = fuse_cx_images
        ? build_linear_lookup(cx_images.value())
        : LinearLookup{};
    const int64_t cx_shift_mode = fuse_cx_images
        ? linear_shift_mode(cx_images.value())
        : 0;
    const uint64_t cx_state_mask = (uint64_t{1} << n_wires) - 1;
    const real_t* angle_data = angles.const_data_ptr<real_t>();
    const real_t* observable_weight_data = nullptr;
    if (seed_observable) {
      TORCH_CHECK(
          observable_weights.value().scalar_type() ==
              c10::CppTypeToScalarType<real_t>::value,
          "observable weight dtype must match state precision");
      observable_weight_data =
          observable_weights.value().const_data_ptr<real_t>();
    }
    const real_t* rzz_angle_data = nullptr;
    std::vector<scalar_t> rzz_inverses;
    if (fuse_rzz) {
      TORCH_CHECK(rzz_angles.value().scalar_type() == c10::CppTypeToScalarType<real_t>::value,
                  "fused RZZ angle dtype must match state precision");
      rzz_angle_data = rzz_angles.value().const_data_ptr<real_t>();
      for (int64_t gate = 1; gate < rzz_count; ++gate) {
        TORCH_CHECK(rzz_angle_data[gate] == rzz_angle_data[0],
                    "fused RZZ gates must share one angle");
      }
      rzz_inverses.resize(rzz_count + 1);
      for (int64_t transitions = 0; transitions <= rzz_count; ++transitions) {
        const real_t sign_sum = static_cast<real_t>(rzz_count - 2 * transitions);
        const real_t phase = real_t{0.5} * sign_sum * rzz_angle_data[0];
        rzz_inverses[transitions] = scalar_t(std::cos(phase), std::sin(phase));
      }
    }
    std::vector<real_t> cosines(gate_count);
    std::vector<real_t> sines(gate_count);
    std::vector<real_t> full_cosines(gate_count);
    std::vector<real_t> full_sines(gate_count);
    std::vector<scalar_t> euler_inverses(gate_count * 4);
    for (int64_t gate = 0; gate < gate_count; ++gate) {
      cosines[gate] = std::cos(angle_data[gate] * real_t{0.5});
      sines[gate] = std::sin(angle_data[gate] * real_t{0.5});
      if (enable_flat_pair_simd) {
        full_cosines[gate] = std::cos(angle_data[gate]);
        full_sines[gate] = std::sin(angle_data[gate]);
      }
    }
    for (int64_t gate = 0;
         restore_state && enable_flat_pair_simd && gate + 2 < gate_count;
         ++gate) {
      if (kind_data[gate] == 2 && kind_data[gate + 1] == 1 &&
          kind_data[gate + 2] == 0 &&
          wire_data[gate] == wire_data[gate + 1] &&
          wire_data[gate] == wire_data[gate + 2]) {
        scalar_t column_zero{1, 0};
        scalar_t column_one{0, 0};
        apply_inverse_rotation_pair<2>(
            cosines[gate], sines[gate], column_zero, column_one);
        apply_inverse_rotation_pair<1>(
            cosines[gate + 1], sines[gate + 1], column_zero, column_one);
        apply_inverse_rotation_pair<0>(
            cosines[gate + 2], sines[gate + 2], column_zero, column_one);
        euler_inverses[gate * 4] = column_zero;
        euler_inverses[gate * 4 + 2] = column_one;
        column_zero = scalar_t{0, 0};
        column_one = scalar_t{1, 0};
        apply_inverse_rotation_pair<2>(
            cosines[gate], sines[gate], column_zero, column_one);
        apply_inverse_rotation_pair<1>(
            cosines[gate + 1], sines[gate + 1], column_zero, column_one);
        apply_inverse_rotation_pair<0>(
            cosines[gate + 2], sines[gate + 2], column_zero, column_one);
        euler_inverses[gate * 4 + 1] = column_zero;
        euler_inverses[gate * 4 + 3] = column_one;
      }
    }
    const int64_t thread_count = at::get_num_threads();
    const int64_t gradient_count = aggregate_shared_parameter ? 1 : gate_count;
    const int64_t result_gradient_count = gradient_count + (fuse_rzz ? 1 : 0);
    at::Tensor partial = at::zeros(
        {thread_count, result_gradient_count},
        ket.options().dtype(c10::CppTypeToScalarType<real_t>::value));
    real_t* partial_data = partial.data_ptr<real_t>();

    int64_t segment_start = 0;
    while (segment_start < gate_count) {
      std::array<int64_t, 11> unique_wires{};
      std::array<int64_t, 11> unique_masks{};
      std::array<int64_t, 11> bit_positions{};
      std::vector<int64_t> local_masks(gate_count);
      int64_t unique_count = 0;
      int64_t segment_end = segment_start;
      while (segment_end < gate_count) {
        int64_t local_wire = -1;
        for (int64_t candidate = 0; candidate < unique_count; ++candidate) {
          if (unique_wires[candidate] == wire_data[segment_end]) {
            local_wire = candidate;
            break;
          }
        }
        if (local_wire < 0) {
          if (unique_count == max_tile_wires) {
            break;
          }
          local_wire = unique_count++;
          unique_wires[local_wire] = wire_data[segment_end];
          const int64_t bit_position = n_wires - wire_data[segment_end] - 1;
          unique_masks[local_wire] = int64_t{1} << bit_position;
          bit_positions[local_wire] = bit_position;
        }
        local_masks[segment_end] = int64_t{1} << local_wire;
        ++segment_end;
      }
      std::sort(bit_positions.begin(), bit_positions.begin() + unique_count);
      const int64_t local_size = int64_t{1} << unique_count;
      std::array<int64_t, 2048> offsets{};
      for (int64_t local = 0; local < local_size; ++local) {
        int64_t offset = 0;
        for (int64_t wire = 0; wire < unique_count; ++wire) {
          if ((local & (int64_t{1} << wire)) != 0) {
            offset |= unique_masks[wire];
          }
        }
        offsets[local] = offset;
      }
      const int64_t blocks_per_row = amplitudes >> unique_count;
      const int64_t item_count = ket.size(0) * blocks_per_row;
      const int64_t effective_parallel_grain =
          parallel_grain == 0
          ? (item_count >= 1024
                 ? 128
                 : std::max<int64_t>(1, item_count / (4 * thread_count)))
          : parallel_grain;
      at::parallel_for(
          int64_t{0},
          item_count,
          effective_parallel_grain,
          [&](int64_t begin, int64_t end) {
        std::array<scalar_t, 2048> ket_values{};
        std::array<scalar_t, 2048> adjoint_values{};
        real_t* local_gradients =
            partial_data + at::get_thread_num() * result_gradient_count;
        for (int64_t item = begin; item < end; ++item) {
          const int64_t row = item / blocks_per_row;
          int64_t base = item - row * blocks_per_row;
          for (int64_t wire = 0; wire < unique_count; ++wire) {
            const int64_t position = bit_positions[wire];
            const int64_t lower_mask = (int64_t{1} << position) - 1;
            base = (base & lower_mask) | ((base & ~lower_mask) << 1);
          }
          base += row * amplitudes;
          for (int64_t local = 0; local < local_size; ++local) {
            const int64_t item_index = base + offsets[local];
            const bool load_source = segment_start == 0 || !restore_state;
            int64_t source_index = item_index;
            if (fuse_cx && load_source) {
              const int64_t destination = item_index - row * amplitudes;
              const int64_t source = cx_index32 != nullptr
                  ? static_cast<int64_t>(cx_index32[destination])
                  : cx_index64 != nullptr
                      ? cx_index64[destination]
                      : apply_linear_lookup(
                            static_cast<uint64_t>(destination),
                            cx_lookup,
                            cx_chunk_count,
                            cx_shift_mode,
                            cx_state_mask);
              source_index = row * amplitudes + source;
            }
            ket_values[local] = load_source
                ? source_ket_data[source_index]
                : ket_data[item_index];
            if (seed_observable && load_source) {
              const int64_t basis = source_index - row * amplitudes;
              adjoint_values[local] = ket_values[local] *
                  (real_t{2} * observable_weight_data[basis]);
            } else {
              adjoint_values[local] = load_source
                  ? source_adjoint_data[source_index]
                  : adjoint_data[item_index];
            }
          }
          for (int64_t gate = segment_start; gate < segment_end; ++gate) {
            const int64_t local_mask = local_masks[gate];
            const bool euler_triple = enable_pair_fast_path &&
                gate + 2 < segment_end &&
                local_masks[gate + 1] == local_mask &&
                local_masks[gate + 2] == local_mask &&
                kind_data[gate] == 2 && kind_data[gate + 1] == 1 &&
                kind_data[gate + 2] == 0;
            if (euler_triple) {
              real_t gradient_rz = real_t{0};
              real_t gradient_ry = real_t{0};
              real_t gradient_rx = real_t{0};
              const int64_t local_block = local_mask << 1;
              if (enable_flat_pair_simd) {
                if (enable_euler_post_reduction) {
                  real_t raw_gradient_x = real_t{0};
                  real_t raw_gradient_y = real_t{0};
                  real_t raw_gradient_z = real_t{0};
                  FQ_OPENMP_SIMD_PRAGMA(
                      reduction(+:raw_gradient_x,raw_gradient_y,raw_gradient_z))
                  for (int64_t pair = 0; pair < local_size / 2; ++pair) {
                    const int64_t local = (pair & (local_mask - 1)) |
                        ((pair & ~(local_mask - 1)) << 1);
                    const int64_t one = local + local_mask;
                    scalar_t ket_zero = ket_values[local];
                    scalar_t ket_one = ket_values[one];
                    scalar_t adjoint_zero = adjoint_values[local];
                    scalar_t adjoint_one = adjoint_values[one];
                    real_t pair_gradient_x;
                    real_t pair_gradient_y;
                    real_t pair_gradient_z;
                    rotation_pair_gradients(
                        ket_zero,
                        ket_one,
                        adjoint_zero,
                        adjoint_one,
                        pair_gradient_x,
                        pair_gradient_y,
                        pair_gradient_z);
                    raw_gradient_x += pair_gradient_x;
                    raw_gradient_y += pair_gradient_y;
                    raw_gradient_z += pair_gradient_z;
                    if (restore_state) {
                      const scalar_t* inverse = euler_inverses.data() + gate * 4;
                      ket_values[local] =
                          inverse[0] * ket_zero + inverse[1] * ket_one;
                      ket_values[one] =
                          inverse[2] * ket_zero + inverse[3] * ket_one;
                      adjoint_values[local] =
                          inverse[0] * adjoint_zero + inverse[1] * adjoint_one;
                      adjoint_values[one] =
                          inverse[2] * adjoint_zero + inverse[3] * adjoint_one;
                    }
                  }
                  gradient_rz = raw_gradient_z;
                  gradient_ry =
                      -full_sines[gate] * raw_gradient_x +
                      full_cosines[gate] * raw_gradient_y;
                  gradient_rx =
                      full_cosines[gate + 1] * full_cosines[gate] *
                          raw_gradient_x +
                      full_cosines[gate + 1] * full_sines[gate] *
                          raw_gradient_y -
                      full_sines[gate + 1] * raw_gradient_z;
                } else {
                  FQ_OPENMP_SIMD_PRAGMA(
                      reduction(+:gradient_rz,gradient_ry,gradient_rx))
                  for (int64_t pair = 0; pair < local_size / 2; ++pair) {
                    const int64_t local = (pair & (local_mask - 1)) |
                        ((pair & ~(local_mask - 1)) << 1);
                    const int64_t one = local + local_mask;
                    scalar_t ket_zero = ket_values[local];
                    scalar_t ket_one = ket_values[one];
                    scalar_t adjoint_zero = adjoint_values[local];
                    scalar_t adjoint_one = adjoint_values[one];
                    real_t gradient_x;
                    real_t gradient_y;
                    real_t gradient_z;
                    rotation_pair_gradients(
                        ket_zero,
                        ket_one,
                        adjoint_zero,
                        adjoint_one,
                        gradient_x,
                        gradient_y,
                        gradient_z);
                    gradient_rz += gradient_z;
                    gradient_ry +=
                        -full_sines[gate] * gradient_x +
                        full_cosines[gate] * gradient_y;
                    gradient_rx +=
                        full_cosines[gate + 1] * full_cosines[gate] *
                            gradient_x +
                        full_cosines[gate + 1] * full_sines[gate] * gradient_y -
                        full_sines[gate + 1] * gradient_z;
                    if (restore_state) {
                      const scalar_t* inverse = euler_inverses.data() + gate * 4;
                      ket_values[local] =
                          inverse[0] * ket_zero + inverse[1] * ket_one;
                      ket_values[one] =
                          inverse[2] * ket_zero + inverse[3] * ket_one;
                      adjoint_values[local] =
                          inverse[0] * adjoint_zero + inverse[1] * adjoint_one;
                      adjoint_values[one] =
                          inverse[2] * adjoint_zero + inverse[3] * adjoint_one;
                    }
                  }
                }
              } else {
                for (int64_t block = 0; block < local_size;
                     block += local_block) {
                  FQ_OPENMP_SIMD_PRAGMA(
                      reduction(+:gradient_rz,gradient_ry,gradient_rx))
                  for (int64_t offset = 0; offset < local_mask; ++offset) {
                    const int64_t local = block + offset;
                    const int64_t one = local + local_mask;
                    scalar_t ket_zero = ket_values[local];
                    scalar_t ket_one = ket_values[one];
                    scalar_t adjoint_zero = adjoint_values[local];
                    scalar_t adjoint_one = adjoint_values[one];
                    gradient_rz += rotation_pair_gradient<2>(
                        ket_zero, ket_one, adjoint_zero, adjoint_one);
                    if (restore_state) {
                      apply_inverse_rotation_pair<2>(
                          cosines[gate], sines[gate], ket_zero, ket_one);
                      apply_inverse_rotation_pair<2>(
                          cosines[gate], sines[gate], adjoint_zero, adjoint_one);
                    }
                    gradient_ry += rotation_pair_gradient<1>(
                        ket_zero, ket_one, adjoint_zero, adjoint_one);
                    if (restore_state) {
                      apply_inverse_rotation_pair<1>(
                          cosines[gate + 1], sines[gate + 1], ket_zero, ket_one);
                      apply_inverse_rotation_pair<1>(
                          cosines[gate + 1], sines[gate + 1],
                          adjoint_zero, adjoint_one);
                    }
                    gradient_rx += rotation_pair_gradient<0>(
                        ket_zero, ket_one, adjoint_zero, adjoint_one);
                    if (restore_state) {
                      apply_inverse_rotation_pair<0>(
                          cosines[gate + 2], sines[gate + 2], ket_zero, ket_one);
                      apply_inverse_rotation_pair<0>(
                          cosines[gate + 2], sines[gate + 2],
                          adjoint_zero, adjoint_one);
                      ket_values[local] = ket_zero;
                      ket_values[one] = ket_one;
                      adjoint_values[local] = adjoint_zero;
                      adjoint_values[one] = adjoint_one;
                    }
                  }
                }
              }
              local_gradients[aggregate_shared_parameter ? 0 : gate] +=
                  gradient_rz;
              local_gradients[aggregate_shared_parameter ? 0 : gate + 1] +=
                  gradient_ry;
              local_gradients[aggregate_shared_parameter ? 0 : gate + 2] +=
                  gradient_rx;
              gate += 2;
              continue;
            }
            auto apply_gate = [&](auto kind_tag) {
              constexpr int64_t gate_kind = decltype(kind_tag)::value;
              real_t gate_gradient = real_t{0};
              if (enable_pair_fast_path) {
                const int64_t local_block = local_mask << 1;
                if (enable_flat_pair_simd) {
                  FQ_OPENMP_SIMD_PRAGMA(reduction(+:gate_gradient))
                  for (int64_t pair = 0; pair < local_size / 2; ++pair) {
                    const int64_t local = (pair & (local_mask - 1)) |
                        ((pair & ~(local_mask - 1)) << 1);
                    const int64_t one = local + local_mask;
                    const scalar_t ket_zero = ket_values[local];
                    const scalar_t ket_one = ket_values[one];
                    const scalar_t adjoint_zero = adjoint_values[local];
                    const scalar_t adjoint_one = adjoint_values[one];
                    gate_gradient += rotation_pair_gradient<gate_kind>(
                        ket_zero, ket_one, adjoint_zero, adjoint_one);
                    if (restore_state) {
                      apply_inverse_rotation_pair<gate_kind>(
                          cosines[gate], sines[gate],
                          ket_values[local], ket_values[one]);
                      apply_inverse_rotation_pair<gate_kind>(
                          cosines[gate], sines[gate],
                          adjoint_values[local], adjoint_values[one]);
                    }
                  }
                } else {
                  for (int64_t block = 0; block < local_size;
                       block += local_block) {
                    FQ_OPENMP_SIMD_PRAGMA(reduction(+:gate_gradient))
                    for (int64_t offset = 0; offset < local_mask; ++offset) {
                      const int64_t local = block + offset;
                      const int64_t one = local + local_mask;
                      const scalar_t ket_zero = ket_values[local];
                      const scalar_t ket_one = ket_values[one];
                      const scalar_t adjoint_zero = adjoint_values[local];
                      const scalar_t adjoint_one = adjoint_values[one];
                      gate_gradient += rotation_pair_gradient<gate_kind>(
                          ket_zero, ket_one, adjoint_zero, adjoint_one);
                      if (restore_state) {
                        apply_inverse_rotation_pair<gate_kind>(
                            cosines[gate], sines[gate],
                            ket_values[local], ket_values[one]);
                        apply_inverse_rotation_pair<gate_kind>(
                            cosines[gate], sines[gate],
                            adjoint_values[local], adjoint_values[one]);
                      }
                    }
                  }
                }
              } else {
                for (int64_t local = 0; local < local_size; ++local) {
                  if ((local & local_mask) != 0) {
                    continue;
                  }
                  const int64_t one = local | local_mask;
                  const scalar_t ket_zero = ket_values[local];
                  const scalar_t ket_one = ket_values[one];
                  const scalar_t adjoint_zero = adjoint_values[local];
                  const scalar_t adjoint_one = adjoint_values[one];
                  gate_gradient += rotation_pair_gradient<gate_kind>(
                      ket_zero, ket_one, adjoint_zero, adjoint_one);
                  if (restore_state) {
                    apply_inverse_rotation_pair<gate_kind>(
                        cosines[gate], sines[gate],
                        ket_values[local], ket_values[one]);
                    apply_inverse_rotation_pair<gate_kind>(
                        cosines[gate], sines[gate],
                        adjoint_values[local], adjoint_values[one]);
                  }
                }
              }
              local_gradients[aggregate_shared_parameter ? 0 : gate] +=
                  gate_gradient;
            };
            if (kind_data[gate] == 0) {
              apply_gate(std::integral_constant<int64_t, 0>{});
            } else if (kind_data[gate] == 1) {
              apply_gate(std::integral_constant<int64_t, 1>{});
            } else {
              apply_gate(std::integral_constant<int64_t, 2>{});
            }
          }
          if (fuse_rzz && segment_end == gate_count) {
            real_t rzz_gradient = real_t{0};
            for (int64_t local = 0; local < local_size; ++local) {
              const int64_t basis = base + offsets[local] - row * amplitudes;
              const uint64_t transitions =
                  (static_cast<uint64_t>(basis) ^
                   (static_cast<uint64_t>(basis) >> 1)) & rzz_transition_mask;
              const int64_t transition_count = popcount(transitions);
              const real_t sign_sum = static_cast<real_t>(
                  rzz_count - 2 * transition_count);
              const scalar_t ket_value = ket_values[local];
              const scalar_t adjoint_value = adjoint_values[local];
              const real_t cross = adjoint_value.real() * ket_value.imag() -
                  adjoint_value.imag() * ket_value.real();
              rzz_gradient += real_t{0.5} * sign_sum * cross;
              const scalar_t inverse = rzz_inverses[transition_count];
              ket_values[local] = inverse * ket_value;
              adjoint_values[local] = inverse * adjoint_value;
            }
            local_gradients[gradient_count] += rzz_gradient;
          }
          if (fuse_preceding_hadamards && segment_end == gate_count) {
            for (int64_t wire = 0; wire < unique_count; ++wire) {
              const int64_t local_mask = int64_t{1} << wire;
              for (int64_t block = 0; block < local_size;
                   block += 2 * local_mask) {
                FQ_OPENMP_SIMD_PRAGMA()
                for (int64_t offset = 0; offset < local_mask; ++offset) {
                  const int64_t local = block + offset;
                  const int64_t one = local + local_mask;
                  const scalar_t ket_zero = ket_values[local];
                  const scalar_t ket_one = ket_values[one];
                  ket_values[local] = scalar_t(
                      (ket_zero.real() + ket_one.real()) * inverse_sqrt_two,
                      (ket_zero.imag() + ket_one.imag()) * inverse_sqrt_two);
                  ket_values[one] = scalar_t(
                      (ket_zero.real() - ket_one.real()) * inverse_sqrt_two,
                      (ket_zero.imag() - ket_one.imag()) * inverse_sqrt_two);
                  const scalar_t adjoint_zero = adjoint_values[local];
                  const scalar_t adjoint_one = adjoint_values[one];
                  adjoint_values[local] = scalar_t(
                      (adjoint_zero.real() + adjoint_one.real()) * inverse_sqrt_two,
                      (adjoint_zero.imag() + adjoint_one.imag()) * inverse_sqrt_two);
                  adjoint_values[one] = scalar_t(
                      (adjoint_zero.real() - adjoint_one.real()) * inverse_sqrt_two,
                      (adjoint_zero.imag() - adjoint_one.imag()) * inverse_sqrt_two);
                }
              }
            }
          }
          if (restore_state) {
            for (int64_t local = 0; local < local_size; ++local) {
              ket_data[base + offsets[local]] = ket_values[local];
              adjoint_data[base + offsets[local]] = adjoint_values[local];
            }
          }
        }
      });
      segment_start = segment_end;
    }
    result = at::zeros(
        {result_gradient_count}, ket.options().dtype(c10::CppTypeToScalarType<real_t>::value));
    real_t* result_data = result.data_ptr<real_t>();
    for (int64_t thread = 0; thread < thread_count; ++thread) {
      for (int64_t gradient = 0; gradient < result_gradient_count; ++gradient) {
        result_data[gradient] +=
            partial_data[thread * result_gradient_count + gradient];
      }
    }
  });
  return std::make_tuple(result, transformed_ket, transformed_adjoint);
}

at::Tensor fused_rzz_segment_adjoint_cpu(
    at::Tensor ket,
    at::Tensor adjoint,
    const at::Tensor& angles,
    const at::Tensor& first_wires,
    const at::Tensor& second_wires,
    int64_t n_wires,
    bool aggregate_shared_parameter) {
  TORCH_CHECK(ket.device().is_cpu() && adjoint.device().is_cpu(), "states must be on CPU");
  TORCH_CHECK(
      angles.device().is_cpu() && first_wires.device().is_cpu() && second_wires.device().is_cpu(),
      "segment metadata must be on CPU");
  TORCH_CHECK(ket.is_contiguous() && adjoint.is_contiguous(), "states must be contiguous");
  TORCH_CHECK(
      angles.is_contiguous() && first_wires.is_contiguous() && second_wires.is_contiguous(),
      "segment metadata must be contiguous");
  TORCH_CHECK(ket.dim() == 2 && adjoint.sizes() == ket.sizes(), "state shapes must match");
  TORCH_CHECK(adjoint.scalar_type() == ket.scalar_type(), "state dtypes must match");
  TORCH_CHECK(
      ket.scalar_type() == at::kComplexFloat || ket.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(first_wires.scalar_type() == at::kLong, "first_wires must be int64");
  TORCH_CHECK(second_wires.scalar_type() == at::kLong, "second_wires must be int64");
  TORCH_CHECK(angles.dim() == 1 && first_wires.dim() == 1 && second_wires.dim() == 1,
              "segment metadata must be one-dimensional");
  TORCH_CHECK(
      angles.numel() > 1 && first_wires.numel() == angles.numel() &&
          second_wires.numel() == angles.numel(),
      "segment metadata lengths must match and contain at least two gates");
  TORCH_CHECK(n_wires > 1 && n_wires < 63, "n_wires must be in [2, 62]");

  const int64_t amplitudes = int64_t{1} << n_wires;
  TORCH_CHECK(ket.size(1) == amplitudes, "state width does not match n_wires");
  const int64_t gate_count = angles.numel();
  const int64_t item_count = ket.numel();
  const int64_t* first_data = first_wires.const_data_ptr<int64_t>();
  const int64_t* second_data = second_wires.const_data_ptr<int64_t>();
  std::vector<int64_t> first_masks(gate_count);
  std::vector<int64_t> second_masks(gate_count);
  uint64_t transition_mask = 0;
  for (int64_t gate = 0; gate < gate_count; ++gate) {
    TORCH_CHECK(
        first_data[gate] >= 0 && first_data[gate] < n_wires &&
            second_data[gate] >= 0 && second_data[gate] < n_wires &&
            first_data[gate] != second_data[gate],
        "RZZ segment contains an invalid wire pair");
    first_masks[gate] = int64_t{1} << (n_wires - first_data[gate] - 1);
    second_masks[gate] = int64_t{1} << (n_wires - second_data[gate] - 1);
    if (aggregate_shared_parameter) {
      TORCH_CHECK(
          std::abs(first_data[gate] - second_data[gate]) == 1,
          "shared-parameter RZZ fusion requires adjacent wire pairs");
      const int64_t transition_bit =
          std::min(first_masks[gate], second_masks[gate]);
      TORCH_CHECK(
          (transition_mask & transition_bit) == 0,
          "shared-parameter RZZ fusion requires unique wire pairs");
      transition_mask |= transition_bit;
    }
  }

  at::Tensor result;
  AT_DISPATCH_COMPLEX_TYPES(ket.scalar_type(), "fused_rzz_segment_adjoint_cpu", [&] {
    using real_t = typename scalar_t::value_type;
    TORCH_CHECK(
        angles.scalar_type() == c10::CppTypeToScalarType<real_t>::value,
        "angle dtype must match the state precision");
    scalar_t* ket_data = ket.data_ptr<scalar_t>();
    scalar_t* adjoint_data = adjoint.data_ptr<scalar_t>();
    const real_t* angle_data = angles.const_data_ptr<real_t>();
    if (aggregate_shared_parameter) {
      for (int64_t gate = 1; gate < gate_count; ++gate) {
        TORCH_CHECK(
            angle_data[gate] == angle_data[0],
            "shared-parameter RZZ fusion requires equal angles");
      }
    }
    const int64_t thread_count = std::max<int64_t>(1, at::get_num_threads());
    at::Tensor partials = at::zeros(
        {thread_count, gate_count}, angles.options());
    real_t* partial_data = partials.data_ptr<real_t>();

    at::parallel_for(int64_t{0}, item_count, int64_t{4096}, [&](int64_t begin, int64_t end) {
      real_t* local = partial_data + at::get_thread_num() * gate_count;
      for (int64_t item = begin; item < end; ++item) {
        const int64_t basis = item & (amplitudes - 1);
        const scalar_t ket_value = ket_data[item];
        const scalar_t adjoint_value = adjoint_data[item];
        const real_t cross = adjoint_value.real() * ket_value.imag() -
            adjoint_value.imag() * ket_value.real();
        real_t inverse_phase = real_t{0};
        if (aggregate_shared_parameter) {
          const uint64_t transitions =
              (static_cast<uint64_t>(basis) ^ (static_cast<uint64_t>(basis) >> 1)) &
              transition_mask;
          const real_t sign_sum = static_cast<real_t>(
              gate_count - 2 * popcount(transitions));
          local[0] += real_t{0.5} * sign_sum * cross;
          inverse_phase = real_t{0.5} * sign_sum * angle_data[0];
        } else {
          for (int64_t gate = 0; gate < gate_count; ++gate) {
            const bool odd_parity = ((basis & first_masks[gate]) != 0) !=
                ((basis & second_masks[gate]) != 0);
            const real_t sign = odd_parity ? real_t{-1} : real_t{1};
            local[gate] += real_t{0.5} * sign * cross;
            inverse_phase += real_t{0.5} * sign * angle_data[gate];
          }
        }
        const scalar_t inverse(std::cos(inverse_phase), std::sin(inverse_phase));
        ket_data[item] = inverse * ket_value;
        adjoint_data[item] = inverse * adjoint_value;
      }
    });

    result = at::zeros({gate_count}, angles.options());
    real_t* result_data = result.data_ptr<real_t>();
    for (int64_t thread = 0; thread < thread_count; ++thread) {
      for (int64_t gate = 0; gate < gate_count; ++gate) {
        result_data[gate] += partial_data[thread * gate_count + gate];
      }
    }
  });
  return result;
}

}  // namespace

TORCH_LIBRARY(flagquantum_native, library) {
  library.def(
      "fused_observable_expectation(Tensor ket, Tensor weights) -> Tensor");
  library.def(
      "fused_observable_adjoint_seed(Tensor ket, Tensor weights) -> Tensor");
  library.def(
      "fused_rotation_block_forward_(Tensor(a!) state, Tensor matrices, "
      "Tensor wires, int n_wires, Tensor? rzz_angles=None, "
      "Tensor? rzz_first_wires=None, Tensor? rzz_second_wires=None, "
      "bool specialized_rotations=True, Tensor? cx_controls=None, "
      "Tensor? cx_targets=None) -> Tensor(a!)");
  library.def(
      "fused_product_state_initialization_(Tensor(a!) state, Tensor matrices, "
      "Tensor wires, int n_wires, Tensor cx_controls, Tensor cx_targets) "
      "-> Tensor(a!)");
  library.def(
      "fused_static_product_state_initialization_(Tensor(a!) state, "
      "Tensor gate_codes, Tensor wires, int n_wires) -> Tensor(a!)");
  library.def(
      "fused_static_clifford_layer_(Tensor(a!) state, Tensor gate_codes, "
      "Tensor wires, int n_wires) -> Tensor(a!)");
  library.def(
      "fused_hadamard_block_adjoint_(Tensor(a!) ket, Tensor(b!) adjoint, "
      "Tensor wires, int n_wires) -> Tensor(a!)");
  library.def(
      "fused_rotation_adjoint_(Tensor(a!) ket, Tensor(b!) adjoint, "
      "Tensor matrix, int wire, int second_wire, int n_wires, int gate_kind) -> Tensor");
  library.def(
      "fused_rzz_segment_forward_(Tensor(a!) state, Tensor angles, "
      "Tensor first_wires, Tensor second_wires, int n_wires, "
      "bool enable_shared_phase_lookup) -> Tensor(a!)");
  library.def(
      "fused_rotation_segment_adjoint_(Tensor(a!) ket, Tensor(b!) adjoint, "
      "Tensor angles, Tensor gate_kinds, Tensor wires, int n_wires, "
      "bool aggregate_shared_parameter, int max_tile_wires, "
      "int parallel_grain, "
      "bool enable_pair_fast_path, bool enable_flat_pair_simd, "
      "bool enable_euler_post_reduction, bool restore_state, "
      "bool fuse_preceding_hadamards, "
      "Tensor? rzz_angles=None, "
      "Tensor? rzz_first_wires=None, Tensor? rzz_second_wires=None, "
      "Tensor? observable_weights=None, Tensor? cx_index=None, "
      "Tensor? cx_images=None) "
      "-> (Tensor, Tensor, Tensor)");
  library.def(
      "fused_rzz_segment_adjoint_(Tensor(a!) ket, Tensor(b!) adjoint, "
      "Tensor angles, Tensor first_wires, Tensor second_wires, int n_wires, "
      "bool aggregate_shared_parameter) -> Tensor");
}

TORCH_LIBRARY_IMPL(flagquantum_native, CPU, library) {
  library.impl("fused_observable_expectation", &fused_observable_expectation_cpu);
  library.impl(
      "fused_observable_adjoint_seed", &fused_observable_adjoint_seed_cpu);
  library.impl("fused_rotation_block_forward_", &fused_rotation_block_forward_cpu);
  library.impl(
      "fused_product_state_initialization_",
      &fused_product_state_initialization_cpu);
  library.impl(
      "fused_static_product_state_initialization_",
      &fused_static_product_state_initialization_cpu);
  library.impl("fused_static_clifford_layer_", &fused_static_clifford_layer_cpu);
  library.impl("fused_hadamard_block_adjoint_", &fused_hadamard_block_adjoint_cpu);
  library.impl("fused_rotation_adjoint_", &fused_rotation_adjoint_cpu);
  library.impl("fused_rzz_segment_forward_", &fused_rzz_segment_forward_cpu);
  library.impl("fused_rotation_segment_adjoint_", &fused_rotation_segment_adjoint_cpu);
  library.impl("fused_rzz_segment_adjoint_", &fused_rzz_segment_adjoint_cpu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("parallel_build_available", []() {
#ifdef INTRA_OP_PARALLEL
    return true;
#else
    return false;
#endif
  });
}
