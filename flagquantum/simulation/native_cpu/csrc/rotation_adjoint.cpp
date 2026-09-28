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
#include <vector>

namespace {

inline int64_t popcount(uint64_t value) {
  int64_t count = 0;
  while (value != 0) {
    value &= value - 1;
    ++count;
  }
  return count;
}

at::Tensor fused_rotation_block_forward_cpu(
    at::Tensor state,
    const at::Tensor& matrices,
    const at::Tensor& wires,
    int64_t n_wires) {
  TORCH_CHECK(state.device().is_cpu(), "state must be on CPU");
  TORCH_CHECK(matrices.device().is_cpu(), "matrices must be on CPU");
  TORCH_CHECK(wires.device().is_cpu(), "wires must be on CPU");
  TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
  TORCH_CHECK(matrices.is_contiguous(), "matrices must be contiguous");
  TORCH_CHECK(wires.is_contiguous(), "wires must be contiguous");
  TORCH_CHECK(state.dim() == 2, "state must have shape [batch, amplitudes]");
  TORCH_CHECK(matrices.dim() == 3, "matrices must have shape [gates, 2, 2]");
  TORCH_CHECK(
      matrices.size(1) == 2 && matrices.size(2) == 2,
      "every rotation matrix must be 2 by 2");
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

  const int64_t gate_count = matrices.size(0);
  TORCH_CHECK(
      gate_count >= 2 && gate_count <= 6,
      "a fused rotation block must contain between two and six wires");
  TORCH_CHECK(wires.numel() == gate_count, "wire and matrix counts must match");
  const int64_t amplitudes = int64_t{1} << n_wires;
  TORCH_CHECK(state.size(1) == amplitudes, "state width does not match n_wires");

  const int64_t* wire_data = wires.const_data_ptr<int64_t>();
  std::array<int64_t, 6> masks{};
  std::array<int64_t, 6> bit_positions{};
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

  const int64_t local_size = int64_t{1} << gate_count;
  std::array<int64_t, 64> offsets{};
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
  const int64_t item_count = state.size(0) * blocks_per_row;
  AT_DISPATCH_COMPLEX_TYPES(state.scalar_type(), "fused_rotation_block_forward_cpu", [&] {
    scalar_t* state_data = state.data_ptr<scalar_t>();
    const scalar_t* matrix_data = matrices.const_data_ptr<scalar_t>();
    at::parallel_for(int64_t{0}, item_count, int64_t{256}, [&](int64_t begin, int64_t end) {
      std::array<scalar_t, 64> values{};
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
          values[local] = state_data[base + offsets[local]];
        }
        for (int64_t gate = 0; gate < gate_count; ++gate) {
          const scalar_t* matrix = matrix_data + gate * 4;
          const int64_t local_mask = int64_t{1} << gate;
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
        }
        for (int64_t local = 0; local < local_size; ++local) {
          state_data[base + offsets[local]] = values[local];
        }
      }
    });
  });
  return state;
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
      "fused_rotation_block_forward_(Tensor(a!) state, Tensor matrices, "
      "Tensor wires, int n_wires) -> Tensor(a!)");
  library.def(
      "fused_rotation_adjoint_(Tensor(a!) ket, Tensor(b!) adjoint, "
      "Tensor matrix, int wire, int second_wire, int n_wires, int gate_kind) -> Tensor");
  library.def(
      "fused_rzz_segment_adjoint_(Tensor(a!) ket, Tensor(b!) adjoint, "
      "Tensor angles, Tensor first_wires, Tensor second_wires, int n_wires, "
      "bool aggregate_shared_parameter) -> Tensor");
}

TORCH_LIBRARY_IMPL(flagquantum_native, CPU, library) {
  library.impl("fused_rotation_block_forward_", &fused_rotation_block_forward_cpu);
  library.impl("fused_rotation_adjoint_", &fused_rotation_adjoint_cpu);
  library.impl("fused_rzz_segment_adjoint_", &fused_rzz_segment_adjoint_cpu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
