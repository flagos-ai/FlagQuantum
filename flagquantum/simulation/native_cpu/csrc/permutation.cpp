#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "linear_permutation.h"

namespace {

using flagquantum_native::LinearLookup;
using flagquantum_native::apply_linear_lookup;
using flagquantum_native::build_linear_lookup;
using flagquantum_native::linear_shift_mode;

inline uint64_t insert_zero_bit(uint64_t value, int64_t position) {
  const uint64_t low_mask = (uint64_t{1} << position) - 1;
  return (value & low_mask) | ((value & ~low_mask) << 1);
}

template <typename scalar_t, typename index_t>
void gather_state_out(
    const scalar_t* state,
    const index_t* index,
    scalar_t* output,
    int64_t batch,
    int64_t width) {
  if (batch == 1) {
    at::parallel_for(
        int64_t{0}, width, int64_t{4096}, [&](int64_t begin, int64_t end) {
          for (int64_t destination = begin; destination < end; ++destination) {
            output[destination] = state[static_cast<int64_t>(index[destination])];
          }
        });
    return;
  }
  at::parallel_for(
      int64_t{0}, batch * width, int64_t{4096}, [&](int64_t begin, int64_t end) {
        for (int64_t item = begin; item < end; ++item) {
          const int64_t row = item / width;
          const int64_t destination = item - row * width;
          const int64_t source =
              row * width + static_cast<int64_t>(index[destination]);
          output[item] = state[source];
        }
      });
}

at::Tensor& fused_cx_gather_out_cpu(
    const at::Tensor& state,
    const at::Tensor& index,
    at::Tensor& output) {
  TORCH_CHECK(state.device().is_cpu(), "state must be on CPU");
  TORCH_CHECK(index.device().is_cpu(), "index must be on CPU");
  TORCH_CHECK(output.device().is_cpu(), "output must be on CPU");
  TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
  TORCH_CHECK(index.is_contiguous(), "index must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(state.dim() == 2, "state must have shape [batch, amplitudes]");
  TORCH_CHECK(output.sizes() == state.sizes(), "output shape must match state");
  TORCH_CHECK(output.scalar_type() == state.scalar_type(), "output dtype must match state");
  TORCH_CHECK(
      state.scalar_type() == at::kComplexFloat ||
          state.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(index.dim() == 1, "index must be one-dimensional");
  TORCH_CHECK(index.numel() == state.size(1), "index width must match the state");
  TORCH_CHECK(
      index.scalar_type() == at::kInt || index.scalar_type() == at::kLong,
      "index must be int32 or int64");
  TORCH_CHECK(!state.is_alias_of(output), "state and output must not alias");

  const int64_t width = state.size(1);
  const int64_t batch = state.size(0);
  AT_DISPATCH_COMPLEX_TYPES(state.scalar_type(), "fused_cx_gather_out_cpu", [&] {
    const scalar_t* state_data = state.const_data_ptr<scalar_t>();
    scalar_t* output_data = output.data_ptr<scalar_t>();
    if (index.scalar_type() == at::kInt) {
      gather_state_out(
          state_data, index.const_data_ptr<int32_t>(), output_data, batch, width);
    } else {
      gather_state_out(
          state_data, index.const_data_ptr<int64_t>(), output_data, batch, width);
    }
  });
  return output;
}

at::Tensor& fused_compact_cx_gather_out_cpu(
    const at::Tensor& state,
    const at::Tensor& images,
    at::Tensor& output) {
  TORCH_CHECK(state.device().is_cpu(), "state must be on CPU");
  TORCH_CHECK(images.device().is_cpu(), "CX images must be on CPU");
  TORCH_CHECK(output.device().is_cpu(), "output must be on CPU");
  TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
  TORCH_CHECK(images.is_contiguous(), "CX images must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(state.dim() == 2, "state must have shape [batch, amplitudes]");
  TORCH_CHECK(output.sizes() == state.sizes(), "output shape must match state");
  TORCH_CHECK(output.scalar_type() == state.scalar_type(), "output dtype must match state");
  TORCH_CHECK(
      state.scalar_type() == at::kComplexFloat ||
          state.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(images.dim() == 1, "CX images must be one-dimensional");
  TORCH_CHECK(images.scalar_type() == at::kLong, "CX images must be int64");
  TORCH_CHECK(!state.is_alias_of(output), "state and output must not alias");
  const int64_t n_wires = images.numel();
  TORCH_CHECK(n_wires > 1 && n_wires < 63, "CX image count must be in [2, 62]");
  const int64_t width = state.size(1);
  TORCH_CHECK(width == (int64_t{1} << n_wires),
              "CX image count must match state width");
  const int64_t batch = state.size(0);
  const int64_t chunk_count = (n_wires + 7) / 8;
  const LinearLookup lookup = build_linear_lookup(images);
  const int64_t shift_mode = linear_shift_mode(images);
  const uint64_t state_mask = (uint64_t{1} << n_wires) - 1;

  AT_DISPATCH_COMPLEX_TYPES(
      state.scalar_type(), "fused_compact_cx_gather_out_cpu", [&] {
        const scalar_t* state_data = state.const_data_ptr<scalar_t>();
        scalar_t* output_data = output.data_ptr<scalar_t>();
        at::parallel_for(
            int64_t{0}, batch * width, int64_t{4096},
            [&](int64_t begin, int64_t end) {
              for (int64_t item = begin; item < end; ++item) {
                const int64_t row = item / width;
                const int64_t destination = item - row * width;
                const int64_t source = row * width + apply_linear_lookup(
                    static_cast<uint64_t>(destination),
                    lookup,
                    chunk_count,
                    shift_mode,
                    state_mask);
                output_data[item] = state_data[source];
              }
            });
      });
  return output;
}

at::Tensor& fused_compact_cx_rzz_swap_out_cpu(
    const at::Tensor& state,
    const at::Tensor& images,
    at::Tensor& output,
    int64_t first_wire,
    int64_t second_wire,
    double angle) {
  TORCH_CHECK(state.device().is_cpu() && images.device().is_cpu() &&
                  output.device().is_cpu(), "CX/RZZ/SWAP inputs must be on CPU");
  TORCH_CHECK(state.is_contiguous() && images.is_contiguous() &&
                  output.is_contiguous(), "CX/RZZ/SWAP inputs must be contiguous");
  TORCH_CHECK(state.dim() == 2, "state must have shape [batch, amplitudes]");
  TORCH_CHECK(output.sizes() == state.sizes(), "output shape must match state");
  TORCH_CHECK(output.scalar_type() == state.scalar_type(),
              "output dtype must match state");
  TORCH_CHECK(
      state.scalar_type() == at::kComplexFloat ||
          state.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(images.dim() == 1 && images.scalar_type() == at::kLong,
              "images must be a one-dimensional int64 tensor");
  TORCH_CHECK(!state.is_alias_of(output), "state and output must not alias");
  const int64_t n_wires = images.numel();
  TORCH_CHECK(n_wires > 1 && n_wires < 63, "image count must be in [2, 62]");
  TORCH_CHECK(first_wire >= 0 && first_wire < n_wires && second_wire >= 0 &&
                  second_wire < n_wires && first_wire != second_wire,
              "RZZ wires must be distinct and in range");
  const int64_t width = state.size(1);
  TORCH_CHECK(width == (int64_t{1} << n_wires),
              "image count must match state width");
  const int64_t batch = state.size(0);
  const int64_t chunk_count = (n_wires + 7) / 8;
  const LinearLookup lookup = build_linear_lookup(images);
  const int64_t shift_mode = linear_shift_mode(images);
  const uint64_t state_mask = (uint64_t{1} << n_wires) - 1;
  const uint64_t first_mask = uint64_t{1} << (n_wires - first_wire - 1);
  const uint64_t second_mask = uint64_t{1} << (n_wires - second_wire - 1);

  AT_DISPATCH_COMPLEX_TYPES(
      state.scalar_type(), "fused_compact_cx_rzz_swap_out_cpu", [&] {
        using real_t = typename scalar_t::value_type;
        const scalar_t* state_data = state.const_data_ptr<scalar_t>();
        scalar_t* output_data = output.data_ptr<scalar_t>();
        const real_t half_angle = static_cast<real_t>(angle * 0.5);
        const scalar_t even_phase(std::cos(half_angle), -std::sin(half_angle));
        const scalar_t odd_phase(std::cos(half_angle), std::sin(half_angle));
        at::parallel_for(
            int64_t{0}, batch * width, int64_t{4096},
            [&](int64_t begin, int64_t end) {
              for (int64_t item = begin; item < end; ++item) {
                const int64_t row = item / width;
                const uint64_t destination = static_cast<uint64_t>(item - row * width);
                const int64_t source = apply_linear_lookup(
                    destination, lookup, chunk_count, shift_mode, state_mask);
                const bool odd = ((destination & first_mask) != 0) !=
                    ((destination & second_mask) != 0);
                output_data[item] = state_data[row * width + source] *
                    (odd ? odd_phase : even_phase);
              }
            });
      });
  return output;
}

void fused_cx_rzz_swap_sequence_inplace_cpu(
    at::Tensor& state,
    const at::Tensor& controls,
    const at::Tensor& targets,
    int64_t n_wires,
    int64_t first_wire,
    int64_t second_wire,
    int64_t swap_left,
    int64_t swap_right,
    double angle) {
  TORCH_CHECK(state.device().is_cpu() && controls.device().is_cpu() &&
                  targets.device().is_cpu(),
              "CX/RZZ/SWAP inputs must be on CPU");
  TORCH_CHECK(state.is_contiguous() && controls.is_contiguous() &&
                  targets.is_contiguous(),
              "CX/RZZ/SWAP inputs must be contiguous");
  TORCH_CHECK(state.dim() == 2, "state must have shape [batch, amplitudes]");
  TORCH_CHECK(
      state.scalar_type() == at::kComplexFloat ||
          state.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(controls.dim() == 1 && controls.scalar_type() == at::kLong,
              "controls must be a one-dimensional int64 tensor");
  TORCH_CHECK(targets.sizes() == controls.sizes() &&
                  targets.scalar_type() == at::kLong,
              "targets must match controls and use int64");
  TORCH_CHECK(n_wires > 1 && n_wires < 63, "wire count must be in [2, 62]");
  TORCH_CHECK(first_wire >= 0 && first_wire < n_wires && second_wire >= 0 &&
                  second_wire < n_wires && first_wire != second_wire,
              "RZZ wires must be distinct and in range");
  TORCH_CHECK(swap_left >= 0 && swap_left < n_wires && swap_right >= 0 &&
                  swap_right < n_wires && swap_left != swap_right,
              "SWAP wires must be distinct and in range");
  const int64_t width = state.size(1);
  TORCH_CHECK(width == (int64_t{1} << n_wires),
              "wire count must match state width");
  const int64_t batch = state.size(0);
  const uint64_t first_mask = uint64_t{1} << (n_wires - first_wire - 1);
  const uint64_t second_mask = uint64_t{1} << (n_wires - second_wire - 1);
  const uint64_t swap_left_mask = uint64_t{1} << (n_wires - swap_left - 1);
  const uint64_t swap_right_mask = uint64_t{1} << (n_wires - swap_right - 1);
  const int64_t* control_data = controls.const_data_ptr<int64_t>();
  const int64_t* target_data = targets.const_data_ptr<int64_t>();

  AT_DISPATCH_COMPLEX_TYPES(
      state.scalar_type(), "fused_cx_rzz_swap_sequence_inplace_cpu", [&] {
        using real_t = typename scalar_t::value_type;
        scalar_t* state_data = state.data_ptr<scalar_t>();
        bool adjacent_chain = controls.numel() > 0;
        for (int64_t gate = 0; gate < controls.numel(); ++gate) {
          const int64_t control = control_data[gate];
          const int64_t target = target_data[gate];
          TORCH_CHECK(control >= 0 && control < n_wires,
                      "control wire is out of range");
          TORCH_CHECK(target >= 0 && target < n_wires,
                      "target wire is out of range");
          TORCH_CHECK(control != target, "control and target wires must differ");
          adjacent_chain = adjacent_chain && target == control + 1 &&
              (gate == 0 || control == control_data[gate - 1] + 1);
        }
        if (adjacent_chain) {
          constexpr int64_t adjacent_group_size = 4;
          constexpr size_t adjacent_block_size =
              size_t{1} << (adjacent_group_size + 1);
          for (int64_t first_gate = 0; first_gate < controls.numel();
               first_gate += adjacent_group_size) {
            const int64_t gate_count = std::min(
                adjacent_group_size, controls.numel() - first_gate);
            const int64_t affected_bits = gate_count + 1;
            const int64_t local_width = int64_t{1} << affected_bits;
            const int64_t low_position =
                n_wires - target_data[first_gate + gate_count - 1] - 1;
            const uint64_t low_mask = (uint64_t{1} << low_position) - 1;
            const int64_t block_count = width / local_width;
            std::array<int64_t, adjacent_block_size> destinations{};
            for (int64_t source = 0; source < local_width; ++source) {
              int64_t destination = source;
              for (int64_t gate = 0; gate < gate_count; ++gate) {
                const int64_t control_bit = gate_count - gate;
                if ((destination & (int64_t{1} << control_bit)) != 0) {
                  destination ^= int64_t{1} << (control_bit - 1);
                }
              }
              destinations[static_cast<size_t>(source)] = destination;
            }
            at::parallel_for(
                int64_t{0}, batch * block_count, int64_t{64},
                [&](int64_t begin, int64_t end) {
                  std::array<scalar_t, adjacent_block_size> block;
                  for (int64_t item = begin; item < end; ++item) {
                    const int64_t row = item / block_count;
                    const uint64_t compact =
                        static_cast<uint64_t>(item - row * block_count);
                    const uint64_t base = (compact & low_mask) |
                        ((compact & ~low_mask) << affected_bits);
                    for (int64_t source = 0; source < local_width; ++source) {
                      const int64_t destination =
                          destinations[static_cast<size_t>(source)];
                      const int64_t source_index = row * width +
                          static_cast<int64_t>(base) +
                          (source << low_position);
                      block[static_cast<size_t>(destination)] =
                          state_data[source_index];
                    }
                    for (int64_t destination = 0;
                         destination < local_width; ++destination) {
                      const int64_t destination_index = row * width +
                          static_cast<int64_t>(base) +
                          (destination << low_position);
                      state_data[destination_index] =
                          block[static_cast<size_t>(destination)];
                    }
                  }
                });
          }
        } else {
          for (int64_t gate = 0; gate < controls.numel(); ++gate) {
            const int64_t control = control_data[gate];
            const int64_t target = target_data[gate];
            const int64_t control_position = n_wires - control - 1;
            const int64_t target_position = n_wires - target - 1;
            const int64_t low_position =
                std::min(control_position, target_position);
            const int64_t high_position =
                std::max(control_position, target_position);
            const uint64_t control_mask = uint64_t{1} << control_position;
            const uint64_t target_mask = uint64_t{1} << target_position;
            const int64_t pair_count = width / 4;
            at::parallel_for(
                int64_t{0}, batch * pair_count, int64_t{4096},
                [&](int64_t begin, int64_t end) {
                  for (int64_t item = begin; item < end; ++item) {
                    const int64_t row = item / pair_count;
                    uint64_t basis = insert_zero_bit(
                        static_cast<uint64_t>(item - row * pair_count),
                        low_position);
                    basis = insert_zero_bit(basis, high_position) | control_mask;
                    const int64_t index =
                        row * width + static_cast<int64_t>(basis);
                    const int64_t peer =
                        index ^ static_cast<int64_t>(target_mask);
                    std::swap(state_data[index], state_data[peer]);
                  }
                });
          }
        }

        const real_t half_angle = static_cast<real_t>(angle * 0.5);
        const scalar_t even_phase(std::cos(half_angle), -std::sin(half_angle));
        const scalar_t odd_phase(std::cos(half_angle), std::sin(half_angle));
        at::parallel_for(
            int64_t{0}, batch * width, int64_t{4096},
            [&](int64_t begin, int64_t end) {
              for (int64_t item = begin; item < end; ++item) {
                const int64_t row = item / width;
                const uint64_t basis = static_cast<uint64_t>(item - row * width);
                const bool left_set = (basis & swap_left_mask) != 0;
                const bool right_set = (basis & swap_right_mask) != 0;
                if (left_set && !right_set) {
                  const int64_t peer = item ^ static_cast<int64_t>(
                      swap_left_mask | swap_right_mask);
                  const uint64_t peer_basis = basis ^
                      (swap_left_mask | swap_right_mask);
                  const bool basis_odd = ((basis & first_mask) != 0) !=
                      ((basis & second_mask) != 0);
                  const bool peer_odd = ((peer_basis & first_mask) != 0) !=
                      ((peer_basis & second_mask) != 0);
                  const scalar_t value = state_data[item];
                  const scalar_t peer_value = state_data[peer];
                  state_data[item] = peer_value *
                      (peer_odd ? odd_phase : even_phase);
                  state_data[peer] = value *
                      (basis_odd ? odd_phase : even_phase);
                } else if (left_set == right_set) {
                  const bool odd = ((basis & first_mask) != 0) !=
                      ((basis & second_mask) != 0);
                  state_data[item] *= odd ? odd_phase : even_phase;
                }
              }
            });
      });
}

at::Tensor& fused_clifford_matching_out_cpu(
    const at::Tensor& state,
    const at::Tensor& cx_mapping,
    const at::Tensor& cz_edges,
    at::Tensor& output,
    int64_t n_wires,
    bool phase_encoded) {
  TORCH_CHECK(state.device().is_cpu(), "state must be on CPU");
  TORCH_CHECK(cx_mapping.device().is_cpu(), "CX mapping must be on CPU");
  TORCH_CHECK(cz_edges.device().is_cpu(), "CZ edges must be on CPU");
  TORCH_CHECK(output.device().is_cpu(), "output must be on CPU");
  TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
  TORCH_CHECK(cx_mapping.is_contiguous(), "CX mapping must be contiguous");
  TORCH_CHECK(cz_edges.is_contiguous(), "CZ edges must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(state.dim() == 2, "state must have shape [batch, amplitudes]");
  TORCH_CHECK(output.sizes() == state.sizes(), "output shape must match state");
  TORCH_CHECK(output.scalar_type() == state.scalar_type(), "output dtype must match state");
  TORCH_CHECK(
      state.scalar_type() == at::kComplexFloat ||
          state.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(cx_mapping.dim() == 1, "CX mapping must be one-dimensional");
  TORCH_CHECK(
      cx_mapping.scalar_type() == at::kInt ||
          cx_mapping.scalar_type() == at::kLong,
      "CX mapping must be int32 or int64");
  TORCH_CHECK(
      cz_edges.dim() == 2 && cz_edges.size(1) == 2,
      "CZ edges must have shape [edges, 2]");
  TORCH_CHECK(cz_edges.scalar_type() == at::kLong, "CZ edges must be int64");
  TORCH_CHECK(!state.is_alias_of(output), "state and output must not alias");

  TORCH_CHECK(n_wires > 1 && n_wires < 63, "wire count must be in [2, 62]");
  const int64_t width = state.size(1);
  TORCH_CHECK(width == (int64_t{1} << n_wires), "wire count must match state width");
  const bool compact_mapping = cx_mapping.numel() == n_wires;
  TORCH_CHECK(
      compact_mapping || cx_mapping.numel() == width,
      "CX mapping must contain one image per wire or one index per amplitude");
  TORCH_CHECK(
      !compact_mapping || cx_mapping.scalar_type() == at::kLong,
      "compact CX images must be int64");
  TORCH_CHECK(
      !phase_encoded || (!compact_mapping && cz_edges.size(0) == 0),
      "phase-encoded mappings must be full and have no CZ edges");
  const int64_t batch = state.size(0);
  const int64_t chunk_count = (n_wires + 7) / 8;
  const LinearLookup lookup =
      compact_mapping ? build_linear_lookup(cx_mapping) : LinearLookup{};
  const int64_t shift_mode =
      compact_mapping ? linear_shift_mode(cx_mapping) : int64_t{0};
  const uint64_t state_mask = (uint64_t{1} << n_wires) - 1;
  const int64_t* edge_data = cz_edges.const_data_ptr<int64_t>();
  std::vector<std::pair<uint64_t, uint64_t>> phase_masks;
  phase_masks.reserve(static_cast<size_t>(cz_edges.size(0)));
  for (int64_t edge = 0; edge < cz_edges.size(0); ++edge) {
    const int64_t left = edge_data[edge * 2];
    const int64_t right = edge_data[edge * 2 + 1];
    TORCH_CHECK(left >= 0 && left < n_wires, "CZ wire is out of range");
    TORCH_CHECK(right >= 0 && right < n_wires, "CZ wire is out of range");
    TORCH_CHECK(left != right, "CZ wires must differ");
    phase_masks.emplace_back(
        uint64_t{1} << (n_wires - left - 1),
        uint64_t{1} << (n_wires - right - 1));
  }

  AT_DISPATCH_COMPLEX_TYPES(
      state.scalar_type(), "fused_clifford_matching_out_cpu", [&] {
        const scalar_t* state_data = state.const_data_ptr<scalar_t>();
        scalar_t* output_data = output.data_ptr<scalar_t>();
        const auto apply_signed = [&](const auto* index) {
          at::parallel_for(
              int64_t{0}, batch * width, int64_t{4096},
              [&](int64_t begin, int64_t end) {
                for (int64_t item = begin; item < end; ++item) {
                  const int64_t destination = item & (width - 1);
                  const int64_t encoded = static_cast<int64_t>(index[destination]);
                  const bool negative = encoded < 0;
                  const int64_t source = negative ? -encoded - 1 : encoded;
                  const scalar_t value = state_data[item - destination + source];
                  output_data[item] = negative ? -value : value;
                }
              });
        };
        const auto apply = [&](const auto& source_at) {
          at::parallel_for(
              int64_t{0}, batch * width, int64_t{4096},
              [&](int64_t begin, int64_t end) {
              for (int64_t item = begin; item < end; ++item) {
                const int64_t row = item / width;
                const uint64_t destination =
                    static_cast<uint64_t>(item - row * width);
                const int64_t source = source_at(destination);
                bool negative = false;
                for (const auto& masks : phase_masks) {
                  negative ^= (destination & masks.first) != 0 &&
                              (destination & masks.second) != 0;
                }
                const scalar_t value =
                    state_data[row * width + source];
                output_data[item] = negative ? -value : value;
              }
            });
        };
        if (phase_encoded && cx_mapping.scalar_type() == at::kInt) {
          apply_signed(cx_mapping.const_data_ptr<int32_t>());
        } else if (phase_encoded) {
          apply_signed(cx_mapping.const_data_ptr<int64_t>());
        } else if (compact_mapping) {
          apply([&](uint64_t destination) {
            return apply_linear_lookup(
                destination,
                lookup,
                chunk_count,
                shift_mode,
                state_mask);
          });
        } else if (cx_mapping.scalar_type() == at::kInt) {
          const int32_t* index = cx_mapping.const_data_ptr<int32_t>();
          apply([&](uint64_t destination) {
            return static_cast<int64_t>(index[destination]);
          });
        } else {
          const int64_t* index = cx_mapping.const_data_ptr<int64_t>();
          apply([&](uint64_t destination) { return index[destination]; });
        }
      });
  return output;
}

std::tuple<at::Tensor, at::Tensor> fused_cx_adjoint_gather_cpu(
    const at::Tensor& ket,
    const at::Tensor& adjoint,
    const at::Tensor& index) {
  TORCH_CHECK(ket.device().is_cpu(), "ket must be on CPU");
  TORCH_CHECK(adjoint.device().is_cpu(), "adjoint must be on CPU");
  TORCH_CHECK(index.device().is_cpu(), "index must be on CPU");
  TORCH_CHECK(ket.is_contiguous(), "ket must be contiguous");
  TORCH_CHECK(adjoint.is_contiguous(), "adjoint must be contiguous");
  TORCH_CHECK(index.is_contiguous(), "index must be contiguous");
  TORCH_CHECK(ket.dim() == 2, "ket must have shape [batch, amplitudes]");
  TORCH_CHECK(adjoint.sizes() == ket.sizes(), "adjoint shape must match ket");
  TORCH_CHECK(adjoint.scalar_type() == ket.scalar_type(), "state dtypes must match");
  TORCH_CHECK(
      ket.scalar_type() == at::kComplexFloat || ket.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(index.dim() == 1, "index must be one-dimensional");
  TORCH_CHECK(index.numel() == ket.size(1), "index width must match the state");
  TORCH_CHECK(
      index.scalar_type() == at::kInt || index.scalar_type() == at::kLong,
      "index must be int32 or int64");

  at::Tensor gathered_ket = at::empty_like(ket);
  at::Tensor gathered_adjoint = at::empty_like(adjoint);
  const int64_t width = ket.size(1);
  const int64_t batch = ket.size(0);

  AT_DISPATCH_COMPLEX_TYPES(ket.scalar_type(), "fused_cx_adjoint_gather_cpu", [&] {
    const scalar_t* ket_data = ket.const_data_ptr<scalar_t>();
    const scalar_t* adjoint_data = adjoint.const_data_ptr<scalar_t>();
    scalar_t* gathered_ket_data = gathered_ket.data_ptr<scalar_t>();
    scalar_t* gathered_adjoint_data = gathered_adjoint.data_ptr<scalar_t>();

    const auto gather = [&](const auto* index_data) {
      if (batch == 1) {
        at::parallel_for(int64_t{0}, width, int64_t{4096}, [&](int64_t begin, int64_t end) {
          for (int64_t destination = begin; destination < end; ++destination) {
            const int64_t source = static_cast<int64_t>(index_data[destination]);
            gathered_ket_data[destination] = ket_data[source];
            gathered_adjoint_data[destination] = adjoint_data[source];
          }
        });
        return;
      }
      at::parallel_for(int64_t{0}, batch * width, int64_t{4096}, [&](int64_t begin, int64_t end) {
        for (int64_t item = begin; item < end; ++item) {
          const int64_t row = item / width;
          const int64_t destination = item - row * width;
          const int64_t source = row * width +
              static_cast<int64_t>(index_data[destination]);
          gathered_ket_data[item] = ket_data[source];
          gathered_adjoint_data[item] = adjoint_data[source];
        }
      });
    };
    if (index.scalar_type() == at::kInt) {
      gather(index.const_data_ptr<int32_t>());
    } else {
      gather(index.const_data_ptr<int64_t>());
    }
  });
  return std::make_tuple(gathered_ket, gathered_adjoint);
}

std::tuple<at::Tensor, at::Tensor> fused_compact_cx_adjoint_gather_cpu(
    const at::Tensor& ket,
    const at::Tensor& adjoint,
    const at::Tensor& images) {
  TORCH_CHECK(ket.device().is_cpu(), "ket must be on CPU");
  TORCH_CHECK(adjoint.device().is_cpu(), "adjoint must be on CPU");
  TORCH_CHECK(images.device().is_cpu(), "CX images must be on CPU");
  TORCH_CHECK(ket.is_contiguous(), "ket must be contiguous");
  TORCH_CHECK(adjoint.is_contiguous(), "adjoint must be contiguous");
  TORCH_CHECK(images.is_contiguous(), "CX images must be contiguous");
  TORCH_CHECK(ket.dim() == 2, "ket must have shape [batch, amplitudes]");
  TORCH_CHECK(adjoint.sizes() == ket.sizes(), "adjoint shape must match ket");
  TORCH_CHECK(adjoint.scalar_type() == ket.scalar_type(), "state dtypes must match");
  TORCH_CHECK(
      ket.scalar_type() == at::kComplexFloat || ket.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(images.dim() == 1, "CX images must be one-dimensional");
  TORCH_CHECK(images.scalar_type() == at::kLong, "CX images must be int64");
  const int64_t n_wires = images.numel();
  TORCH_CHECK(n_wires > 1 && n_wires < 63, "CX image count must be in [2, 62]");
  TORCH_CHECK(
      ket.size(1) == (int64_t{1} << n_wires),
      "CX image count must match state width");

  at::Tensor gathered_ket = at::empty_like(ket);
  at::Tensor gathered_adjoint = at::empty_like(adjoint);
  const int64_t width = ket.size(1);
  const int64_t batch = ket.size(0);
  const int64_t chunk_count = (n_wires + 7) / 8;
  const LinearLookup lookup = build_linear_lookup(images);
  const int64_t shift_mode = linear_shift_mode(images);
  const uint64_t state_mask = (uint64_t{1} << n_wires) - 1;

  AT_DISPATCH_COMPLEX_TYPES(
      ket.scalar_type(), "fused_compact_cx_adjoint_gather_cpu", [&] {
        const scalar_t* ket_data = ket.const_data_ptr<scalar_t>();
        const scalar_t* adjoint_data = adjoint.const_data_ptr<scalar_t>();
        scalar_t* gathered_ket_data = gathered_ket.data_ptr<scalar_t>();
        scalar_t* gathered_adjoint_data = gathered_adjoint.data_ptr<scalar_t>();
        at::parallel_for(
            int64_t{0}, batch * width, int64_t{4096},
            [&](int64_t begin, int64_t end) {
              for (int64_t item = begin; item < end; ++item) {
                const int64_t row = item / width;
                const int64_t destination = item - row * width;
                const int64_t source = row * width + apply_linear_lookup(
                    static_cast<uint64_t>(destination),
                    lookup,
                    chunk_count,
                    shift_mode,
                    state_mask);
                gathered_ket_data[item] = ket_data[source];
                gathered_adjoint_data[item] = adjoint_data[source];
              }
            });
      });
  return std::make_tuple(gathered_ket, gathered_adjoint);
}

void fused_cx_adjoint_inplace_cpu(
    at::Tensor& ket,
    at::Tensor& adjoint,
    const at::Tensor& controls,
    const at::Tensor& targets,
    int64_t n_wires) {
  TORCH_CHECK(ket.device().is_cpu(), "ket must be on CPU");
  TORCH_CHECK(adjoint.device().is_cpu(), "adjoint must be on CPU");
  TORCH_CHECK(controls.device().is_cpu(), "controls must be on CPU");
  TORCH_CHECK(targets.device().is_cpu(), "targets must be on CPU");
  TORCH_CHECK(ket.is_contiguous(), "ket must be contiguous");
  TORCH_CHECK(adjoint.is_contiguous(), "adjoint must be contiguous");
  TORCH_CHECK(controls.is_contiguous(), "controls must be contiguous");
  TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
  TORCH_CHECK(ket.dim() == 2, "ket must have shape [batch, amplitudes]");
  TORCH_CHECK(adjoint.sizes() == ket.sizes(), "adjoint shape must match ket");
  TORCH_CHECK(adjoint.scalar_type() == ket.scalar_type(), "state dtypes must match");
  TORCH_CHECK(
      ket.scalar_type() == at::kComplexFloat || ket.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(controls.dim() == 1, "controls must be one-dimensional");
  TORCH_CHECK(targets.sizes() == controls.sizes(), "targets shape must match controls");
  TORCH_CHECK(controls.scalar_type() == at::kLong, "controls must be int64");
  TORCH_CHECK(targets.scalar_type() == at::kLong, "targets must be int64");
  TORCH_CHECK(n_wires > 1 && n_wires < 63, "wire count must be in [2, 62]");
  const int64_t width = ket.size(1);
  TORCH_CHECK(width == (int64_t{1} << n_wires), "wire count must match state width");
  const int64_t batch = ket.size(0);
  const int64_t* control_data = controls.const_data_ptr<int64_t>();
  const int64_t* target_data = targets.const_data_ptr<int64_t>();

  AT_DISPATCH_COMPLEX_TYPES(ket.scalar_type(), "fused_cx_adjoint_inplace_cpu", [&] {
    scalar_t* ket_data = ket.data_ptr<scalar_t>();
    scalar_t* adjoint_data = adjoint.data_ptr<scalar_t>();
    for (int64_t gate = 0; gate < controls.numel(); ++gate) {
      const int64_t control = control_data[gate];
      const int64_t target = target_data[gate];
      TORCH_CHECK(control >= 0 && control < n_wires, "control wire is out of range");
      TORCH_CHECK(target >= 0 && target < n_wires, "target wire is out of range");
      TORCH_CHECK(control != target, "control and target wires must differ");
      const int64_t control_position = n_wires - control - 1;
      const int64_t target_position = n_wires - target - 1;
      const int64_t low_position = std::min(control_position, target_position);
      const int64_t high_position = std::max(control_position, target_position);
      const uint64_t control_mask = uint64_t{1} << control_position;
      const uint64_t target_mask = uint64_t{1} << target_position;
      const int64_t pair_count = width / 4;
      const auto swap_pair = [&](int64_t row, uint64_t ordinal) {
        uint64_t basis = insert_zero_bit(ordinal, low_position);
        basis = insert_zero_bit(basis, high_position) | control_mask;
        const int64_t item = row * width + static_cast<int64_t>(basis);
        const int64_t peer = item ^ static_cast<int64_t>(target_mask);
        std::swap(ket_data[item], ket_data[peer]);
        std::swap(adjoint_data[item], adjoint_data[peer]);
      };
      if (batch == 1) {
        at::parallel_for(
            int64_t{0}, pair_count, int64_t{4096}, [&](int64_t begin, int64_t end) {
              for (int64_t pair = begin; pair < end; ++pair) {
                swap_pair(0, static_cast<uint64_t>(pair));
              }
            });
      } else {
        at::parallel_for(
            int64_t{0}, batch * pair_count, int64_t{4096},
            [&](int64_t begin, int64_t end) {
              for (int64_t item = begin; item < end; ++item) {
                const int64_t row = item / pair_count;
                swap_pair(row, static_cast<uint64_t>(item - row * pair_count));
              }
            });
      }
    }
  });
}

void fused_compact_cx_adjoint_inplace_cpu(
    at::Tensor& ket,
    at::Tensor& adjoint,
    const at::Tensor& images) {
  TORCH_CHECK(ket.device().is_cpu(), "ket must be on CPU");
  TORCH_CHECK(adjoint.device().is_cpu(), "adjoint must be on CPU");
  TORCH_CHECK(images.device().is_cpu(), "CX images must be on CPU");
  TORCH_CHECK(ket.is_contiguous(), "ket must be contiguous");
  TORCH_CHECK(adjoint.is_contiguous(), "adjoint must be contiguous");
  TORCH_CHECK(images.is_contiguous(), "CX images must be contiguous");
  TORCH_CHECK(ket.dim() == 2, "ket must have shape [batch, amplitudes]");
  TORCH_CHECK(adjoint.sizes() == ket.sizes(), "adjoint shape must match ket");
  TORCH_CHECK(adjoint.scalar_type() == ket.scalar_type(), "state dtypes must match");
  TORCH_CHECK(
      ket.scalar_type() == at::kComplexFloat || ket.scalar_type() == at::kComplexDouble,
      "only complex64 and complex128 are supported");
  TORCH_CHECK(images.dim() == 1, "CX images must be one-dimensional");
  TORCH_CHECK(images.scalar_type() == at::kLong, "CX images must be int64");
  const int64_t n_wires = images.numel();
  TORCH_CHECK(n_wires > 1 && n_wires < 31, "CX image count must be in [2, 30]");
  const int64_t width = ket.size(1);
  TORCH_CHECK(width == (int64_t{1} << n_wires),
              "CX image count must match state width");

  const int64_t batch = ket.size(0);
  const int64_t chunk_count = (n_wires + 7) / 8;
  const LinearLookup lookup = build_linear_lookup(images);
  const int64_t shift_mode = linear_shift_mode(images);
  const uint64_t state_mask = (uint64_t{1} << n_wires) - 1;
  std::vector<int32_t> permutation(static_cast<size_t>(width));
  at::parallel_for(
      int64_t{0}, width, int64_t{4096}, [&](int64_t begin, int64_t end) {
        for (int64_t destination = begin; destination < end; ++destination) {
          permutation[static_cast<size_t>(destination)] = static_cast<int32_t>(
              apply_linear_lookup(
                  static_cast<uint64_t>(destination),
                  lookup,
                  chunk_count,
                  shift_mode,
                  state_mask));
        }
      });

  std::vector<uint8_t> visited(static_cast<size_t>(width), uint8_t{0});
  std::vector<int32_t> leaders;
  leaders.reserve(static_cast<size_t>(width / 16));
  for (int64_t start = 0; start < width; ++start) {
    if (visited[static_cast<size_t>(start)] != 0) {
      continue;
    }
    leaders.push_back(static_cast<int32_t>(start));
    int32_t current = static_cast<int32_t>(start);
    do {
      TORCH_CHECK(
          visited[static_cast<size_t>(current)] == 0,
          "CX images must define a bijection");
      visited[static_cast<size_t>(current)] = uint8_t{1};
      current = permutation[static_cast<size_t>(current)];
    } while (current != start);
  }

  AT_DISPATCH_COMPLEX_TYPES(
      ket.scalar_type(), "fused_compact_cx_adjoint_inplace_cpu", [&] {
        scalar_t* ket_data = ket.data_ptr<scalar_t>();
        scalar_t* adjoint_data = adjoint.data_ptr<scalar_t>();
        const int64_t cycle_count = static_cast<int64_t>(leaders.size());
        at::parallel_for(
            int64_t{0}, batch * cycle_count, int64_t{64},
            [&](int64_t begin, int64_t end) {
              for (int64_t item = begin; item < end; ++item) {
                const int64_t row = item / cycle_count;
                const int32_t leader = leaders[static_cast<size_t>(
                    item - row * cycle_count)];
                int32_t current = leader;
                const scalar_t first_ket = ket_data[row * width + leader];
                const scalar_t first_adjoint =
                    adjoint_data[row * width + leader];
                int32_t next = permutation[static_cast<size_t>(current)];
                while (next != leader) {
                  ket_data[row * width + current] =
                      ket_data[row * width + next];
                  adjoint_data[row * width + current] =
                      adjoint_data[row * width + next];
                  current = next;
                  next = permutation[static_cast<size_t>(current)];
                }
                ket_data[row * width + current] = first_ket;
                adjoint_data[row * width + current] = first_adjoint;
              }
            });
      });
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(flagquantum_native, library) {
  library.def(
      "fused_cx_gather_out(Tensor state, Tensor index, Tensor(a!) output) "
      "-> Tensor(a!)");
  library.def(
      "fused_compact_cx_gather_out(Tensor state, Tensor images, "
      "Tensor(a!) output) -> Tensor(a!)");
  library.def(
      "fused_compact_cx_rzz_swap_out(Tensor state, Tensor images, "
      "Tensor(a!) output, int first_wire, int second_wire, float angle) "
      "-> Tensor(a!)");
  library.def(
      "fused_cx_rzz_swap_sequence_inplace_(Tensor(a!) state, Tensor controls, "
      "Tensor targets, int n_wires, int first_wire, int second_wire, "
      "int swap_left, int swap_right, float angle) -> ()");
  library.def(
      "fused_clifford_matching_out(Tensor state, Tensor cx_mapping, "
      "Tensor cz_edges, Tensor(a!) output, int n_wires, bool phase_encoded) "
      "-> Tensor(a!)");
  library.def(
      "fused_cx_adjoint_gather(Tensor ket, Tensor adjoint, Tensor index) "
      "-> (Tensor, Tensor)");
  library.def(
      "fused_compact_cx_adjoint_gather(Tensor ket, Tensor adjoint, "
      "Tensor images) -> (Tensor, Tensor)");
  library.def(
      "fused_cx_adjoint_inplace_(Tensor(a!) ket, Tensor(b!) adjoint, "
      "Tensor controls, Tensor targets, int n_wires) -> ()");
  library.def(
      "fused_compact_cx_adjoint_inplace_(Tensor(a!) ket, Tensor(b!) adjoint, "
      "Tensor images) -> ()");
}

TORCH_LIBRARY_IMPL(flagquantum_native, CPU, library) {
  library.impl("fused_cx_gather_out", fused_cx_gather_out_cpu);
  library.impl("fused_compact_cx_gather_out", fused_compact_cx_gather_out_cpu);
  library.impl(
      "fused_compact_cx_rzz_swap_out", fused_compact_cx_rzz_swap_out_cpu);
  library.impl(
      "fused_cx_rzz_swap_sequence_inplace_",
      fused_cx_rzz_swap_sequence_inplace_cpu);
  library.impl("fused_clifford_matching_out", fused_clifford_matching_out_cpu);
  library.impl("fused_cx_adjoint_gather", fused_cx_adjoint_gather_cpu);
  library.impl(
      "fused_compact_cx_adjoint_gather",
      fused_compact_cx_adjoint_gather_cpu);
  library.impl("fused_cx_adjoint_inplace_", fused_cx_adjoint_inplace_cpu);
  library.impl(
      "fused_compact_cx_adjoint_inplace_",
      fused_compact_cx_adjoint_inplace_cpu);
}
