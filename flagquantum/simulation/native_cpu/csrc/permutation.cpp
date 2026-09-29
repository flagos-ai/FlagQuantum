#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <cstdint>
#include <tuple>

namespace {

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

}  // namespace

TORCH_LIBRARY_FRAGMENT(flagquantum_native, library) {
  library.def(
      "fused_cx_gather_out(Tensor state, Tensor index, Tensor(a!) output) "
      "-> Tensor(a!)");
  library.def(
      "fused_cx_adjoint_gather(Tensor ket, Tensor adjoint, Tensor index) "
      "-> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(flagquantum_native, CPU, library) {
  library.impl("fused_cx_gather_out", fused_cx_gather_out_cpu);
  library.impl("fused_cx_adjoint_gather", fused_cx_adjoint_gather_cpu);
}
