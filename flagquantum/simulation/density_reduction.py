# SPDX-License-Identifier: BSD-3-Clause
# Adapted arithmetic ordering from NumPy 2.5.4 loops_utils.h.src.
# Copyright (c) 2005-2025, NumPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.
#
#     * Neither the name of the NumPy Developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Differentiable complex traces with a fixed four-lane reduction order.

NumPy 2.5.4's complex pairwise reduction uses four complex accumulators, balanced
combination and blocks of at most 64 complex entries. This native Torch owner
retains that arithmetic order without calling a reference runtime.
"""

import torch


def _pairwise_complex_sum(values: torch.Tensor) -> torch.Tensor:
    count = values.shape[-1]
    if count < 4:
        result = -values[..., :0].sum(dim=-1)
        for index in range(count):
            result = result + values[..., index]
        return result
    if count > 64:
        middle = (count // 2) // 4 * 4
        return _pairwise_complex_sum(values[..., :middle]) + _pairwise_complex_sum(
            values[..., middle:]
        )
    end = count - count % 4
    lanes = values[..., :4]
    for start in range(4, end, 4):
        lanes = lanes + values[..., start : start + 4]
    result = (lanes[..., 0] + lanes[..., 1]) + (lanes[..., 2] + lanes[..., 3])
    for index in range(end, count):
        result = result + values[..., index]
    return result


def _pairwise_complex_trace(matrix: torch.Tensor) -> torch.Tensor:
    """Reduce each diagonal with NumPy-compatible complex pairwise ordering."""
    diagonal = matrix.diagonal(dim1=-2, dim2=-1)
    # The outer reduction starts from its additive identity, including empty input.
    return _pairwise_complex_sum(diagonal) + diagonal.new_zeros(diagonal.shape[:-1])
