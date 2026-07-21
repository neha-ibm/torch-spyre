# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import unittest
import torch
import torch.nn.functional as F

from utils_inductor import ParameterizedTestMeta, compare_with_cpu


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestNormFunctionalAPI(unittest.TestCase, metaclass=ParameterizedTestMeta):
    T = 64
    D = 256

    def setUp(self):
        super().setUp()
        torch.manual_seed(4242)

    def test_layernorm_with_weight_and_bias(self):
        def fn(x, w, b):
            return F.layer_norm(x, x.shape[1:], weight=w, bias=b)

        compare_with_cpu(
            fn,
            torch.randn(self.T, self.D, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
        )

    def test_layernorm_with_weight_bias_and_eps(self):
        def fn(x, w, b):
            return F.layer_norm(x, x.shape[1:], weight=w, bias=b, eps=1e-3)

        compare_with_cpu(
            fn,
            torch.randn(self.T, self.D, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
        )

    def test_layernorm_fused_residual(self):
        def fn(x, r, w, b):
            return F.layer_norm(x + r, (x + r).shape[1:], weight=w, bias=b)

        compare_with_cpu(
            fn,
            torch.randn(self.T, self.D, dtype=torch.float16),
            torch.randn(self.T, self.D, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
        )

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1889
    @pytest.mark.xfail(
        reason="#1889: aten::native_batch_norm not implemented for the 'spyre' backend "
        "(runtime NotImplementedError before compilation)"
    )
    def test_batch_norm_functional(self):
        def fn(x, rm, rv, w, b):
            return F.batch_norm(x, rm, rv, weight=w, bias=b, training=False)

        compare_with_cpu(
            fn,
            torch.randn(self.T, self.D, dtype=torch.float16),
            torch.zeros(self.D, dtype=torch.float16),
            torch.ones(self.D, dtype=torch.float16),
            torch.ones(self.D, dtype=torch.float16),
            torch.zeros(self.D, dtype=torch.float16),
        )

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/3287
    @pytest.mark.xfail(
        reason="#3287: TorchInductor compilation failure in Spyre lowering pass — "
        "KeyError 'No FX node for buf11' in split_multi_ops.py"
    )
    def test_group_norm_functional(self):
        def fn(x, w, b):
            return F.group_norm(x, 8, weight=w, bias=b)

        compare_with_cpu(
            fn,
            torch.randn(self.T, self.D, 16, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
        )

    def test_rmsnorm_manual(self):
        def fn(x, w):
            rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + 1e-5)
            return x / rms * w

        compare_with_cpu(
            fn,
            torch.randn(self.T, self.D, dtype=torch.float16),
            torch.ones(self.D, dtype=torch.float16),
        )
