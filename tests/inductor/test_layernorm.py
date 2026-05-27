import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_inductor import compare_with_cpu


# PART 1: NEGATIVE & REGRESSION TESTS


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestLayerNormNegative:
    torch.manual_seed(0xAFFE)

    @pytest.mark.xfail(
        reason="Spyre backend: views not supported for non-contiguous tensors"
    )
    def test_non_contiguous_input(self, execution_mode):
        """Non-contiguous input (transposed) — backend consistency check."""

        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        x = torch.randn(10, 2, dtype=torch.float16).transpose(0, 1)
        compare_with_cpu(
            fn,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: NaN propagation returns zeros instead of NaN"
    )
    def test_nan_input(self, execution_mode):
        """NaN in input tensor — numerical propagation parity."""

        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        x = torch.randn(2, 10, dtype=torch.float16)
        x[0, 0] = float("nan")
        compare_with_cpu(
            fn,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: returns constant garbage values (0.5400) instead of normalized output"
    )
    def test_large__values(self, execution_mode):
        """Large magnitude values — stability under scale."""

        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        x = torch.randn(2, 10, dtype=torch.float16) * 1e3
        compare_with_cpu(
            fn,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: illegal device layout error for single element tensors"
    )
    def test_1d_tensor_edge_case(self, execution_mode):
        """1D tensor — valid boundary acceptance."""

        def fn(x):
            return F.layer_norm(x, x.shape)

        x = torch.randn(10, dtype=torch.float16)
        compare_with_cpu(
            fn,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Issue #2: Numerical accuracy failure - 85-100% element mismatch vs CPU"
    )
    def test_exactly_6_args_boundary(self, execution_mode):
        """Exactly 6 arguments — core regression guard for PR #337."""

        def fn(x, w, b):
            return F.layer_norm(x, x.shape[1:], weight=w, bias=b, eps=1e-5)

        compare_with_cpu(
            fn,
            torch.randn(2, 10, dtype=torch.float16),
            torch.ones(10, dtype=torch.float16),
            torch.zeros(10, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Issue #1 (compiled) / Issue #2 (eager): C++ compilation error in compiled mode, numerical accuracy failure in eager mode"
    )
    def test_5_args_no_regression(self, execution_mode):
        """5 arguments — no regression below fix boundary."""

        def fn(x, w):
            return F.layer_norm(x, x.shape[1:], weight=w, eps=1e-5)

        compare_with_cpu(
            fn,
            torch.randn(2, 10, dtype=torch.float16),
            torch.ones(10, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Issue #1 (compiled) / Issue #2 (eager): C++ compilation error in compiled mode, numerical accuracy failure in eager mode"
    )
    def test_4_args_no_regression(self, execution_mode):
        """4 arguments — no regression below fix boundary."""

        def fn(x):
            return F.layer_norm(x, x.shape[1:], eps=1e-5)

        compare_with_cpu(
            fn,
            torch.randn(2, 10, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Issue #2: Numerical accuracy failure - 85-100% element mismatch vs CPU"
    )
    def test_3_args_minimal(self, execution_mode):
        """3 arguments — minimal path (lowest supported path remains valid)."""

        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        compare_with_cpu(
            fn,
            torch.randn(2, 10, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )


# PART 2: ARGUMENT SCALING & NORM COMPARISONS


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestNormArgScaling:
    T = 64
    D = 256

    @pytest.mark.xfail(
        reason="Issue #2: Numerical accuracy failure - 85-100% element mismatch vs CPU"
    )
    def test_layernorm_1_arg(self, execution_mode):
        torch.manual_seed(4242)

        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        compare_with_cpu(
            fn,
            torch.randn(self.T, self.D, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Issue #2: Numerical accuracy failure - 85-100% element mismatch vs CPU"
    )
    def test_layernorm_2_arg(self, execution_mode):
        torch.manual_seed(4242)

        def fn(x):
            return F.layer_norm(x, x.shape[1:], weight=None, bias=None)

        compare_with_cpu(
            fn,
            torch.randn(self.T, self.D, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Issue #2: Numerical accuracy failure - 85-100% element mismatch vs CPU"
    )
    def test_layernorm_3_arg(self, execution_mode):
        torch.manual_seed(4242)

        def fn(x, w):
            return F.layer_norm(x, x.shape[1:], weight=w, bias=None)

        compare_with_cpu(
            fn,
            torch.randn(self.T, self.D, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Issue #1 (compiled) / Issue #2 (eager): C++ compilation error in compiled mode, numerical accuracy failure in eager mode"
    )
    def test_layernorm_4_arg(self, execution_mode):
        torch.manual_seed(4242)

        def fn(x, w, b):
            return F.layer_norm(x, x.shape[1:], weight=w, bias=b)

        compare_with_cpu(
            fn,
            torch.randn(self.T, self.D, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Issue #1 (compiled) / Issue #2 (eager): C++ compilation error in compiled mode, numerical accuracy failure in eager mode"
    )
    def test_layernorm_5_arg(self, execution_mode):
        torch.manual_seed(4242)

        def fn(x, w, b):
            return F.layer_norm(x, x.shape[1:], weight=w, bias=b, eps=1e-3)

        compare_with_cpu(
            fn,
            torch.randn(self.T, self.D, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: device mismatch — module weights on CPU while input on Spyre device"
    )
    def test_layernorm_6_arg_module(self, execution_mode):
        torch.manual_seed(4242)
        layer = nn.LayerNorm([self.D])
        layer.weight.data = torch.randn([self.D], dtype=torch.float16)
        layer.bias.data = torch.randn([self.D], dtype=torch.float16)

        def fn(x):
            return layer(x)

        compare_with_cpu(
            fn,
            torch.randn(self.T, self.D, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Issue #3: Missing BatchNorm support - aten::native_batch_norm not implemented for spyre backend"
    )
    def test_batchnorm_safe(self, execution_mode):
        torch.manual_seed(4242)

        def fn(x, rm, rv, w, b):
            return F.batch_norm(x, rm, rv, weight=w, bias=b, training=False)

        compare_with_cpu(
            fn,
            torch.randn(self.T, self.D, dtype=torch.float16),
            torch.zeros(self.D, dtype=torch.float16),
            torch.ones(self.D, dtype=torch.float16),
            torch.ones(self.D, dtype=torch.float16),
            torch.zeros(self.D, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: C++ compilation error — missing operator== in VectorizedN<float, 2>"
    )
    def test_groupnorm(self, execution_mode):
        torch.manual_seed(4242)

        def fn(x, w, b):
            return F.group_norm(x, 8, weight=w, bias=b)

        compare_with_cpu(
            fn,
            torch.randn(self.T, self.D, 16, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: C++ compilation error — missing operator== in VectorizedN<float, 2>"
    )
    def test_instancenorm(self, execution_mode):
        torch.manual_seed(4242)

        def fn(x, w, b):
            return F.instance_norm(x, weight=w, bias=b, use_input_stats=True)

        compare_with_cpu(
            fn,
            torch.randn(8, self.D, 16, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    def test_rmsnorm(self, execution_mode):
        torch.manual_seed(4242)

        def fn(x, w):
            rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + 1e-5)
            return x / rms * w

        compare_with_cpu(
            fn,
            torch.randn(self.T, self.D, dtype=torch.float16),
            torch.ones(self.D, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Issue #1 (compiled) / Issue #2 (eager): C++ compilation error in compiled mode, numerical accuracy failure in eager mode"
    )
    def test_fused_layernorm(self, execution_mode):
        torch.manual_seed(4242)

        def fn(x, r, w, b):
            return F.layer_norm(x + r, (x + r).shape[1:], weight=w, bias=b)

        compare_with_cpu(
            fn,
            torch.randn(self.T, self.D, dtype=torch.float16),
            torch.randn(self.T, self.D, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
            torch.randn(self.D, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )


# PART 3: BACKEND DIVERGENCE & STRESS TESTS


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestLayerNormBackendStress:
    torch.manual_seed(0xBEEF)

    @pytest.mark.xfail(
        reason="Spyre backend: views not supported for non-contiguous tensors with parameters"
    )
    def test_noncontiguous_with_params(self, execution_mode):
        def fn(x, w, b):
            return F.layer_norm(x, x.shape[1:], weight=w, bias=b)

        x = torch.randn(10, 2, dtype=torch.float16).transpose(0, 1)
        w = torch.ones(10, dtype=torch.float16)
        b = torch.zeros(10, dtype=torch.float16)
        compare_with_cpu(
            fn,
            x,
            w,
            b,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: returns constant garbage values for large FP16 inputs"
    )
    def test_fp16_overflow(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        x = (torch.randn(2, 10) * 1e4).to(torch.float16)
        compare_with_cpu(
            fn,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: returns zeros for constant input (zero variance)"
    )
    def test_zero_variance(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        x = torch.ones(2, 10, dtype=torch.float16) * 5.0
        compare_with_cpu(
            fn,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: returns constant garbage values with extreme epsilon"
    )
    def test_extreme_eps(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:], eps=1e-12)

        x = torch.randn(2, 10, dtype=torch.float16)
        compare_with_cpu(
            fn,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: only supports normalized_shape of length 1, not multi-dimensional"
    )
    def test_high_dim_tensor(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        x = torch.randn(2, 3, 4, 5, 10, dtype=torch.float16)
        compare_with_cpu(
            fn,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.skip(reason="FATAL: Spyre backend segfaults on int32 casting")
    def test_int_cast_path(self, execution_mode):
        def fn(x):
            return F.layer_norm(x.float(), x.float().shape[1:])

        x = torch.randint(0, 10, (2, 10), dtype=torch.int32)
        compare_with_cpu(
            fn,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )


# PART 4: COMPREHENSIVE EDGE CASES


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestLayerNormEdgeCases:
    torch.manual_seed(0xAFFE)

    @pytest.mark.xfail(
        reason="Spyre backend: illegal device layout error for single element tensors"
    )
    def test_single_element(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape)

        compare_with_cpu(
            fn,
            torch.randn(1, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: illegal device layout error for single element tensors with 6 args"
    )
    def test_single_element_6arg(self, execution_mode):
        def fn(x, w, b):
            return F.layer_norm(x, x.shape, weight=w, bias=b)

        compare_with_cpu(
            fn,
            torch.randn(1, dtype=torch.float16),
            torch.ones(1, dtype=torch.float16),
            torch.zeros(1, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: returns constant garbage values for very wide tensors"
    )
    def test_very_wide(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        compare_with_cpu(
            fn,
            torch.randn(1, 50000, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: returns constant garbage values for very wide tensors with 6 args"
    )
    def test_very_wide_6arg(self, execution_mode):
        def fn(x, w, b):
            return F.layer_norm(x, x.shape[1:], weight=w, bias=b)

        compare_with_cpu(
            fn,
            torch.randn(1, 50000, dtype=torch.float16),
            torch.ones(50000, dtype=torch.float16),
            torch.zeros(50000, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: returns constant garbage values for very tall tensors"
    )
    def test_very_tall(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        compare_with_cpu(
            fn,
            torch.randn(100000, 10, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: only supports normalized_shape of length 1, not multi-dimensional [4, 5]"
    )
    def test_5d_tensor(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[2:])

        compare_with_cpu(
            fn,
            torch.randn(2, 3, 4, 5, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: only supports normalized_shape of length 1, not multi-dimensional [4, 5] with 6 args"
    )
    def test_5d_tensor_6arg(self, execution_mode):
        def fn(x, w, b):
            return F.layer_norm(x, x.shape[2:], weight=w, bias=b)

        compare_with_cpu(
            fn,
            torch.randn(2, 3, 4, 5, dtype=torch.float16),
            torch.ones(4, 5, dtype=torch.float16),
            torch.zeros(4, 5, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Issue #2: Numerical accuracy failure - 85-100% element mismatch vs CPU"
    )
    def test_large_batch(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        compare_with_cpu(
            fn,
            torch.randn(2048, 128, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Issue #2: Numerical accuracy failure - 85-100% element mismatch vs CPU"
    )
    def test_non_power_of_two(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        compare_with_cpu(
            fn,
            torch.randn(2, 130, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(reason="Spyre backend: returns zeros for all-zero input")
    def test_all_zeros(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        compare_with_cpu(
            fn,
            torch.zeros(2, 10, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: returns zeros for all-zero input with 6 args"
    )
    def test_all_zeros_6arg(self, execution_mode):
        def fn(x, w, b):
            return F.layer_norm(x, x.shape[1:], weight=w, bias=b)

        compare_with_cpu(
            fn,
            torch.zeros(2, 10, dtype=torch.float16),
            torch.ones(10, dtype=torch.float16),
            torch.zeros(10, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: returns zeros for constant input (all ones)"
    )
    def test_all_ones(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        compare_with_cpu(
            fn,
            torch.ones(2, 10, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: returns zeros for constant input (value 5.0)"
    )
    def test_constant_value(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        compare_with_cpu(
            fn,
            torch.full((2, 10), 5.0, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: returns constant garbage values for large magnitude inputs"
    )
    def test_large_values(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        compare_with_cpu(
            fn,
            torch.randn(2, 10, dtype=torch.float16) * 1e3,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Issue #2: Numerical accuracy failure - 85-100% element mismatch vs CPU"
    )
    def test_small_values(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        compare_with_cpu(
            fn,
            torch.randn(2, 10, dtype=torch.float16) * 1e-3,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: returns constant garbage values for large mean with small variance"
    )
    def test_large_mean_small_variance(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        x = (
            torch.full((2, 10), 1e3, dtype=torch.float16)
            + torch.randn(2, 10, dtype=torch.float16) * 1e-2
        )
        compare_with_cpu(
            fn,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: NaN propagation returns zeros instead of NaN"
    )
    def test_nan(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        x = torch.randn(2, 10, dtype=torch.float16)
        x[0, 0] = float("nan")
        compare_with_cpu(
            fn,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: Inf propagation returns zeros instead of Inf"
    )
    def test_pos_inf(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        x = torch.randn(2, 10, dtype=torch.float16)
        x[0, 0] = float("inf")
        compare_with_cpu(
            fn,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: -Inf propagation returns zeros instead of -Inf"
    )
    def test_neg_inf(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        x = torch.randn(2, 10, dtype=torch.float16)
        x[0, 0] = float("-inf")
        compare_with_cpu(
            fn,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(reason="Spyre backend: mixed NaN/Inf propagation returns zeros")
    def test_mixed_inf_nan(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        x = torch.randn(2, 10, dtype=torch.float16)
        x[0, 0] = float("nan")
        x[0, 1] = float("inf")
        x[0, 2] = float("-inf")
        compare_with_cpu(
            fn,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: returns zeros for constant input with tiny epsilon"
    )
    def test_tiny_eps(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:], eps=1e-12)

        compare_with_cpu(
            fn,
            torch.ones(2, 10, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: returns constant garbage values with large epsilon"
    )
    def test_large_eps(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:], eps=1.0)

        compare_with_cpu(
            fn,
            torch.randn(2, 10, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: returns constant garbage values with extreme epsilon"
    )
    def test_extreme__eps(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:], eps=10.0)

        compare_with_cpu(
            fn,
            torch.randn(2, 10, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: returns zeros for constant input (zero variance)"
    )
    def test_zero__variance(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        compare_with_cpu(
            fn,
            torch.ones(2, 10, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: views not supported for non-contiguous tensors"
    )
    def test_non_contiguous(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        x = torch.randn(10, 2, dtype=torch.float16).transpose(0, 1)
        compare_with_cpu(
            fn,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(reason="Spyre backend: views not supported for strided tensors")
    def test_strided_slice(self, execution_mode):
        def fn(x):
            return F.layer_norm(x, x.shape[1:])

        x = torch.randn(20, 10, dtype=torch.float16)[::2]
        compare_with_cpu(
            fn,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: mixed dtype not supported — unexpected argument PointwiseOp(op='to_dtype')"
    )
    def test_mixed_dtype(self, execution_mode):
        def fn(x, w):
            return F.layer_norm(x, x.shape[1:], weight=w)

        compare_with_cpu(
            fn,
            torch.randn(2, 10, dtype=torch.float16),
            torch.ones(10, dtype=torch.float32),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Issue #2: Numerical accuracy failure - 85-100% element mismatch vs CPU"
    )
    def test_6arg_dynamic_shape(self, execution_mode):
        """6-arg path with normalized_shape derived from input dim at runtime."""

        def fn(x, w, b):
            return F.layer_norm(x, (x.shape[-1],), weight=w, bias=b)

        compare_with_cpu(
            fn,
            torch.randn(2, 10, dtype=torch.float16),
            torch.ones(10, dtype=torch.float16),
            torch.zeros(10, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.xfail(
        reason="Spyre backend: only supports normalized_shape of length 1, not multi-dimensional [4, 5] with 6 args"
    )
    def test_6arg_multi_dim(self, execution_mode):
        def fn(x, w, b):
            return F.layer_norm(x, x.shape[2:], weight=w, bias=b)

        compare_with_cpu(
            fn,
            torch.randn(2, 3, 4, 5, dtype=torch.float16),
            torch.ones(4, 5, dtype=torch.float16),
            torch.zeros(4, 5, dtype=torch.float16),
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
