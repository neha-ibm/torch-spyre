import pytest
import unittest
import torch

from utils_inductor import (
    ParameterizedTestMeta,
    cached_randn,
)


class TestOps(unittest.TestCase, metaclass=ParameterizedTestMeta):

    torch.manual_seed(0xAFFE)  # seeds cached_randn/cached_xavier calls in PARAMS below

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    PARAMS = {
        (
            "test_linear_decomposition_graph",
            "_linear_decomposition_graph",
        ): {
            "param_sets": {
                "2d": (
                    cached_randn((67, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((128, 256), dtype=torch.float16).to("spyre"),
                    None,
                ),
                "3d": (
                    cached_randn((2, 67, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((128, 256), dtype=torch.float16).to("spyre"),
                    None,
                ),
                "2d_bias": (
                    cached_randn((67, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((128, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((128,), dtype=torch.float16).to("spyre"),
                ),
                "3d_bias": (
                    cached_randn((67, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((128, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((128,), dtype=torch.float16).to("spyre"),
                ),
            },
        },
        (
            "test_unflatten_bmm_pass_graph",
            "_unflatten_bmm_pass_graph",
        ): {
            "param_sets": {
                "3d_2d": (
                    cached_randn((2, 67, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((256, 128), dtype=torch.float16).to("spyre"),
                ),
                "3d_3d": (
                    cached_randn((2, 67, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((2, 256, 128), dtype=torch.float16).to("spyre"),
                ),
                "3d_3d_bcast": (
                    cached_randn((4, 67, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((1, 256, 128), dtype=torch.float16).to("spyre"),
                ),
                "4d_4d": (
                    cached_randn((3, 17, 128, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((3, 17, 256, 128), dtype=torch.float16).to("spyre"),
                ),
                "4d_4d_bcast": (
                    cached_randn((3, 1, 128, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((1, 17, 256, 128), dtype=torch.float16).to("spyre"),
                ),
            },
        },
    }



    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def _linear_decomposition_graph(
        self, x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None
    ):
        from torch._dynamo.testing import InductorAndRecordGraphs, normalize_gm
        import torch._inductor.config as config

        config.force_disable_caches = True

        def linear_test(x, w, bias=None):
            return torch.nn.functional.linear(x, w, bias)

        torch.compiler.reset()
        backend = InductorAndRecordGraphs()
        cmp = torch.compile(linear_test, backend=backend)
        cmp(x, w, bias)
        inductor_graph_str = normalize_gm(
            backend.inductor_graphs[0].print_readable(print_output=False)
        )
        if x.dim() == 2:
            assert "aten.mm.default" in inductor_graph_str, (
                "Expected aten.mm.default in 2D linear decomposition graph"
            )
        elif x.dim() == 3:
            assert "aten.bmm.default" in inductor_graph_str, (
                "Expected aten.bmm.default in 3D linear decomposition graph"
            )
        assert "aten.addmm" not in inductor_graph_str, (
            "Custom linear decomp should avoid addmm"
        )

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def _unflatten_bmm_pass_graph(self, x: torch.Tensor, w: torch.Tensor):
        from torch._dynamo.testing import InductorAndRecordGraphs, normalize_gm
        import torch._inductor.config as config

        config.force_disable_caches = True

        def fn(x, w):
            return x @ w

        torch.compiler.reset()
        backend = InductorAndRecordGraphs()
        cmp = torch.compile(fn, backend=backend)
        cmp(x.to("spyre"), w.to("spyre"))
        inductor_graph_str = normalize_gm(
            backend.inductor_graphs[0].print_readable(print_output=False)
        )
        has_batched_matmul = (
            "aten.bmm.default" in inductor_graph_str
            or "spyre.batched_matmul" in inductor_graph_str
        )
        assert has_batched_matmul, (
            "Expected aten.bmm.default or spyre.batched_matmul after passes"
        )
        assert "aten.mm.default" not in inductor_graph_str, (
            "aten.mm.default should be replaced by bmm/batched_matmul after passes"
        )

    def test_mixed_device_seq(self):
        model = torch.compile(torch.sin)
        cpu_1 = torch._inductor.utils.get_code(model, torch.randn(5))[0]
        model = torch.compile(torch.sin)
        spyre_1 = torch._inductor.utils.get_code(
            model, torch.randn(5, device="spyre")
        )[0]
        torch._dynamo.reset()
        model = torch.compile(torch.sin)
        cpu_2 = torch._inductor.utils.get_code(model, torch.randn(5))[0]
        assert cpu_1.split("\n", 1)[1] == cpu_2.split("\n", 1)[1], (
            "CPU graph should be the same across compilations"
        )
        assert spyre_1 != cpu_1, "SPYRE graph should differ from CPU graph"



def _exp_sigmoid(x):
    return torch.sigmoid(torch.exp(x))

def _relu_tanh(x):
    return torch.tanh(torch.relu(x))

def _neg_abs(x):
    return torch.neg(torch.abs(x))

def _mul(x, y):
    return x * y

def _add(x, y):
    return x + y

def _sub(x, y):
    return x - y

def _where(x, y):
    return torch.where(x > 0, x, y)

def _relu(x):
    return torch.relu(x)

def _chained(x, y):
    a = x * y
    b = torch.sigmoid(a)
    c = b * 2.0
    return c + a



_ATOL = 1e-1


class TestPointwiseChunking(unittest.TestCase):
    """Correctness tests for the chunk_large_tensors inductor FX pass."""

    # ------------------------------------------------------------------
    # Core helper – compiles fn, runs on Spyre, checks against CPU ref.
    # ------------------------------------------------------------------

    def _check(self, fn, spyre_inputs, tag="", ref_fn=None):
        """Compile *fn*, run on Spyre tensors, compare result to a CPU reference.

        Args:
            fn:           Callable to compile and test.
            spyre_inputs: Tuple of tensors already on the "spyre" device.
            tag:          Short label included in the assertion message.
            ref_fn:       Optional alternative callable for the CPU reference.
                          Defaults to *fn* itself.
        """
        compiled = torch.compile(fn)
        z = compiled(*spyre_inputs)

        cpu_inputs = tuple(t.cpu().float() for t in spyre_inputs)
        ref = (ref_fn if ref_fn is not None else fn)(*cpu_inputs)

        max_diff = (ref - z.cpu().float()).abs().max().item()
        self.assertLessEqual(max_diff, _ATOL, f"{tag} max_diff={max_diff}")

    # ------------------------------------------------------------------
    # Group 1 – below the 8 GB limit, chunking should NOT be triggered
    # ------------------------------------------------------------------

    def test_1_1_small_tensor_exp_sigmoid(self):
        """1.1 Small tensor (~32 KB) – well under limit."""
        x = torch.randn(1, 128, 128, dtype=torch.float16, device="spyre")
        self._check(_exp_sigmoid, (x,), tag="[1.1]")

    def test_1_2_medium_tensor_exp_sigmoid(self):
        """1.2 Medium tensor (~64 MB) – comfortably under limit."""
        x = torch.randn(32, 1024, 1024, dtype=torch.float16, device="spyre")
        self._check(_exp_sigmoid, (x,), tag="[1.2]")

    def test_1_3_just_under_8gb_exp_sigmoid(self):
        """1.3 Just under 8 GB (~4 GB in float16) – no chunking expected."""
        x = torch.randn(32, 8192, 8192, dtype=torch.float16, device="spyre")
        self._check(_exp_sigmoid, (x,), tag="[1.3]")

    def test_1_4_exactly_8gb_exp_sigmoid(self):
        """1.4 Exactly 8 GB in float16 – no chunking expected.

        8 GB = 8,589,934,592 bytes; float16 = 2 bytes/element.
        64 × 8192 × 8192 = 4,294,967,296 elements → 8 GB exactly.
        """
        x = torch.randn(64, 8192, 8192, dtype=torch.float16, device="spyre")
        self._check(_exp_sigmoid, (x,), tag="[1.4]")

    # ------------------------------------------------------------------
    # Group 2 – above the 8 GB limit, chunking SHOULD be triggered
    # ------------------------------------------------------------------

    @pytest.mark.xfail(
        raises=Exception,
        reason="EAR overflow: chunk_large_tensors pass not yet splitting correctly for this shape",
        strict=False,
    )
    def test_2_1_basic_chunking_exp_sigmoid(self):
        """2.1 Basic chunking – ~9 GB in float16."""
        x = torch.randn(32, 8192, 17408, dtype=torch.float16, device="spyre")
        self._check(_exp_sigmoid, (x,), tag="[2.1]")

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="Numerical mismatch after chunking: max_diff=49.5 >> atol=0.1; chunking produces wrong results for relu+tanh at this size",
        strict=True,
    )
    def test_2_2_slightly_over_8gb_relu_tanh(self):
        """2.2 Slightly over 8 GB (~8.06 GB) – just crosses the threshold."""
        x = torch.randn(32, 8192, 16512, dtype=torch.float16, device="spyre")
        self._check(_relu_tanh, (x,), tag="[2.2]")

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="OutOfMemory: 27 GB tensor allocation fails on device; test requires hardware with sufficient memory",
        strict=False,
    )
    def test_2_3_very_large_abs_neg(self):
        """2.3 Very large – heavily over limit, multiple chunks expected."""
        x = torch.randn(32, 24576, 17408, dtype=torch.float16, device="spyre")
        self._check(_neg_abs, (x,), tag="[2.3]")

    @pytest.mark.xfail(
        raises=Exception,
        reason="EAR overflow: chunk_large_tensors pass not yet splitting correctly for this shape",
        strict=False,
    )
    def test_2_4_large_batch_dim_as_split_dim(self):
        """2.4 Large batch dim as split dim – chunking splits along dim 0."""
        x = torch.randn(35651584, 1, 128, dtype=torch.float16, device="spyre")
        self._check(_exp_sigmoid, (x,), tag="[2.4]")

    # ------------------------------------------------------------------
    # Group 3 – broadcast operands
    # ------------------------------------------------------------------

    @pytest.mark.xfail(
        raises=Exception,
        reason="EAR overflow: chunk_large_tensors pass not yet splitting correctly for broadcast batch dim",
        strict=False,
    )
    def test_3_1_broadcast_batch_dim(self):
        """3.1 y broadcasts across batch dim (1 → 32), ~9 GB total."""
        x = torch.randn(32, 8192, 17408, dtype=torch.float16, device="spyre")
        y = torch.randn(1, 8192, 17408, dtype=torch.float16, device="spyre")
        self._check(_mul, (x, y), tag="[3.1]")

    @pytest.mark.xfail(
        raises=Exception,
        reason="EAR overflow: chunk_large_tensors pass not yet splitting correctly for broadcast M dim",
        strict=False,
    )
    def test_3_2_broadcast_m_dim(self):
        """3.2 y broadcasts across M dim (1 → 8192), ~9 GB total."""
        x = torch.randn(32, 8192, 17408, dtype=torch.float16, device="spyre")
        y = torch.randn(32, 1, 17408, dtype=torch.float16, device="spyre")
        self._check(_add, (x, y), tag="[3.2]")

    @pytest.mark.xfail(
        raises=Exception,
        reason="EAR overflow: chunk_large_tensors pass not yet splitting correctly for broadcast N dim",
        strict=False,
    )
    def test_3_3_broadcast_n_dim(self):
        """3.3 y broadcasts across N dim (1 → 17408), ~9 GB total."""
        x = torch.randn(32, 8192, 17408, dtype=torch.float16, device="spyre")
        y = torch.randn(32, 8192, 1, dtype=torch.float16, device="spyre")
        self._check(_sub, (x, y), tag="[3.3]")

    # ------------------------------------------------------------------
    # Group 4 – spatial dims that are NOT multiples of 64
    # ------------------------------------------------------------------

    @pytest.mark.xfail(
        raises=Exception,
        reason="EAR overflow: M=8193 (non-multiple of 64) triggers span overflow even on small tensors",
        strict=False,
    )
    def test_4_1_m_not_multiple_of_64_small(self):
        """4.1 M=8193 (not multiple of 64), small tensor – no chunking."""
        x = torch.randn(32, 8193, 1740, dtype=torch.float16, device="spyre")
        y = torch.randn(32, 8193, 1740, dtype=torch.float16, device="spyre")
        self._check(_mul, (x, y), tag="[4.1 small]")

    @pytest.mark.xfail(
        raises=Exception,
        reason="EAR overflow: M=8193 (non-multiple of 64) triggers span overflow on large tensors",
        strict=False,
    )
    def test_4_1_m_not_multiple_of_64_large(self):
        """4.1 M=8193 (not multiple of 64), large tensor – chunking triggered."""
        x = torch.randn(32, 8193, 17408, dtype=torch.float16, device="spyre")
        y = torch.randn(32, 8193, 17408, dtype=torch.float16, device="spyre")
        self._check(_mul, (x, y), tag="[4.1 large]")

    def test_4_2_n_not_multiple_of_64_small(self):
        """4.2 N=17409 (not multiple of 64), small tensor – no chunking."""
        x = torch.randn(8, 128, 17409, dtype=torch.float16, device="spyre")
        y = torch.randn(8, 128, 17409, dtype=torch.float16, device="spyre")
        self._check(_add, (x, y), tag="[4.2 small]")

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="OutOfMemory: N=17409 large tensor (~9 GB) fails to allocate on device",
        strict=False,
    )
    def test_4_2_n_not_multiple_of_64_large(self):
        """4.2 N=17409 (not multiple of 64), large tensor – chunking triggered."""
        x = torch.randn(32, 8192, 17409, dtype=torch.float16, device="spyre")
        y = torch.randn(32, 8192, 17409, dtype=torch.float16, device="spyre")
        self._check(_add, (x, y), tag="[4.2 large]")

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="OutOfMemory: M=8193 N=17409 tensor (~9.1 GB) fails to allocate on device",
        strict=False,
    )
    def test_4_3_both_m_and_n_not_multiple_of_64(self):
        """4.3 M=8193 and N=17409 – both non-multiples of 64, chunking triggered."""
        x = torch.randn(32, 8193, 17409, dtype=torch.float16, device="spyre")
        y = torch.randn(32, 8193, 17409, dtype=torch.float16, device="spyre")
        self._check(_sub, (x, y), tag="[4.3]")

    # ------------------------------------------------------------------
    # Group 5 – edge batch sizes
    # ------------------------------------------------------------------

    def test_5_1_batch1_small(self):
        """5.1 batch=1, small (~272 MB) – no chunking."""
        x = torch.randn(1, 8192, 17408, dtype=torch.float16, device="spyre")
        y = torch.randn(1, 8192, 17408, dtype=torch.float16, device="spyre")
        self._check(_mul, (x, y), tag="[5.1 small]")

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="OutOfMemory: batch=1 large tensor (~8.5 GB) fails to allocate on device",
        strict=False,
    )
    def test_5_1_batch1_large(self):
        """5.1 batch=1, large (~8.5 GB via scaled M) – chunking triggered."""
        x = torch.randn(1, 262144, 17408, dtype=torch.float16, device="spyre")
        y = torch.randn(1, 262144, 17408, dtype=torch.float16, device="spyre")
        self._check(_mul, (x, y), tag="[5.1 large]")

    def test_5_2_batch2_small(self):
        """5.2 batch=2, small (~544 MB) – no chunking."""
        x = torch.randn(2, 8192, 17408, dtype=torch.float16, device="spyre")
        y = torch.randn(2, 8192, 17408, dtype=torch.float16, device="spyre")
        self._check(_add, (x, y), tag="[5.2 small]")

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="OutOfMemory: batch=2 large tensor (~8.5 GB) fails to allocate on device",
        strict=False,
    )
    def test_5_2_batch2_large(self):
        """5.2 batch=2, large (~8.5 GB via scaled M) – chunking triggered."""
        x = torch.randn(2, 131072, 17408, dtype=torch.float16, device="spyre")
        y = torch.randn(2, 131072, 17408, dtype=torch.float16, device="spyre")
        self._check(_add, (x, y), tag="[5.2 large]")

    def test_5_3_batch32_small_m_small(self):
        """5.3 batch=32, small M, small (~1.0 GB) – no chunking."""
        x = torch.randn(32, 256, 65536, dtype=torch.float16, device="spyre")
        y = torch.randn(32, 256, 65536, dtype=torch.float16, device="spyre")
        self._check(_sub, (x, y), tag="[5.3 small]")

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="OutOfMemory: batch=32 small-M large-N tensor (~8.5 GB) fails to allocate on device",
        strict=False,
    )
    def test_5_3_batch32_small_m_large(self):
        """5.3 batch=32, small M, large (~8.5 GB via scaled N) – chunking triggered."""
        x = torch.randn(32, 256, 557056, dtype=torch.float16, device="spyre")
        y = torch.randn(32, 256, 557056, dtype=torch.float16, device="spyre")
        self._check(_sub, (x, y), tag="[5.3 large]")

    # ------------------------------------------------------------------
    # Group 6 – prime / odd dimensions
    # ------------------------------------------------------------------

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="OutOfMemory: batch=31 tensor (~8.7 GB) fails to allocate on device",
        strict=False,
    )
    def test_6_1_prime_batch_dim(self):
        """6.1 batch=31 (prime) – chunking triggered (~8.73 GB)."""
        x = torch.randn(31, 8192, 17408, dtype=torch.float16, device="spyre")
        y = torch.randn(31, 8192, 17408, dtype=torch.float16, device="spyre")
        self._check(_mul, (x, y), tag="[6.1]")

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="OutOfMemory: M=8191 tensor (~9 GB) fails to allocate on device",
        strict=False,
    )
    def test_6_2_prime_m_dim(self):
        """6.2 M=8191 (prime) – chunking triggered (~8.99 GB)."""
        x = torch.randn(32, 8191, 17408, dtype=torch.float16, device="spyre")
        y = torch.randn(32, 8191, 17408, dtype=torch.float16, device="spyre")
        self._check(_add, (x, y), tag="[6.2]")

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="OutOfMemory: all-odd dims tensor (~9.4 GB) fails to allocate on device",
        strict=False,
    )
    def test_6_3_all_odd_dims(self):
        """6.3 batch=33, M=8193, N=17409 – all odd, chunking triggered."""
        x = torch.randn(33, 8193, 17409, dtype=torch.float16, device="spyre")
        y = torch.randn(33, 8193, 17409, dtype=torch.float16, device="spyre")
        self._check(_sub, (x, y), tag="[6.3]")

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="OutOfMemory: float32 M=8191 tensor (~18 GB) fails to allocate on device",
        strict=False,
    )
    def test_6_4_prime_m_dim_float32(self):
        """6.4 M=8191 (prime) – chunking triggered (~8.99 GB), float32."""
        x = torch.randn(32, 8191, 17408, dtype=torch.float32, device="spyre")
        y = torch.randn(32, 8191, 17408, dtype=torch.float32, device="spyre")
        self._check(_add, (x, y), tag="[6.4]")

    # ------------------------------------------------------------------
    # Group 7 – size-1 dimensions
    # ------------------------------------------------------------------

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="OutOfMemory: batch=1 M=1 large-N tensor (~8.5 GB) fails to allocate on device",
        strict=False,
    )
    def test_7_1_batch1_m1_only_n_splittable(self):
        """7.1 batch=1, M=1 – only N is splittable; fallback to max() picks N."""
        x = torch.randn(1, 1, 4563402752, dtype=torch.float16, device="spyre")
        y = torch.randn(1, 1, 4563402752, dtype=torch.float16, device="spyre")
        self._check(_mul, (x, y), tag="[7.1]")

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="OutOfMemory: batch=1 large-M tensor (~8.5 GB) fails to allocate on device",
        strict=False,
    )
    def test_7_2_multiple_size1_dims_batch1(self):
        """7.2 batch=1 – _find_split_dim skips batch, picks M (dim 1)."""
        x = torch.randn(1, 262144, 17408, dtype=torch.float16, device="spyre")
        y = torch.randn(1, 262144, 17408, dtype=torch.float16, device="spyre")
        self._check(_add, (x, y), tag="[7.2]")

    # ------------------------------------------------------------------
    # Group 8 – very large tensors requiring many chunks
    # ------------------------------------------------------------------

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="OutOfMemory: ~25 GB tensor fails to allocate on device",
        strict=False,
    )
    def test_8_1_25gb_three_or_more_chunks(self):
        """8.1 ~25 GB output – needs 3+ chunks."""
        x = torch.randn(32, 24576, 17408, dtype=torch.float16, device="spyre")
        y = torch.randn(32, 24576, 17408, dtype=torch.float16, device="spyre")
        self._check(_mul, (x, y), tag="[8.1]")

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="OutOfMemory: ~50 GB tensor fails to allocate on device",
        strict=False,
    )
    def test_8_2_50gb_six_or_more_chunks(self):
        """8.2 ~50 GB output – needs 6+ chunks."""
        x = torch.randn(32, 49152, 17408, dtype=torch.float16, device="spyre")
        y = torch.randn(32, 49152, 17408, dtype=torch.float16, device="spyre")
        self._check(_add, (x, y), tag="[8.2]")

    # ------------------------------------------------------------------
    # Group 9 – various pointwise op types
    # ------------------------------------------------------------------

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="OutOfMemory: ~9 GB tensor fails to allocate on device",
        strict=False,
    )
    def test_9_1_addition_pointwise(self):
        """9.1 Binary addition – chunking triggered (~9 GB)."""
        x = torch.randn(32, 8192, 17408, dtype=torch.float16, device="spyre")
        y = torch.randn(32, 8192, 17408, dtype=torch.float16, device="spyre")
        self._check(_add, (x, y), tag="[9.1]")

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="OutOfMemory: ~9 GB tensor fails to allocate on device",
        strict=False,
    )
    def test_9_2_relu_unary(self):
        """9.2 Unary ReLU – no broadcast concern, chunking triggered (~9 GB)."""
        x = torch.randn(32, 8192, 17408, dtype=torch.float16, device="spyre")
        self._check(_relu, (x,), tag="[9.2]")

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="OutOfMemory: ~9 GB tensor fails to allocate on device",
        strict=False,
    )
    def test_9_3_where_conditional(self):
        """9.3 Ternary torch.where – chunking triggered (~9 GB)."""
        x = torch.randn(32, 8192, 17408, dtype=torch.float16, device="spyre")
        y = torch.randn(32, 8192, 17408, dtype=torch.float16, device="spyre")
        self._check(_where, (x, y), tag="[9.3]")

    # ------------------------------------------------------------------
    # Group 10 – exact 8 GB boundary
    # ------------------------------------------------------------------

    def test_10_1_exactly_8gb_should_not_chunk(self):
        """10.1 Exactly 8 GB – should NOT trigger chunking.

        32 * 8192 * 16384 * 2 bytes = 8,589,934,592 bytes = exactly 8 GB.
        """
        x = torch.randn(32, 8192, 16384, dtype=torch.float16, device="spyre")
        y = torch.randn(32, 8192, 16384, dtype=torch.float16, device="spyre")
        self._check(_mul, (x, y), tag="[10.1]")

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="Numerical mismatch after chunking: max_diff=15.15 >> atol=0.1; chunking produces wrong results at N=16448",
        strict=True,
    )
    def test_10_2_one_stick_over_8gb_should_chunk(self):
        """10.2 One stick (64 elements) over 8 GB – SHOULD trigger chunking.

        32 * 8192 * 16448 * 2 bytes = 8,623,489,024 bytes (~8.03 GB).
        N = 16384 + 64 = 16448.
        """
        x = torch.randn(32, 8192, 16448, dtype=torch.float16, device="spyre")
        y = torch.randn(32, 8192, 16448, dtype=torch.float16, device="spyre")
        self._check(_add, (x, y), tag="[10.2]")

    # ------------------------------------------------------------------
    # Group 11 – 4-D tensors
    # ------------------------------------------------------------------

    @pytest.mark.xfail(
        raises=Exception,
        reason="EAR overflow: 4-D tensor chunking triggers SIGABRT in dxp_standalone bundler",
        strict=False,
    )
    def test_11_1_4d_pointwise_mul(self):
        """11.1 4-D pointwise multiply – ~8.50 GB, chunking triggered.

        2 * 32 * 4096 * 17408 * 2 bytes = 9,126,805,504 bytes (~8.50 GB).
        """
        x = torch.randn(2, 32, 4096, 17408, dtype=torch.float16, device="spyre")
        y = torch.randn(2, 32, 4096, 17408, dtype=torch.float16, device="spyre")
        self._check(_mul, (x, y), tag="[11.1]")

    # ------------------------------------------------------------------
    # Group 12 – Awkward batch=7
    # ------------------------------------------------------------------

    def test_12_awkward_batch_exp_sigmoid(self):
        """12.1 Awkward batch=7 (7, 8193, 1740) – non-power-of-2 dimensions.

        Shape: 7 × 8193 × 1740 = 99,746,220 elements (~190 MB in float16).
        Tests compiler handling of irregular tensor shapes.
        """
        x = torch.randn(7, 8193, 1740, dtype=torch.float16, device="spyre")
        self._check(_exp_sigmoid, (x,), tag="[12.1]")

    # ------------------------------------------------------------------
    # Group 13 – Single giant dimension
    # ------------------------------------------------------------------

    @pytest.mark.xfail(
        raises=Exception,
        reason="EAR overflow: seq_len=65537 (2^16+1) triggers SIGABRT in dxp_standalone bundler",
        strict=False,
    )
    def test_13_long_sequence_exp_sigmoid(self):
        """13.1 Long sequence (1, 65537, 4096) – tests large sequence length.

        Shape: 1 × 65537 × 4096 = 268,435,456 elements (~512 MB in float16).
        65537 = 2^16 + 1.
        """
        x = torch.randn(1, 65537, 4096, dtype=torch.float16, device="spyre")
        self._check(_exp_sigmoid, (x,), tag="[13.1]")

    # ------------------------------------------------------------------
    # Group 14 – Chained Ops
    # ------------------------------------------------------------------

    @pytest.mark.xfail(
        raises=Exception,
        reason="EAR overflow: chained ops (32, 8193, 1740) triggers SIGABRT in dxp_standalone bundler",
        strict=False,
    )
    def test_14_chained_ops(self):
        """14.1 Chained operations – tests fusion and intermediate allocations.

        Shape: 32 × 8193 × 1740 = 456,499,200 elements/tensor (~870 MB in float16).
        """
        x = torch.randn(32, 8193, 1740, dtype=torch.float16, device="spyre")
        y = torch.randn(32, 8193, 1740, dtype=torch.float16, device="spyre")
        self._check(_chained, (x, y), tag="[14.1]")


if __name__ == "__main__":
    unittest.main()