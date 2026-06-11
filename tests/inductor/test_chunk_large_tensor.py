import os
import pytest
import torch

from utils_inductor import (
    _compile_and_run,
    cached_randn,
    cached_xavier,
    compare_with_cpu,
    make_param_dict,
    unique_randn_along_dim,
    shapes2key,
)
import utils_inductor
from torch_spyre._inductor.dtype_ops import DtypeOpTable
from torch_spyre._inductor.constants import IDENTITY_OP

os.environ.setdefault("CHUNK_LARGE_TENSORS", "1")
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")
os.environ.setdefault("SPYRE_INDUCTOR_LOG", "1")
os.environ.setdefault("SPYRE_INDUCTOR_LOG_LEVEL", "DEBUG")
os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _exp_sigmoid(x): return torch.sigmoid(torch.exp(x))
def _relu_tanh(x): return torch.tanh(torch.relu(x))
def _neg_abs(x): return torch.neg(torch.abs(x))
def _mul(x, y): return x * y
def _add(x, y): return x + y
def _sub(x, y): return x - y
def _where(x, y): return torch.where(x > 0, x, y)
def _relu(x): return torch.relu(x)

def _chained(x, y): return torch.cos(torch.sin(x) * torch.nn.functional.silu(y)) + x
def _chained_relu_add(x, y): return torch.sigmoid(torch.where(x > 0.0, torch.nn.functional.gelu(x), y))
def _chained_mul_sub_tanh(x, y):
    x_pos, y_pos = x.abs() + 0.1, y.abs() + 0.1
    return torch.tanh(torch.clamp(torch.div(torch.pow(x_pos, y_pos), x_pos), min=-1.0, max=1.0))
def _chained_neg_abs_add(x, y):
    return torch.neg(torch.reciprocal(torch.sqrt(torch.log(x.abs() + 0.5).abs() + 0.1))) + torch.rsqrt(y.abs() + 0.5)
def _chained_deep(x, y):
    return torch.where(torch.floor(torch.nn.functional.mish(torch.nn.functional.softplus(x))) <= 0.0, x, y)

_ATOL = 1e-1

# ---------------------------------------------------------------------------
# Parameterization helpers
# ---------------------------------------------------------------------------

def _expand_dict(raw_dict):
    """Normalise cores=None → cores=1 and suffix the key."""
    expanded = {}
    for k, v in raw_dict.items():
        if v[-1] is not None:
            expanded[f"{k}-cores{v[-1]}"] = v
        else:
            expanded[f"{k}-cores1"] = (*v[:-1], 1)
    return expanded

def _expand_xfails(raw_list, raw_dict):
    expanded = []
    for k in raw_list:
        v = raw_dict[k]
        if v[-1] is not None:
            expanded.append(f"{k}-cores{v[-1]}")
        else:
            expanded.append(f"{k}-cores1")
    return expanded

def _make_custom_params(param_sets, expect_fail=None):
    expect_fail = expect_fail or {}
    result = []
    for param_id, (factory_fn, fn, ref_fn) in param_sets.items():
        marks = []
        if param_id in expect_fail:
            action, reason = expect_fail[param_id]
            if action == "skip":
                marks.append(pytest.mark.skip(reason=reason))
            else:
                marks.append(pytest.mark.xfail(reason=reason))
        result.append(
            pytest.param(factory_fn, fn, ref_fn, id=param_id, marks=marks)
        )
    return result

def _make_params(param_sets, ops_dict=None, expect_fail=None):
    expect_fail = expect_fail or {}
    result = []

    if ops_dict:
        for param_id, args in param_sets.items():
            if not isinstance(args, (tuple, list)):
                args = (args,)
            for op_name, op in ops_dict.items():
                full_id = f"{op_name}-{param_id}"
                marks = []
                if full_id in expect_fail:
                    action, reason = expect_fail[full_id]
                    if action == "skip":
                        marks.append(pytest.mark.skip(reason=reason))
                    else:
                        marks.append(pytest.mark.xfail(reason=reason))
                result.append(
                    pytest.param(op_name, op, *args, id=full_id, marks=marks)
                )
    else:
        for param_id, args in param_sets.items():
            if not isinstance(args, (tuple, list)):
                args = (args,)
            marks = []
            if param_id in expect_fail:
                action, reason = expect_fail[param_id]
                if action == "skip":
                    marks.append(pytest.mark.skip(reason=reason))
                else:
                    marks.append(pytest.mark.xfail(reason=reason))
            result.append(
                pytest.param(*args, id=param_id, marks=marks)
            )

    return result

# ---------------------------------------------------------------------------
# 1-D Chunking
# ---------------------------------------------------------------------------

CHUNKING_1D_PARAM_SETS = {
    # ── span > 256 MB, total < 8 GB ──────────────────────────────────────────
    # even element count, float16, unary
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_even_fp16":        (_exp_sigmoid, (134217728,),  None,           torch.float16,   1),
    # odd element count, bfloat16, unary  (different shape from above)
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_odd_bf16":         (_relu_tanh,   (134217757,),  None,           torch.bfloat16, 32),
    # prime element count, float32, unary
    # TODO: DtException: EAR overflow detected
    "span_gt256_lt8gb_prime_fp32":       (_neg_abs,     (134217761,),  None,           torch.float32,   7),

    # ── total > 8 GB ──────────────────────────────────────────────────────────
    # odd, float32, unary
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_odd_fp32":              (_relu_tanh,   (4412345679,), None,           torch.float32,   7),
    # prime, float16, unary
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_prime_fp16":            (_neg_abs,     (5123456789,), None,           torch.float16,  32),

    # ── span > 256 MB AND total > 8 GB ───────────────────────────────────────
    # prime, float16, unary
    # TODO: DtException: isValidDimParam
    "span_gt256_and_total_gt8gb_prime_fp16": (_neg_abs, (5123456789,), None,           torch.float16,  32),

    # ── exact boundary: total == 8 GB (float16 → 8*2^30 / 2 = 2^32 elems) ──
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "exact_8gb_fp16":                    (_exp_sigmoid, (4294967296,), None,           torch.float16,   1),

    # ── exact boundary: span == 256 MB (float32 → 256*2^20 / 4 = 2^26 elems) ─
    # TODO: Error in codegen for ComputedBuffer during pointwise operations (issue #2612)
    "exact_span_256mb_fp32":             (_relu_tanh,   (67108864,),   None,           torch.float32,   1),

    # ── floor-division boundary ───────────────────────────────────────────────
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "floor_division_boundary_fp16":      (_exp_sigmoid, (402653184,),  None,           torch.float16,   1),

    # ── dtype coverage ───────────────────────────────────────────────────────
    # TODO: FlexAllocator OutOfMemory
    "bool_unary_prime":                  (torch.logical_not, (4987654321,), None,      torch.bool,     32),
    # TODO: Missing Backend Codegen for Dtype/Op
    "int32_binary_even":                 (torch.add,    (134217729,),  (134217729,),   torch.int32,     7),
    # TODO: FlexAllocator OutOfMemory
    "int64_add_prime":                   (torch.add,    (134217761,),  (134217761,),   torch.int64,    32),
}

EXPECTED_FAILURES_1D = {
    "span_gt256_lt8gb_even_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "span_gt256_lt8gb_odd_bf16-cores32": ("xfail", "DtException: No valid prime factor"),
    "span_gt256_lt8gb_prime_fp32-cores7": ("xfail", "DtException: EAR overflow detected"),
    "total_gt8gb_odd_fp32-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_prime_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "span_gt256_and_total_gt8gb_prime_fp16-cores32": ("xfail", "DtException: isValidDimParam"),
    "exact_8gb_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "exact_span_256mb_fp32-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "floor_division_boundary_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "bool_unary_prime-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "int32_binary_even-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int64_add_prime-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "span_gt256_lt8gb_odd_bf16-cores7": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "span_gt256_lt8gb_prime_fp32-cores32": ("xfail", "DtException: EAR overflow detected"),
    "dim0_1_dim1_max_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "dim0_max_dim1_1_fp16-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_2d_1d_int32-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "broadcast_2d_1d_fp32-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "bool_logical_or_even-cores1": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int32_sub_prime-cores32": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int64_add_odd-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "bf16_add_even-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "mixed_fp16_bf16_add-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "mixed_fp16_fp32_add-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "exact_8gb_span_256mb_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "threshold_just_under_8gb_fp16-cores7": ("xfail", "DtException: Invalid xlat size"),
    "threshold_just_over_8gb_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_even_fp16_c1-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_odd_fp32_c7-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_prime_bf16_c32-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "dev_large_batch_large_m-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "dev_large_total_bytes_trigger-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_batch_dim_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_m_dim_fp32-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_3d_1d_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "basic_chunking_bf16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "both_m_n_not_multiple_64_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "prime_m_dim_fp16-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "bool_logical_and-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "int32_add_even-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "float32_large_mul-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "deep_five_op_chain-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "prime_m_prime_n_fp16-cores7": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
}
CHUNKING_1D_PARAM_SETS = _expand_dict(CHUNKING_1D_PARAM_SETS)

# ---------------------------------------------------------------------------
# 2-D Chunking
# ---------------------------------------------------------------------------

CHUNKING_2D_PARAM_SETS = {
    # ── span > 256 MB, total < 8 GB ──────────────────────────────────────────
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_even_fp16":        (_exp_sigmoid, (36, 4000000),    None,                        torch.float16,  1),
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_odd_bf16":         (_relu_tanh,   (36, 4000001),    None,                        torch.bfloat16, 7),
    # TODO: DtException: EAR overflow detected
    "span_gt256_lt8gb_prime_fp32":       (_neg_abs,     (36, 4000037),    None,                        torch.float32, 32),

    # ── total > 8 GB ─────────────────────────────────────────────────────────
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_odd_fp32":              (_relu_tanh,   (63, 70312501),   None,                        torch.float32,  7),
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_prime_fp16":            (_neg_abs,     (37, 121621621),  None,                        torch.float16, 32),

    # ── dim extremes ────────────────────────────────────────
    # dim0=1, dim1 as large as possible within < 8 GB for float16
    # TODO: FlexAllocator OutOfMemory
    "dim0_1_dim1_max_fp16":              (_add,         (1, 4294967296),  (1, 4294967296),             torch.float16,  1),
    # dim0=max, dim1=1
    # TODO: FlexAllocator OutOfMemory
    "dim0_max_dim1_1_fp16":              (_add,         (5500123456, 1),  (5500123456, 1),             torch.float16,  7),

    # ── broadcast: 2-D + 1-D ───────────────
    # TODO: Missing Backend Codegen for Dtype/Op
    "broadcast_2d_1d_int32":             (torch.add,    (42000, 31231),   (31231,),                    torch.int32,    7),
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "broadcast_2d_1d_fp32":              (_add,         (21000, 321),     (321,),                      torch.float32,  1),

    # ── dtype coverage ───────────────────────────────────────────────────────
    # TODO: Missing Backend Codegen for Dtype/Op
    "bool_logical_or_even":              (torch.logical_or,  (37, 4000000),   (37, 4000000),           torch.bool,     1),
    # TODO: Missing Backend Codegen for Dtype/Op
    "int32_sub_prime":                   (torch.sub,    (37, 4000037),    (37, 4000037),               torch.int32,   32),
    # TODO: Missing Backend Codegen for Dtype/Op
    "int64_add_odd":                     (torch.add,    (36, 4000001),    (36, 4000001),               torch.int64,    7),
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "bf16_add_even":                     (_add,         (36, 4000000),    (36, 4000000),               torch.bfloat16, 1),

    # ── mixed precision  ──────────
    # TODO: Error in codegen for ComputedBuffer during pointwise operations (issue #2612)
    "mixed_fp16_bf16_add":               (torch.add,    (128, 32),        (128, 32),                   (torch.float16, torch.bfloat16), 1),
    # TODO: Error in codegen for ComputedBuffer during pointwise operations (issue #2612)
    "mixed_fp16_fp32_add":               (torch.add,    (128, 32),        (128, 32),                   (torch.float16, torch.float32),  1),
}

EXPECTED_FAILURES_2D = {
    "span_gt256_lt8gb_even_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "span_gt256_lt8gb_odd_bf16-cores32": ("xfail", "DtException: No valid prime factor"),
    "span_gt256_lt8gb_prime_fp32-cores7": ("xfail", "DtException: EAR overflow detected"),
    "total_gt8gb_odd_fp32-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_prime_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "span_gt256_and_total_gt8gb_prime_fp16-cores32": ("xfail", "DtException: isValidDimParam"),
    "exact_8gb_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "exact_span_256mb_fp32-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "floor_division_boundary_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "bool_unary_prime-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "int32_binary_even-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int64_add_prime-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "span_gt256_lt8gb_odd_bf16-cores7": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "span_gt256_lt8gb_prime_fp32-cores32": ("xfail", "DtException: EAR overflow detected"),
    "dim0_1_dim1_max_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "dim0_max_dim1_1_fp16-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_2d_1d_int32-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "broadcast_2d_1d_fp32-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "bool_logical_or_even-cores1": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int32_sub_prime-cores32": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int64_add_odd-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "bf16_add_even-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "mixed_fp16_bf16_add-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "mixed_fp16_fp32_add-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "exact_8gb_span_256mb_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "threshold_just_under_8gb_fp16-cores7": ("xfail", "DtException: Invalid xlat size"),
    "threshold_just_over_8gb_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_even_fp16_c1-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_odd_fp32_c7-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_prime_bf16_c32-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "dev_large_batch_large_m-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "dev_large_total_bytes_trigger-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_batch_dim_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_m_dim_fp32-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_3d_1d_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "basic_chunking_bf16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "both_m_n_not_multiple_64_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "prime_m_dim_fp16-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "bool_logical_and-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "int32_add_even-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "float32_large_mul-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "deep_five_op_chain-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "prime_m_prime_n_fp16-cores7": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
}
CHUNKING_2D_PARAM_SETS = _expand_dict(CHUNKING_2D_PARAM_SETS)

# ---------------------------------------------------------------------------
# 3-D Chunking
# ---------------------------------------------------------------------------

CHUNKING_3D_PARAM_SETS = {
    # ── No chunking (sanity) ──────────────────────────────────────────────────
    "small_no_chunk":                    (_mul,  (32, 1024, 1024),   (32, 1024, 1024),   torch.float16, 32),
    "medium_no_chunk":                   (_mul,  (32, 4096, 4096),   (32, 4096, 4096),   torch.float16, 32),
    "clearly_under_limit":               (_mul,  (32, 8192, 1024),   (32, 8192, 1024),   torch.float16, 32),

    # ── exact boundaries ──────────────────────────────────────────────────────
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "exact_8gb_span_256mb_fp16":         (_sub,  (32, 8192, 16384),  (32, 8192, 16384),  torch.float16,  1),
    # TODO: DtException: Invalid xlat size
    "threshold_just_under_8gb_fp16":     (_add,  (32, 8192, 16383),  (32, 8192, 16383),  torch.float16,  7),
    # TODO: FlexAllocator OutOfMemory
    "threshold_just_over_8gb_fp16":      (_mul,  (32, 8192, 16385),  (32, 8192, 16385),  torch.float16, 32),

    # ── span > 256 MB, total < 8 GB ──────────────────────────────────────────
    # even batch
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_even_fp16":        (_exp_sigmoid, (36, 2000, 2000),  None,          torch.float16,  1),
    # odd inner dims
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_odd_bf16":         (_relu_tanh,   (33, 2001, 2001),  None,          torch.bfloat16, 7),
    # prime inner dims
    # TODO: DtException: EAR overflow detected
    "span_gt256_lt8gb_prime_fp32":       (_neg_abs,     (31, 2003, 2003),  None,          torch.float32, 32),

    # ── total > 8 GB ─────────────────────────────────────────────────────────
    # even – cores 1
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_even_fp16_c1":          (_relu_tanh,   (32, 8192, 17408),  (32, 8192, 17408), torch.float16,  1),
    # odd – cores 7
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_odd_fp32_c7":           (_add,         (31, 8191, 17409),  (31, 8191, 17409), torch.float32,  7),
    # prime – cores 32
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_prime_bf16_c32":        (_mul,         (37, 8191, 17407),  (37, 8191, 17407), torch.bfloat16, 32),
    # Developer test: large batch + large M
    # TODO: FlexAllocator OutOfMemory
    "dev_large_batch_large_m":           (_mul,         (64, 8192, 17408),  (64, 8192, 17408), torch.float16, 32),
    # TODO: FlexAllocator OutOfMemory
    "dev_large_total_bytes_trigger":     (_mul,         (32, 8192, 17408),  (32, 8192, 17408), torch.float16, 32),

    # ── Chunking triggered (M > 8192 or N > 65536) ───────────────────────────
    "dev_prime_m_per_core_trigger":      (_mul,  (32, 8193, 1740),   (32, 8193, 1740),   torch.float16, 32),
    "dev_awkward_batch_7":               (_mul,  (7, 8193, 1740),    (7, 8193, 1740),    torch.float16, 32),
    "dev_case2_m1_giant_n":              (_mul,  (32, 1, 524288),    (32, 1, 524288),    torch.float16, 32),
    "dev_case4_batch1_m1_flat_giant":    (_mul,  (1, 1, 8388608),    (1, 1, 8388608),    torch.float16, 32),
    "dev_case5_large_batch_prime_m":     (_mul,  (256, 8193, 1740),  (256, 8193, 1740),  torch.float16, 32),
    "dev_case1_batch1_large_N_total":    (_mul,  (1, 8192, 17408),   (1, 8192, 17408),   torch.float16, 32),
    "dev_single_giant_dim":              (_mul,  (1, 65537, 4096),   (1, 65537, 4096),   torch.float16, 32),
    "dev_float32_prime_M":               (_mul,  (32, 8193, 1740),   (32, 8193, 1740),   torch.float32, 32),

    # ── broadcast ─────────────────────────────────────────────────────────────
    # TODO: FlexAllocator OutOfMemory
    "broadcast_batch_dim_fp16":          (_mul,  (32, 8192, 17408),  (1, 8192, 17408),   torch.float16,  1),
    # TODO: FlexAllocator OutOfMemory
    "broadcast_m_dim_fp32":              (_add,  (32, 8192, 17408),  (32, 1, 17408),     torch.float32,  1),
    # 3D + 1D different ndim (review request)
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "broadcast_3d_1d_fp16":              (_add,  (21000, 321, 64),   (64,),              torch.float16,  1),
    "dev_broadcast":                     (_mul,  (32, 8193, 1740),   (1, 8193, 1740),    torch.float16, 32),

    # ── basic binary / unary  ────────────────────────────────────────
    # TODO: FlexAllocator OutOfMemory
    "basic_chunking_bf16":               (_add,  (32, 8192, 17408),  (32, 8192, 17408),  torch.bfloat16, 1),
    # TODO: FlexAllocator OutOfMemory
    "both_m_n_not_multiple_64_fp16":     (_sub,  (32, 8193, 17409),  (32, 8193, 17409),  torch.float16, 32),
    # TODO: FlexAllocator OutOfMemory
    "prime_m_dim_fp16":                  (_add,  (32, 8191, 17408),  (32, 8191, 17408),  torch.float16,  7),

    # ── dtype coverage (one per dtype, distinct shape) ────────────────────────
    # TODO: FlexAllocator OutOfMemory
    "bool_logical_and":                  (torch.logical_and, (32, 8192, 17408), (32, 8192, 17408), torch.bool,    1),
    # TODO: FlexAllocator OutOfMemory
    "int32_add_even":                    (torch.add,  (32, 8192, 17408), (32, 8192, 17408), torch.int32,   7),
    # TODO: FlexAllocator OutOfMemory
    "int64_add_prime":                   (torch.add,  (32, 8192, 17408), (32, 8192, 17408), torch.int64,  32),

    # ── additional shapes / ops ───────────────────────────────────────────────
    # TODO: FlexAllocator OutOfMemory
    "float32_large_mul":                 (_mul,  (32, 8192, 8192),   (32, 8192, 8192),   torch.float32,  7),
    # TODO: Error in codegen for ComputedBuffer during pointwise operations (issue #2612)
    "deep_five_op_chain":                (_chained_deep, (32, 8192, 1740), (32, 8192, 1740), torch.float16, 1),
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "prime_m_prime_n_fp16":              (_mul,  (1, 8193, 8193),    (1, 8193, 8193),    torch.float16,  7),
    "empty_tensor_fp16":                 (_add,  (32, 0, 17408),     (32, 0, 17408),     torch.float16,  1),
    "single_core_unary":                 (_exp_sigmoid, (32, 8192, 8192), None,           torch.float16,  1),
    "thirtytwo_cores_unary":             (_exp_sigmoid, (31, 8191, 8191), None,           torch.float16, 32),
    "seven_cores_unary":                 (_exp_sigmoid, (31, 8191, 8191), None,           torch.float16,  7),
    "all_ones_except_m":                 (_add,  (1, 4310002000, 1), (1, 4310002000, 1), torch.float16,  1),
    "all_ones_except_batch":             (_add,  (5500123456, 1, 1), (5500123456, 1, 1), torch.float16,  7),
    "perfect_cube_fp16":                 (_mul,  (2048, 2048, 2048), (2048, 2048, 2048), torch.float16,  1),
    "prime_sandwich_fp16":               (_mul,  (32768, 7, 32768),  (32768, 7, 32768),  torch.float16, 32),
    "batch1_m1_only_n_splittable":       (_mul,  (1, 1, 4820001024),(1, 1, 4820001024),  torch.float16,  1),
    "25gb_three_or_more_chunks":         (_mul,  (32, 24576, 17408), (32, 24576, 17408), torch.float16,  1),
    "where_conditional":                 (_where,(32, 8192, 17408),  (32, 8192, 17408),  torch.float16,  1),
    "trailing_padding":                  (_add,  (8192, 524288, 1),  (8192, 524288, 1),  torch.float16,  1),
    "donut_padding":                     (_mul,  (256, 1, 16777216), (256, 1, 16777216), torch.float16,  1),
    "batch1_exp_sigmoid_fp16":           (_exp_sigmoid, (64, 8192, 8192), None,           torch.float16,  1),
}

EXPECTED_FAILURES_3D = {
    "span_gt256_lt8gb_even_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "span_gt256_lt8gb_odd_bf16-cores32": ("xfail", "DtException: No valid prime factor"),
    "span_gt256_lt8gb_prime_fp32-cores7": ("xfail", "DtException: EAR overflow detected"),
    "total_gt8gb_odd_fp32-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_prime_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "span_gt256_and_total_gt8gb_prime_fp16-cores32": ("xfail", "DtException: isValidDimParam"),
    "exact_8gb_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "exact_span_256mb_fp32-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "floor_division_boundary_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "bool_unary_prime-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "int32_binary_even-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int64_add_prime-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "span_gt256_lt8gb_odd_bf16-cores7": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "span_gt256_lt8gb_prime_fp32-cores32": ("xfail", "DtException: EAR overflow detected"),
    "dim0_1_dim1_max_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "dim0_max_dim1_1_fp16-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_2d_1d_int32-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "broadcast_2d_1d_fp32-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "bool_logical_or_even-cores1": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int32_sub_prime-cores32": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int64_add_odd-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "bf16_add_even-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "mixed_fp16_bf16_add-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "mixed_fp16_fp32_add-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "exact_8gb_span_256mb_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "threshold_just_under_8gb_fp16-cores7": ("xfail", "DtException: Invalid xlat size"),
    "threshold_just_over_8gb_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_even_fp16_c1-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_odd_fp32_c7-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_prime_bf16_c32-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "dev_large_batch_large_m-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "dev_large_total_bytes_trigger-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_batch_dim_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_m_dim_fp32-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_3d_1d_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "basic_chunking_bf16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "both_m_n_not_multiple_64_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "prime_m_dim_fp16-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "bool_logical_and-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "int32_add_even-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "float32_large_mul-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "deep_five_op_chain-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "prime_m_prime_n_fp16-cores7": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "dev_prime_m_per_core_trigger-cores32": ("skip", "Crashes with SIGABRT"),
    "dev_case5_large_batch_prime_m-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "dev_single_giant_dim-cores32": ("skip", "Crashes with SIGABRT"),
    "dev_float32_prime_M-cores32": ("skip", "Crashes with SIGABRT"),
    "dev_broadcast-cores32": ("skip", "Crashes with SIGABRT"),
    "single_core_unary-cores1": ("skip", "Crashes with SIGABRT"),
    "thirtytwo_cores_unary-cores32": ("skip", "Crashes with SIGABRT"),
    "seven_cores_unary-cores7": ("skip", "Crashes with SIGABRT"),
    "all_ones_except_m-cores1": ("skip", "Crashes with SIGABRT"),
    "all_ones_except_batch-cores7": ("skip", "Crashes with SIGABRT"),
    "perfect_cube_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "prime_sandwich_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "batch1_m1_only_n_splittable-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "25gb_three_or_more_chunks-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "where_conditional-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "trailing_padding-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "donut_padding-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "batch1_exp_sigmoid_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
}
CHUNKING_3D_PARAM_SETS = _expand_dict(CHUNKING_3D_PARAM_SETS)

# ---------------------------------------------------------------------------
# 4-D Chunking
# ---------------------------------------------------------------------------

CHUNKING_4D_PARAM_SETS = {
    # ── span > 256 MB, total < 8 GB ──────────────────────────────────────────
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_even_fp16":        (_exp_sigmoid, (36, 100, 200, 200),    None,                         torch.float16,  1),
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_odd_bf16":         (_relu_tanh,   (35, 101, 201, 201),    None,                         torch.bfloat16, 7),
    # TODO: DtException: EAR overflow detected
    "span_gt256_lt8gb_prime_fp32":       (_neg_abs,     (37, 101, 199, 199),    None,                         torch.float32, 32),

    # ── total > 8 GB ─────────────────────────────────────────────────────────
    # even – cores 1
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_even_fp16_c1":          (_relu_tanh,   (7, 5, 8192, 17408),    None,                         torch.float16,  1),
    # odd – cores 7
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_odd_fp32_c7":           (_add,         (7, 5, 8191, 17409),    (7, 5, 8191, 17409),          torch.float32,  7),
    # prime – cores 32
    "total_gt8gb_prime_fp16_c32":        (_neg_abs,     (37, 1, 8192, 14801),   None,                         torch.float16, 32),
    "dev_4D_large":                      (_mul,         (2, 32, 4096, 17408),   (2, 32, 4096, 17408),         torch.float16,  32),

    # ── dtype coverage ───────────────────────────────────────────────────────
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "bf16_add_even":                     (torch.add,    (2, 16, 1024, 1048576),  (2, 16, 1024, 1048576),       torch.bfloat16, 1),
    "bool_logical_or":                   (torch.logical_or, (7, 5, 8192, 17408),(7, 5, 8192, 17408),          torch.bool,     7),
    # TODO: Missing Backend Codegen for Dtype/Op
    "int32_sub_prime":                   (torch.sub,    (37, 1, 8192, 14801),    (37, 1, 8192, 14801),         torch.int32,   32),
    "int64_mul_even":                    (_mul,         (4, 8, 8192, 4096),      (4, 8, 8192, 4096),           torch.int64,    1),
    "float32_add":                       (torch.add,    (4, 8, 8192, 4096),      (4, 8, 8192, 4096),           torch.float32,  1),

    # ── shapes / ops ─────────────────────────────────────────────────────────
    "single_giant_prime_dim_fp16":       (_exp_sigmoid, (1, 1, 65537, 4096),     None,                         torch.float16, 32),
    "seven_cores_unary":                 (_exp_sigmoid, (3, 7, 8191, 8191),      None,                         torch.float16,  7),
    "waterfall_ascending_fp16":          (_sub,         (2, 16, 1024, 1048576),  (2, 16, 1024, 1048576),       torch.float16,  1),
    "waterfall_descending_fp16":         (_add,         (1048576, 1024, 16, 2),  (1048576, 1024, 16, 2),       torch.float16,  1),
    "trailing_padding_fp32":             (_mul,         (32, 8192, 17408, 1),    (32, 8192, 17408, 1),         torch.float32,  1),
    "interleaved_padding_fp16":          (_mul,         (65536, 1, 65536, 1),    (65536, 1, 65536, 1),         torch.float16,  1),
}

EXPECTED_FAILURES_4D = {
    "span_gt256_lt8gb_even_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "span_gt256_lt8gb_odd_bf16-cores32": ("xfail", "DtException: No valid prime factor"),
    "span_gt256_lt8gb_prime_fp32-cores7": ("xfail", "DtException: EAR overflow detected"),
    "total_gt8gb_odd_fp32-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_prime_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "span_gt256_and_total_gt8gb_prime_fp16-cores32": ("xfail", "DtException: isValidDimParam"),
    "exact_8gb_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "exact_span_256mb_fp32-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "floor_division_boundary_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "bool_unary_prime-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "int32_binary_even-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int64_add_prime-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "span_gt256_lt8gb_odd_bf16-cores7": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "span_gt256_lt8gb_prime_fp32-cores32": ("xfail", "DtException: EAR overflow detected"),
    "dim0_1_dim1_max_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "dim0_max_dim1_1_fp16-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_2d_1d_int32-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "broadcast_2d_1d_fp32-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "bool_logical_or_even-cores1": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int32_sub_prime-cores32": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int64_add_odd-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "bf16_add_even-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "mixed_fp16_bf16_add-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "mixed_fp16_fp32_add-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "exact_8gb_span_256mb_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "threshold_just_under_8gb_fp16-cores7": ("xfail", "DtException: Invalid xlat size"),
    "threshold_just_over_8gb_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_even_fp16_c1-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_odd_fp32_c7-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_prime_bf16_c32-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "dev_large_batch_large_m-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "dev_large_total_bytes_trigger-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_batch_dim_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_m_dim_fp32-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_3d_1d_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "basic_chunking_bf16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "both_m_n_not_multiple_64_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "prime_m_dim_fp16-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "bool_logical_and-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "int32_add_even-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "float32_large_mul-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "deep_five_op_chain-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "prime_m_prime_n_fp16-cores7": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "total_gt8gb_prime_fp16_c32-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "dev_4D_large-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "bool_logical_or-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int64_mul_even-cores1": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "float32_add-cores1": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "single_giant_prime_dim_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "seven_cores_unary-cores7": ("skip", "Crashes with SIGABRT"),
    "waterfall_ascending_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "waterfall_descending_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "trailing_padding_fp32-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "interleaved_padding_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
}
CHUNKING_4D_PARAM_SETS = _expand_dict(CHUNKING_4D_PARAM_SETS)

# ---------------------------------------------------------------------------
# 5-D Chunking
# ---------------------------------------------------------------------------

CHUNKING_5D_PARAM_SETS = {
    # ── span > 256 MB, total < 8 GB ──────────────────────────────────────────
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_even_fp16":        (_exp_sigmoid, (36, 10, 10, 200, 200),  None,                         torch.float16,  1),
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_odd_bf16":         (_relu_tanh,   (35, 11, 11, 201, 201),  None,                         torch.bfloat16, 7),
    # TODO: DtException: EAR overflow detected
    "span_gt256_lt8gb_prime_fp32":       (_neg_abs,     (37, 11, 11, 199, 199),  None,                         torch.float32, 32),

    # ── total > 8 GB ─────────────────────────────────────────────────────────
    # even – cores 1
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_even_fp16_c1":          (_relu_tanh,   (2, 3, 5, 8192, 17408),  None,                         torch.float16,  1),
    # odd – cores 7
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_odd_fp32_c7":           (_add,         (2, 3, 3, 8191, 14201),  (2, 3, 3, 8191, 14201),       torch.float32,  7),
    # prime – cores 32
    "total_gt8gb_prime_fp16_c32":        (_neg_abs,     (37, 1, 1, 8192, 14801), None,                         torch.float16, 32),

    # ── dtype coverage ───────────────────────────────────────────────────────
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "bf16_add_even":                     (torch.add,    (2, 4, 4, 8192, 4096),   (2, 4, 4, 8192, 4096),        torch.bfloat16, 1),
    "bool_logical_and_prime":            (torch.logical_and, (3, 3, 3, 8191, 4091), (3, 3, 3, 8191, 4091),    torch.bool,     7),
    "int64_mul_buried_core":             (_mul,         (1, 1, 6100200301, 1, 1), (1, 1, 6100200301, 1, 1),   torch.int64,   32),
    "float32_add_odd":                   (torch.add,    (3, 3, 3, 8191, 4091),   (3, 3, 3, 8191, 4091),       torch.float32,  7),

    # ── shapes / ops ─────────────────────────────────────────────────────────
    "single_giant_prime_dim_fp16":       (_exp_sigmoid, (1, 1, 1, 65537, 4096),  None,                         torch.float16, 32),
    "single_core_unary":                 (_exp_sigmoid, (2, 4, 4, 8192, 4096),   None,                         torch.float16,  1),
    "seven_cores_binary_fp32":           (torch.add,    (3, 3, 3, 8191, 4091),   (3, 3, 3, 8191, 4091),       torch.float32,  7),
    "trailing_padding_fp16":             (_add,         (32, 8192, 17408, 1, 1), (32, 8192, 17408, 1, 1),     torch.float16,  1),
    "interleaved_padding_fp16":          (_add,         (1, 65536, 1, 65536, 1), (1, 65536, 1, 65536, 1),     torch.float16,  1),
    "donut_padding_fp16":                (_add,         (32, 64, 1, 128, 17408), (32, 64, 1, 128, 17408),     torch.float16,  1),
}

EXPECTED_FAILURES_5D = {
    "span_gt256_lt8gb_even_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "span_gt256_lt8gb_odd_bf16-cores32": ("xfail", "DtException: No valid prime factor"),
    "span_gt256_lt8gb_prime_fp32-cores7": ("xfail", "DtException: EAR overflow detected"),
    "total_gt8gb_odd_fp32-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_prime_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "span_gt256_and_total_gt8gb_prime_fp16-cores32": ("xfail", "DtException: isValidDimParam"),
    "exact_8gb_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "exact_span_256mb_fp32-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "floor_division_boundary_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "bool_unary_prime-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "int32_binary_even-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int64_add_prime-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "span_gt256_lt8gb_odd_bf16-cores7": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "dim0_1_dim1_max_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "dim0_max_dim1_1_fp16-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_2d_1d_int32-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "broadcast_2d_1d_fp32-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "bool_logical_or_even-cores1": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int32_sub_prime-cores32": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int64_add_odd-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "bf16_add_even-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "mixed_fp16_bf16_add-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "mixed_fp16_fp32_add-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "exact_8gb_span_256mb_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "threshold_just_under_8gb_fp16-cores7": ("xfail", "DtException: Invalid xlat size"),
    "threshold_just_over_8gb_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_even_fp16_c1-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_odd_fp32_c7-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_prime_bf16_c32-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "dev_large_batch_large_m-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "dev_large_total_bytes_trigger-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_batch_dim_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_m_dim_fp32-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_3d_1d_fp16-cores1": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "basic_chunking_bf16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "both_m_n_not_multiple_64_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "prime_m_dim_fp16-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "bool_logical_and-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "int32_add_even-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "float32_large_mul-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "deep_five_op_chain-cores1": ("xfail", "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)"),
    "prime_m_prime_n_fp16-cores7": ("xfail", "C++ compiler crashes with std::out_of_range map::at (issue #2614)"),
    "total_gt8gb_prime_fp16_c32-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "bool_logical_and_prime-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int64_mul_buried_core-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "float32_add_odd-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "single_giant_prime_dim_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "single_core_unary-cores1": ("skip", "Crashes with SIGABRT"),
    "seven_cores_binary_fp32-cores7": ("skip", "Crashes with SIGABRT"),
    "trailing_padding_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "interleaved_padding_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "donut_padding_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
}
CHUNKING_5D_PARAM_SETS = _expand_dict(CHUNKING_5D_PARAM_SETS)

# ---------------------------------------------------------------------------
# Custom / edge-case parameter sets
# ---------------------------------------------------------------------------

CUSTOM_OP_PARAM_SETS = {
    "float8_e4m3fn_relu_small": (
        lambda: (torch.randn(1, 128, 128, dtype=torch.float16, device='spyre').to(torch.float8_e4m3fn),),
        lambda t: torch.relu(t.to(torch.float16)).to(torch.float8_e4m3fn),
        lambda t: torch.relu(t.float()),
    ),
    "float8_e4m3fn_relu_large": (
        lambda: (torch.randn(32, 8192, 34816, dtype=torch.float16, device='spyre').to(torch.float8_e4m3fn),),
        lambda t: torch.relu(t.to(torch.float16)).to(torch.float8_e4m3fn),
        lambda t: torch.relu(t.float()),
    ),
    "float8_e4m3fn_relu_span_large": (
        lambda: (torch.randn(1, 1, 300000000, dtype=torch.float16, device='spyre').to(torch.float8_e4m3fn),),
        lambda t: torch.relu(t.to(torch.float16)).to(torch.float8_e4m3fn),
        lambda t: torch.relu(t.float()),
    ),
    "float8_e4m3fn_relu_span_total_large": (
        lambda: (torch.randn(37, 1, 300000000, dtype=torch.float16, device='spyre').to(torch.float8_e4m3fn),),
        lambda t: torch.relu(t.to(torch.float16)).to(torch.float8_e4m3fn),
        lambda t: torch.relu(t.float()),
    ),
    "float8_e5m2_relu_small": (
        lambda: (torch.randn(1, 128, 128, dtype=torch.float16, device='spyre').to(torch.float8_e5m2),),
        lambda t: torch.relu(t.to(torch.float16)).to(torch.float8_e5m2),
        lambda t: torch.relu(t.float()),
    ),
    "float8_e5m2_relu_large": (
        lambda: (torch.randn(32, 8192, 34816, dtype=torch.float16, device='spyre').to(torch.float8_e5m2),),
        lambda t: torch.relu(t.to(torch.float16)).to(torch.float8_e5m2),
        lambda t: torch.relu(t.float()),
    ),
    "float8_e5m2_relu_span_large": (
        lambda: (torch.randn(1, 1, 300000000, dtype=torch.float16, device='spyre').to(torch.float8_e5m2),),
        lambda t: torch.relu(t.to(torch.float16)).to(torch.float8_e5m2),
        lambda t: torch.relu(t.float()),
    ),
    "float8_e5m2_relu_span_total_large": (
        lambda: (torch.randn(37, 1, 300000000, dtype=torch.float16, device='spyre').to(torch.float8_e5m2),),
        lambda t: torch.relu(t.to(torch.float16)).to(torch.float8_e5m2),
        lambda t: torch.relu(t.float()),
    ),
    "float8_e4m3fn_add_small": (
        lambda: (
            torch.randn(32, 128, 128, dtype=torch.float16, device='spyre').to(torch.float8_e4m3fn),
            torch.randn(32, 128, 128, dtype=torch.float16, device='spyre').to(torch.float8_e4m3fn)
        ),
        lambda a, b: (a.to(torch.float16) + b.to(torch.float16)).to(torch.float8_e4m3fn),
        lambda a, b: a.float() + b.float(),
    ),
    "float8_e4m3fn_add_large": (
        lambda: (
            torch.randn(32, 8192, 34816, dtype=torch.float16, device='spyre').to(torch.float8_e4m3fn),
            torch.randn(32, 8192, 34816, dtype=torch.float16, device='spyre').to(torch.float8_e4m3fn)
        ),
        lambda a, b: (a.to(torch.float16) + b.to(torch.float16)).to(torch.float8_e4m3fn),
        lambda a, b: a.float() + b.float(),
    ),
    "float8_e4m3fn_add_span_large": (
        lambda: (
            torch.randn(1, 1, 300000000, dtype=torch.float16, device='spyre').to(torch.float8_e4m3fn),
            torch.randn(1, 1, 300000000, dtype=torch.float16, device='spyre').to(torch.float8_e4m3fn)
        ),
        lambda a, b: (a.to(torch.float16) + b.to(torch.float16)).to(torch.float8_e4m3fn),
        lambda a, b: a.float() + b.float(),
    ),
    "float8_e4m3fn_add_span_total_large": (
        lambda: (
            torch.randn(37, 1, 300000000, dtype=torch.float16, device='spyre').to(torch.float8_e4m3fn),
            torch.randn(37, 1, 300000000, dtype=torch.float16, device='spyre').to(torch.float8_e4m3fn)
        ),
        lambda a, b: (a.to(torch.float16) + b.to(torch.float16)).to(torch.float8_e4m3fn),
        lambda a, b: a.float() + b.float(),
    ),
    "int8_neg_abs_small": (
        lambda: (torch.randint(-127, 127, (1, 128, 128), dtype=torch.int8, device='spyre'),),
        lambda t: torch.neg(torch.abs(t)),
        lambda t: torch.neg(torch.abs(t.float())),
    ),
    "int8_neg_abs_large": (
        lambda: (torch.randint(-127, 127, (32, 8192, 34816), dtype=torch.int8, device='spyre'),),
        lambda t: torch.neg(torch.abs(t)),
        lambda t: torch.neg(torch.abs(t.float())),
    ),
    "int8_neg_abs_span_large": (
        lambda: (torch.randint(-127, 127, (1, 1, 300000000), dtype=torch.int8, device='spyre'),),
        lambda t: torch.neg(torch.abs(t)),
        lambda t: torch.neg(torch.abs(t.float())),
    ),
    "int8_neg_abs_span_total_large": (
        lambda: (torch.randint(-127, 127, (37, 1, 300000000), dtype=torch.int8, device='spyre'),),
        lambda t: torch.neg(torch.abs(t)),
        lambda t: torch.neg(torch.abs(t.float())),
    ),
    "int8_add_small": (
        lambda: (
            torch.randint(-127, 127, (32, 128, 128), dtype=torch.int8, device='spyre'),
            torch.randint(-127, 127, (32, 128, 128), dtype=torch.int8, device='spyre')
        ),
        lambda a, b: a + b,
        lambda a, b: a.float() + b.float(),
    ),
    "int8_add_large": (
        lambda: (
            torch.randint(-127, 127, (32, 8192, 34816), dtype=torch.int8, device='spyre'),
            torch.randint(-127, 127, (32, 8192, 34816), dtype=torch.int8, device='spyre')
        ),
        lambda a, b: a + b,
        lambda a, b: a.float() + b.float(),
    ),
    "int8_add_span_large": (
        lambda: (
            torch.randint(-127, 127, (1, 1, 300000000), dtype=torch.int8, device='spyre'),
            torch.randint(-127, 127, (1, 1, 300000000), dtype=torch.int8, device='spyre')
        ),
        lambda a, b: a + b,
        lambda a, b: a.float() + b.float(),
    ),
    "int8_add_span_total_large": (
        lambda: (
            torch.randint(-127, 127, (37, 1, 300000000), dtype=torch.int8, device='spyre'),
            torch.randint(-127, 127, (37, 1, 300000000), dtype=torch.int8, device='spyre')
        ),
        lambda a, b: a + b,
        lambda a, b: a.float() + b.float(),
    ),
    "non_contiguous_slice_small": (
        lambda: (
            torch.randn(32, 128, 256, dtype=torch.float16, device='spyre')[:, :, ::2],
            torch.randn(32, 128, 256, dtype=torch.float16, device='spyre')[:, :, ::2]
        ),
        _add,
        None,
    ),
    "non_contiguous_slice_large": (
        lambda: (
            torch.randn(32, 8192, 34816, dtype=torch.float16, device='spyre')[:, :, ::2],
            torch.randn(32, 8192, 34816, dtype=torch.float16, device='spyre')[:, :, ::2]
        ),
        _add,
        None,
    ),
    "non_contiguous_slice_span_large": (
        lambda: (
            torch.randn(1, 1, 300000000, dtype=torch.float16, device='spyre')[:, :, ::2],
            torch.randn(1, 1, 300000000, dtype=torch.float16, device='spyre')[:, :, ::2]
        ),
        _add,
        None,
    ),
    "non_contiguous_slice_span_total_large": (
        lambda: (
            torch.randn(37, 1, 300000000, dtype=torch.float16, device='spyre')[:, :, ::2],
            torch.randn(37, 1, 300000000, dtype=torch.float16, device='spyre')[:, :, ::2]
        ),
        _add,
        None,
    ),
    "expand_zero_stride_small": (
        lambda: (
            torch.randn(1, 1, 1, dtype=torch.float16, device='spyre').expand(32, 128, 128),
            torch.randn(32, 128, 128, dtype=torch.float16, device='spyre')
        ),
        _add,
        None,
    ),
    "expand_zero_stride_large": (
        lambda: (
            torch.randn(1, 1, 1, dtype=torch.float16, device='spyre').expand(32, 8192, 17408),
            torch.randn(32, 8192, 17408, dtype=torch.float16, device='spyre')
        ),
        _add,
        None,
    ),
    "expand_zero_stride_span_large": (
        lambda: (
            torch.randn(1, 1, 1, dtype=torch.float16, device='spyre').expand(1, 1, 150000000),
            torch.randn(1, 1, 150000000, dtype=torch.float16, device='spyre')
        ),
        _add,
        None,
    ),
    "expand_zero_stride_span_total_large": (
        lambda: (
            torch.randn(1, 1, 1, dtype=torch.float16, device='spyre').expand(37, 1, 150000000),
            torch.randn(37, 1, 150000000, dtype=torch.float16, device='spyre')
        ),
        _add,
        None,
    ),
    "in_place_mutation_conflict_small": (
        lambda: (
            torch.randn(32, 128, 128, dtype=torch.float16, device='spyre'),
            torch.randn(1, 1, 1, dtype=torch.float16, device='spyre')
        ),
        lambda a, b: a.add_(b),
        lambda a, b: a.clone().add_(b),
    ),
    "in_place_mutation_conflict_large": (
        lambda: (
            torch.randn(32, 8192, 17408, dtype=torch.float16, device='spyre'),
            torch.randn(1, 1, 1, dtype=torch.float16, device='spyre')
        ),
        lambda a, b: a.add_(b),
        lambda a, b: a.clone().add_(b),
    ),
    "in_place_mutation_conflict_span_large": (
        lambda: (
            torch.randn(1, 1, 150000000, dtype=torch.float16, device='spyre'),
            torch.randn(1, 1, 1, dtype=torch.float16, device='spyre')
        ),
        lambda a, b: a.add_(b),
        lambda a, b: a.clone().add_(b),
    ),
    "in_place_mutation_conflict_span_total_large": (
        lambda: (
            torch.randn(37, 1, 150000000, dtype=torch.float16, device='spyre'),
            torch.randn(1, 1, 1, dtype=torch.float16, device='spyre')
        ),
        lambda a, b: a.add_(b),
        lambda a, b: a.clone().add_(b),
    ),
    "mixed_precision_fp16_fp32_add": (
        lambda: (
            torch.randn(32, 128, 128, dtype=torch.float16, device='spyre'),
            torch.randn(32, 128, 128, dtype=torch.float32, device='spyre')
        ),
        lambda a, b: a + b,
        lambda a, b: a.float() + b.float(),
    ),
    "bool_logical_and_small": (
        lambda: (
            torch.randint(0, 2, (32, 128, 128), dtype=torch.bool, device='spyre'),
            torch.randint(0, 2, (32, 128, 128), dtype=torch.bool, device='spyre')
        ),
        lambda a, b: torch.logical_and(a, b),
        None,
    ),
    "bool_logical_and_large": (
        lambda: (
            torch.randint(0, 2, (32, 8192, 34816), dtype=torch.bool, device='spyre'),
            torch.randint(0, 2, (32, 8192, 34816), dtype=torch.bool, device='spyre')
        ),
        lambda a, b: torch.logical_and(a, b),
        None,
    ),
    "bool_logical_and_span_large": (
        lambda: (
            torch.randint(0, 2, (1, 1, 300000000), dtype=torch.bool, device='spyre'),
            torch.randint(0, 2, (1, 1, 300000000), dtype=torch.bool, device='spyre')
        ),
        lambda a, b: torch.logical_and(a, b),
        None,
    ),
    "bool_logical_and_span_total_large": (
        lambda: (
            torch.randint(0, 2, (37, 1, 300000000), dtype=torch.bool, device='spyre'),
            torch.randint(0, 2, (37, 1, 300000000), dtype=torch.bool, device='spyre')
        ),
        lambda a, b: torch.logical_and(a, b),
        None,
    ),
    "bf16_add_small": (
        lambda: (
            torch.randn(32, 128, 128, dtype=torch.bfloat16, device='spyre'),
            torch.randn(32, 128, 128, dtype=torch.bfloat16, device='spyre')
        ),
        lambda a, b: a + b,
        None,
    ),
    "bf16_add_large": (
        lambda: (
            torch.randn(32, 8192, 34816, dtype=torch.bfloat16, device='spyre'),
            torch.randn(32, 8192, 34816, dtype=torch.bfloat16, device='spyre')
        ),
        lambda a, b: a + b,
        None,
    ),
    "bf16_add_span_large": (
        lambda: (
            torch.randn(1, 1, 300000000, dtype=torch.bfloat16, device='spyre'),
            torch.randn(1, 1, 300000000, dtype=torch.bfloat16, device='spyre')
        ),
        lambda a, b: a + b,
        None,
    ),
    "bf16_add_span_total_large": (
        lambda: (
            torch.randn(37, 1, 300000000, dtype=torch.bfloat16, device='spyre'),
            torch.randn(37, 1, 300000000, dtype=torch.bfloat16, device='spyre')
        ),
        lambda a, b: a + b,
        None,
    ),
    "int32_add_small": (
        lambda: (
            torch.randint(-100, 100, (32, 128, 128), dtype=torch.int32, device='spyre'),
            torch.randint(-100, 100, (32, 128, 128), dtype=torch.int32, device='spyre')
        ),
        lambda a, b: a + b,
        None,
    ),
    "int32_add_large": (
        lambda: (
            torch.randint(-100, 100, (32, 8192, 34816), dtype=torch.int32, device='spyre'),
            torch.randint(-100, 100, (32, 8192, 34816), dtype=torch.int32, device='spyre')
        ),
        lambda a, b: a + b,
        None,
    ),
    "int32_add_span_large": (
        lambda: (
            torch.randint(-100, 100, (1, 1, 300000000), dtype=torch.int32, device='spyre'),
            torch.randint(-100, 100, (1, 1, 300000000), dtype=torch.int32, device='spyre')
        ),
        lambda a, b: a + b,
        None,
    ),
    "int32_add_span_total_large": (
        lambda: (
            torch.randint(-100, 100, (37, 1, 300000000), dtype=torch.int32, device='spyre'),
            torch.randint(-100, 100, (37, 1, 300000000), dtype=torch.int32, device='spyre')
        ),
        lambda a, b: a + b,
        None,
    ),
}

EXPECTED_FAILURES_CUSTOM = {
    "float8_e4m3fn_relu_small": ("xfail", "Spyre backend does not support: type conversion from torch.float16 to torch.float8_e4m3fn"),
    "float8_e4m3fn_relu_large": ("xfail", "FlexAllocator OutOfMemory"),
    "float8_e4m3fn_relu_span_large": ("xfail", "Spyre backend does not support: type conversion from torch.float16 to torch.float8_e4m3fn"),
    "float8_e4m3fn_relu_span_total_large": ("xfail", "FlexAllocator OutOfMemory"),
    "float8_e5m2_relu_small": ("xfail", "Spyre backend does not support dtype Float8_e5m2"),
    "float8_e5m2_relu_large": ("xfail", "FlexAllocator OutOfMemory"),
    "float8_e5m2_relu_span_large": ("xfail", "Spyre backend does not support dtype Float8_e5m2"),
    "float8_e5m2_relu_span_total_large": ("xfail", "FlexAllocator OutOfMemory"),
    "float8_e4m3fn_add_small": ("xfail", "Spyre backend does not support: type conversion from torch.float16 to torch.float8_e4m3fn"),
    "float8_e4m3fn_add_large": ("xfail", "FlexAllocator OutOfMemory"),
    "float8_e4m3fn_add_span_large": ("xfail", "Spyre backend does not support: type conversion from torch.float16 to torch.float8_e4m3fn"),
    "float8_e4m3fn_add_span_total_large": ("xfail", "FlexAllocator OutOfMemory"),
    "int8_neg_abs_small": ("xfail", "Spyre backend does not support dtype Int8"),
    "int8_neg_abs_large": ("xfail", "FlexAllocator OutOfMemory"),
    "int8_neg_abs_span_large": ("xfail", "Spyre backend does not support dtype Int8"),
    "int8_neg_abs_span_total_large": ("xfail", "FlexAllocator OutOfMemory"),
    "int8_add_small": ("xfail", "Spyre backend does not support dtype Int8"),
    "int8_add_large": ("xfail", "FlexAllocator OutOfMemory"),
    "int8_add_span_large": ("xfail", "Spyre backend does not support dtype Int8"),
    "int8_add_span_total_large": ("xfail", "FlexAllocator OutOfMemory"),
    "non_contiguous_slice_large": ("xfail", "FlexAllocator OutOfMemory"),
    "non_contiguous_slice_span_total_large": ("xfail", "FlexAllocator OutOfMemory"),
    "expand_zero_stride_large": ("xfail", "FlexAllocator OutOfMemory"),
    "expand_zero_stride_span_total_large": ("xfail", "FlexAllocator OutOfMemory"),
    "in_place_mutation_conflict_small": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "in_place_mutation_conflict_large": ("xfail", "FlexAllocator OutOfMemory"),
    "in_place_mutation_conflict_span_large": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "in_place_mutation_conflict_span_total_large": ("xfail", "FlexAllocator OutOfMemory"),
    "mixed_precision_fp16_fp32_add": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "bool_logical_and_small": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "bool_logical_and_large": ("xfail", "FlexAllocator OutOfMemory"),
    "bool_logical_and_span_large": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "bool_logical_and_span_total_large": ("xfail", "FlexAllocator OutOfMemory"),
    "bf16_add_large": ("xfail", "FlexAllocator OutOfMemory"),
    "bf16_add_span_total_large": ("xfail", "FlexAllocator OutOfMemory"),
    "int32_add_small": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int32_add_large": ("xfail", "FlexAllocator OutOfMemory"),
    "int32_add_span_large": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int32_add_span_total_large": ("xfail", "FlexAllocator OutOfMemory"),
}

# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

torch.manual_seed(0xAFFE)


class TestOps:

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    def compare_with_cpu(self, *args, **kwargs):
        kwargs["run_eager"] = False
        kwargs["cpu_compile"] = True
        return utils_inductor.compare_with_cpu(*args, **kwargs)

    def _check(self, fn, spyre_inputs, tag="", ref_fn=None):
        cpu_inputs = tuple(t.cpu() for t in spyre_inputs)
        if ref_fn is not None:
            def custom_fn(*args):
                if any(t.device.type == "spyre" for t in args):
                    return fn(*args)
                else:
                    return ref_fn(*tuple(t.float() for t in args))
            self.compare_with_cpu(custom_fn, *cpu_inputs, atol=_ATOL)
        else:
            self.compare_with_cpu(fn, *cpu_inputs, atol=_ATOL)

    def run_chunking_test(self, fn, shape_x, shape_y, dtype, expected_cores):
        if isinstance(dtype, tuple):
            dtype_x, dtype_y = dtype
        else:
            dtype_x = dtype_y = dtype

       
        if isinstance(dtype_x, str): dtype_x = eval(dtype_x)
        if isinstance(dtype_y, str): dtype_y = eval(dtype_y)

        def _make_tensor(shape, dt):
            if dt == torch.bool:
                return torch.randint(0, 2, shape, dtype=dt, device="spyre")
            elif dt in (torch.int32, torch.int64, torch.int8):
                return torch.randint(-100, 100, shape, dtype=dt, device="spyre")
            else:
                return torch.randn(*shape, dtype=dt, device="spyre")

        x = _make_tensor(shape_x, dtype_x)
        y = _make_tensor(shape_y, dtype_y) if shape_y is not None else None

        inputs = (x, y) if y is not None else (x,)

        if expected_cores is not None:
            os.environ["SENCORES"] = str(expected_cores)

        try:
            self._check(fn, inputs, ref_fn=None)
        finally:
            if expected_cores is not None:
                del os.environ["SENCORES"]

    @pytest.mark.parametrize(
        "fn, shape_x, shape_y, dtype, cores",
        _make_params(CHUNKING_1D_PARAM_SETS, expect_fail=EXPECTED_FAILURES_1D)
    )
    def test_chunking_1d(self, fn, shape_x, shape_y, dtype, cores):
        self.run_chunking_test(fn, shape_x, shape_y, dtype, cores)

    @pytest.mark.parametrize(
        "fn, shape_x, shape_y, dtype, cores",
        _make_params(CHUNKING_2D_PARAM_SETS, expect_fail=EXPECTED_FAILURES_2D)
    )
    def test_chunking_2d(self, fn, shape_x, shape_y, dtype, cores):
        self.run_chunking_test(fn, shape_x, shape_y, dtype, cores)

    @pytest.mark.parametrize(
        "fn, shape_x, shape_y, dtype, cores",
        _make_params(CHUNKING_3D_PARAM_SETS, expect_fail=EXPECTED_FAILURES_3D)
    )
    def test_chunking_3d(self, fn, shape_x, shape_y, dtype, cores):
        self.run_chunking_test(fn, shape_x, shape_y, dtype, cores)

    @pytest.mark.parametrize(
        "fn, shape_x, shape_y, dtype, cores",
        _make_params(CHUNKING_4D_PARAM_SETS, expect_fail=EXPECTED_FAILURES_4D)
    )
    def test_chunking_4d(self, fn, shape_x, shape_y, dtype, cores):
        self.run_chunking_test(fn, shape_x, shape_y, dtype, cores)

    @pytest.mark.parametrize(
        "fn, shape_x, shape_y, dtype, cores",
        _make_params(CHUNKING_5D_PARAM_SETS, expect_fail=EXPECTED_FAILURES_5D)
    )
    def test_chunking_5d(self, fn, shape_x, shape_y, dtype, cores):
        self.run_chunking_test(fn, shape_x, shape_y, dtype, cores)

    @pytest.mark.parametrize(
        "factory_fn, fn, ref_fn",
        _make_custom_params(CUSTOM_OP_PARAM_SETS, expect_fail=EXPECTED_FAILURES_CUSTOM)
    )
    def test_custom_edge_cases(self, factory_fn, fn, ref_fn):
        inputs = factory_fn()
        self._check(fn, inputs, ref_fn=ref_fn)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    run_tests()
