# Copyright 2025 The Torch-Spyre Authors.
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

import math

# --- GLOBAL DICTIONARIES FOR PARAMETERIZATION ---
def _get_intelligent_cores(shape_x, shape_y, dtype):
    """Dynamically calculates the minimal, high-value core counts to test."""
    elems_x = math.prod(shape_x) if shape_x else 0
    elems_y = math.prod(shape_y) if shape_y else 0
    max_elems = max(elems_x, elems_y)
    
    bytes_per_elem = 2
    if dtype == torch.float32: bytes_per_elem = 4
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8): bytes_per_elem = 1
    
    total_bytes = max_elems * bytes_per_elem
    shape_to_use = shape_x if elems_x >= elems_y else shape_y
    span_bytes = shape_to_use[-1] * bytes_per_elem if shape_to_use else 0
    
    # 1. Non-chunking cases (too small for both total chunking and span chunking)
    if total_bytes < 8 * (1024**3) and span_bytes <= 256 * (1024**2):
        return [1, 32]
        
    # 2. Chunking triggered cases
    cores_to_test = {1, 32}
    max_dim = max(shape_to_use) if shape_to_use else 1
    
    # Divisible core count (prefer classic hardware sizes like 8, 16, 4, 2)
    for c in [8, 16, 4, 2]:
        if max_dim % c == 0:
            cores_to_test.add(c)
            break
            
    # Indivisible/prime core count (forces remainder logic testing)
    for p in [7, 3, 5, 17, 31]:
        if max_dim % p != 0:
            cores_to_test.add(p)
            break
            
    return sorted(list(cores_to_test))

def _expand_dict(raw_dict):
    expanded = {}
    for k, v in raw_dict.items():
        if v[-1] is not None:
            expanded[f"{k}-cores{v[-1]}"] = v
        else:
            cores_list = _get_intelligent_cores(v[1], v[2], v[3])
            for c in cores_list:
                expanded[f"{k}-cores{c}"] = (*v[:-1], c)
    return expanded

def _expand_xfails(raw_list, raw_dict):
    expanded = []
    for k in raw_list:
        v = raw_dict[k]
        if v[-1] is not None:
            expanded.append(f"{k}-cores{v[-1]}")
        else:
            cores_list = _get_intelligent_cores(v[1], v[2], v[3])
            for c in cores_list:
                expanded.append(f"{k}-cores{c}")
    return expanded

CHUNKING_1D_PARAM_SETS = {
    "cat1_non_chunking": (_exp_sigmoid, (1024,), None, torch.float16, None),
    "cat2_span_gt_256_total_lt_8gb": (_exp_sigmoid, (134217757,), None, torch.float16, None),
    "cat3_total_gt_8gb": (_relu_tanh, (4500000000,), None, torch.float16, None),
    "cat4_span_gt_256_total_gt_8gb": (_neg_abs, (4500000011,), None, torch.float16, None),
    "22_1_prime_span_collapse": (_exp_sigmoid, (134217757,), None, torch.float16, 32),
    "22_2_floor_division_boundary": (_exp_sigmoid, (402653185,), None, torch.float16, 3),
}

CHUNKING_2D_PARAM_SETS = {
    "cat1_non_chunking": (_exp_sigmoid, (128, 128), None, torch.float16, None),
    "cat2_span_gt_256_total_lt_8gb": (_exp_sigmoid, (37, 4000000), None, torch.float16, None),
    "cat3_total_gt_8gb": (_relu_tanh, (64, 70312500), None, torch.float16, None),
    "cat4_span_gt_256_total_gt_8gb": (_neg_abs, (37, 121621621), None, torch.float16, None),
    "13_5_mixed_precision_add": (torch.add, (128, 32), (128, 32), torch.float32, None),
    "13_7_mixed_precision_fp16_bf16_add": (torch.add, (128, 32), (128, 32), torch.bfloat16, None),
}

CHUNKING_3D_PARAM_SETS = {
    "cat1_non_chunking": (_exp_sigmoid, (1, 128, 128), None, torch.float16, None),
    "cat2_span_gt_256_total_lt_8gb": (_exp_sigmoid, (37, 2000, 2000), None, torch.float16, None),
    "cat3_total_gt_8gb": (_relu_tanh, (32, 8192, 17408), None, torch.float16, None),
    "cat4_span_gt_256_total_gt_8gb": (_neg_abs, (37, 8192, 14800), None, torch.float16, None),
    "1_1_small_tensor_exp_sigmoid": (_exp_sigmoid, (1, 128, 128), None, torch.float16, None),
    "1_4_exactly_8gb_exp_sigmoid": (_exp_sigmoid, (64, 8192, 8192), None, torch.float16, None),
    "2_1_basic_chunking_exp_sigmoid": (_exp_sigmoid, (32, 8192, 17408), None, torch.float16, None),
    "3_1_broadcast_batch_dim": (_mul, (32, 8192, 17408), (1, 8192, 17408), torch.float16, None),
    "3_2_broadcast_m_dim": (_add, (32, 8192, 17408), (32, 1, 17408), torch.float16, None),
    "4_3_both_m_and_n_not_multiple_of_64": (_sub, (32, 8193, 17409), (32, 8193, 17409), torch.float16, None),
    "6_1_prime_batch_dim": (_mul, (31, 8192, 17408), (31, 8192, 17408), torch.float16, None),
    "6_2_prime_m_dim": (_add, (32, 8191, 17408), (32, 8191, 17408), torch.float16, None),
    "7_1_batch1_m1_only_n_splittable": (_mul, (1, 1, 4563402752), (1, 1, 4563402752), torch.float16, None),
    "8_1_25gb_three_or_more_chunks": (_mul, (32, 24576, 17408), (32, 24576, 17408), torch.float16, None),
    "9_3_where_conditional": (_where, (32, 8192, 17408), (32, 8192, 17408), torch.float16, None),
    "dtype_bfloat16_large_chunking_triggered": (_add, (32, 8192, 17408), (32, 8192, 17408), torch.bfloat16, None),
    "13_3_float32_large_chunking_triggered": (_mul, (32, 8192, 8192), (32, 8192, 8192), torch.float32, None),
    "14_5_deep_five_op_chain": (_chained_deep, (32, 8193, 1740), (32, 8193, 1740), torch.float16, None),
    "16_4_batch1_m1_extreme_n": (_mul, (1, 1, 4563402752), (1, 1, 4563402752), torch.float16, None),
    "17_2_prime_m_prime_n": (_mul, (1, 8193, 8193), (1, 8193, 8193), torch.float16, None),
    "18_3_empty_tensor_trap": (_add, (32, 0, 17408), (32, 0, 17408), torch.float16, None),
    "19_1_single_core_chunking_3d": (_exp_sigmoid, (32, 8192, 8192), None, torch.float16, 1),
    "19_2_two_cores_chunking_3d": (_add, (32, 8192, 8192), (32, 8192, 8192), torch.float16, 2),
    "19_3_four_cores_chunking_3d": (torch.mul, (16, 8192, 8192), (16, 8192, 8192), torch.float32, 4),
    "19_4_eight_cores_chunking_3d": (_exp_sigmoid, (32, 8192, 8192), None, torch.float16, 8),
    "19_5_sixteen_cores_chunking_3d": (torch.add, (16, 8192, 8192), (16, 8192, 8192), torch.float32, 16),
    "19_6_thirtytwo_cores_chunking_3d": (_exp_sigmoid, (32, 8192, 8192), None, torch.float16, 32),
    "19_7_three_cores_3d": (_exp_sigmoid, (32, 8192, 8192), None, torch.float16, 3),
    "19_7_seventeen_cores_3d": (torch.add, (16, 8192, 8192), (16, 8192, 8192), torch.float32, 17),
    "19_8_no_chunking_needed_cores1": (_exp_sigmoid, (32, 1024, 1024), None, torch.float16, 1),
    "22_3_empty_tensor_32_cores": (_exp_sigmoid, (32, 0, 17408), None, torch.float16, 32),
    "20_2_all_ones_except_m": (_add, (1, 4500000000, 1), (1, 4500000000, 1), torch.float16, None),
    "20_3_all_ones_except_batch": (_add, (4500000000, 1, 1), (4500000000, 1, 1), torch.float16, None),
    "20_4_perfect_cube": (_mul, (2048, 2048, 2048), (2048, 2048, 2048), torch.float16, None),
    "20_7_prime_sandwich": (_mul, (32768, 7, 32768), (32768, 7, 32768), torch.float16, None),
    "20_8_threshold_just_under_8gb": (_add, (32, 8192, 16383), (32, 8192, 16383), torch.float16, None),
    "20_9_threshold_exactly_8gb": (_sub, (32, 8192, 16384), (32, 8192, 16384), torch.float16, None),
    "20_10_threshold_just_over_8gb": (_mul, (32, 8192, 16385), (32, 8192, 16385), torch.float16, None),
    "21_1_trailing_padding_3d": (_add, (8192, 524288, 1), (8192, 524288, 1), torch.float16, None),
    "21_6_donut_padding_3d": (_mul, (256, 1, 16777216), (256, 1, 16777216), torch.float16, None),
}

CHUNKING_4D_PARAM_SETS = {
    "cat1_non_chunking": (_exp_sigmoid, (2, 4, 64, 64), None, torch.float16, None),
    "cat2_span_gt_256_total_lt_8gb": (_exp_sigmoid, (37, 100, 200, 200), None, torch.float16, None),
    "cat3_total_gt_8gb": (_relu_tanh, (8, 4, 8192, 17408), None, torch.float16, None),
    "cat4_span_gt_256_total_gt_8gb": (_neg_abs, (37, 1, 8192, 14800), None, torch.float16, None),
    "8_2_4d_50gb_six_or_more_chunks": (_add, (4, 8, 49152, 17408), (4, 8, 49152, 17408), torch.float16, None),
    "17_1_4d_single_giant_prime_dim": (_exp_sigmoid, (1, 1, 65537, 4096), None, torch.float16, None),
    "19_8_no_chunking_needed_cores8": (_mul, (4, 8, 1024, 1024), (4, 8, 1024, 1024), torch.float16, 8),
    "19_1_single_core_chunking_4d": (torch.add, (4, 8, 8192, 4096), (4, 8, 8192, 4096), torch.float32, 1),
    "19_7_five_cores_4d": (_exp_sigmoid, (4, 8, 8192, 8192), None, torch.float16, 5),
    "20_5_waterfall_ascending": (_sub, (2, 16, 1024, 1048576), (2, 16, 1024, 1048576), torch.float16, None),
    "20_6_waterfall_descending": (_add, (1048576, 1024, 16, 2), (1048576, 1024, 16, 2), torch.float16, None),
    "21_2_trailing_padding_4d": (_mul, (32, 8192, 17408, 1), (32, 8192, 17408, 1), torch.float16, None),
    "21_4_interleaved_padding_4d": (_mul, (65536, 1, 65536, 1), (65536, 1, 65536, 1), torch.float16, None),
}

CHUNKING_5D_PARAM_SETS = {
    "cat1_non_chunking": (_exp_sigmoid, (2, 2, 4, 32, 32), None, torch.float16, None),
    "cat2_span_gt_256_total_lt_8gb": (_exp_sigmoid, (37, 10, 10, 200, 200), None, torch.float16, None),
    "cat3_total_gt_8gb": (_relu_tanh, (2, 4, 4, 8192, 17408), None, torch.float16, None),
    "cat4_span_gt_256_total_gt_8gb": (_neg_abs, (37, 1, 1, 8192, 14800), None, torch.float16, None),
    "8_2_5d_50gb_six_or_more_chunks": (_add, (2, 2, 4, 49152, 8704), (2, 2, 4, 49152, 8704), torch.float16, None),
    "17_1_5d_single_giant_prime_dim": (_exp_sigmoid, (1, 1, 1, 65537, 4096), None, torch.float16, None),
    "19_8_no_chunking_needed_cores32": (_add, (2, 2, 4, 256, 256), (2, 2, 4, 256, 256), torch.float32, 32),
    "19_1_single_core_chunking_5d": (_exp_sigmoid, (2, 4, 4, 8192, 4096), None, torch.float16, 1),
    "19_7_seven_cores_5d": (torch.add, (2, 4, 4, 8192, 4096), (2, 4, 4, 8192, 4096), torch.float32, 7),
    "21_3_trailing_padding_5d": (_add, (32, 8192, 17408, 1, 1), (32, 8192, 17408, 1, 1), torch.float16, None),
    "21_5_interleaved_padding_5d": (_add, (1, 65536, 1, 65536, 1), (1, 65536, 1, 65536, 1), torch.float16, None),
    "21_7_donut_padding_5d": (_add, (32, 64, 1, 128, 17408), (32, 64, 1, 128, 17408), torch.float16, None),
    "21_9_buried_core_5d": (_mul, (1, 1, 4500000000, 1, 1), (1, 1, 4500000000, 1, 1), torch.float16, None),
}

EXPECTED_FAILURES_1D = []
CHUNKING_1D_PARAM_SETS = _expand_dict(CHUNKING_1D_PARAM_SETS)

EXPECTED_FAILURES_2D = []
CHUNKING_2D_PARAM_SETS = _expand_dict(CHUNKING_2D_PARAM_SETS)

EXPECTED_FAILURES_3D = []
CHUNKING_3D_PARAM_SETS = _expand_dict(CHUNKING_3D_PARAM_SETS)

EXPECTED_FAILURES_4D = []
CHUNKING_4D_PARAM_SETS = _expand_dict(CHUNKING_4D_PARAM_SETS)

EXPECTED_FAILURES_5D = []
CHUNKING_5D_PARAM_SETS = _expand_dict(CHUNKING_5D_PARAM_SETS)

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
}

EXPECTED_FAILURES_CUSTOM = set()

def _make_custom_params(param_sets, expect_fail=None):
    expect_fail = expect_fail or set()
    result = []
    for param_id, (factory_fn, fn, ref_fn) in param_sets.items():
        marks = [pytest.mark.xfail] if param_id in expect_fail else []
        result.append(
            pytest.param(factory_fn, fn, ref_fn, id=param_id, marks=marks)
        )
    return result

def _make_params(param_sets, ops_dict=None, expect_fail=None):
    expect_fail = set(expect_fail or [])
    result = []

    if ops_dict:
        for param_id, args in param_sets.items():
            if not isinstance(args, (tuple, list)):
                args = (args,)
            for op_name, op in ops_dict.items():
                full_id = f"{op_name}-{param_id}"
                marks = [pytest.mark.xfail] if param_id in expect_fail else []
                result.append(
                    pytest.param(op_name, op, *args, id=full_id, marks=marks)
                )
    else:
        for param_id, args in param_sets.items():
            if not isinstance(args, (tuple, list)):
                args = (args,)
            marks = [pytest.mark.xfail] if param_id in expect_fail else []
            result.append(
                pytest.param(*args, id=param_id, marks=marks)
            )

    return result



# ---------------------------------------------------------------------------
# TestChunking – Optimal Pruned Suite
# ---------------------------------------------------------------------------

torch.manual_seed(0xAFFE)

class TestOps:

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    def compare_with_cpu(self, *args, **kwargs):
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
        if isinstance(dtype, str):
            dtype = eval(dtype)
            
        x = torch.randn(*shape_x, dtype=dtype, device="spyre")
        y = torch.randn(*shape_y, dtype=dtype, device="spyre") if shape_y is not None else None
        
        inputs = (x, y) if y is not None else (x,)
        
        if expected_cores is not None:
            os.environ["SENCORES"] = str(expected_cores)
            
        try:
            self._check(fn, inputs)
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
