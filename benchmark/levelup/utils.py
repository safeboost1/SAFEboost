import json
import math
from typing import List, Optional
import numpy as np
import os
def get_smallest_pow_2(x):
    return 1 << (x - 1).bit_length()
def left_rotate_bin(context, data, inverval, gs):
    binary_list = []
    while gs > 1:
        if gs % 2 == 1:
            binary_list.append(1)
        else:
            binary_list.append(0)
        gs = gs // 2
    binary_list.append(gs)
    i = len(binary_list) - 1
    sdind = 0
    while i >= 0:
        if binary_list[i] == 1:
            ind = 0
            s = inverval
            tmp = data.deepcopy()
            if data.on_gpu:
                tmp.to_device()
            while ind < i:
                rot = tmp << s
                tmp = tmp + rot
                s = s * 2
                ind = ind + 1
            if sdind > 0:
                tmp = tmp << sdind
            if i == len(binary_list) - 1:
                res = tmp
            else:
                res = tmp + res
            sdind = sdind + s
        i = i - 1
    return res
def right_rotate_bin(context, data, interval, gs):
    binary_list = []
    while gs > 1:
        if gs % 2 == 1:
            binary_list.append(1)
        else:
            binary_list.append(0)
        gs = gs // 2
    binary_list.append(gs)
    i = len(binary_list) - 1
    sdind = 0
    while i >= 0:
        if binary_list[i] == 1:
            ind = 0
            s = interval
            tmp = data.deepcopy()
            if data.on_gpu:
                tmp.to_device()
            while ind < i:
                rot = tmp >> s
                tmp = tmp + rot
                s = s * 2
                ind = ind + 1
            if sdind > 0:
                tmp = tmp >> sdind
            if i == len(binary_list) - 1:
                res = tmp
            else:
                res = tmp + res
            sdind = sdind + s
        i = i - 1
    return res
def log2_choose(n: int, k: int) -> float:
    """return log2(n choose k)"""
    if k > n:
        raise ValueError("k cannot be greater than n")
    acc = 0.0
    for d in range(1, k + 1):
        acc += math.log2(n) - math.log2(d)
        n -= 1
    return acc
def get_OP_CHaW_encoding(number, encoding_size: int, hamming_weight: int, verbose: bool = False) -> List[int]:
    """
    OP-CHaW encoding: represent 'number' as a combination index among
    subsets of size 'hamming_weight' in a universe of 'encoding_size' elements.
    Returns a binary vector (list of 0/1) length encoding_size with exactly
    hamming_weight ones.
    """
    ans = [0] * encoding_size
    if number is None:
        return ans
    number += 1
    log2_mod_size = log2_choose(encoding_size, hamming_weight)
    if math.log2(number) >= log2_mod_size:
        raise ValueError(f"number={number} too large to encode with size={encoding_size}, weight={hamming_weight}")
    if verbose:
        print(f"Encoding number={number} into size={encoding_size}, weight={hamming_weight}")
    remainder = number
    k_prime = hamming_weight
    for pointer in range(encoding_size - 1, -1, -1):
        c = math.comb(pointer, k_prime)
        if remainder >= c:
            ans[pointer] = 1
            remainder -= c
            k_prime -= 1
            if verbose:
                print(
                    f"  pick pos {pointer}: subtract comb({pointer},{k_prime + 1})={c}, new remainder={remainder}, k'={k_prime}"
                )
        if k_prime == 0:
            break
    if verbose:
        print("Resulting vector:", ans)
    return ans
def find_code_length(n: int, k: int) -> int:
    """
    Find the smallest m such that log2_choose(m, k) > n, with an extra +1.
    Translates the C++:
        uint64_t find_code_length(uint64_t n, uint64_t k)
    Steps:
    1) If k == 0, return 0.
    2) Compute cur_n = 2^n.
    3) Compute double_m = cur_n^(1/k) * ∏_{i=2..k} i^(1/k).
    4) Initialize m = ceil(double_m) + k.
    5) Increment m while log2_choose(m, k) <= n.
    6) Decrement m while log2_choose(m, k) > n.
    7) Return m + 1.
    """
    if k == 0:
        return 0
    cur_n = 2**n
    double_m = cur_n ** (1.0 / k)
    for i in range(2, k + 1):
        double_m *= i ** (1.0 / k)
    m = math.ceil(double_m) + k
    while log2_choose(m, k) <= n:
        m += 1
    while log2_choose(m, k) > n:
        m -= 1
    return m + 1
def PE(value: int, n: int) -> list:

    if value is None:
        return [None] * (n + 1)
    result = []
    cur = value
    for _ in range(n):
        result.append(cur)
        cur //= 2
    result.append(0)
    return result
def comp_range_smallest(a: int, largest: int, n: int, return_early: bool = False) -> List[int]:
    """
    Range Cover[0,a]
    Args:
        a: the lower bound of the range (unsigned)
        largest: the maximum value (unsigned)
        n: tree height parameter
        return_early: if True, initializes and returns the array immediately
    Returns:
        A list `range_arr` of length n+1, where each entry is either
        largest+1 (default) or a computed value at certain indices.
    """
    range_arr = [largest + 1] * (n + 1)
    if return_early:
        return range_arr
    exponents: List[int] = []
    powers_sum = 0  
    while (a + 1) - powers_sum != 0:
        i = a - powers_sum
        j = 0
        while (1 << j) - 1 <= i:
            j += 1
        j -= 1
        value = sum(1 << (k - j) for k in exponents)
        range_arr[j] = value
        exponents.append(j)
        powers_sum = sum(1 << q for q in exponents)
    for i, x in enumerate(range_arr):
        if x == largest + 1:
            range_arr[i] = None
    return range_arr
def comp_range_largest(a: int, largest: int, n: int) -> List[int]:
    """
    Range Cover[a,2**n -1 ]
    """
    range_arr = [largest + 1] * (n + 1)
    if a == 0:
        range_arr[n] = 0
        return range_arr
    exponents: List[int] = []
    curr_largest = largest
    while curr_largest >= a:
        j = 0
        while curr_largest >= (1 << j) and curr_largest - (1 << j) + 1 >= a:
            j += 1
        j -= 1  
        value = (largest + 1) // (1 << j)
        for k in exponents:
            value -= 1 << (k - j)
        range_arr[j] = value - 1
        exponents.append(j)
        powers_sum = sum(1 << q for q in exponents)
        curr_largest = largest - powers_sum
    for i, x in enumerate(range_arr):
        if x == largest + 1:
            range_arr[i] = None
    return range_arr
def tournament_product(arr: List[int]) -> int:
    if not arr:
        return 1  
    while len(arr) > 1:
        next_round = []
        for i in range(0, len(arr), 2):
            if i + 1 < len(arr):
                next_round.append(arr[i] * arr[i + 1])
            else:
                next_round.append(arr[i])  
        arr = next_round
    return arr[0]
def evaluate_model(model_path, input_features: list) -> int:

    json_path = os.path.join(model_path ,"splitcond.json")
    with open(json_path, "r") as f:
        model = json.load(f)
    leaf_idx = 0  
    max_depth = max(map(int, model.keys()))
    for depth in range(max_depth + 1):
        split_info = model[str(depth)][leaf_idx]  
        feature = split_info["feature"]
        threshold = split_info["condition"]
        feature_index = int(feature[1:])
        if input_features[feature_index] <= threshold:
            leaf_idx = leaf_idx * 2
        else:
            leaf_idx = leaf_idx * 2 + 1
    return leaf_idx
def print_matrix_b(ctxt, length, base=0.9):
    if ctxt.encrypted:
        tmp = ctxt.deepcopy()
        tmp.to_host()
        msg = tmp.decrypt(inplace=False)
    else:
        print("[Warning] This Ciphertext is already decrypted")
        msg = ctxt
    for blocklist_idx in range(len(ctxt)):  
        for block_idx in range(len(ctxt[0])):  
            for slot_idx in range(length):
                if msg[blocklist_idx][block_idx][slot_idx] > base:
                    print(
                        f"[{blocklist_idx}][{block_idx}][{slot_idx}]",
                        msg[blocklist_idx][block_idx][slot_idx],
                        flush=True,
                    )
def print_ctxt_b(
    c,
    num_print: int = 10,
    base: float = 0.00001,
):
    
    tmp = c.deepcopy()
    tmp.to_host()
    if getattr(tmp, "encrypted", True):
        msg = tmp.decrypt()
        for i in range(num_print):
            r = msg[i]
            if math.hypot(r) >= base:  
                print(i, f"{r}")
    else:
        for i in range(num_print):
            if tmp[i] > base:
                print(i, tmp[i])
def split_vector(vec, cut_idx):
    chunks = []
    for i in range(0, len(vec), cut_idx):
        chunk = vec[i : i + cut_idx]
        if len(chunk) < cut_idx:
            pad = np.zeros(cut_idx - len(chunk), dtype=vec.dtype)
            chunk = np.concatenate([chunk, pad])
        chunks.append(chunk)
    return chunks
def _decrypt(obj):
    
    return obj.decrypt(inplace=False) if getattr(obj, "encrypted", False) else obj
def print_list_b(ctxt, length, base: float = 0.9, is_complex: Optional[bool] = False, li_len=5):

    if is_complex is None:
        is_complex = getattr(ctxt, "is_complex", False)
    li_len = min(li_len, len(ctxt))
    msg = _decrypt(ctxt)
    for blk_idx in range(li_len):
        for slot_idx in range(length):
            val = msg[blk_idx][slot_idx]
            if val >= base:
                print(f"[{blk_idx}][{slot_idx}] {val}", flush=True)
def check_ct_pred(ct_pred, server):
    tree_slot = server.node_itv * (2 ** (server.max_depth) - 1)
    one_ct_tree = server.n_tree_one_ct
    ct_pred = ct_pred.decrypt()
    result = []
    n_ct = math.ceil(server.n_estimators / one_ct_tree)
    for i in range(0, n_ct):
        for j in range(one_ct_tree):
            if len(result) == server.n_estimators:
                break
            msg = ct_pred[i]
            for k in range(2 ** (server.max_depth)):
                idx = j * tree_slot + k
                if msg[idx].real < 0.5:
                    result.append(idx % tree_slot)
    return result