import json
import math
import os
import pickle
import random
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd
import psutil
from heaan_stat.core import Block
from natsort import natsorted
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score  
random.seed(42)
np.random.seed(42)
def _decrypt(obj):
    
    return obj.decrypt(inplace=False) if getattr(obj, "encrypted", False) else obj
def _get_re_im(ctxt: Union["Block"]):

    real2x, imag2x = ctxt.twice_of_real_and_imag_parts()
    return real2x * 0.5, imag2x * 0.5  
def print_ctxt_step(c, step=9, num_print=300):
    tmp = c.deepcopy()
    tmp.decrypt()
    for i in range(0, num_print, step):
        print(i, tmp[i])
def print_ctxt(
    c,
    num_print: int = 300,
    is_complex: Optional[bool] = False,
):

    tmp = c.deepcopy()
    if is_complex is None:
        is_complex = getattr(tmp, "is_complex", False)
    if is_complex:
        re_obj, im_obj = _get_re_im(tmp)  
        re_msg = _decrypt(re_obj)
        im_msg = _decrypt(im_obj)
        for i in range(num_print):
            print(i, f"{re_msg[i]} + {im_msg[i]}j")
    else:
        if getattr(tmp, "encrypted", False):
            tmp = list(tmp.decrypt(inplace=False))
        for i in range(num_print):
            print(i, tmp[i])
def print_ctxt_not_zero(c, num_print: int = 10):
    tmp = c.deepcopy()
    if tmp.encrypted:
        tmp.decrypt()
    else:
        print("[Warning] This Ciphertext is already decrypted")
    for i in range(num_print):
        if abs(tmp[i]) > 0.000000001:
            print(i, tmp[i])
def print_ctxt_b(
    c,
    num_print: int = 10,
    base: float = 0.00001,
    is_complex: Optional[bool] = False,
):

    tmp = c.deepcopy()
    if is_complex is None:
        is_complex = getattr(tmp, "is_complex", False)
    if is_complex:
        re_obj, im_obj = _get_re_im(tmp)
        re_msg = _decrypt(re_obj)
        im_msg = _decrypt(im_obj)
        for i in range(num_print):
            r = re_msg[i]
            im = im_msg[i]
            if math.hypot(r, im) >= base:  
                print(i, f"{r} + {im}j")
    else:
        if getattr(tmp, "encrypted", False):
            tmp.decrypt()
        for i in range(num_print):
            if tmp[i] > base:
                print(i, tmp[i])
def print_matrix_b(ctxt, length, base=0.9):
    if ctxt.encrypted:
        tmp = ctxt.deepcopy()
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
def print_matrix_not_zero(ctxt, mat_len=30, length=100, is_complex: Optional[bool] = False):

    if getattr(ctxt, "encrypted", False):
        tmp = ctxt.deepcopy()
        msg = tmp.decrypt(inplace=False)
    else:
        print("[Warning] This Ciphertext is already decrypted")
        msg = ctxt
    mat_len = min(mat_len, len(msg))
    if is_complex:
        re_obj, im_obj = _get_re_im(ctxt)
        re_msg = _decrypt(re_obj)
        im_msg = _decrypt(im_obj)
        for bl in range(mat_len):
            for bk in range(len(re_msg[bl])):
                for slot in range(length):
                    r = re_msg[bl][bk][slot]
                    im = im_msg[bl][bk][slot]
                    if abs(r) > 1e-5 or abs(im) > 1e-5:
                        print(f"[{bl}][{bk}][{slot}] {r} + {im}j", flush=True)
    else:
        for bl in range(mat_len):
            for bk in range(len(msg[bl])):
                for slot in range(length):
                    val = msg[bl][bk][slot]
                    if abs(val) > 1e-5:
                        print(f"[{bl}][{bk}][{slot}] {val}", flush=True)
def print_matrix(ctxt, mat_len=10, length=10, base=0.9, is_complex: Optional[bool] = False):
    if ctxt.encrypted:
        tmp = ctxt.deepcopy()
        msg = tmp.decrypt(inplace=False)
    else:
        print("[Warning] This Ciphertext is already decrypted")
        msg = ctxt
    mat_len = min(mat_len, len(ctxt))
    for blocklist_idx in range(mat_len):  
        for block_idx in range(len(ctxt[0])):  
            for slot_idx in range(length):
                print(
                    f"[{blocklist_idx}][{block_idx}][{slot_idx}]",
                    msg[blocklist_idx][block_idx][slot_idx],
                    flush=True,
                )
def print_list_b(ctxt, length, base: float = 0.9, is_complex: Optional[bool] = False, li_len=5):

    if is_complex is None:
        is_complex = getattr(ctxt, "is_complex", False)
    li_len = min(li_len, len(ctxt))
    if is_complex:
        re_obj, im_obj = _get_re_im(ctxt)
        re_msg = _decrypt(re_obj)
        im_msg = _decrypt(im_obj)
        for blk_idx in range(li_len):
            for slot_idx in range(length):
                r = re_msg[blk_idx][slot_idx]
                im = im_msg[blk_idx][slot_idx]
                if math.hypot(r, im) >= base:  
                    print(f"[{blk_idx}][{slot_idx}] {r} + {im}j", flush=True)
    else:
        msg = _decrypt(ctxt)
        for blk_idx in range(li_len):
            for slot_idx in range(length):
                val = msg[blk_idx][slot_idx]
                if val >= base:
                    print(f"[{blk_idx}][{slot_idx}] {val}", flush=True)
def check_boolean_list(ctxt, length):

    msg = _decrypt(ctxt)
    count = 0
    for blk_idx in range(len(msg)):
        for slot_idx in range(length):
            if abs(msg[blk_idx][slot_idx]) >= 0.9:
                count += 1
    print(f"Number of values ≥ 0.9: {count}" if count > 0 else "0")
def print_list(ctxt, length, is_complex: Optional[bool] = False):

    if is_complex is None:
        is_complex = getattr(ctxt, "is_complex", False)
    if is_complex:
        re_obj, im_obj = _get_re_im(ctxt)
        re_msg = _decrypt(re_obj)
        im_msg = _decrypt(im_obj)
        for blk_idx in range(len(re_msg)):
            for slot_idx in range(length):
                r = re_msg[blk_idx][slot_idx]
                im = im_msg[blk_idx][slot_idx]
                print(f"[{blk_idx}][{slot_idx}] {r} + {im}j", flush=True)
    else:
        msg = _decrypt(ctxt)
        for blk_idx in range(len(msg)):
            for slot_idx in range(length):
                print(f"[{blk_idx}][{slot_idx}] {msg[blk_idx][slot_idx]}", flush=True)
def print_list_not_zero(ctxt, length, is_complex: Optional[bool] = False):

    if is_complex is None:
        is_complex = getattr(ctxt, "is_complex", False)
    if ctxt.encrypted:
        msg = ctxt.decrypt(inplace=False)
    else:
        print("[Warning] This Ciphertext is already decrypted")
        msg = ctxt
    if is_complex:
        re_obj, im_obj = _get_re_im(ctxt)
        re_msg = _decrypt(re_obj)
        im_msg = _decrypt(im_obj)
        for blk_idx in range(len(re_msg)):
            for slot_idx in range(length):
                r = re_msg[blk_idx][slot_idx]
                im = im_msg[blk_idx][slot_idx]
                if abs(r) > 0.00001 or abs(im) > 0.00001:
                    print(f"[{blk_idx}][{slot_idx}] {r} + {im}j", flush=True)
    else:
        for blk_idx in range(len(msg)):
            for slot_idx in range(length):
                val = msg[blk_idx][slot_idx]
                if abs(val) > 0.00001:
                    print(f"[{blk_idx}][{slot_idx}] {val}", flush=True)
def combine_real_imag_to_complex(real_obj, imag_obj):
    from dataset.data_set import DataList

    if isinstance(real_obj, Block) and isinstance(imag_obj, Block):
        comp_blk = real_obj.deepcopy()  
        comp_blk += imag_obj * (1j)  
        comp_blk.is_complex = True  
        return comp_blk
    if isinstance(real_obj, DataList) and isinstance(imag_obj, DataList):
        if len(real_obj) != len(imag_obj):
            raise ValueError("The number of blocks in the two DataLists does not match")
        complex_blocks = [
            combine_real_imag_to_complex(r_blk, i_blk)  
            for r_blk, i_blk in zip(real_obj.block_list, imag_obj.block_list)
        ]
        gh_dl = DataList(real_obj.context, encrypted=real_obj.encrypted, is_complex=True)
        gh_dl.set_block_list(complex_blocks)
        return gh_dl
    raise TypeError("Both real_obj and imag_obj must be either Block or DataList")
def fill_slots_if_one_present(context, block, inplace=True):
    """
    vec: numpy array representing a single block
    If any slot in the block contains 1, fill all slots with 1.
    """
    num_slots = context.num_slots
    if not inplace:
        res = block.deepcopy()
        r = 1
        while r < num_slots:
            rot = res << r
            res += rot
            r *= 2
        return res
    else:
        r = 1
        while r < num_slots:
            rot = block << r
            block += rot
            r *= 2
def get_sorted_folders(directory_path):
    folders = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
    folders = natsorted(folders)
    return folders
def get_sorted_files(directory_path):
    files = [name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))]
    files = natsorted(files)
    return files
def F2P(x):
    return 1 / (1 + np.exp(-x))
def print_inferec_res(context, model):
    y_pred, y_prob = decrypt_res_by_one_row(context, model)
    return y_pred, y_prob
def decrypt_res_by_one_row(context, model):  

    from dataset.data_set import DataMatrix
    k = 100
    rows = model.inf_ndata
    lr = model.eta
    itv = model.node_itv
    one_ctxt_node = int(model.one_ctxt_node)
    n_estimators = model.n_estimators
    y_predict_list = [0] * rows
    y_prob_list = [0] * rows
    if one_ctxt_node < 1:
        one_ctxt_node = 1
    pred_path = f"./{model.main_folder}/predict_ctxt"
    for i in range(rows):
        print("row:", i)
        row_path = Path(f"{pred_path}/row_{i}")
        Fm = 0
        tmp = DataMatrix.from_path(context, row_path, True)
        for dlist in tmp:
            for data in dlist:
                m = data.decrypt()
                for j in range(0, context.num_slots):
                    idx = j
                    if 10**-6 < abs(m[idx].real) < 49:
                        y_predict = m[idx].real
                        Fm += lr * y_predict
                        print("predict:", y_predict)
                y_prob_list[i] = F2P(Fm)
                y_predict_list[i] = round(F2P(Fm))
    return y_predict_list, y_prob_list
def check_ct_pred(context, ct_pred):  

    ct_pred_list=[]
    for dlist in ct_pred:
        for data in dlist:
            m = data.decrypt()
            for j in range(0, context.num_slots):
                idx = j
                if 10**-6 < abs(m[idx].real) < 49:
                    y_predict = m[idx].real
                    ct_pred_list.append(y_predict)
                    print("predict:", y_predict)
    return ct_pred_list
def save_metrics(y_true, y_pred, y_prob, dataset_name, n_estimators, max_bin, max_depth, save_path):

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_prob)
    val_loss = log_loss(y_true, y_prob)
    print(f"acc: {acc:.4f}, f1: {f1:.4f}, auc: {auc:.4f}, val_loss: {val_loss:.4f}")
    df = pd.DataFrame(
        [
            {
                "n_estimators": n_estimators,
                "bin": max_bin,
                "depth": max_depth,
                "accuracy": acc,
                "f1_score": f1,
                "auc": auc,
                "val_loss": val_loss,
            }
        ]
    )
    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, f"{dataset_name}_metrics(train_only).csv")
    header = not os.path.exists(out_file)
    df.to_csv(out_file, index=False, mode="a", header=header)
    print(f"✅ Metrics saved to {out_file}")
def accuracy(y_hat_list, y_test):
    label_list = y_test
    print("y_list", list(label_list))
    print("y_pred", y_hat_list)
    sum = 0
    is_correct = []
    for i in range(len(y_hat_list)):
        if label_list[i] == (y_hat_list[i]):
            is_correct.append(1)
            sum += 1
        else:
            is_correct.append(0)
    res_accurate = sum / len(is_correct)
    print("is correct :", is_correct)
    print("ratio:", res_accurate)
    return res_accurate, is_correct
def set_metadata(df):
    col = df.columns
    for cname in col:
        df[cname] = df[cname].astype("category")
        tmp_cat = np.sort(df[cname].unique())
        for j in range(len(tmp_cat)):
            tmp_cat[j] = j + 1
        df[cname].values.set_categories = tmp_cat
    df = df.astype("int64")
    ndata = df.shape[0]
    d = len(col) - 1
    n = find_max_cat_x(df)
    t = len(df["label"].unique())
    return n, d, t, ndata
def find_max_cat_x(df):
    col = df.columns
    max_values = []
    for i in col.drop("label"):
        max_values.append(max(df[i]))
    n = int(max(max_values))
    return n
def ensure_directory_exists(directory_path):

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
def get_smallest_pow_2(x):
    return 1 << (x - 1).bit_length()
def make_tree_structure(name, max_depth, tree_structure=None):
    if tree_structure is None:
        tree_structure = {}
    current_depth = len(name) - 1
    if current_depth in tree_structure:
        tree_structure[current_depth].append(name)
    else:
        tree_structure[current_depth] = [name]
    if current_depth == max_depth:
        return tree_structure
    new = name + "l"
    make_tree_structure(new, max_depth, tree_structure)
    new = name + "r"
    make_tree_structure(new, max_depth, tree_structure)
    return tree_structure
def calc_slot_per_node(x):
    n_slot = x + x % 2
    return n_slot
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
def rotate_bin_datalist(context, data_list, interval, gs, direction):
    res = data_list.deepcopy()
    if direction == "left":
        rotate_bin = left_rotate_bin
    if direction == "right":
        rotate_bin = right_rotate_bin
    for idx in range(len(res)):
        res[idx] = rotate_bin(context, data_list[idx], interval, gs)
    return res
def random_mode(lst):

    counter = Counter(lst)
    most_common = counter.most_common()
    if all(count == most_common[0][1] for _, count in most_common):
        return random.choice(lst)
    modes = [value for value, count in most_common if count == most_common[0][1]]
    return random.choice(modes)
def compute_bit_counts(depth):

    leaf_count = 2**depth
    bit_counts = {}
    for L in range(leaf_count):
        bin_str = format(L, f"0{depth}b")
        ones = bin_str.count("1")
        zeros = bin_str.count("0")
        bit_counts[L] = {"ones": ones, "zeros": zeros}
    return bit_counts
def split_vector(vec, cut_idx):
    if isinstance(vec, list):
        vec = np.array(vec)
    chunks = []
    for i in range(0, len(vec), cut_idx):
        chunk = vec[i : i + cut_idx]
        if len(chunk) < cut_idx:
            pad = np.zeros(cut_idx - len(chunk), dtype=vec.dtype)
            chunk = np.concatenate([chunk, pad])
        chunks.append(chunk)
    return chunks
def find_max_value(context, c, num_comp, slot_per_node, num_node, elapsed=0, msg_mgr=None, msg_idx=0):
    if num_comp == 1:
        print(f"[TIME]find max value time: {elapsed:.4f} s")
        return c, elapsed
    if num_comp % 4 != 0:
        i = num_comp
        while i % 4 != 0:
            i += 1
        num_comp = i
    t0 = time.time()
    c = c + msg_mgr.empty_slot_masking[msg_idx]
    elapsed += time.time() - t0
    t1 = time.time()
    if c.need_bootstrap(3):
        c.bootstrap()
    masked_value = []
    masked_value.append(c * msg_mgr.find_max_value_msg1[msg_idx])
    for i in range(1, 4):
        ctmp1 = c << (i * num_comp) // 4
        masked_value.append(ctmp1 * msg_mgr.find_max_value_msg1[msg_idx])
    a1, b1, c1, d1 = masked_value
    compare_round1 = [a1 - b1, b1 - c1, c1 - d1, d1 - a1, a1 - c1, b1 - d1]
    ctmp1 = compare_round1[0]
    for i in range(1, len(compare_round1)):  
        ctmp2 = compare_round1[i] >> ((i * num_comp) // 4)
        ctmp1 = ctmp1 + ctmp2
    c0 = ctmp1.sign(inplace=False)
    if c0.need_bootstrap(9):
        c0.bootstrap()
    c0_c = c0.deepcopy()
    c0 = 1 + c0
    c0 = c0 * 0.5
    ceq = c0_c * c0_c
    ceq = ceq * -1
    ceq = ceq + 1
    elapsed += time.time() - t1
    t2 = time.time()
    c_neg = c0 * -1
    c_neg = c_neg + 1
    compare_round2 = []
    ctmp1 = c0 * msg_mgr.find_max_value_mk[msg_idx][0]
    c_ab = ctmp1.deepcopy()
    ctmp2 = c_neg * msg_mgr.find_max_value_mk[msg_idx][3]
    c_ab = ctmp1.deepcopy()  
    ctmp2 = ctmp2 << ((3 * num_comp) // 4)
    ctmp1 = ctmp1 * ctmp2
    ctmp2 = c0 * msg_mgr.find_max_value_mk[msg_idx][4]  
    ctmp2 = ctmp2 << num_comp
    compare_round2.append(ctmp1 * ctmp2)
    ctmp1 = c_neg * msg_mgr.find_max_value_mk[msg_idx][0]
    ctmp2 = c0 * msg_mgr.find_max_value_mk[msg_idx][1]
    ctmp2 = ctmp2 << (num_comp // 4)
    c_bc = ctmp2.deepcopy()
    ctmp1 = ctmp1 * ctmp2
    ctmp2 = c0 * msg_mgr.find_max_value_mk[msg_idx][5]
    ctmp2 = ctmp2 << (num_comp * 5 // 4)
    compare_round2.append(ctmp1 * ctmp2)
    ctmp1 = c_neg * msg_mgr.find_max_value_mk[msg_idx][1]
    ctmp1 = ctmp1 << (num_comp // 4)
    ctmp2 = c0 * msg_mgr.find_max_value_mk[msg_idx][2]
    ctmp2 = ctmp2 << (num_comp // 2)
    c_cd = ctmp2.deepcopy()
    ctmp1 = ctmp1 * ctmp2
    ctmp2 = c_neg * msg_mgr.find_max_value_mk[msg_idx][4]
    ctmp2 = ctmp2 << (num_comp)
    compare_round2.append(ctmp1 * ctmp2)
    ctmp1 = c_neg * msg_mgr.find_max_value_mk[msg_idx][2]
    ctmp1 = ctmp1 << (num_comp // 2)
    ctmp2 = c0 * msg_mgr.find_max_value_mk[msg_idx][3]
    ctmp2 = ctmp2 << (3 * num_comp // 4)
    cda = ctmp2.deepcopy()
    ctmp1 = ctmp1 * ctmp2
    ctmp2 = c_neg * msg_mgr.find_max_value_mk[msg_idx][5]
    ctmp2 = ctmp2 << (5 * num_comp // 4)
    compare_round2.append(ctmp1 * ctmp2)
    cout = compare_round2[0] * masked_value[0]
    for i in range(1, len(compare_round2)):
        tmp = compare_round2[i] * masked_value[i]
        cout = cout + tmp
    if cout.need_bootstrap(9):
        cout.bootstrap()
    cneq = ceq * -1
    cneq = cneq + 1
    cneq_da = cneq << (3 * num_comp // 4)
    cneq_da = cneq_da * msg_mgr.find_max_value_mk[msg_idx][0]
    ceq_ab = ceq * msg_mgr.find_max_value_mk[msg_idx][0]
    ceq_bc = ceq << (num_comp // 4)
    ceq_bc = ceq_bc * msg_mgr.find_max_value_mk[msg_idx][0]
    ceq_cd = ceq << (num_comp // 2)
    ceq_cd = ceq_cd * msg_mgr.find_max_value_mk[msg_idx][0]
    ceq_da = cneq_da * -1
    ceq_da = ceq_da + msg_mgr.find_max_value_mk[msg_idx][0]
    ctmp2 = ceq_ab * ceq_bc
    ctmp1 = ctmp2 * c_cd
    c_cond3 = ctmp1.deepcopy()
    ctmp1 = ceq_bc * ceq_cd
    ctmp1 = ctmp1 * cda
    c_cond3 = c_cond3 + ctmp1
    ctmp1 = ceq_cd * ceq_da
    ctmp1 = ctmp1 * c_ab
    c_cond3 = c_cond3 + ctmp1
    ctmp1 = ceq_ab * ceq_da
    ctmp1 = ctmp1 * c_bc
    c_cond3 = c_cond3 + ctmp1
    c_cond4 = ctmp2 * ceq_cd
    c_tba = c_cond3 * (1 / 3)
    c_tba = c_tba + 1
    ctmp1 = c_cond4 + 1
    c_tba = c_tba * ctmp1
    cout = cout * c_tba
    elapsed += time.time() - t2
    return find_max_value(context, cout, num_comp // 4, slot_per_node, num_node, elapsed, msg_mgr, msg_idx + 1)
def select_one_random_pos(context, c_red, num_comp, slot_per_node, num_node):
    rando = np.random.permutation(num_comp)
    ctmp1 = c_red.deepcopy()
    m0_ = [0] * context.num_slots
    c_sel = Block(context, encrypted=True, data=m0_)
    for i in range(num_node):
        m0_[i * (slot_per_node)] = 1
    empty_msg = Block(context, encrypted=False, data=m0_)
    t0 = time.time()
    for l in range(num_comp):
        if ctmp1.level <= 5:
            ctmp1.bootstrap()
            c_sel.bootstrap()
        if l > 0:
            ctmp1 = ctmp1 << l
            ctmp2 = ctmp1 * c_sel
            ctmp1 = ctmp1 - ctmp2
            ctmp2 = ctmp1 * empty_msg
            ctmp1 = ctmp1 >> l
            c_sel = c_sel + ctmp2
        else:
            ctmp2 = c_sel * ctmp1
            ctmp1 = ctmp1 - ctmp2
            ctmp2 = ctmp1 * empty_msg
            c_sel = c_sel + ctmp2
    return ctmp1, time.time() - t0
def find_biggerone(context, ctxt1, ctxt2):
    if ctxt1.level < 4:
        ctxt1.bootstrap()
    if ctxt2.level < 4:
        ctxt2.bootstrap()
    add = ctxt1 + ctxt2
    sub = ctxt1 - ctxt2
    sgn_ab = sub.sign(inplace=False)
    if sgn_ab.level < 4:
        sgn_ab.bootstrap()
    sub_mult_sgn_ab = sub * sgn_ab
    res1 = add + sub_mult_sgn_ab
    if res1.level < 4:
        res1.bootstrap()
    res1 = res1 * 0.5
    if res1.level < 4:
        res1.bootstrap()
    return res1
def find_max_ciplist(context, cip_list):
    ctxtnum = len(cip_list)
    interval = 2 ** (math.ceil(math.log2(ctxtnum)))
    interval = interval // 2
    init_time = 0
    t0 = time.time()
    while 0 < interval:
        for i in range(interval):
            if len(cip_list) <= (i + interval):
                pass
            else:
                cip_list[i] = find_biggerone(context, cip_list[i], cip_list[i + interval])
        interval = interval // 2
    init_time += time.time() - t0
    return cip_list[0], init_time
def find_max_pos(context, c, num_comp, slot_per_node, num_node, msg_mgr):
    if c.need_bootstrap(9):
        c.bootstrap()
    masking = ([1] * num_comp + [0]) * num_node
    masking = Block(context, False, masking)
    if c.need_bootstrap(1):
        c.bootstrap()
    c = c * masking
    if c.need_bootstrap(1):
        c.bootstrap()
    find_max_pos_time = 0
    if c.need_bootstrap(1):
        c.bootstrap()
    cmax, f_max_val_time = find_max_onecip(context, c, num_comp, slot_per_node, num_node, msg_mgr)
    find_max_pos_time += f_max_val_time
    t0 = time.time()
    c = c + msg_mgr.max_pos_empty_slot_masking
    c = c - cmax
    c = c + (1 / (2**18))
    c_red = c.sign(inplace=False)
    if c_red.need_bootstrap(9):
        c_red.bootstrap()
    c_red = c_red * 0.8
    c_red = c_red.sign(inplace=False)
    if c_red.need_bootstrap(9):
        c_red.bootstrap()
    c_red = c_red + 1
    c_red = c_red * 0.5
    find_max_pos_time += time.time() - t0
    c_out, csel, sel_pos_time = select_one_pos_bin(context, c_red, num_comp, slot_per_node, num_node, msg_mgr)
    find_max_pos_time += sel_pos_time
    t1 = time.time()
    if c_out.need_bootstrap(9):
        c_out.bootstrap()
    find_max_pos_time += time.time() - t1
    print(f"[TIME]find max pos Time: {find_max_pos_time:.4f} s")
    return c_out, find_max_pos_time
def find_max_onecip(context, cipher_text=None, num_comp=32768, slot_per_node=32768, num_node=1, msg_mgr=None):
    res = cipher_text.deepcopy()
    if context.num_slots < num_comp:
        num_comp = context.num_slots
    if res.need_bootstrap(1):
        res.bootstrap()
    init_time = 0
    t0 = time.time()
    res = res * msg_mgr.scale_gain_msg_masking_list[num_node - 1]
    if res.level < 4:
        res.bootstrap()
    tmp_num_comp = num_comp
    is_first = True
    while 1 < tmp_num_comp:
        tmp_num_comp = tmp_num_comp + (tmp_num_comp % 2)
        tmp_num_comp = tmp_num_comp // 2
        res_rot = res << tmp_num_comp
        res = find_biggerone(context, res, res_rot)
        if is_first:
            is_first = False
            if res.level < 4:
                res.bootstrap()
            res = res * msg_mgr.is_first_block_dict[num_node]
            if res.level < 4:
                res.bootstrap()
    init_time += time.time() - t0
    t1 = time.time()
    if res.level < 4:
        res.bootstrap()
    init_time += time.time() - t1
    t2 = time.time()
    res = res * msg_mgr.one_slot_per_node[num_node]
    init_time += time.time() - t2
    res = right_rotate_bin(context, res, 1, slot_per_node)
    init_time += time.time() - t1
    del res_rot
    return res, init_time
def noise_clean(ctxt):
    if ctxt.level < 6:
        ctxt.bootstrap()
    a = ctxt * ctxt
    b = a * ctxt
    a = a * 3
    b = b * -2
    res = a + b
    if ctxt.level < 4:
        ctxt.bootstrap()
    return res
def select_one_random_pos_iter(context, c_red, num_comp, slot_per_chunk, slot_per_node, num_chunk, num_node, msg_mgr):

    rando = np.random.permutation(num_comp)
    ctmp1 = c_red.deepcopy()
    c_sel = Block(context, encrypted=True, data=[0])
    empty_msg = msg_mgr.select_one_random_pos_iter_msg_dict[num_node]
    t0 = time.time()
    if ctmp1.level <= 6:
        ctmp1.bootstrap()
    ctmp2 = ctmp1 << 1
    ctmp3 = ctmp1 * ctmp2
    c_sel = ctmp1 + ctmp2
    c_sel = c_sel - ctmp3
    c_sel = c_sel * empty_msg
    ctmp3 = ctmp3 * empty_msg
    ctmp3 = ctmp3 >> rando[0]
    ctmp1 = ctmp1 - ctmp3
    if ctmp1.level <= 4:
        ctmp1.bootstrap()
        c_sel.bootstrap()
    end_time = time.time() - t0
    return ctmp1, c_sel, end_time
def select_one_random_pos_interval(context, c_red, num_comp, slot_per_node, num_node, interval):
    rando = np.random.permutation(num_comp)
    ctmp1 = c_red.deepcopy()
    m0_ = [0] * context.num_slots
    c_sel = Block(context, encrypted=True, data=m0_)
    for i in range(num_node):
        m0_[i * (slot_per_node)] = 1
    empty_msg = Block(context, encrypted=False, data=m0_)
    for i in range(num_comp):
        l = rando[i]
        if ctmp1.level <= 5:
            ctmp1.bootstrap()
            c_sel.bootstrap()
        if l > 0:
            ctmp1 = ctmp1 << l * interval
            ctmp2 = ctmp1 * c_sel
            ctmp1 = ctmp1 - ctmp2
            ctmp2 = ctmp1 * empty_msg
            ctmp1 = ctmp1 >> l * interval
            c_sel = c_sel + ctmp2
        else:
            ctmp2 = c_sel * ctmp1
            ctmp1 = ctmp1 - ctmp2
            ctmp2 = ctmp1 * empty_msg
            c_sel = c_sel + ctmp2
    return ctmp1, c_sel
def select_one_random_pos_iter_bin_interval(
    context, c_red, num_comp, slot_per_chunk, slot_per_node, num_chunk, num_node, interval
):
    rando = np.random.permutation(num_comp)
    ctmp1 = c_red.deepcopy()
    m0_ = [0] * context.num_slots
    c_sel = Block(context, encrypted=True, data=m0_)
    for j in range(num_node):
        for i in range(num_chunk):
            m0_[j * (slot_per_node) + i * (slot_per_chunk)] = 1
    empty_msg = Block(context, encrypted=False, data=m0_)
    for i in range(num_comp):
        l = rando[i]
        if ctmp1.level <= 5:
            ctmp1.bootstrap()
            c_sel.bootstrap()
        ctmp1 = ctmp1 << (l * interval)
        ctmp2 = ctmp1 * c_sel
        ctmp1 = ctmp1 - ctmp2
        ctmp2 = ctmp1 * empty_msg
        ctmp1 = ctmp1 >> (l * interval)
        c_sel = c_sel + ctmp2
    return ctmp1, c_sel
def reverse_binary(num):
    binary_str = bin(num)[2:]
    binary_list = list(binary_str)[::-1]
    return "".join(binary_list)
def remain_slot_one(context, ctxt1, ctxt2):
    if ctxt1.level <= 3:
        ctxt1.bootstrap()
    if ctxt2.level <= 3:
        ctxt2.bootstrap()
    ctxt3 = ctxt2 * ctxt1
    if ctxt3.level <= 3:
        ctxt3.bootstrap()
    ctxt2 = ctxt2 - ctxt3
    ctxt1 = ctxt1 + ctxt2
    return ctxt1
def weighted_order(n):

    binary_str = bin(n)[2:]
    bits = [2 ** (len(binary_str) - 1 - i) for i, bit in enumerate(binary_str) if bit == "1"]
    result = []
    while bits:
        total = sum(bits)
        r = random.random() * total
        cumulative = 0
        for i, value in enumerate(bits):
            cumulative += value
            if r < cumulative:
                selected = bits.pop(i)
                result.append(selected)
                break
    return result
def cumulative_sum_larger(numbers):

    result = []
    for x in numbers:
        total = sum(y for y in numbers if y > x)
        result.append(total)
    return result
def select_one_random_pos_interval_list(context, c_red, slot_per_node, num_node, interval_list):
    ctmp1 = c_red.deepcopy()
    m0_ = [0] * context.num_slots
    c_sel = Block(context, encrypted=True, data=m0_)
    for i in range(num_node):
        m0_[i * (slot_per_node)] = 1
    empty_msg = Block(context, encrypted=False, data=m0_)
    for l in interval_list:
        if ctmp1.level <= 5:
            ctmp1.bootstrap()
            c_sel.bootstrap()
        if l > 0:
            ctmp1 = ctmp1 << l
            ctmp2 = ctmp1 * c_sel
            ctmp1 = ctmp1 - ctmp2
            ctmp2 = ctmp1 * empty_msg
            ctmp1 = ctmp1 >> l
            c_sel = c_sel + ctmp2
        else:
            ctmp2 = c_sel * ctmp1
            ctmp1 = ctmp1 - ctmp2
            ctmp2 = ctmp1 * empty_msg
            c_sel = c_sel + ctmp2
    return ctmp1, c_sel
def select_one_pos_bin(context, c_red, num_comp, slot_per_node, num_node, msg_mgr):

    init_time = 0
    c_sel_list = []
    num_chunk = math.ceil(num_comp / 2)
    ctmp1, c_sel, sel_one_rand_pos_iter_time = select_one_random_pos_iter(
        context, c_red, 2, 2, slot_per_node, num_chunk, num_node, msg_mgr
    )
    init_time += sel_one_rand_pos_iter_time
    interval = 2
    reverse_bin = reverse_binary(num_comp)
    binary_str = bin(num_comp)[2:]
    binary_list = list(binary_str)
    csel_msg = []
    num_chunk = num_comp // 4
    ctmp_list = []
    iter_round = 0
    start_org = 0
    msg_idx = 0
    msg = Block(context, encrypted=False, data=[0])
    t0 = time.time()
    c_sel_org = c_sel.deepcopy()
    for iteration in range(math.ceil(math.log2(num_comp)) - 1):
        ctmp, c_sel = select_one_random_pos_iter_bin_interval(
            context, c_sel, 2, interval * 2, slot_per_node, num_chunk, num_node, interval
        )
        ctmp = right_rotate_bin(context, ctmp, 1, interval)
        if int(reverse_bin[iter_round]) == 1:
            msg_idx = msg_idx + 1
        ctmp = ctmp + msg_mgr.select_one_bin_msg[num_node][msg_idx]
        c_sel_list.append(c_sel)
        ctmp_list.append(ctmp)
        interval = interval * 2
        num_chunk = num_chunk // 2
        iter_round = iter_round + 1
    if ctmp1.level <= 3:
        ctmp1.bootstrap()
    for i in ctmp_list:
        if i.level <= 3:
            i.bootstrap()
        ctmp1 = ctmp1 * i
        if ctmp1.level <= 3:
            ctmp1.bootstrap()
    for i in c_sel_list:
        c_sel_org = remain_slot_one(context, c_sel_org, i)
    c_sel_last = c_sel_org * msg_mgr.csel_dict[num_node]
    init_time += time.time() - t0
    select_list = cumulative_sum_larger(weighted_order(num_comp))
    t1 = time.time()
    c_sel_last, is_one = select_one_random_pos_interval_list(context, c_sel_last, slot_per_node, num_node, select_list)
    msk_c_sel_list = []
    one_idx = 0
    for i in range(len(binary_list)):
        if int(binary_list[i]) == 1:
            msg = [0] * context.num_slots
            for index_count in range(num_node):
                msg[(index_count) * slot_per_node + one_idx] = 1
            masking_msg = Block(context, False, data=msg)
            one_idx = one_idx + 2 ** (len(binary_list) - i - 1)
            add_ctxt = c_sel_last * masking_msg
            add_ctxt = right_rotate_bin(context, add_ctxt, 1, 2 ** (len(binary_list) - i - 1))
            msk_c_sel_list.append(add_ctxt)
    mul_ctxt = msk_c_sel_list[0]
    a = 0
    for i in msk_c_sel_list[1:]:
        a = a + 1
        mul_ctxt = mul_ctxt + i
    ctmp1 = ctmp1 * mul_ctxt
    init_time += time.time() - t1
    return ctmp1, is_one, init_time
def find_max_value_fix(context, c, num_comp, slot_per_node, num_node):
    if num_comp == 1:
        return c
    if num_comp % 4 != 0:
        i = num_comp
        m = [0] * (context.num_slots)
        while i % 4 != 0:
            for j in range(num_node):
                m[i + j * slot_per_node] = 2**-18
            i += 1
        num_comp = i
        msg = Block(context, encrypted=False, data=m)
        c = c + msg
    masking_slot_per_node = Block(context, True, [1] * slot_per_node)
    msg_numcomp = Block(context, True, [2**-18] * slot_per_node)
    m = [0] * (context.num_slots)
    for j in range(num_node):
        for i in range(num_comp // 4):
            m[j * slot_per_node + i] = 1
    msg1 = Block(context, encrypted=False, data=m)
    if c.need_bootstrap(3):
        c.bootstrap()
    masked_value = []
    masked_value.append(c * msg1)
    for i in range(1, 4):
        ctmp1 = c << (i * num_comp) // 4
        masked_value.append(ctmp1 * msg1)
    a1, b1, c1, d1 = masked_value
    compare_round1 = [a1 - b1, b1 - c1, c1 - d1, d1 - a1, a1 - c1, b1 - d1]
    ctmp1 = compare_round1[0]
    for i in range(1, len(compare_round1)):
        ctmp2 = compare_round1[i] >> ((i * num_comp) // 4)
        ctmp1 = ctmp1 + ctmp2
    ctmp1 = ctmp1 + msg_numcomp
    c0 = ctmp1.sign()
    c0.bootstrap()
    c0_c = c0.deepcopy()
    c0 = 1 + c0
    c0 = c0 * 0.5
    c0 = c0 * c0_c
    c0 = noise_clean(c0)
    c0.bootstrap()
    ceq = c0_c * c0_c
    ceq = ceq * -1
    ceq = ceq + 1
    ceq = ceq * masking_slot_per_node
    mk = []
    mk.append(msg1)
    for i in range(1, 6):
        m = [0] * context.num_slots
        for j in range(num_node):
            for k in range((i * num_comp) // 4, (i + 1) * num_comp // 4):
                m[j * (slot_per_node) + k] = 1
        mk.append(Block(context, encrypted=False, data=m))
    c_neg = c0 * -1
    c_neg = c_neg + 1
    c_neg = c_neg - ceq
    c0_add_ceq = c0 + ceq
    c_neg_add_ceq = c_neg + ceq
    compare_round2 = []
    ctmp1 = c0_add_ceq * mk[0]
    ctmp2 = c_neg_add_ceq * mk[3]
    ctmp2 = ctmp2 << ((3 * num_comp) // 4)
    ctmp1 = ctmp1 * ctmp2
    ctmp2 = c0_add_ceq * mk[4]
    ctmp2 = ctmp2 << num_comp
    compare_round2.append(ctmp1 * ctmp2)
    ctmp1 = c_neg * mk[0]
    ctmp2 = c0_add_ceq * mk[1]
    ctmp2 = ctmp2 << (num_comp // 4)
    ctmp1 = ctmp1 * ctmp2
    ctmp2 = c0_add_ceq * mk[5]
    ctmp2 = ctmp2 << (num_comp * 5 // 4)
    compare_round2.append(ctmp1 * ctmp2)
    ctmp1 = c_neg * mk[1]
    ctmp1 = ctmp1 << (num_comp // 4)
    ctmp2 = c0_add_ceq * mk[2]
    ctmp2 = ctmp2 << (num_comp // 2)
    ctmp1 = ctmp1 * ctmp2
    ctmp2 = c_neg * mk[4]
    ctmp2 = ctmp2 << (num_comp)
    compare_round2.append(ctmp1 * ctmp2)
    ctmp1 = c_neg * mk[2]
    ctmp1 = ctmp1 << (num_comp // 2)
    ctmp2 = c0 * mk[3]
    ctmp2 = ctmp2 << (3 * num_comp // 4)
    ctmp1 = ctmp1 * ctmp2
    ctmp2 = c_neg * mk[5]
    ctmp2 = ctmp2 << (5 * num_comp // 4)
    compare_round2.append(ctmp1 * ctmp2)
    cout = compare_round2[0] * masked_value[0]
    for i in range(1, len(compare_round2)):
        tmp = compare_round2[i] * masked_value[i]
        cout = cout + tmp
    cout.bootstrap()
    return find_max_value_fix(context, cout, num_comp // 4, slot_per_node, num_node)
def find_max_pos_extend_log(context, block_list, num_comp, slot_per_node, num_node=1):
    print(">>> find max pos extend log start<<<")
    from dataset.data_set import DataList

    org_num_comp = num_comp
    t0 = time.time()
    block_list_org = DataList(context, True)
    for i in block_list:
        block_list_org.append(i.deepcopy())
    if len(block_list) < 1:
        print("[Error] block list lenght is 0")
        return
    if 1 < len(block_list):
        c, cip_time = find_max_ciplist(context, block_list)
        num_comp = context.num_slots
    else:
        c = block_list[0]
    if (context.num_slots - 1) <= slot_per_node:
        c2 = c << (num_comp // 2)
        c = find_biggerone(context, c, c2)
        num_comp = math.ceil(num_comp / 2)
        slot_per_node = calc_slot_per_node(num_comp)
    cmax, onecip_time = find_max_onecip(context, c)
    cmax = cmax - (1 / 2**18)
    position_list = []
    iteration = -1
    for block_idx in range(len(block_list)):
        iteration = iteration + 1
        tmp_block = block_list_org[block_idx] - cmax
        tmp_block = tmp_block.sign()
        if tmp_block.level < 4:
            tmp_block.bootstrap()
        tmp_block2 = tmp_block + 1
        tmp_block = tmp_block2 * tmp_block
        tmp_block = tmp_block * 0.5
        tmp_block = noise_clean(tmp_block)
        tmp_block = tmp_block * tmp_block
        if tmp_block.level < 4:
            tmp_block.bootstrap()
        position_list.append(tmp_block)
    cout_list = DataList(context, True)
    csel_list = DataList(context, True)
    iteration = 0
    num = 1
    for c_red in position_list:
        c_out, c_sel, sel_pos_time = select_one_pos_bin(context, c_red, context.num_slots, context.num_slots, num_node)
        c_out = noise_clean(c_out)
        cout_list.append(c_out)
        csel_list.append(c_sel)
        iteration = iteration + 1
    if len(block_list) == 1:
        return cout_list
    is_one = csel_list[0].deepcopy()
    selct_round2 = DataList(context, True)
    selct_round2.append(Block(context, True, [1] * context.num_slots))
    round_ = 1
    for ctmp in csel_list[1:]:
        if is_one.level < 4:
            is_one.bootstrap()
        if ctmp.level < 4:
            is_one.bootstrap()
        ctmp2 = ctmp * is_one
        ctmp = ctmp - ctmp2
        is_one = is_one + ctmp
        rot = right_rotate_bin(context, ctmp, 1, context.num_slots)
        rot = noise_clean(rot)
        selct_round2.append(rot)
        round_ = round_ + 1
    if cout_list[0].level < 4:
        cout_list.bootstrap()
    if selct_round2[0].level < 4:
        selct_round2.bootstrap()
    cout_list = cout_list * selct_round2
    if cout_list.need_bootstrap(9):
        cout_list.bootstrap()
    print(f"[TIME]find max pos Time: {time.time() - t0:.4f} s")
    return cout_list
def scaling_max(context, c, num_comp, slot_per_node, num_node, scale_value=10000, msg_mgr=None):

    init_time = 0
    tmp_cip, cip_time = find_max_ciplist(context, c)
    init_time += cip_time
    tmp_cip, onecip_time = find_max_onecip(context, tmp_cip, num_comp, slot_per_node, num_node, msg_mgr)
    init_time += onecip_time
    t1 = time.time()
    if tmp_cip.level < 4:
        tmp_cip.bootstrap()
    tmp_cip = tmp_cip * scale_value
    tmp_cip = tmp_cip.inverse()  
    scale_value = int(scale_value * 0.9)
    tmp_cip = tmp_cip + (1 / 2**18)
    tmp_cip = tmp_cip * scale_value  
    tmp_cip = tmp_cip * msg_mgr.scale_gain_msg_masking_list[num_node - 1]
    if tmp_cip.level < 4:
        tmp_cip.bootstrap()
    init_time += time.time() - t1
    return tmp_cip, init_time
def select_one_random_pos_sqrt(context, c_red, num_comp, slot_per_node, num_node):
    num_comp_sq = math.ceil(math.sqrt(num_comp))
    slot_per_chunk = math.ceil(num_comp / num_comp_sq)
    num_chunk = math.ceil(num_comp / num_comp_sq)
    ctmp1, c_sel, pos_iter_time = select_one_random_pos_iter(
        context, c_red, num_comp_sq, slot_per_chunk, slot_per_node, num_chunk, num_node
    )
    ctmp2, c_sel = select_one_random_pos_interval(
        context, c_sel, slot_per_chunk, slot_per_node, num_node, slot_per_chunk
    )
    c_res2 = right_rotate_bin(context, ctmp2, 1, slot_per_chunk)
    res = c_res2 * ctmp1
    return res, c_sel
def scaling_approx_max(context, ctxt_list, iter=7, num_comp=32768):

    cop_ctxt_list = ctxt_list.deepcopy()
    K = 100
    scale_num = num_comp // K
    cop_ctxt_list = cop_ctxt_list * scale_num
    for i in range(iter):
        allsum = Block.zeros(context, True)
        for j in range(len(cop_ctxt_list)):
            cop_ctxt_list[j] = cop_ctxt_list[j] * cop_ctxt_list[j]
            if cop_ctxt_list[j].level <= 4:
                cop_ctxt_list[j].bootstrap()
            allsum = allsum + cop_ctxt_list[j]
            print_ctxt(cop_ctxt_list[j], 5)
        allsum = right_rotate_bin(context, allsum, 1, context.num_slots)
        mult_val = (2**17) * (K / num_comp) ** 2
        allsum = allsum * mult_val
        inv_ctxt = allsum.inverse(inplace=False)
        if inv_ctxt.level <= cop_ctxt_list[0].level:
            inv_ctxt.bootstrap()
        if i < (iter - 1):
            fix_number = scale_num * mult_val
        else:
            fix_number = mult_val
        inv_ctxt = inv_ctxt * fix_number
        for j in range(len(cop_ctxt_list)):
            cop_ctxt_list[j] = cop_ctxt_list[j] * inv_ctxt
            if cop_ctxt_list[j].level <= 4:
                cop_ctxt_list[j].bootstrap()
    allsum = Block.zeros(context, True)
    for j in range(len(cop_ctxt_list)):
        tmp1 = ctxt_list[j] * cop_ctxt_list[j]
        allsum = allsum + tmp1
    allsum = right_rotate_bin(context, allsum, 1, context.num_slots)
    allsum = allsum * 10000
    allsum = allsum.inverse()
    if allsum.level < ctxt_list.level:
        allsum.bootstrap()
    allsum = allsum * 9000
    return allsum