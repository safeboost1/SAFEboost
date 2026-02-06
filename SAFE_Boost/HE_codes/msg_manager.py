import copy
import math
from typing import Dict, Tuple, Union
import numpy as np
from dataset.data_set import DataList, DataMatrix
from HE_codes import utils
from heaan_stat.core.block import Block
class MsgManager:
    def __init__(self, context, precompute_lengths=None):
        self.context = context
        self.num_slots = context.num_slots
        self._cache: Dict[Tuple, Union[Block, DataList]] = {}
        self._gpu_loaded: set[tuple] = set()
        if precompute_lengths:
            for L in precompute_lengths:
                self.get_mask_by_length(L)
            self.get_first_slot_mask()
    def get_first_slot_mask(self) -> Block:

        key = ("first",)
        if key not in self._cache:
            data = [1] + [0] * (self.num_slots - 1)
            self._cache[key] = Block(self.context, encrypted=False, data=data)
        return self._ensure_gpu(key)
    def get_mask_by_length(
        self, length: int, return_type=None, encrypted=False, use_gpu=True
    ) -> Union[Block, DataList]:

        key = ("length", length, encrypted)
        if key not in self._cache:
            num_slots = self.context.num_slots
            rows = math.ceil(length / num_slots)
            total_slots = rows * num_slots
            flat_list = [1] * length + [0] * (total_slots - length)
            if rows == 1 and return_type != "datalist":
                mask = Block(self.context, encrypted=encrypted, data=flat_list)
            elif rows == 1 and return_type == "datalist":
                tmp_li = [flat_list]
                mask = DataList(self.context, encrypted=encrypted, data_list=tmp_li)
            else:
                nested_list = [flat_list[i * num_slots : (i + 1) * num_slots] for i in range(rows)]
                mask = DataList(self.context, encrypted=encrypted, data_list=nested_list)
            self._cache[key] = mask
        mask = self._cache[key]
        if use_gpu and self.context.with_gpu:
            if key not in self._gpu_loaded:
                mask.to_device()
                self._gpu_loaded.add(key)
        elif not use_gpu or not self.context.with_gpu:
            if key in self._gpu_loaded:
                mask.to_host()
                self._gpu_loaded.remove(key)
        return mask
    def get_gain_mask(self, B: int, F: int, return_type=None):

        key = ("gain", B, F)
        if key not in self._cache:
            num_slots = self.context.num_slots
            total_len = B * F  
            rows = math.ceil(total_len / num_slots)  
            total_slots = rows * num_slots  
            mask_np = np.ones(total_slots, dtype=np.int8)  
            zero_idx = np.arange(B - 1, total_len, B, dtype=np.int64)
            mask_np[zero_idx] = 0
            flat_list = mask_np.tolist()  
            if rows == 1 and return_type != "datalist":
                mask = Block(self.context, encrypted=False, data=flat_list)
            elif rows == 1 and return_type == "datalist":
                mask = DataList(self.context, encrypted=False, data_list=[flat_list])
            else:
                nested = [flat_list[i * num_slots : (i + 1) * num_slots] for i in range(rows)]
                mask = DataList(self.context, encrypted=False, data_list=nested)
            self._cache[key] = mask
        return self._ensure_gpu(key)
    def _ensure_gpu(self, key: tuple):

        if self.context.with_gpu and key not in self._gpu_loaded:
            self._cache[key].to_device()
            self._gpu_loaded.add(key)
        return self._cache[key]
    def get_feature_mask(self, max_depth, n_max: int, F: int, one_ctxt_node, one_ctxt_feature):
        self.mask_feature = DataMatrix(self.context, False)
        for i in range(max_depth):
            if n_max * F > self.num_slots:
                cut_idx = one_ctxt_feature * n_max
                vec = [0] * n_max * F
            else:
                cut_idx = n_max * F * int(one_ctxt_node)
                vec = [0] * cut_idx
            for j in range(0, len(vec), n_max):
                vec[j] = 1
            split_vec = utils.split_vector(np.array(vec), cut_idx)
            self.mask_feature.append(DataList(self.context, False, split_vec))
    def get_right_path(
        self, max_depth, att_itv, n_max, F, one_ctxt_node, leaf_node_ctxt_num
    ):  
        num_slots = self.num_slots
        self.right_path_mat = DataMatrix(self.context, False)
        need_ones = att_itv
        for i in range(max_depth):
            right_path_li = DataList(self.context, encrypted=False)
            row = [0] * num_slots
            if i >= int(max_depth - np.log2(one_ctxt_node)):
                nodes_per_ctxt_by_depth = 2**i // leaf_node_ctxt_num
                for j in range(nodes_per_ctxt_by_depth):
                    col = need_ones * int(one_ctxt_node // nodes_per_ctxt_by_depth) * j
                    row[col : col + need_ones] = [1] * need_ones
                right_path_li.append(Block(self.context, encrypted=False, data=row))
            elif att_itv <= num_slots:
                row[0:need_ones] = [1] * need_ones
                right_path_li.append(Block(self.context, encrypted=False, data=row))
            elif n_max <= num_slots:
                n_d = math.floor(num_slots / n_max)
                repeat = F // n_d
                for i in range(repeat):
                    row = [1] * (n_max * n_d)
                    right_path_li.append(Block(self.context, encrypted=False, data=row))
                row = [1] * (n_max * (F - (n_d * repeat)))
                right_path_li.append(Block(self.context, encrypted=False, data=row))
            else:
                repeat = math.floor(n_max / num_slots) - 1
                for i in range(F):
                    for _ in range(repeat):
                        right_path_li.append(self.context.ones(False))
                    fill_len = n_max - num_slots * repeat
                    row[:fill_len] = [1] * fill_len
                    right_path_li.append(Block(self.context, encrypted=False, data=row))
            self.right_path_mat.append(right_path_li)
        return
    def bin_edges_by_depth(self, nmax, F, max_depth, att_itv, one_ctxt_node, leaf_node_ctxt_num):
        self.edge_by_depth = DataMatrix(self.context, True)
        scale = nmax
        if nmax * F <= self.context.num_slots:
            nmax_arr = np.array(
                [(i + 0.1) / nmax for i in range(nmax)] * F + [0] * (self.context.num_slots - (nmax * F))
            )
            nmax_arr = utils.split_vector(nmax_arr, self.context.num_slots)
        else:
            one_ctxt_feature_by_bin = min(
                math.floor(self.context.num_slots / (scale)), F
            )  
            nmax_arr = np.array([(i + 0.1) / scale for i in range(nmax)] * F)
            nmax_arr = np.array(utils.split_vector(nmax_arr, int(one_ctxt_feature_by_bin) * nmax))
        for i in range(max_depth):
            can_merge = int(2 ** (i - np.log2(leaf_node_ctxt_num)))
            if i >= int(max_depth - np.log2(one_ctxt_node)):
                merge_num = int(min(one_ctxt_node, 2**i))
                target = copy.deepcopy(nmax_arr)
                for j in range(int(merge_num) - 1):
                    rot_idx = att_itv * 2 ** (max_depth - i) * (j + 1)
                    rotated = np.roll(nmax_arr, rot_idx)
                    target += rotated
                merged_node = [data.tolist() for data in target]
            else:  
                merged_node = [data.tolist() for data in nmax_arr]
            data_list = DataList(self.context, True, merged_node)
            self.edge_by_depth.append(data_list)
        return
    def bin_edges_mask(self, F, nmax, max_depth, att_itv, one_ctxt_node, leaf_node_ctxt_num):
        self.edge_mask = DataMatrix(self.context, True)
        if nmax * F <= self.context.num_slots:
            nmax_arr = np.array(
                ([1 for i in range(nmax - 1)] + [0]) * F + [0] * (self.context.num_slots - ((nmax) * F))
            )
            nmax_arr = utils.split_vector(nmax_arr, self.context.num_slots)
        else:
            one_ctxt_feature_by_bin = min(
                math.floor(self.context.num_slots / (nmax)), F
            )  
            nmax_arr = np.array(([1 for i in range(nmax - 1)] + [0]) * F)
            nmax_arr = np.array(utils.split_vector(nmax_arr, one_ctxt_feature_by_bin * nmax))
        for i in range(max_depth):
            can_merge = int(2 ** (i - np.log2(leaf_node_ctxt_num)))
            if i >= int(max_depth - np.log2(one_ctxt_node)):
                merge_num = int(min(one_ctxt_node, 2**i))
                target = copy.deepcopy(nmax_arr)
                for j in range(can_merge - 1):
                    rot_idx = att_itv * 2 ** (max_depth - i) * (j + 1)
                    rotated = np.roll(nmax_arr, rot_idx)
                    target += rotated
                merged_node = [data.tolist() for data in target]
            else:  
                merged_node = [data.tolist() for data in nmax_arr]
            data_list = DataList(self.context, True, merged_node)
            self.edge_mask.append(data_list)
        return
    def find_max_pos_msg(self, num_comp, max_depth):  
        slot_per_node = utils.calc_slot_per_node(num_comp)
        num_node = math.floor(self.context.num_slots / slot_per_node)
        num_comp_cop = num_comp
        num_comp = num_comp_cop
        m = [0] * self.context.num_slots
        m1 = [0] * self.context.num_slots
        for i in range(num_node):
            m[i * slot_per_node] = 1
            for j in range(math.ceil(num_comp / 4) * 4, slot_per_node):
                m1[i * (slot_per_node) + j] = -0.02
        m = Block(self.context, False, data=m)
        m1 = Block(self.context, False, data=m1)
        self.max_pos_masking = m
        self.max_pos_empty_slot_masking = m1
        is_first_block_dict = {}
        one_slot_per_node = {}
        select_one_bin_msg = {}
        csel_dict = {}
        select_one_random_pos_iter_bin_interval = {}
        reverse_bin = utils.reverse_binary(num_comp)
        num_chunk = math.ceil(num_comp / 2)
        select_one_random_pos_iter_msg_dict = {}
        if num_node < 2 ** (max_depth):
            tmp_num_node = num_node
            m0_ = [0] * self.context.num_slots
            for j in range(tmp_num_node):
                for i in range(num_chunk):
                    m0_[j * (slot_per_node) + i * (2)] = 1
            empty_msg = Block(self.context, encrypted=False, data=m0_)
            select_one_random_pos_iter_msg_dict[tmp_num_node] = empty_msg
            tmp_num_comp = num_comp_cop + (num_comp_cop % 2)
            tmp_num_comp = tmp_num_comp // 2
            masking = ([1] * tmp_num_comp + [0] * (slot_per_node - tmp_num_comp)) * tmp_num_node
            masking = Block(self.context, False, masking)
            is_first_block_dict[tmp_num_node] = masking
            msg = [0] * self.context.num_slots
            for i in range(tmp_num_node):
                one_idx = slot_per_node * i
                msg[one_idx] = 1
            msg = Block(self.context, False, msg)
            one_slot_per_node[tmp_num_node] = msg
            iter_round = 0
            start_org = 0
            msg_list = [Block(self.context, False, [0])]
            for iteration in range(math.ceil(math.log2(num_comp)) - 1):
                if int(reverse_bin[iter_round]) == 1:
                    msg = [0] * self.context.num_slots
                    start_org = start_org + 2**iteration
                    for node_num in range(tmp_num_node):
                        end_idx = node_num * slot_per_node + num_comp
                        start_idx = end_idx - start_org
                        msg[start_idx:end_idx] = [1] * (end_idx - start_idx)
                    msg = Block(self.context, encrypted=False, data=msg)
                    msg_list.append(msg)
                iter_round = iter_round + 1
            select_one_bin_msg[tmp_num_node] = msg_list
            binary_str = bin(num_comp)[2:]
            binary_list = list(binary_str)
            csel_msg = []
            lognum = len(binary_list) - 1
            for i in range(len(binary_list)):
                a = int(binary_list[i])
                if a == 1:
                    csel_msg = csel_msg + [a] + [0] * ((2 ** (lognum - i)) - 1)
            csel_msg = csel_msg + [0] * (slot_per_node - num_comp)
            csel_msg = list(csel_msg) * tmp_num_node
            csel_msg = Block(self.context, encrypted=False, data=csel_msg)
            csel_dict[tmp_num_node] = csel_msg
            interval = 2
            tmp_num_chunk = num_comp_cop + num_comp_cop % 2
            tmp_num_chunk = tmp_num_chunk // 4
            tmp_block_list = []
            for iteration in range(math.ceil(math.log2(num_comp_cop)) - 1):
                m = [0] * self.context.num_slots
                for j in range(tmp_num_node):
                    for i in range(tmp_num_chunk):
                        m[j * (slot_per_node) + i * (interval * 2)] = 1
                tmp_block_list.append(Block(self.context, encrypted=False, data=m))
                interval = interval * 2
                tmp_num_chunk = tmp_num_chunk // 2
            select_one_random_pos_iter_bin_interval[tmp_num_node] = tmp_block_list
        for i in range(max_depth):
            if (2**i) < num_node:
                tmp_num_node = 2**i
            else:
                if (2**i) % num_node == 0:
                    continue
                tmp_num_node = (2**i) % num_node
            m0_ = [0] * self.context.num_slots
            for j in range(tmp_num_node):
                for i in range(num_chunk):
                    m0_[j * (slot_per_node) + i * (2)] = 1
            empty_msg = Block(self.context, encrypted=False, data=m0_)
            select_one_random_pos_iter_msg_dict[tmp_num_node] = empty_msg
            tmp_num_comp = num_comp_cop + (num_comp_cop % 2)
            tmp_num_comp = tmp_num_comp // 2
            masking = ([1] * tmp_num_comp + [0] * (slot_per_node - tmp_num_comp)) * tmp_num_node
            masking = Block(self.context, False, masking)
            is_first_block_dict[tmp_num_node] = masking
            msg = [0] * self.context.num_slots
            for i in range(tmp_num_node):
                one_idx = slot_per_node * i
                msg[one_idx] = 1
            msg = Block(self.context, False, msg)
            one_slot_per_node[tmp_num_node] = msg
            iter_round = 0
            start_org = 0
            msg_list = [Block(self.context, False, [0])]
            for iteration in range(math.ceil(math.log2(num_comp)) - 1):
                if int(reverse_bin[iter_round]) == 1:
                    msg = [0] * self.context.num_slots
                    start_org = start_org + 2**iteration
                    for node_num in range(tmp_num_node):
                        end_idx = node_num * slot_per_node + num_comp
                        start_idx = end_idx - start_org
                        msg[start_idx:end_idx] = [1] * (end_idx - start_idx)
                    msg = Block(self.context, encrypted=False, data=msg)
                    msg_list.append(msg)
                iter_round = iter_round + 1
            select_one_bin_msg[tmp_num_node] = msg_list
            binary_str = bin(num_comp)[2:]
            binary_list = list(binary_str)
            csel_msg = []
            lognum = len(binary_list) - 1
            for i in range(len(binary_list)):
                a = int(binary_list[i])
                if a == 1:
                    csel_msg = csel_msg + [a] + [0] * ((2 ** (lognum - i)) - 1)
            csel_msg = csel_msg + [0] * (slot_per_node - num_comp)
            csel_msg = list(csel_msg) * tmp_num_node
            csel_msg = Block(self.context, encrypted=False, data=csel_msg)
            csel_dict[tmp_num_node] = csel_msg
            interval = 2
            tmp_num_chunk = num_comp_cop + num_comp_cop % 2
            tmp_num_chunk = tmp_num_chunk // 4
            tmp_block_list = []
            for iteration in range(math.ceil(math.log2(num_comp_cop)) - 1):
                m = [0] * self.context.num_slots
                for j in range(tmp_num_node):
                    for i in range(tmp_num_chunk):
                        m[j * (slot_per_node) + i * (interval * 2)] = 1
                tmp_block_list.append(Block(self.context, encrypted=False, data=m))
                interval = interval * 2
                tmp_num_chunk = tmp_num_chunk // 2
            select_one_random_pos_iter_bin_interval[tmp_num_node] = tmp_block_list
        self.select_one_random_pos_iter_msg_dict = select_one_random_pos_iter_msg_dict
        self.is_first_block_dict = is_first_block_dict
        self.one_slot_per_node = one_slot_per_node
        self.select_one_bin_msg = select_one_bin_msg
        self.csel_dict = csel_dict
        self.select_one_random_pos_iter_bin_interval = select_one_random_pos_iter_bin_interval
    def scale_max_msg(self, num_comp):
        slot_per_node = utils.calc_slot_per_node(num_comp)
        num_node = math.floor(self.context.num_slots / slot_per_node)
        msg = [0] * self.context.num_slots
        for i in range(num_node):
            one_idx = slot_per_node * i
            msg[one_idx] = 1
        msg = Block(self.context, False, msg)
        self.scale_masking = msg
        masking_msg_list = []
        for i in range(1, num_node + 1):
            masking_msg = ([1] * num_comp + [0] * (slot_per_node - num_comp)) * i
            masking_block = Block(self.context, False, masking_msg)
            masking_msg_list.append(masking_block)
        self.scale_gain_msg_masking_list = masking_msg_list