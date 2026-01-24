import gc
import math
import os
import time
from pathlib import Path
import HE_codes.node as node
import HE_codes.utils as utils
import numpy as np
from dataset.data_set import DataList, DataMatrix
from heaan_stat.core.block import Block
class MyXGBClassificationTree:
    def __init__(
        self,
        context,
        msg_mgr,
        tree_idx,
        max_depth,
        reg_lambda,
        prune_gamma,
        tree_method="hist",
        n_bins=4,
        main_folder=None,
    ):
        self.context = context
        self.num_slot = context.num_slots
        self.msg_mgr = msg_mgr
        self.reg_lambda = reg_lambda
        self.max_depth = max_depth
        self.prune_gamma = prune_gamma
        self.tree_method = tree_method
        self.n_bins = n_bins
        self.time = 0
        self.debug = True
        self.tree_idx = tree_idx
        self._parent_cache = {}  
        self._cache_level = -1  
        self.hist = None
        self.node_id_counter = 0
        self.estimator = None
        self.main_folder = main_folder
        self.save_path = Path(f"{self.main_folder}/eval_ctxt/tree{tree_idx + 1}/")
        self.model_path = Path(f"{self.main_folder}/model_ctxt/tree{tree_idx + 1}/")
        self.ctxt_path = Path(f"{self.main_folder}/enc_data/tree{tree_idx + 1}/")
        self.tree_path_save = Path(self.save_path, "path")
        self.inf_leaf_weight_save = Path(self.save_path, "leaf_weight")
        self.key_file_path = Path(f"{self.main_folder}/keys_FGb")
        self.json_path = Path(f"{self.main_folder}/Metadata.json")
    def _load_parent_info(self, pid: str):
        if pid in self._parent_cache:
            return self._parent_cache[pid]
        info = {}
        base_path_model = self.model_path / pid
        base_path_enc = self.ctxt_path / pid
        info["gh"] = DataList.from_path(self.context, str(base_path_enc / "gh"), encrypted=True)
        if pid != "o":  
            info["is"] = DataList.from_path(self.context, base_path_model / "is_data", encrypted=True)
        info["sel"] = DataList.from_path(self.context, base_path_model / "selected_bin", encrypted=True)
        self._parent_cache[pid] = info
        return info
    def fit_from_histogram(self, hist_obj):
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.ctxt_path, exist_ok=True)
        self.hist = hist_obj
        self.max_bin = hist_obj.max_bin
        self.d = hist_obj.d
        self.n_max = hist_obj.n_max
        self.att_itv = self.n_max * self.d
        self.one_ctxt_feature = min(math.floor(self.num_slot / self.n_max), self.d)
        self.ctxt_num_by_feature = math.ceil((self.n_max * self.one_ctxt_feature) / self.num_slot)
        
        self.leaf_node_ctxt_num = utils.get_smallest_pow_2(
            int((np.ceil((self.att_itv * (2**self.max_depth)) / self.num_slot)) // self.ctxt_num_by_feature)
        )
        self.one_ctxt_node = 2 ** (self.max_depth) / (
            self.leaf_node_ctxt_num * self.ctxt_num_by_feature
        )  
        nodes_dict = {}
        tree_structure = utils.make_tree_structure("o", self.max_depth)
        for cur_depth in range(self.max_depth):
            if self._cache_level != cur_depth - 1:
                self._parent_cache.clear()
                gc.collect()
                self._cache_level = cur_depth - 1
            self.train_nonleaf(cur_depth, nodes_dict, tree_structure)
            if cur_depth >= 1:  
                parent_level = tree_structure[cur_depth - 1]
                for pid in parent_level:
                    if pid in nodes_dict:
                        nodes_dict[pid].gh = None
                        del nodes_dict[pid]
                gc.collect()
        if self._cache_level != cur_depth - 1:
            self._parent_cache.clear()
            gc.collect()
            self._cache_level = cur_depth - 1
        self.train_leaf(self.max_depth, nodes_dict, tree_structure)
        nodes_dict.clear()
        del nodes_dict
        self._parent_cache.clear()
        gc.collect()
        return
    def train_nonleaf(self, cur_level, nodes_dict, tree_structure):
        print(f">>> train {cur_level} level start <<<", flush=True)
        num_slots = self.context.num_slots
        cur_level_nodes = tree_structure[cur_level]
        num_gains_per_node = (self.n_bins - 1) * self.hist.d  
        comp_gain_slot = utils.calc_slot_per_node(num_gains_per_node)
        total_max_gain = []
        if comp_gain_slot > num_slots / 2:
            for node_idx in range(len(cur_level_nodes)):
                node_id = cur_level_nodes[node_idx]
                if node_id not in nodes_dict:
                    current_node = node.Node(p_maxn=self.n_bins, p_d=self.hist.d)
                    current_node.id = node_id
                    nodes_dict[node_id] = current_node
                current_node = nodes_dict[node_id]
                if node_id == "o" and current_node.gh is None:
                    current_node.gh = self.hist.gh
                elif node_id != "o" and current_node.gh is None:
                    if not (self.ctxt_path / current_node.id).exists():
                        os.mkdir(self.ctxt_path / current_node.id)
                    self.update_data(node=current_node)
                gain = self.calc_gain(node=current_node)
                scaled_gain, gain_scale_time = self.gain_scaling_one_node(gain, node=current_node)
                print(f"[TIME]gain scaling Time: {gain_scale_time:.4f} s", flush=True)
                self.time += gain_scale_time
                max_gain = utils.find_max_pos_extend_log(
                    self.context,
                    scaled_gain,
                    num_comp=num_gains_per_node,
                    slot_per_node=comp_gain_slot,
                )
                total_max_gain.append(max_gain)
                self.update_node(node=current_node, max_gain=max_gain)
        else:
            max_nodes_per_batch = int(num_slots // comp_gain_slot)
            one_msg = self.msg_mgr.get_mask_by_length(num_gains_per_node)
            for node_idx in range(int(np.ceil(len(cur_level_nodes) / max_nodes_per_batch))):
                gain_total = self.context.zeros(encrypted=True)
                gain_sigma_total = self.context.zeros(encrypted=True)
                last_idx = max_nodes_per_batch * (node_idx + 1)
                node_per_batch = cur_level_nodes[max_nodes_per_batch * node_idx : last_idx]
                total_gain_scale_time = 0  
                for i in range(len(node_per_batch)):
                    node_id = node_per_batch[i]
                    if node_id not in nodes_dict:
                        current_node = node.Node(p_maxn=self.n_bins, p_d=self.hist.d)
                        current_node.id = node_id
                        nodes_dict[node_id] = current_node
                    current_node = nodes_dict[node_id]
                    if node_id == "o" and current_node.gh is None:
                        current_node.gh = self.hist.gh
                    elif node_id != "o" and current_node.gh is None:
                        if not (self.ctxt_path / current_node.id).exists():
                            os.mkdir(self.ctxt_path / current_node.id)
                        self.update_data(node=current_node)
                    gain = self.calc_gain(node=current_node)  
                    tt = time.time()
                    tmp_gain = gain >> (comp_gain_slot * i)
                    gain_total = gain_total + tmp_gain
                    self.time += time.time() - tt
                    gain_scale_t = time.time()
                    r = 1
                    while r < num_slots:
                        rot_gain = gain << r
                        gain += rot_gain
                        r *= 2
                    sigma_gain = gain * one_msg
                    sigma_gain = sigma_gain >> (comp_gain_slot * i)
                    gain_sigma_total = gain_sigma_total + sigma_gain
                    total_gain_scale_time += time.time() - gain_scale_t
                gain_total, gain_scale_time = self.gain_scaling(
                    gain_total, gain_sigma_total, num_gains_per_node, comp_gain_slot, len(node_per_batch), self.msg_mgr
                )
                total_gain_scale_time += gain_scale_time
                self.time += total_gain_scale_time
                print(f"[TIME]gain scaling Time: {total_gain_scale_time:.4f} s")  
                max_gain_per_node, find_max_pos_time = utils.find_max_pos(
                    self.context,
                    c=gain_total,
                    num_comp=num_gains_per_node,
                    slot_per_node=comp_gain_slot,
                    num_node=len(node_per_batch),
                    msg_mgr=self.msg_mgr,
                )  
                self.time += find_max_pos_time
                for i in range(len(node_per_batch)):
                    node_id = node_per_batch[i]
                    current_node = nodes_dict[node_id]
                    masking_gain_msg = self.msg_mgr.get_mask_by_length(num_gains_per_node)
                    t3 = time.time()
                    max_gain = max_gain_per_node << (comp_gain_slot * i)
                    max_gain = max_gain * masking_gain_msg
                    self.time += time.time() - t3
                    if isinstance(max_gain, Block):
                        max_gain_list = DataList(self.context, encrypted=True)
                        max_gain_list.append(max_gain)
                    total_max_gain.append(max_gain_list)
                    if self.debug:
                        print(
                            f"[LEVEL] {current_node.id} max gain level: ",
                            max_gain.level,
                        )
                        print(f"---{current_node.id} max gain---")
                    self.update_node(node=current_node, max_gain=max_gain_list)
        self.pre_process(total_max_gain, cur_level)
    def calc_gain(self, node):
        except_time = 0
        num_slots = self.context.num_slots
        masking_msg = self.msg_mgr.get_first_slot_mask()
        num_gains_per_node = (self.n_bins - 1) * self.hist.d  
        add_to_gain = Block(
            self.context, False, [0.0000001] * num_gains_per_node + [0] * (num_slots - num_gains_per_node)
        )
        copy_slot = min(num_gains_per_node, num_slots)
        left_GH = DataList(self.context, True)
        t0 = time.time()
        GH = node.gh.sum_block()  
        r = 1
        while r < num_slots:
            rot_gh = GH << r
            GH += rot_gh
            r *= 2
        GH = GH * masking_msg
        trans_GH = utils.right_rotate_bin(self.context, GH, 1, copy_slot)  
        if self.debug:
            print("[LEVEL]trans_GH level : ", GH.level)
        et0 = time.time()
        if GH.on_gpu:
            GH.to_host()
        del GH
        gc.collect()
        except_time += time.time() - et0
        left_gh = self.hist.bin_indexes *(node.gh)  
        et5 = time.time()
        except_time += time.time() - et5
        if len(left_gh) > 1:
            for list_idx in range(len(left_gh)):
                tmp_gh_block = left_gh[list_idx].sum_block()  
                r = 1
                while r < num_slots:
                    rot_gh = tmp_gh_block << r
                    tmp_gh_block += rot_gh
                    r *= 2
                tmp_gh_block = tmp_gh_block * masking_msg  
                left_GH.append(tmp_gh_block)  
        else:
            left_GH = left_gh
        st = time.time()
        trans_left_GH = self.transform_datalist(left_GH, inplace=True)  
        print(f"[TIME] transform datalist Time: {time.time() - st:.4f} s")
        if self.debug:
            print(f"[LEVEL] {node.id}  trans_left_GH level : ", trans_left_GH.level, flush=True)
        if trans_left_GH.need_bootstrap(3):
            print("[BOOTSTRAP] trans_left_GH level (before G,H): ", trans_left_GH.level, flush=True)
            left_G, left_H = trans_left_GH.twice_of_real_and_imag_parts()
            left_G *= 0.5
            left_H *= 0.5
            Block.bootstrap_two_ctxts(left_G, left_H)
            G, H = trans_left_GH.twice_of_real_and_imag_parts()
            G *= 0.5
            H *= 0.5
            Block.bootstrap_two_ctxts(G, H)
        else:
            G, H = trans_GH.twice_of_real_and_imag_parts()
            G *= 0.5
            H *= 0.5
            left_G, left_H = trans_left_GH.twice_of_real_and_imag_parts()
            left_G *= 0.5
            left_H *= 0.5
        right_G = G - left_G  
        tmp_H = 2 * H - left_H  
        tmp2_H = H + left_H
        tmp_gain_left = left_G * tmp_H
        tmp_gain_left *= tmp_gain_left
        tmp_gain_right = right_G * tmp2_H
        tmp_gain_right *= tmp_gain_right
        tmp_gain_left += tmp_gain_right  
        gain = tmp_gain_left
        if gain.need_bootstrap(2):
            print("[BOOTSTRAP] gain level (after masking) : ", gain.level, flush=True)
            gain.bootstrap()
        gain += add_to_gain
        print(f"[TIME]calc gain Time: {time.time() - t0 - except_time:.4f} s")
        self.time += time.time() - t0 - except_time
        return gain
    def gain_scaling(self, total_gain, sigma_gain, num_comp, slot_per_node, num_node, msg_mgr):

        init_time = 0
        t0 = time.time()
        inv_sigma_gain = sigma_gain.inverse()  
        scaled_gain = inv_sigma_gain * total_gain
        init_time += time.time() - t0
        scaled_gain = DataList(self.context, True, [scaled_gain])
        gain_for_find_max, scale_max_time = utils.scaling_max(
            self.context, scaled_gain, num_comp, slot_per_node, num_node, 10000, msg_mgr
        )
        init_time += scale_max_time
        t1 = time.time()
        init_time += scale_max_time
        total_gain = scaled_gain[0] * gain_for_find_max
        total_gain = total_gain * msg_mgr.scale_gain_msg_masking_list[num_node - 1]
        init_time += time.time() - t1
        return total_gain, init_time
    def gain_scaling_one_node(self, gain, node):

        num_slots = self.context.num_slots
        num_comp = (self.max_bin - 1) * self.d
        slot_per_node = utils.calc_slot_per_node(num_comp)
        t0 = time.time()
        sigma_gain = gain.sum_block()  
        r = 1
        while r < num_slots:
            rot = sigma_gain << r
            sigma_gain += rot
            r *= 2
        print("[LEVEL] sigma_gain level : ", sigma_gain.level, flush=True)
        inv_gain = sigma_gain.inverse()
        print("[LEVEL] inv_gain level : ", inv_gain.level, flush=True)
        gain *= inv_gain  
        if gain.need_bootstrap(1):
            print("[BOOTSTRAP] gain level (after scaling) : ", gain.level, flush=True)
            if isinstance(gain, Block):
                gain.bootstrap()
            else:
                Block.bootstrap_batch(gain.block_list)
        print("[LEVEL] inv scaling gain level : ", gain.level, flush=True)
        gain_scale_ctxt = utils.scaling_max(self.context, gain, num_comp, slot_per_node, 1, 10000)
        print("[LEVEL] gain_scale_ctxt level : ", gain_scale_ctxt.level, flush=True)
        gain *= gain_scale_ctxt  
        if gain.need_bootstrap(1):
            print("[BOOTSTRAP] gain level (after scaling) : ", gain.level, flush=True)
            if isinstance(gain, Block):
                gain.bootstrap()
            else:
                Block.bootstrap_batch(gain.block_list)
        print("[LEVEL]final gain level : ", gain.level, flush=True)
        print(f"[TIME]gain scaling Time: {time.time() - t0:.4f} s")
        return gain
    def transform_datalist(self, input_list: DataList, *, inplace: bool = True):

        num_slots = self.context.num_slots  
        total_items = len(input_list)  
        if inplace:
            new_blocks = []
        else:
            res_list = DataList(self.context, encrypted=True)
        current_idx = 0  
        slot_offset = 0  
        tmp = self.context.zeros(encrypted=True)
        while current_idx < total_items:
            tmp += input_list[current_idx] >> slot_offset
            current_idx += 1
            slot_offset += 1
            if slot_offset >= num_slots:
                if inplace:
                    new_blocks.append(tmp)
                else:
                    res_list.append(tmp)
                tmp = self.context.zeros(encrypted=True)
                slot_offset = 0
        if slot_offset > 0:
            if inplace:
                new_blocks.append(tmp)
            else:
                res_list.append(tmp)
        if inplace:
            input_list.block_list = new_blocks
            return input_list[0]
        else:
            return res_list[0]
    def update_node(self, node, max_gain):

        node_data_save_path = self.model_path / node.id
        split_condition_save_path = node_data_save_path / "split_condition"
        selected_bin_save_path = node_data_save_path / "selected_bin"
        for path in [split_condition_save_path, selected_bin_save_path]:
            os.makedirs(path, exist_ok=True)
        selected_bin = self.select_bin_index(max_gain)
        node.selected_bin = selected_bin
        selected_bin.save(selected_bin_save_path)
    def select_bin_index(self, max_gain: DataList):

        if self.debug:
            print("[LEVEL] max gain level : ", max_gain.level)
            print("[LEVEL] self.hist.bin_indexes level : ", self.hist.bin_indexes.level)
        num_slots = self.context.num_slots  
        total_bins = (self.n_bins - 1) * self.hist.d  
        slots_per_blk = min(total_bins, num_slots)  
        masking_msg = self.msg_mgr.get_first_slot_mask()
        selected_bin = DataList(self.context, encrypted=True)
        empty_block = self.context.zeros(encrypted=True)
        for _ in range(len(self.hist.bin_indexes[0])):
            selected_bin.append(empty_block)
        t0 = time.time()
        for i in range(len(self.hist.bin_indexes)):
            blk_idx = i // slots_per_blk
            rot_idx = i % slots_per_blk
            tmp = max_gain[blk_idx] << rot_idx
            tmp = tmp * masking_msg
            utils.fill_slots_if_one_present(self.context, tmp, inplace=True)
            selected_bin = selected_bin + tmp * self.hist.bin_indexes[i]
        print(f"[TIME]Update node Time: {time.time() - t0:.4f} s")
        self.time += time.time() - t0
        if self.debug:
            print("[LEVEL] selected_bin level : ", selected_bin.level)
        return selected_bin
    def update_data(self, node):

        parent_id = node.id[: len(node.id) - 1]
        parent = self._load_parent_info(parent_id)
        save_path = self.ctxt_path / node.id  
        for child_ctxt_path in [save_path / "gh"]:
            if not os.path.exists(child_ctxt_path):
                os.makedirs(child_ctxt_path)
        if node.id != "o":
            child_is_data_path = self.model_path / node.id / "is_data"
            os.makedirs(child_is_data_path, exist_ok=True)
        t0 = time.time()
        if node.id[-1] == "l":
            node.is_data = parent["sel"] if parent_id == "o" else parent["is"] * parent["sel"]
        else:
            m1 = self.msg_mgr.get_mask_by_length(self.hist.ndata, encrypted=True, use_gpu=True)
            right_select_bin = m1 - parent["sel"]  
            node.is_data = right_select_bin if parent_id == "o" else parent["is"] * right_select_bin
        update_gh = node.is_data * parent["gh"]
        if update_gh.need_bootstrap(3):
            print("[BOOTSTRAP] update_gh level : ", update_gh.level)
            re, im = update_gh.twice_of_real_and_imag_parts()
            re *= 0.5
            im *= 0.5
            Block.bootstrap_batch(re.block_list)
            Block.bootstrap_batch(im.block_list)
            update_gh = utils.combine_real_imag_to_complex(re, im)
            if isinstance(node.is_data, Block):
                node.is_data.bootstrap()
            else:
                Block.bootstrap_batch(node.is_data.block_list)
        print(f"[TIME]Update data Time: {time.time() - t0:.4f} s")
        self.time += time.time() - t0
        node.is_data.save(child_is_data_path)  
        update_gh.save(save_path / "gh")
        node.gh = update_gh
    def train_leaf(self, cur_level, nodes_dict, tree_structure):
        print(">>> train leaf level start <<<", flush=True)
        cur_level_nodes = tree_structure[cur_level]
        leaf_nodes_gh = DataMatrix(self.context, encrypted=True)
        for i in range(len(cur_level_nodes)):
            node_id = cur_level_nodes[i]
            if node_id not in nodes_dict:
                current_node = node.Node(p_maxn=self.n_bins, p_d=self.hist.d)
                current_node.id = node_id
                if not (self.ctxt_path / current_node.id).exists():
                    os.mkdir(self.ctxt_path / current_node.id)
            gh = self.update_data_leaf(node=current_node)  
            leaf_nodes_gh.append(gh)
        self.calc_weight(leaf_nodes_gh)
    def update_data_leaf(self, node):

        parent_id = node.id[: len(node.id) - 1]
        parent = self._load_parent_info(parent_id)
        save_path = self.ctxt_path / node.id  
        child_is_data_path = self.model_path / node.id / "is_data"
        for child_ctxt_path in [save_path / "gh", child_is_data_path]:
            if not os.path.exists(child_ctxt_path):
                os.makedirs(child_ctxt_path, exist_ok=True)
        if self.debug:
            print("[LEVEL] parent is_data level : ", parent["is"].level)
            print("[LEVEL] parent selected bin : ", parent["sel"].level)
        t0 = time.time()
        if node.id[-1] == "l":
            node.is_data = parent["is"] * parent["sel"]
        else:
            m1 = self.msg_mgr.get_mask_by_length(self.hist.ndata, encrypted=True, use_gpu=True)
            right_select_bin = m1 - parent["sel"]
            node.is_data = parent["is"] * right_select_bin
        if self.debug:
            print("[LEVEL] node.is_data level : ", node.is_data.level)
        update_gh = node.is_data * parent["gh"]
        print(f"[TIME]Update data leaf Time: {time.time() - t0:.4f} s")
        self.time += time.time() - t0
        node.is_data.save(child_is_data_path)  
        update_gh.save(save_path / "gh")
        node.gh = update_gh
        return update_gh
    def calc_weight(self, gh_matrix):
        num_slots = self.context.num_slots
        masking_msg = self.msg_mgr.get_first_slot_mask()
        GH = DataList(self.context, True)
        t0 = time.time()
        for list_idx in range(len(gh_matrix)):
            tmp_gh_block = gh_matrix[list_idx].sum_block()  
            r = 1
            while r < num_slots:
                rot_gh = tmp_gh_block << r
                tmp_gh_block += rot_gh
                r *= 2
            tmp_gh_block *= masking_msg  
            GH.append(tmp_gh_block)  
        if self.debug:
            print("[LEVEL]GH level : ", GH.level, flush=True)
        G, H = GH.twice_of_real_and_imag_parts()
        G *= 0.5
        H *= 0.5
        if G.need_bootstrap(1):
            print("[BOOTSTRAP] G level : ", G.level, flush=True)
            G.bootstrap()  
            H.bootstrap()
        tmp_denominator = H + self.reg_lambda
        denominator = tmp_denominator.inverse()
        child_nodes_weight = G * denominator
        print(f"[TIME]calc weight Time: {time.time() - t0:.4f} s")
        self.time += time.time() - t0
        if child_nodes_weight.need_bootstrap(1):
            print("[BOOTSTRAP] child_nodes_weight level : ", child_nodes_weight.level, flush=True)
            child_nodes_weight.bootstrap()
        self.make_leaf_weight(child_nodes_weight)
        save_path = self.model_path / "leaf_weights"  
        if not (save_path).exists():
            os.mkdir(save_path)
        child_nodes_weight.save(save_path)
    def pre_process(self, total_max_gain, cur_level):
        t0 = time.time()
        merged_gain = self.n_max_merge_gain(total_max_gain, cur_level)
        left_path, right_path = self.make_split_cond(merged_gain, cur_level)
        self.make_path(left_path, right_path, cur_level)
        print(f"[TIME]Preprocess Time: {time.time() - t0:.4f} s")
    def make_leaf_weight(self, leaf_weight):
        if self.one_ctxt_node > 1:
            inference_leaf_weight = DataList(
                self.context, True, [self.context.zeros(encrypted=True) for _ in range(self.leaf_node_ctxt_num)]
            )
            for i in range(self.leaf_node_ctxt_num):
                for j in range(int(self.one_ctxt_node)):
                    rot_idx = self.att_itv * j  
                    inference_leaf_weight[i] += leaf_weight[i * int(self.one_ctxt_node) + j] >> rot_idx
        else:
            inference_leaf_weight = leaf_weight
        inference_leaf_weight.save(self.inf_leaf_weight_save)
    def make_split_cond(self, mered_gains, cur_level):
        find_idx = self.findNmaxidx(mered_gains, cur_level)
        cv = find_idx.left_rotate_bin(1, self.n_max)
        ca = cv*(self.msg_mgr.mask_feature[cur_level])
        ca = ca.right_rotate_bin(1, self.n_max)
        left_path = ca * cv
        right_path = ca - left_path
        return left_path, right_path
    def rotate_copy_path(self, path_ctxt, edge_cnt):

        rot_idx = self.att_itv
        res = path_ctxt.right_rotate_bin(rot_idx, 2**edge_cnt)
        return res
    def make_path(self, left_path, right_path, cur_level):

        if cur_level >= int(self.max_depth - np.log2(self.one_ctxt_node)):
            rot_idx = int(self.att_itv * 2 ** (self.max_depth - (cur_level + 1)))
            right_path >>= rot_idx
            all_path = left_path + right_path  
            copy_cnt = self.max_depth - cur_level - 1
            dupli_path = self.rotate_copy_path(all_path, copy_cnt)  
            if cur_level == 0:
                self.sum_path_mat = dupli_path  
            else:
                self.sum_path_mat = self.sum_path_mat + dupli_path
        else:  
            tmp_data_mat = DataMatrix(self.context, True)
            for left, right in zip(left_path, right_path):
                path_copy_cnt = int(np.log2(self.one_ctxt_node))
                if path_copy_cnt < 0:
                    path_copy_cnt = 0
                right = self.rotate_copy_path(right, path_copy_cnt)
                left = self.rotate_copy_path(left, path_copy_cnt)
                ctxt_need_cnt = self.leaf_node_ctxt_num // 2 ** (cur_level)
                for j in range(ctxt_need_cnt):
                    if j < ctxt_need_cnt // 2:
                        tmp_data_mat.append(left)
                    else:
                        tmp_data_mat.append(right)
            if cur_level == 0:
                self.sum_path_mat = (
                    tmp_data_mat  
                )
            else:
                self.sum_path_mat = self.sum_path_mat + tmp_data_mat  
        if cur_level == self.max_depth - 1:
            self.sum_path_mat.save(self.tree_path_save, 3)
        return
    def maxbin2Nmax(self, gain):

        mask_block = self.msg_mgr.get_first_slot_mask()
        one_ctxt_feature_by_bin = min(
            math.floor(self.num_slot / (self.max_bin - 1)), self.d
        )  
        bin_remain = self.d - (
            one_ctxt_feature_by_bin * (len(gain) - 1)
        )  
        n_max_remain = self.d - (
            self.one_ctxt_feature * (self.ctxt_num_by_feature - 1)
        )  
        bin_chunk = [one_ctxt_feature_by_bin] * (len(gain) - 1) + [bin_remain]
        n_max_chunk = [self.one_ctxt_feature] * (self.ctxt_num_by_feature - 1) + [n_max_remain]
        result = DataList(self.context, True)
        if len(gain) > 1:  
            target = None
            n_max_chunk_idx = 0
            reset_idx = 0
            for i in len(gain):  
                rot_gain = gain[i]
                for j in range(bin_chunk[i]):  
                    if reset_idx == 0:
                        target = rot_gain * mask_block
                        rot_gain <<= self.max_bin - 1
                        reset_idx += 1
                        continue
                    tmp = rot_gain * mask_block
                    tmp >>= self.n_max * reset_idx
                    target += tmp
                    reset_idx += 1
                    if reset_idx == n_max_chunk[n_max_chunk_idx] - 1:
                        result.append(target)
                        reset_idx = 0
                        target = None
                        n_max_chunk_idx += 1
        else:  
            gain[0] = utils.left_rotate_bin(self.context, gain[0], 1, self.max_bin - 1)
            target = gain[0] * mask_block
            for i in range(1, self.d):  
                gain[0] <<= self.max_bin - 1
                tmp = gain[0] * mask_block
                tmp >>= self.n_max * i
                target += tmp
            result.append(target)
        return result
    def split_gains(self, gain):

        one_ctxt_feature_by_bin = min(
            math.floor(self.num_slot / (self.max_bin - 1)), self.d
        )  
        block_info = []
        sum_d = 0
        for i in range(len(gain)):
            front_split = 0
            full = 0
            back_split = 0
            if len(block_info) != 0:
                info = block_info[i - 1]
                front_split = self.max_bin - 1 - info[2]
            if self.d - sum_d - one_ctxt_feature_by_bin > 0:
                full = one_ctxt_feature_by_bin
                back_split = self.num_slot - front_split - full * (self.max_bin - 1)
            else:
                full = self.d - sum_d
                back_split = 0
            tmp = [front_split, full, back_split]
            block_info.append(tmp)
        result = DataList(self.context, True)
        for i in range(len(gain)):
            front_split = block_info[i][0]
            full = block_info[i][1]
            back_split = block_info[i][2]
            mask_block1 = self.msg_mgr.get_mask_by_length(front_split + full * (self.max_bin - 1))
            tmp = gain[i] * mask_block1
            if len(result) == 0:
                result.append(tmp)
            else:
                tmp >> block_info[i - 1][2]  
                result[-1] += tmp
            if back_split > 0:
                mask_block2 = self.msg_mgr.get_mask_by_length(back_split)
                tmp = gain[i] >> back_split
                tmp *= mask_block2
                result.append(tmp)
        return result
    def n_max_merge_gain(self, max_gain_list, cur_level):
        merged_node = []
        if cur_level >= int(self.max_depth - np.log2(self.one_ctxt_node)):
            can_merge = int(2 ** (cur_level - np.log2(self.leaf_node_ctxt_num)))
            repeat_num = int(len(max_gain_list) // can_merge)
            for j in range(repeat_num):
                idx = j * can_merge
                target = self.maxbin2Nmax(
                    max_gain_list[idx] * self.hist.bin_edges
                )  
                for k in range(can_merge - 1):
                    rot_idx = self.att_itv * 2 ** (self.max_depth - cur_level) * (k + 1)
                    tmp = self.maxbin2Nmax(max_gain_list[idx + k + 1] * self.hist.bin_edges)
                    tmp = tmp >> rot_idx
                    target += tmp
                merged_node.append(target)
        else:  
            for j in range(len(max_gain_list)):
                if (self.max_bin - 1) * self.d > self.num_slot:
                    splited = self.split_gains(max_gain_list[j])
                    n_maxed = self.maxbin2Nmax(splited * self.hist.bin_edges)
                    merged_node.append(splited)
                else:
                    n_maxed = self.maxbin2Nmax(max_gain_list[j] * self.hist.bin_edges)
                    merged_node.append(n_maxed)
        data_mat = DataMatrix(self.context, True, merged_node)
        return data_mat
    def findNmaxidx(self, merged_gain, cur_level):
        result = DataMatrix(self.context, True)
        for dlist in merged_gain:
            tmp = dlist.right_rotate_bin(1, self.n_max - 1)
            find_idx = self.findca(tmp, self.msg_mgr.edge_by_depth[cur_level], self.msg_mgr.edge_mask[cur_level])
            result.append(find_idx)
        return result
    def findca(self, ctxt_input, ctxt_eval, masking, num_iter_g: int = 8, num_iter_f: int = 3):
        start = time.time()
        ctxt_ca = ctxt_input.compare(ctxt_eval, numiter_g=num_iter_g, numiter_f=num_iter_f)
        if ctxt_ca.need_bootstrap(9):
            ctxt_ca.bootstrap()
        ctxt_ca = ctxt_ca * ctxt_ca
        ctxt_tmp = ctxt_ca << (1)
        ctxt_ca = ctxt_ca - ctxt_tmp
        ctxt_ca = ctxt_ca * masking
        find_ca_time = time.time() - start
        return ctxt_ca
    def predict_contribution(self):
        num_slots = self.context.num_slots
        tree_structure = utils.make_tree_structure("o", self.max_depth)
        cur_level_nodes = tree_structure[self.max_depth]
        leaf_nodes_is_data = DataMatrix(self.context, True)
        for node_name in cur_level_nodes:
            is_data_path = self.model_path / node_name / "is_data"
            tmp = DataList.from_path(self.context, encrypted=True, path=is_data_path)
            leaf_nodes_is_data.append(tmp)
        weight_path = self.model_path / "leaf_weights"
        leaf_nodes_weight = DataList.from_path(self.context, encrypted=True, path=weight_path)
        tmp_y_hat = DataMatrix(self.context, True)
        empty_block = self.context.zeros(encrypted=True)
        y_hat = DataList(self.context, True)
        n_block = math.ceil(self.hist.ndata / num_slots)
        for _ in range(n_block):
            y_hat.append(empty_block)
        if self.debug:
            print("[LEVEL] leaf_nodes_weight level : ", leaf_nodes_weight.level)
            print("[LEVEL] leaf_nodes_is_data level : ", leaf_nodes_is_data.level)
        t0 = time.time()
        copy_leaf_nodes_weight = leaf_nodes_weight.right_rotate_bin(1, min(num_slots, self.hist.ndata))  

        for li_idx in range(len(copy_leaf_nodes_weight)):
            tmp_y_hat.append(copy_leaf_nodes_weight[li_idx] * leaf_nodes_is_data[li_idx])  
        for idx in range(len(tmp_y_hat)):  
            y_hat = y_hat + tmp_y_hat[idx]  
        if y_hat.need_bootstrap(1):
            
            y_hat.bootstrap()
        total_time = time.time() - t0
        print(f"[TIME]Predict contribution Time: {total_time:.4f} s")
        return y_hat, total_time
    def get_tree_structure(self):
        def traverse(nd, nodeid, depth):
            s = '{ "nodeid": ' + str(nodeid) + ', "depth": ' + str(depth)
            if nd.is_leaf():
                s += ', "leaf": ' + str(nd.leaf)
                s += ', "cover": ' + str(nd.cover)
                s += " }"
                return s
            else:
                s += ', "split": "f' + str(nd.fid) + '", '
                s += '"split_condition": ' + str(nd.split_point)
                s += ', "gain": ' + str(nd.gain)
                s += ', "cover": ' + str(nd.cover)
                left_str = traverse(nd.left, 2 * nodeid + 1, depth + 1)
                right_str = traverse(nd.right, 2 * nodeid + 2, depth + 1)
                s += ', "children": [' + left_str + ", " + right_str + "] }"
                return s
        return traverse(self.estimator, 0, 0)