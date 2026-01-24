import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import utils
from dataset.data_set import DataList, DataMatrix
from heaan_stat.core import Block
class MyXGBClassifier:
    def __init__(self, context, n_estimators=1, max_depth=2, learning_rate=0.3,d=None, n_max=None):
        self.context = context
        self.num_slot = self.context.num_slots
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.eta = learning_rate
        self.d=d
        self.n_max = n_max
        print("\n=== Hyperparameters ===")
        print(f"n_estimators: {self.n_estimators}")
        print(f"max_depth: {self.max_depth}")
        print(f"learning_rate: {self.eta}")
        print(f"n_max: {self.n_max}")
    def copy_input(self, input_ctxt):

        rot_idx = self.node_itv
        if self.one_ctxt_node > 1:
            copyed_input = input_ctxt.right_rotate_bin(rot_idx, self.one_ctxt_node)
            if self.one_ctxt_batch_tree > 1:
                copyed_input = copyed_input.right_rotate_bin(
                    self.node_itv * 2**self.max_depth, self.one_ctxt_batch_tree
                )
        else:
            copyed_input = input_ctxt
        return copyed_input
    def generate_random_input(self):
        input_list = []
        for i in range(self.d):
            val = random.randint(0, self.n_max-1)
            input_list.append(val)
        return input_list
    def encoding_input(self, input):
        onehot = []
        for val in input:
            v = [0] * (self.n_max)
            v[val] = 1
            onehot.extend(v)
      
        if len(onehot) <= self.num_slot:
            split_row = [onehot]
        else:
            cut_idx = math.floor(self.num_slot / self.n_max)
            split_row = [onehot[i : i + self.num_slot] for i in range(0, len(onehot), cut_idx)]
        return  DataList(self.context, True, split_row)
    def rotate_copy_path(self, path_ctxt, edge_cnt):

        rot_idx = self.node_itv
        res = path_ctxt.right_rotate_bin(rot_idx, 2**edge_cnt)
        return res
    def load_path(self, n_tree):
        path_matirx = DataMatrix(context=self.context, encrypted=True)
        cacv_path = Path(f"{self.save_path}/tree{str(n_tree)}/path")
        path_matirx.load(cacv_path, False)
        if path_matirx.encrypted:
            if path_matirx.level < self.need_depth:
                path_matirx.bootstrap()
                path_matirx.level_down(self.need_depth)
            else:
                path_matirx.level_down(self.need_depth)
        return path_matirx
    def plain_make_path(self, cacv_mat):
        for i in range(self.max_depth):
            cacv_per_depth = cacv_mat[i]
            self.make_path(cacv_per_depth, i)

        return self.sum_path_mat.encrypt()
    def encode_split_condition(self, feature_index, split_bin_idx):

        total_size = self.node_itv  
        vec_size = max(self.num_slot, total_size)
        vec = np.zeros(vec_size, dtype=int)
        base_idx = feature_index * self.n_max
        for b in range(split_bin_idx):
            vec[base_idx + b] = 1
        return vec
    def split_vector(self, vec, cut_idx):
        chunks = []
        for i in range(0, len(vec), cut_idx):
            chunk = vec[i : i + cut_idx]
            if len(chunk) < cut_idx:
                pad = np.zeros(cut_idx - len(chunk), dtype=vec.dtype)
                chunk = np.concatenate([chunk, pad])
            chunks.append(chunk)
        return chunks
    def extract_leaf_values(self, node):
        leaves = []
        if "leaf" in node:
            leaves.append(node["leaf"])
        elif "children" in node:
            for child in node["children"]:
                leaves.extend(self.extract_leaf_values(child))
        return leaves
    def extract_leaf_values_from_json(self, json_path):

        with open(json_path, "r") as f:
            tree_data = json.load(f)
        all_leaves = []
        if isinstance(tree_data, list):
            for tree in tree_data:
                all_leaves.append(self.extract_leaf_values(tree))
            return all_leaves
        else:
            return self.extract_leaf_values(tree_data)
    def left_rotate_list(self, my_list, a):
        rotated_list = my_list[a:] + my_list[:a]
        return rotated_list
    def make_right_path(self, split_feature):  
        total_size = self.node_itv  
        vec_size = max(self.num_slot, total_size)
        vec = np.zeros(vec_size, dtype=int)
        base_idx = split_feature * self.n_max
        vec[base_idx : base_idx + self.n_max] = [1] * self.n_max
        return vec
    def encode_leaf_weight(self, leaf_values):
        stride = min(self.node_itv, self.num_slot)
        vec_size = stride * len(leaf_values)  
        vec = np.zeros(vec_size, dtype=float)
        for i, val in enumerate(leaf_values):
            vec[i * stride] = val
        if self.one_ctxt_node >= 1:
            cut_idx = int(stride * self.one_ctxt_node)
        else:
            cut_idx = self.num_slot
        split_list = [data.tolist() for data in self.split_vector(vec, cut_idx)]
        cy_list = DataList(self.context, True, split_list)
        return cy_list
    def encode_split_cond(self, json_path):

        with open(json_path, "r") as f:
            depth_data = json.load(f)
        split_result = []
        right_path_res = []
        for depth, splits in depth_data.items():
            split_group = []  
            right_split_group = []
            for split in splits:
                feature_str = split["feature"]
                bin_idx = int(split["condition"])
                try:
                    feature_index = int(feature_str[1:])  
                except ValueError:
                    print(f"❗ Warning: Invalid feature name '{feature_str}'")
                    continue
                full_vec = self.encode_split_condition(feature_index, bin_idx)
                right_path = self.make_right_path(feature_index)
                if self.node_itv <= self.num_slot:
                    sliced_vecs = self.split_vector(full_vec, self.num_slot)
                    sliced_vecs2 = self.split_vector(right_path, self.num_slot)
                elif self.n_max <= self.num_slot:
                    sliced_vecs = []
                    sliced_vecs2 = []
                    n_d = math.floor(self.num_slot / self.n_max)
                    for i in range(0, len(full_vec), n_d * self.n_max):
                        sliced_vecs.append(full_vec[i : i + (n_d * self.n_max)])
                    for i in range(0, len(right_path), n_d * self.n_max):
                        sliced_vecs.append(right_path[i : i + (n_d * self.n_max)])
                else:
                    sliced_vecs = []
                    for i in range(0, len(full_vec), self.n_max):
                        tmp = full_vec[i : i + self.n_max]
                        tmp2 = self.split_vector(tmp, self.num_slot)
                        for li in tmp2:
                            sliced_vecs.append(li)
                split_group.append(sliced_vecs)
                right_split_group.append(sliced_vecs2)
            split_result.append(split_group)
            right_path_res.append(right_split_group)
        return split_result, right_path_res
    def split_cond_merge(self, split_conds, right_paths):
        merged_cacv = []
        self.right_path_mat = []
        for i in range(self.max_depth):
            print("depth:", i)
            merged_node = []
            tmp_merged_right = []
            if i >= int(self.max_depth - np.log2(self.one_ctxt_node)):
                can_merge = int(2 ** (i - np.log2(self.leaf_node_ctxt_num)))
                repeat_num = int(len(split_conds[i]) // can_merge)
                for j in range(repeat_num):
                    idx = j * can_merge
                    target = split_conds[i][idx]
                    right_target = right_paths[i][idx]
                    for k in range(can_merge - 1):
                        rot_idx = self.node_itv * 2 ** (self.max_depth - i) * (k + 1)
                        for l in range(len(split_conds[i][idx])):
                            rotated = np.roll(split_conds[i][idx + k + 1][l], rot_idx)
                            right_rotated = np.roll(right_paths[i][idx + k + 1][l], rot_idx)
                            target[l] += rotated
                            right_target[l] += right_rotated
                    list_target = [data.tolist() for data in target]
                    list_right_target = [data.tolist() for data in right_target]
                    merged_node.append(list_target)
                    tmp_merged_right.append(list_right_target)
            else:  
                for j in range(len(split_conds[i])):
                    merged_node.append([data.tolist() for data in split_conds[i][j]])
                    tmp_merged_right.append([data.tolist() for data in right_paths[i][j]])
            data_mat = DataMatrix(self.context, False, merged_node)
            tmp_right_mat = DataMatrix(self.context, False, tmp_merged_right)
            merged_cacv.append(data_mat)
            self.right_path_mat.append(tmp_right_mat)
        return merged_cacv
    def make_path(self, cacv_mat, cur_level):

        if cur_level >= int(self.max_depth - np.log2(self.one_ctxt_node)):
            left_path = cacv_mat
            rot_idx = self.node_itv * 2 ** (self.max_depth - (cur_level + 1))
            right_path = (
                (self.right_path_mat[cur_level] - left_path) >> rot_idx
            )  
            all_path = left_path + right_path  
            copy_cnt = self.max_depth - cur_level - 1
            dupli_path = self.rotate_copy_path(all_path, copy_cnt)  
            if cur_level == 0:
                self.sum_path_mat = dupli_path  
            else:
                self.sum_path_mat = self.sum_path_mat + dupli_path
       
        else:  
            tmp_data_mat = DataMatrix(self.context, True)
            for left, right_msg in zip(cacv_mat, self.right_path_mat[cur_level]):
                right_path = right_msg - left
                path_copy_cnt = int(np.log2(self.one_ctxt_node))
                if path_copy_cnt < 0:
                    path_copy_cnt = 0
                right_path = self.rotate_copy_path(right_path, path_copy_cnt)
                left_path = self.rotate_copy_path(left, path_copy_cnt)
                ctxt_need_cnt = self.leaf_node_ctxt_num // 2 ** (cur_level)
                for j in range(ctxt_need_cnt):
                    if j < ctxt_need_cnt // 2:
                        tmp_data_mat.append(left_path)
                    else:
                        tmp_data_mat.append(right_path)
            if cur_level == 0:
                self.sum_path_mat = tmp_data_mat  
            else:
                self.sum_path_mat = self.sum_path_mat + tmp_data_mat  
        return
    def cal_path(self, copyed_data, path_list):  
        c1 =path_list*copyed_data
        sum_node = DataList(self.context, True, [sum(row) for row in c1])
        sum_rot = min(self.node_itv, self.num_slot)
        sum_node = sum_node.left_rotate_bin(1, sum_rot)
        sum_node -= self.max_depth
        sum_node *= self.inf_masking_msg
        return sum_node
    def get_predict(self, path, leaf_weight):  

        path += leaf_weight
        return path
    def predict(self, model_path,input):
        num_slots = self.num_slot
        self.need_depth = 2
        self.t = 1  
        self.d = self.d
        self.n_max = self.n_max
        self.leaf_node_num = 2**self.max_depth
        print("self.n_max", self.n_max)
        print("self.d:", self.d)
        self.node_itv = self.n_max * self.d
        print("self.node_itv:", self.node_itv)
        self.ctxt_num_by_feature = math.ceil(self.node_itv / num_slots)
        print("ctxt_num_by_feature:", self.ctxt_num_by_feature)
        self.leaf_node_ctxt_num = utils.get_smallest_pow_2(
            int((np.ceil((self.node_itv * self.leaf_node_num) / num_slots)) // self.ctxt_num_by_feature)
        )
        self.one_ctxt_node = self.leaf_node_num / (
            self.leaf_node_ctxt_num * self.ctxt_num_by_feature
        )  
        self.one_node_trees = (
            self.node_itv
        )  
        self.one_res_block = (
            self.num_slot // (self.node_itv * self.leaf_node_num)
        )  
        self.one_ctxt_batch_tree = self.num_slot // (self.node_itv * self.leaf_node_num // self.leaf_node_ctxt_num)
        self.one_ctxt_trees = min(
            (self.num_slot // (self.node_itv * self.leaf_node_num // self.leaf_node_ctxt_num)) * self.one_node_trees,
            self.n_estimators,
        )
        print("leaf_ctxt_num:", self.leaf_node_ctxt_num)
        print("one_ctxt_node", self.one_ctxt_node)
        print("one_node_trees:", self.one_node_trees)
        print("one_res_block:", self.one_res_block)
        print("one_ctxt_trees:", self.one_ctxt_trees)
        print("one_ctxt_batch_trees:", self.one_ctxt_batch_tree)
        self.k = (
            self.t * 100
        )  
        m = [0] * num_slots
        if self.one_ctxt_node > 1:
            for i in range(int(self.one_ctxt_node) * self.one_ctxt_batch_tree):
                m[self.node_itv * i] = random.randint(50, 100)
        else:
            m[0] = random.randint(50, 100)
        self.inf_masking_msg = Block(self.context, encrypted=False, data=m)  
        m1 = [1] * (num_slots)
        if self.one_ctxt_node > 1:
            for i in range(0, int(self.one_ctxt_node * self.node_itv), self.node_itv):
                m1[i : i + self.n_estimators] = [0] * self.n_estimators
        else:
            m1[: self.n_estimators] = [0] * self.n_estimators
        self.inf_msg1 = Block(self.context, encrypted=False, data=m1)
        cy_json_path = os.path.join(model_path, "model.json")
        cacv_json_path = os.path.join(model_path, "splitcond.json")
        plain_cacv_mat, plain_right_paths = self.encode_split_cond(cacv_json_path)
        cacv_mat = self.split_cond_merge(plain_cacv_mat, plain_right_paths)
        inference_path = self.plain_make_path(cacv_mat)
        cy_list = self.extract_leaf_values_from_json(cy_json_path)
        y_value = self.encode_leaf_weight(cy_list)
        path_n_leaf = {}
        for n_tree in range(self.n_estimators):
            path_n_leaf[n_tree] = {"path": inference_path, "leaf": y_value}
        merged_leaf = DataMatrix(self.context, True)
        tmp = None
        for i in range(self.n_estimators):
            cur_block = i % self.one_ctxt_batch_tree
            rot_idx = i // self.one_ctxt_batch_tree
            block_rot = cur_block * self.node_itv * 2**self.max_depth
            leaf_val = path_n_leaf[i]["leaf"]
            if tmp is None:
                tmp = leaf_val
            else:
                leaf_val >>= block_rot
                tmp += leaf_val >> rot_idx
        if tmp is not None:
            merged_leaf.append(tmp)
        merged_path = DataMatrix(self.context, True)
        tmp = None
        for i in range(self.n_estimators):
            cur_block = i % self.one_ctxt_batch_tree
            rot_idx = i // self.one_ctxt_batch_tree
            if i != 0 and cur_block == 0:
                merged_path.append(tmp)
                tmp = None
            block_rot = cur_block * self.node_itv * 2**self.max_depth
            leaf_val = path_n_leaf[i]["path"]
            if tmp is None:
                tmp = leaf_val
            else:
                tmp += leaf_val >> block_rot
        if tmp is not None:
            merged_path.append(tmp)
        total_time = 0
        encoded_input=self.encoding_input(input)
        encoded_input.level_down(self.need_depth)
        copy_ctxt_li = self.copy_input(encoded_input)  
        total_pred = DataMatrix(self.context, True)
        tmp_pred = None
        for n_path in range(len(merged_path)):
            path_list = merged_path[n_path]
            copy_ctxt = copy_ctxt_li.deepcopy()
            tmp_time = time.time()
            total_path = self.cal_path(copy_ctxt, path_list) 
            total_time += time.time() - tmp_time
            print("cal_path_time:", time.time() - tmp_time)
            if tmp_pred is None:
                tmp_pred = total_path
            else:
                tmp_pred += total_path >> n_path
        if tmp_pred is not None:
            total_pred.append(tmp_pred)
            print("append done")
        tmp_time = time.time()
        y_predict = self.get_predict(total_pred, merged_leaf)  
        total_time += time.time() - tmp_time
        print("get_pred_time:", time.time() - tmp_time)
        print("inference done", flush=True)
        self.inf_time = total_time
        return y_predict,total_time