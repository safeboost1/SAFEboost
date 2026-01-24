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
class Server:
    def __init__(self, context, n_estimators=1, max_depth=2, n_bits=None, d=None, hamming_weight=2):
        self.debug = False
        self.context = context
        self.num_slot = self.context.num_slots
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.precision_bits = n_bits
        self.cw_ele = self.precision_bits + 1
        self.code_length = utils.find_code_length(self.precision_bits, hamming_weight)
        self.d = d
        self.hamming = hamming_weight
        self.main_folder = "./levelup_test"
        self.node_itv = self.code_length * self.cw_ele * d
        self.one_ct_node = min(
            2 ** (self.max_depth) - 1, math.floor(self.num_slot / (self.node_itv))
        )  
        self.one_ct_features = min(
            self.d, math.floor(self.num_slot / self.code_length)
        )  
        self.n_ct_one_node = math.ceil(self.node_itv / self.num_slot)  
        self.n_tree_one_ct = max(
            1, min(self.n_estimators, math.floor(self.num_slot / (self.node_itv * (2 ** (self.max_depth) - 1))))
        )  
        self.save_path = Path(f"{self.main_folder}/eval_ctxt")  
        print("\n=== server Hyperparameters ===")
        print(f"n_estimators       : {self.n_estimators}")
        print(f"max_depth          : {self.max_depth}")
        print(f"precision_bits     : {self.precision_bits}")
        print(f"code_length        : {self.code_length}")
        print(f"d (num features)   : {self.d}")
        print(f"hamming_weight     : {self.hamming}")
        print(f"num_slot           : {self.num_slot}")
        print(f"node_itv           : {self.node_itv}")   
        print(f"one_ct_node        : {self.one_ct_node}")  
        print(f"one_ct_features    : {self.one_ct_features}")  
        print(f"n_ct_one_node      : {self.n_ct_one_node}")  
        print(f"n_tree_one_ct      : {self.n_tree_one_ct}")  
        print(f"save_path          : {self.save_path}")
        self.precomput()
    def precomput(self):
        self.first_slot_mask = Block(self.context, encrypted=False, data=[1])
        self.inverse_ham = 1 / (math.factorial(self.hamming))  
        use_slots = self.node_itv * self.one_ct_node * self.n_tree_one_ct
        self.node_first_slot_mask = Block(
            self.context, encrypted=False, data=[1 if _ % self.node_itv == 0 else 0 for _ in range(0, use_slots)]
        )
        self.arith_mask = Block(
            self.context, encrypted=False, data=[1 if _ % self.code_length == 0 else 0 for _ in range(0, use_slots)]
        )
        self.tree_first_slot_mask = Block(
            self.context,
            encrypted=False,
            data=[1 if _ % (self.node_itv * self.one_ct_node) == 0 else 0 for _ in range(0, use_slots)],
        )
        self.random_msg = Block(
            self.context, encrypted=False, data=[random.randint(50, 100) for _ in range(self.num_slot)]
        )
    def build_node_position_dict(self, encoded_conds: list) -> dict:

        self.position_dict = {}
        iter = math.ceil((2 ** (self.max_depth) - 1) / self.one_ct_node)
        for i in range(0, iter):
            for j in range(self.one_ct_node):
                self.position_dict[i * self.one_ct_node + j] = (i, j)
        return self.position_dict
    def encode_model(self, json_path):
        with open(json_path, "r") as f:
            model = json.load(f)
        encoded_conds = [[] for _ in range(self.n_ct_one_node)]  
        for depth, splits in model.items():
            for split in splits:
                feature_str = split["feature"]
                bin_idx = int(split["condition"])
                try:
                    feature_index = int(feature_str[1:])  
                except ValueError:
                    print(f"❗ Warning: Invalid feature name '{feature_str}'")
                    continue
                tmp = self.encode_split_condition(feature_index, bin_idx)
                for i in range(len(tmp)):
                    encoded_conds[i].append(tmp[i])
        node_position_dict = self.build_node_position_dict(encoded_conds)

        if self.one_ct_node >= 2:  
            grouped_vectors = []
            for i in range(0, len(encoded_conds[0]), self.one_ct_node):
                group = encoded_conds[0][i : i + self.one_ct_node]
                concat_vec = np.concatenate(group, axis=0)
                grouped_vectors.append(concat_vec)
            encoded_conds[0] = grouped_vectors
            return encoded_conds
        else:
            return encoded_conds
    def encode_split_condition(self, feature_index, split_bin_idx):

        vec = np.full(self.d, None, dtype=object)
        vec[feature_index] = split_bin_idx
        encoded_vec_parts = []
        for i in range(self.d):
            pe_encoded = utils.PE(vec[i], self.precision_bits)
            tmp = []
            for j in reversed(range(self.cw_ele)):
                cw_encoded = utils.get_OP_CHaW_encoding(pe_encoded[j], self.code_length, self.hamming)
                tmp.append(cw_encoded)
            encoded_vec_parts.append(np.concatenate(tmp, axis=0))
        encoded_vec = np.concatenate(encoded_vec_parts, axis=0)  
        if len(encoded_vec) <= self.cw_ele * self.code_length * self.one_ct_features:
            return [encoded_vec]
        else:
            return utils.split_vector(encoded_vec, self.cw_ele * self.code_length * self.one_ct_features)
    def merge_models(self, models):

        merged = []
        if self.n_tree_one_ct >= 2:
            flat_models = [model[0][0] for model in models]  
            if self.debug:
                print("flat:", flat_models)
            for i in range(0, len(flat_models), self.n_tree_one_ct):
                group = flat_models[i : i + self.n_tree_one_ct]
                concat_vec = np.concatenate(group, axis=0)
                merged.append([[concat_vec]])
            return merged
        else:
            return models
    def ArithCWEqOp(self, input, models):

        res = DataMatrix(self.context, True)
        for i in range(len(models)):
            tmp_dmat = DataMatrix(self.context, True)
            for dlist, block in zip(models[i], input):
                tmp = dlist * block
                tmp_dmat.append(tmp)
            if self.debug:
                print("mult_with input")
                utils.print_matrix_b(tmp_dmat, self.node_itv * 3)
            tmp_dlist = DataList(self.context, True, sum(tmp_dmat))  
            tmp_dlist = tmp_dlist.left_rotate_bin(1, self.code_length)
            if self.debug:
                print("sum_mult_Res")
                utils.print_list_b(tmp_dlist, self.node_itv * 3)
            tmp_list = [tmp_dlist]
            for j in range(math.ceil(math.log2(self.hamming))):
                tmp2 = tmp_list[j] - self.arith_mask
                if self.debug:
                    print("tmp2")
                    utils.print_list_b(tmp2, self.node_itv * 3)
                tmp_list.append(tmp2)
            tmp3 = utils.tournament_product(tmp_list)
            if self.debug:
                print("tmp3")
                utils.print_list_b(tmp3, self.node_itv * 3)
            tmp3 *= self.inverse_ham
            tmp3 *= self.arith_mask
            if self.debug:
                print("tmp4")
                utils.print_list_b(tmp3, self.node_itv * 3)
            tmp3 = tmp3.left_rotate_bin(1, self.node_itv)
            res.append(tmp3)
        return res
    def sum_comp_res(self, comp_res):

        result = DataList(self.context, True)
        for dlist in comp_res:  
            tmp_list = []
            for i in range(2 ** (self.max_depth) - 1):
                ct_idx, inner_idx = self.position_dict[i]
                curr = dlist[ct_idx] << (inner_idx * self.node_itv)
                if self.debug:
                    print("right")
                    utils.print_ctxt_b(curr, self.node_itv * 3, -1)
                curr_comple = -(curr - self.node_first_slot_mask)
                if len(tmp_list) != 0:
                    parent = tmp_list.pop(0)
                    if self.debug:
                        print("paret")
                        utils.print_ctxt_b(parent, self.node_itv * self.one_ct_node, 0.9)
                    tmp_list.append(curr_comple + parent)
                    tmp_list.append(curr + parent)
                else:
                    tmp_list.append(curr_comple)
                    tmp_list.append(curr)
            for j in range(2 ** (self.max_depth)):
                tmp_list[j] = tmp_list[j] * self.tree_first_slot_mask
                if self.debug:
                    print("tmp_list")
                    utils.print_ctxt_b(tmp_list[j], 2, -1)
                tmp_list[j] = tmp_list[j] >> j
            result.append(sum(tmp_list))
        result *= self.random_msg
        return result
    def model2ctxt(self, models):
        res = []
        for dmat in models:
            tmp = DataMatrix(self.context, True, dmat)
            tmp.level_down(5 + int(math.log2(self.hamming)))
            if self.context.with_gpu:
                tmp.to_device()
            res.append(tmp)
        return res
    def eval_tree(self, input, split_cond_path):
        models = []
        model_path = os.path.join(split_cond_path, "splitcond.json")
        encoded_model = self.encode_model(model_path)
        for n_tree in range(self.n_estimators):
            models.append(encoded_model)
        merged_models = self.merge_models(models)
        if self.debug:
            print("merged_models")
            print(merged_models)
        ct_models = self.model2ctxt(merged_models)
        if self.context.with_gpu:
            input.to_device()
        input.level_down(5 + int(math.log2(self.hamming)))
        start = time.time()
        comp_res = self.ArithCWEqOp(input, ct_models)
        if self.debug:
            print("comp_res")
            print(comp_res)
            utils.print_matrix_b(comp_res, self.node_itv)
        ct_pred = self.sum_comp_res(comp_res)
        end = time.time()
        if self.debug:
            print("ct_pred")
            utils.print_list_b(ct_pred, self.node_itv * 3, -1)
        return ct_pred, end - start