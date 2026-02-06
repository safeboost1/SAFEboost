import math
import random
from pathlib import Path
import numpy as np
import utils
from dataset.data_set import DataList
class Client:
    def __init__(self, context, n_estimators=1, max_depth=2, n_bits=None, d=None, hamming_weight=2):
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
        self.n_tree_one_ct = min(
            self.n_estimators, math.floor(self.num_slot / (self.node_itv * (2 ** (self.max_depth) - 1)))
        )  
        self.save_path = Path(f"{self.main_folder}/eval_ctxt")  
        print("\n=== Hyperparameters ===")
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
    def generate_random_input(self):
        input_list = []
        for i in range(self.d):
            val = random.randint(0, 2 ** (self.precision_bits) - 1)
            input_list.append(val)
        return input_list
    def encode_input(self, input):
        encoded_vec_parts = []
        largest = 1 << self.precision_bits
        for i in range(self.d):
            rc_encoded = utils.comp_range_largest(input[i], (1 << self.precision_bits) - 1, self.precision_bits)
            tmp = []
            for j in reversed(range(self.precision_bits + 1)):
                if rc_encoded[j] == largest:
                    tmp.append([0] * self.code_length)
                else:
                    cw_encoded = utils.get_OP_CHaW_encoding(rc_encoded[j], self.code_length, self.hamming)
                    tmp.append(cw_encoded)
            encoded_vec_parts.append(np.concatenate(tmp, axis=0))
        encoded_vec = np.concatenate(encoded_vec_parts, axis=0)  
        if len(encoded_vec) <= self.cw_ele * self.code_length * self.one_ct_features:
            return [encoded_vec]
        else:
            return utils.split_vector(encoded_vec, self.cw_ele * self.code_length * self.one_ct_features)
    def copy_input(self, encoded_input):
        if self.one_ct_node > 1:
            if self.n_tree_one_ct > 1:
                copyed_input = encoded_input.right_rotate_bin(self.node_itv, self.one_ct_node * self.n_tree_one_ct)
            else:
                copyed_input = encoded_input.right_rotate_bin(self.node_itv, self.one_ct_node)
            return copyed_input
        else:
            return encoded_input
    def make_input(self):
        rand_input = self.generate_random_input()
        encoded_input = self.encode_input(rand_input)
        ct_input = DataList(self.context, encrypted=True, data_list=encoded_input)
        ct_res = self.copy_input(ct_input)
        return rand_input, ct_res