import gc
import json
import math
import os
import random
import time
from pathlib import Path
import HE_codes.utils as utils
import numpy as np
from dataset.data_set import DataList, DataMatrix
from HE_codes.decision_tree import MyXGBClassificationTree
from HE_codes.msg_manager import MsgManager
from heaan_stat.core.block import Block
class MyXGBClassifier:
    def __init__(
        self,
        context,
        msg_mgr=None,
        n_estimators=1,
        max_depth=2,
        learning_rate=0.3,
        prune_gamma=0.0,
        reg_lambda=1.0,
        base_score=0.5,
        tree_method="hist",
        max_bin=4,
        main_folder="./result",
    ):
        self.context = context
        self.msg_mgr = msg_mgr or MsgManager(context)
        self.num_slot = self.context.num_slots
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.eta = learning_rate
        self.prune_gamma = prune_gamma
        self.reg_lambda = reg_lambda
        self.base_score = base_score
        self.tree_method = tree_method
        self.max_bin = max_bin
        self.debug = False
        self.models = []
        self.main_folder = main_folder
        self.predict_ctxt_path = self.main_folder + "/predict_ctxt"  
        self.save_path = Path(f"{self.main_folder}/eval_ctxt")
        print("\n=== Hyperparameters ===")
        print(f"n_estimators: {self.n_estimators}")
        print(f"max_depth: {self.max_depth}")
        print(f"learning_rate: {self.eta}")
        print(f"reg_lambda: {self.reg_lambda}")
        print(f"max_bin: {self.max_bin}")
        print(f"prune_gamma: {self.prune_gamma}")
    def F2P(self, y_hat, ndata):
        masking_msg = self.msg_mgr.get_mask_by_length(ndata, encrypted=True, use_gpu=True)
        start = time.time()

        if y_hat.need_bootstrap(1):
         
            y_hat.bootstrap()
        res = DataList(self.context, True)
        for x in y_hat:
            tmp = x.sigmoid()  
            res.append(tmp)
        
        if res.need_bootstrap(3):
            
            res.bootstrap()
        res = res * masking_msg  
        if res.need_bootstrap(2):
            
            res.bootstrap()
        F2P_time = time.time() - start
        self.time += F2P_time
        
        return res  
    def pre_make_msg(self, ndata, B, F, max_depth):
        self.msg_mgr.get_mask_by_length(B * F)
        self.msg_mgr.get_gain_mask(B, F)
        self.msg_mgr.get_mask_by_length(ndata, encrypted=True, use_gpu=True)
        self.msg_mgr.get_feature_mask(self.max_depth, self.n_max, F, self.one_ctxt_node, self.one_ctxt_feature)
        self.msg_mgr.bin_edges_by_depth(
            self.n_max, F, max_depth, self.node_itv, self.one_ctxt_node, self.leaf_node_ctxt_num
        )
        self.msg_mgr.bin_edges_mask(
            F, self.n_max, max_depth, self.node_itv, self.one_ctxt_node, self.leaf_node_ctxt_num
        )
        self.msg_mgr.find_max_pos_msg((B - 1) * F, max_depth)
        self.msg_mgr.scale_max_msg((B - 1) * F)
    def load_path(self, n_tree):
        path_matirx = DataMatrix(context=self.context, encrypted=True)
        cacv_path = Path(f"{self.save_path}/tree{str(n_tree)}/path")
     
        path_matirx.load(cacv_path, True)
   
        if path_matirx.encrypted:
            if path_matirx.level < self.need_depth:
                path_matirx.bootstrap()
                path_matirx.level_down(self.need_depth)
            else:
                path_matirx.level_down(self.need_depth)
        return path_matirx
    def load_leaf_weight(self, n_tree):  
        cy_path = Path(f"{self.save_path}/tree{n_tree}/leaf_weight")
        cy_ctxt_li = DataList.from_path(self.context, cy_path, True)
        if cy_ctxt_li.encrypted:
            print("enc ==================================")
            for cy_ctxt in cy_ctxt_li:
                if cy_ctxt.level < self.need_depth:
                    cy_ctxt.bootstrap()
                    cy_ctxt.level_down(self.need_depth)
                else:
                    cy_ctxt.level_down(self.need_depth)
        return cy_ctxt_li
    def merge_models(self):
        path_n_leaf = {}
        for n_tree in range(self.n_estimators):
            path_ctxt_li = self.load_path(n_tree + 1)  
            leaf_ctxt_li = self.load_leaf_weight(n_tree + 1)  
            path_n_leaf[n_tree] = {"path": path_ctxt_li, "leaf": leaf_ctxt_li}
        self.merged_leaf = DataMatrix(self.context, True)
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
            self.merged_leaf.append(tmp)
        self.merged_path = DataMatrix(self.context, True)
        tmp = None
        for i in range(self.n_estimators):
            cur_block = i % self.one_ctxt_batch_tree
            rot_idx = i // self.one_ctxt_batch_tree
            if i != 0 and cur_block == 0:
                self.merged_path.append(tmp)
                tmp = None
            block_rot = cur_block * self.node_itv * 2**self.max_depth
            path = path_n_leaf[i]["path"]
            if tmp is None:
                tmp = path
            else:
                tmp += path >> block_rot
        if tmp is not None:
            self.merged_path.append(tmp)
            return
    def fit(self, histogram):
        print("=== XGBoost Training ===")
        self.time = 0
        Fm = histogram.Fm
        self.d = histogram.d
        num_slots = self.num_slot
        self.need_depth = 2
        self.n_max = histogram.n_max
        self.max_bin = histogram.max_bin
        self.leaf_node_num = 2**self.max_depth

        self.node_itv = self.n_max * self.d
        
        self.one_ctxt_feature = min(math.floor(self.num_slot / self.d), self.d)
        self.ctxt_num_by_feature = math.ceil(self.node_itv / num_slots)
      
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
        self.pre_make_msg(histogram.ndata, self.max_bin, histogram.d, self.max_depth)
        masking_msg = self.msg_mgr.get_mask_by_length(histogram.ndata, encrypted=True, use_gpu=True)
        B = self.max_bin - 1
        F = histogram.d
      
        scaling_coef = ((1 << 18) / (9 / 128 * (histogram.ndata**4) * B * F)) ** 0.25
        for tree_idx in range(self.n_estimators):
            print(f"=== {tree_idx}th Tree ===")
            tree = MyXGBClassificationTree(
                context=histogram.context,
                msg_mgr=self.msg_mgr,
                tree_idx=tree_idx,
                max_depth=self.max_depth,
                reg_lambda=self.reg_lambda,
                prune_gamma=self.prune_gamma,
                tree_method=self.tree_method,
                n_bins=self.max_bin,
                main_folder=self.main_folder,
            )
            tree.fit_from_histogram(histogram)
            print(f"[TIME]{tree_idx}th Tree Train Time: {tree.time:.4f} s")
            self.time += tree.time
            gamma, pred_time = tree.predict_contribution()  
            self.time += pred_time
            hist_update_time = 0
            t0 = time.time()
            Fm += self.eta * gamma
            histogram.Fm = Fm
            hist_update_time += time.time() - t0
            y_hat = self.F2P(Fm, histogram.ndata)  
            histogram.y_hat = y_hat
            t1 = time.time()
            residual = histogram.y - y_hat  
            update_h = y_hat * (masking_msg - y_hat)
            if update_h.need_bootstrap(2):
                
                update_h.bootstrap()
            scaled_g = residual * scaling_coef
            scaled_h = update_h * scaling_coef
            if scaled_h.need_bootstrap(2):
                
                if isinstance(scaled_h, Block):
                    Block.bootstrap_two_ctxts(scaled_g, scaled_h)
                else:
                    scaled_g.bootstrap()
                    scaled_h.bootstrap()
            scaled_gh = utils.combine_real_imag_to_complex(scaled_g, scaled_h)
            hist_update_time += time.time() - t1
            self.time += hist_update_time
            
            histogram.gh = scaled_gh
            del residual, update_h, scaled_g, scaled_h, scaled_gh, tree, gamma
            gc.collect()
            histogram.save(tree_idx=tree_idx + 2)
        print('[HE Training] Time:',self.time)
        self.merge_models()
        return
    def make_corr_leafscore(self):  
        bit_counts = utils.compute_bit_counts(self.max_depth)
        tmp_data_list = []
        for i in range(self.leaf_node_ctxt_num):
            tmp_list = [0] * self.num_slot
            if self.one_ctxt_node > 1:
                for j in range(0, int(self.one_ctxt_node)):
                    sub_num = (
                        bit_counts[j + i * self.one_ctxt_node]["zeros"]
                        + bit_counts[j + i * self.one_ctxt_node]["ones"] * self.d
                    )
                    interval = self.node_itv * j
                    tmp_list[interval] = sub_num
            else:
                sub_num = bit_counts[i]["zeros"] + bit_counts[i]["ones"] * self.d
                tmp_list[0] = sub_num
            tmp_data_list.append(tmp_list)
        sub_path_li = DataList(self.context, encrypted=False, data_list=tmp_data_list)
        return sub_path_li
    def save_predict(self, predict, idx):  
        y_pred_path = self.predict_ctxt_path
        rows = "row_" + str(idx)  
        pred_save_path = os.path.join(y_pred_path, rows)  
        utils.ensure_directory_exists(pred_save_path)  

        predict.save(pred_save_path)
        return
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
    def rotate_copy_path(self, path_ctxt, edge_cnt):

        rot_idx = self.node_itv
        res = path_ctxt.right_rotate_bin(rot_idx, 2**edge_cnt)
        return res
    def cal_path(self, copyed_data, path_list):  
        c1 = path_list*(copyed_data)
        if self.debug is True:
            print("c1")
            utils.print_matrix_not_zero(c1, 30, self.num_slot)
        sum_node = DataList(self.context, True, [sum(row) for row in c1])
        sum_rot = min(self.node_itv, self.num_slot)
        sum_node = sum_node.left_rotate_bin(1, sum_rot)
        if self.debug is True:
            print("sum_node")
            utils.print_list_not_zero(sum_node, self.num_slot)
        sum_node -= self.max_depth
        sum_node *= self.inf_masking_msg
        if self.debug is True:
            print("sum_node2")
            utils.print_list_not_zero(sum_node, self.num_slot)
        return sum_node
    def shuffle_list(self, lst):
        random.shuffle(lst)
        return lst
    def get_predict(self, path, leaf_weight):  

        within_random = random.randint(1, self.node_itv)  
        path += leaf_weight
        if self.debug is True:
            print("inf_weight")
            utils.print_matrix_not_zero(path, 30, self.num_slot)
        for i in range(len(path)):
            path[i] <<= within_random
        res = self.shuffle_list(path)
  
        return res
    def predict(self, hist: DataMatrix):
        num_slots = self.num_slot
        self.inf_ndata = hist.ndata

        self.predict_ctxt_path = self.main_folder + "/predict_ctxt"  
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
        total_time = 0
        idx = 0
        for data in hist.bin_indexes:
            data.level_down(self.need_depth)

            copy_ctxt_li = self.copy_input(data)  
            total_pred = DataMatrix(self.context, True)
            tmp_pred = None
            for n_path in range(len(self.merged_path)):
                # print("n_path:", n_path)
                path_list = self.merged_path[n_path]
                copy_ctxt = copy_ctxt_li.deepcopy()
                tmp_time = time.time()
                total_path = self.cal_path(copy_ctxt, path_list)  
                total_time += time.time() - tmp_time
                # print("cal_path_time:", time.time() - tmp_time)
                if tmp_pred is None:
                    tmp_pred = total_path
                else:
                    tmp_pred += total_path >> n_path
            if tmp_pred is not None:
                total_pred.append(tmp_pred)
                # print("append done")
            tmp_time = time.time()

            # print("total_pred type:", type(total_pred))
            y_predict = self.get_predict(total_pred, self.merged_leaf)  
            # print("y_predict type:", type(y_predict))
            total_time += time.time() - tmp_time
            # print("get_pred_time:", time.time() - tmp_time)
            # print("inference done", flush=True)
            self.save_predict(y_predict, idx)  
            idx += 1
        self.inf_time = total_time
      
        print("[HE Inference] total_time:", self.inf_time)
        print("[HE Inference] Amortized time (total/num_rows):", self.inf_time / idx)
        return
    def get_tree_structure(self):
        arr = []
        for t in self.models:
            js = t.get_tree_structure()
            arr.append(json.loads(js))
        return json.dumps(arr, indent=2)
    def get_dump(self, dump_format="json"):
        if dump_format == "json":
            dump_list = []
            for t in self.models:
                dump_list.append(t.get_tree_structure())
            return dump_list
        else:
            raise ValueError("Unsupported dump_format")
    def save_model_trees_to_json(self, filename="my_xgb_trees.json"):
        forest_json = self.get_tree_structure()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(forest_json)
        