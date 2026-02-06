import json
import os
from collections import deque
import numpy as np
class SummaryEntry:
    def __init__(self, value, rmin, rmax, wmin):
        self.value = value
        self.rmin = rmin
        self.rmax = rmax
        self.wmin = wmin
    def RMinNext(self):
        return self.rmin + self.wmin
    def RMaxPrev(self):
        return self.rmax - self.wmin
class Node:
    def __init__(self, cover, G=None, H=None, fid=None, bin_idx=None, split_point=None, gain=None, leaf=None):
        self.cover = cover
        self.G = G
        self.H = H
        self.fid = fid
        self.bin_idx = bin_idx
        self.split_point = split_point
        self.gain = gain
        self.leaf = leaf
        self.left = None
        self.right = None
    def is_leaf(self):
        return self.leaf is not None
class BinEdgeManager:
    def __init__(self, max_bin):
        self.max_bin = max_bin
        self.sketch_data = []
        self.pruned_summary = []
        self.final_bin_edges = []
        self.min_val = 0.0
        self.is_complete = False
    def search_bin(self, value, column_id, ptrs, values):

        beg, end = ptrs[column_id], ptrs[column_id + 1]
        idx = np.searchsorted(values[beg:end], value, side="right") + beg  
        if idx == end:  
            idx -= 1  
        return idx  
    def add_point(self, value, wmin, rmin, rmax):
        entry = SummaryEntry(value, rmin, rmax, wmin)
        self.sketch_data.append(entry)
    def print_queue_contents(self):
        pass
    def set_prune(self, maxsize):
        src = self.sketch_data
        if len(src) == 0:
            self.pruned_summary = []
            return
        if len(src) <= maxsize:
            self.pruned_summary = src[:]
        else:
            begin = src[0].rmax
            range_ = src[-1].rmin - src[0].rmax
            n = maxsize - 1
            data = []
            data.append(src[0])
            size = 1
            i = 1
            lastidx = 0
            for k in range(1, n):
                dx2 = 2.0 * ((k * range_) / n + begin)
                while i < len(src) - 1 and dx2 >= (src[i + 1].rmax + src[i + 1].rmin):
                    i += 1
                if i == len(src) - 1:
                    break
                if dx2 < (src[i].RMinNext() + src[i + 1].RMaxPrev()):
                    if i != lastidx:
                        data.append(src[i])
                        size += 1
                        lastidx = i
                else:
                    if (i + 1) != lastidx:
                        data.append(src[i + 1])
                        size += 1
                        lastidx = i + 1
            if lastidx != len(src) - 1:
                data.append(src[-1])
            self.pruned_summary = data
        first_val = self.pruned_summary[0].value
        self.min_val = first_val - (abs(first_val) + 1e-5)
    def add_cutpoint(self, p_cuts, max_bin):
        summary = self.pruned_summary
        if len(summary) == 0:
            return
        cut_values = p_cuts["cut_values"]
        required_cuts = min(len(summary), max_bin)
        for i in range(1, required_cuts):
            cpt = summary[i].value
            if i == 1 or cpt > cut_values[-1]:
                cut_values.append(cpt)
    def make_cuts(self, p_cuts):
        if not self.is_complete:
            self.print_queue_contents()
            self.is_complete = True
        self.set_prune(self.max_bin + 1)
        p_cuts["cut_values"] = []
        self.add_cutpoint(p_cuts, self.max_bin)
        if len(self.pruned_summary) > 0:
            last_candidate = self.pruned_summary[-1].value
        else:
            last_candidate = self.min_val
        final_last = last_candidate + (abs(last_candidate) + 1e-5)
        self.final_bin_edges = p_cuts["cut_values"]
class MyXGBClassificationTree:
    def __init__(
        self,
        max_depth,
        reg_lambda,
        prune_gamma,
        tree_method="hist",
        n_bins=4,
        cpp_bin_edges=None,
        hist_method="quantile",
    ):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.prune_gamma = prune_gamma
        self.tree_method = tree_method
        self.n_bins = n_bins
        self.bin_indexes = None
        self.cpp_bin_edges = cpp_bin_edges
        self.bin_edges = None
        self.bin_offsets = None
        self.feature = None
        self.residual = None
        self.prev_yhat = None
        self.node_id_counter = 0
        self.estimator = None
        self.hist_method = hist_method
    def node_split(self, did):
        if self.tree_method == "hist":
            return self._node_split_hist(did)
        else:
            raise ValueError("Unknown tree_method")
    def _node_split_hist(self, did):
        if self.bin_indexes is None or self.bin_edges is None:
            self._build_global_histogram()
        node_id = self.node_id_counter
        self.node_id_counter += 1
        # print(f"[DEBUG] Processing Node {node_id}")
        n_total = self.feature.shape[0]
        hessian = [0 for _ in range(n_total)]
        gradient = [0 for _ in range(n_total)]
        B = len(self.bin_edges[0])
        F = self.feature.shape[1]
        # print(f"B : {B}, F: {F}")
        scaling_coef = ((1 << 18) / (9 / 128 * (n_total**4) * (B - 1) * F)) ** 0.25
        # print(f"scaling coef : {scaling_coef}")
        cover = 0.0
        for rid in did:
            p = self.prev_yhat[rid]
            gradient[rid] = self.residual[rid] * scaling_coef
            cover += p * (1.0 - p) * scaling_coef  
        G = sum(gradient)
        H = cover
        r = self.reg_lambda
        parent_score = (G**2) / (H + r) if (H + r) > 1e-12 else 0.0
        max_gain = -np.inf
        best_fid, best_bin = None, None
        best_split_val = None
        d_cols = self.feature.shape[1]
        for k in range(d_cols):
            edges_k = self.bin_edges[k]
            offset_k = self.bin_offsets[k]
            feat_bin_sub = self.bin_indexes[k][did]
            G_in_bin = np.zeros(self.n_bins)
            H_in_bin = np.zeros(self.n_bins)
            for i, rid in enumerate(did):
                local_bin = feat_bin_sub[i] - offset_k
                G_in_bin[local_bin] += self.residual[rid] * scaling_coef  
                pp = self.prev_yhat[rid]
                H_in_bin[local_bin] += pp * (1 - pp) * scaling_coef  
            left_G = np.cumsum(G_in_bin)
            left_H = np.cumsum(H_in_bin)
            right_G = G - left_G
            right_H = H - left_H
            gains = (left_G * (2 * H - left_H)) ** 2 + (right_G * (H + left_H)) ** 2
            b_idx = np.argmax(gains)
            gain = gains[b_idx]
            # print(f"[DEBUG] Node {node_id} Feature {k} Gains per bin index: {gains.tolist()}")
            # print(f"[DEBUG] Max gain at bin index {b_idx} is {gain:.6f}")
            if b_idx + 1 < len(edges_k):
                candidate_threshold = edges_k[b_idx]  
            else:
                candidate_threshold = edges_k[-1]
            if gain > max_gain:
                max_gain = gain
                best_fid = k  
                best_bin = b_idx + 1  
                best_bin = b_idx + 1  
                best_split_val = candidate_threshold  
        # print(
        #     f"[DEBUG] Node {node_id} Maximum gain overall: {max_gain:.6f}, Feature :{best_fid}, bin index :{best_bin}, candidate threshold: {best_split_val}"
        # )
        node = Node(cover, G, H)  
        if max_gain >= self.prune_gamma and best_fid is not None:
            node.fid = best_fid
            node.split_point = float(best_split_val)
            node.gain = float(max_gain)
            node.bin_idx = best_bin
            offset_k = self.bin_offsets[best_fid]
            feat_bin_sub = self.bin_indexes[best_fid][did]
            left_mask = feat_bin_sub <= (best_bin + offset_k)
            b_left = did[left_mask]
            b_right = did[~left_mask]
            node.left = b_left
            node.right = b_right
        else:
            if (H + r) < 1e-12:
                node.leaf = 0.0
            else:
                node.leaf = float(G / (H + r))
        return node
    def _build_global_histogram(self) -> None:

        n_samples, d = self.feature.shape
        self.bin_edges = {}
        self.bin_indexes = {}
        self.bin_offsets = {}
        total_bins = 0
        for k in range(d):
            manager = BinEdgeManager(self.n_bins)
            col = self.feature[:, k]
            if self.hist_method=='quantile':
                edges = np.array(self.cpp_bin_edges[str(k)])  
                ptrs = [0] + list(np.cumsum([len(edges)]))
                X_bin_local = np.array([manager.search_bin(v, 0, ptrs, edges) for v in col], dtype=int)
                X_bin_local = np.clip(X_bin_local, 0, self.n_bins - 1)
                X_bin_local[col >= edges[-1]] = self.n_bins - 1
                tmp_edges = list(edges[:-1])
                if len(tmp_edges) < self.n_bins - 1:
                    if len(edges) == 1:
                        tmp_edges = [edges[0]] * (self.n_bins - 1)
                    else:
                        tmp_edges += [edges[-2]] * (self.n_bins - 1 - len(edges[:-1]))
                edges = np.array(tmp_edges)
            elif self.hist_method=='uniform':
                edges = np.linspace(col.min(), col.max(), self.n_bins + 1)
            offset_k = total_bins
            ptrs = [0] + list(np.cumsum([len(edges)]))  
            X_bin_local = np.array([manager.search_bin(v, 0, ptrs, edges) for v in col])  
            X_bin_local = np.clip(X_bin_local, 0, self.n_bins - 1)
            X_bin_local[col >= edges[-1]] = self.n_bins - 1  
            X_bin_global = X_bin_local + offset_k
            self.bin_offsets[k] = offset_k
            self.bin_indexes[k] = X_bin_global
            self.bin_edges[k] = edges
            total_bins += self.n_bins
    def bfs_split(self, root) -> None:

        queue = deque()
        queue.append((root, 0))
        while queue:
            node, depth = queue.popleft()
            if node.is_leaf():
                continue
            if depth >= self.max_depth:
                r = self.reg_lambda
                if not node.is_leaf():
                    if (node.H + r) < 1e-12:
                        node.leaf = 0.0
                    else:
                        node.leaf = float(node.G / (node.H + r))
                node.left = None
                node.right = None
                continue
            if isinstance(node.left, np.ndarray):
                left_did = node.left
                child = self.node_split(left_did)
                node.left = child
                queue.append((child, depth + 1))
            if isinstance(node.right, np.ndarray):
                right_did = node.right
                child = self.node_split(right_did)
                node.right = child
                queue.append((child, depth + 1))
    def output_leaf(self, node):
        pass
    def fit(self, x, y, prev_yhat):
        self.feature = x
        self.residual = y
        self.prev_yhat = prev_yhat
        if self.tree_method == "hist":
            self._build_global_histogram()
        # print("bin_edges")
        # for feat, edges in self.bin_edges.items():
        #     print(f"Feature {feat}: {edges}")
        root = self.node_split(np.arange(x.shape[0]))
        self.bfs_split(root)
        self.estimator = root
    def predict(self, X):
        if not isinstance(self.estimator, Node):  
            return np.full(X.shape[0], self.estimator, dtype=float)
        preds = []
        for i in range(X.shape[0]):
            # print("row:", i)
            preds.append(self.x_predict(self.estimator, X[i]))  
        return np.array(preds, dtype=float)
    def x_predict(self, node, x_row):
        if node.is_leaf():
            # print("leaf:", node.leaf)
            return node.leaf
        fid = node.fid
        spv = node.split_point
        if x_row[fid] < spv:
            return self.x_predict(node.left, x_row)
        else:
            return self.x_predict(node.right, x_row)
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
                s += ', "split_bin_idx": ' + str(nd.bin_idx)
                s += ', "gain": ' + str(nd.gain)
                s += ', "cover": ' + str(nd.cover)
                left_str = traverse(nd.left, 2 * nodeid + 1, depth + 1)
                right_str = traverse(nd.right, 2 * nodeid + 2, depth + 1)
                s += ', "children": [' + left_str + ", " + right_str + "] }"
                return s
        return traverse(self.estimator, 0, 0)  
class MyXGBClassifier:
    def __init__(
        self,
        n_estimators=1,
        max_depth=2,
        learning_rate=0.3,
        prune_gamma=0.0,
        reg_lambda=0.0,
        base_score=0.5,
        tree_method="hist",
        max_bin=4,
        bin_edges=None,
        hist_method="quantile",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.eta = learning_rate
        self.prune_gamma = prune_gamma
        self.reg_lambda = reg_lambda
        self.base_score = base_score
        self.tree_method = tree_method
        self.max_bin = max_bin
        self.models = []
        self.bin_edges = bin_edges
        self.hist_method = hist_method
        print("\n=== Hyperparameters ===")
        print(f"n_estimators: {self.n_estimators}")
        print(f"max_depth: {self.max_depth}")
        print(f"learning_rate: {self.eta}")
        print(f"reg_lambda: {self.reg_lambda}")
        print(f"max_bin: {self.max_bin}")
        print(f"prune_gamma: {self.prune_gamma}")
    def F2P(self, x):
        return 1 / (1 + np.exp(-x))
    def fit(self, X, y):
        F0 = np.log(self.base_score / (1 - self.base_score))
        Fm = np.full(X.shape[0], F0)
        y_hat = self.F2P(Fm)
        self.models = []
        for tree_idx in range(self.n_estimators):
            print(f"=== {tree_idx}th Tree ===")
            residual = y - y_hat
            tree = MyXGBClassificationTree(
                max_depth=self.max_depth,
                reg_lambda=self.reg_lambda,
                prune_gamma=self.prune_gamma,
                tree_method=self.tree_method,
                n_bins=self.max_bin,
                cpp_bin_edges=self.bin_edges,
                hist_method=self.hist_method,
            )
            tree.fit(X, residual, y_hat)
            gamma = tree.predict(X)
            Fm += self.eta * gamma
            y_hat = self.F2P(Fm)
            self.models.append(tree)
    def predict(self, X, proba=False):
        Fm = np.full(X.shape[0], np.log(self.base_score / (1 - self.base_score)))
        for tree in self.models:
            Fm += self.eta * tree.predict(X)
        y_hat = self.F2P(Fm)
        if proba:
            return y_hat
        else:
            return (y_hat > 0.5).astype(int)
    def predict_proba(self, X):

        y_prob = self.predict(X, proba=True)
        return np.vstack([1 - y_prob, y_prob]).T
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

    def save_each_tree(self, out_dir="trees"):
        os.makedirs(out_dir, exist_ok=True)
        for idx, tree in enumerate(self.models, start=1):
            tree_out_dir = os.path.join(out_dir, f"tree{idx}")
            os.makedirs(tree_out_dir, exist_ok=True)
            tree_json = tree.get_tree_structure()
            data = json.loads(tree_json)
            path = os.path.join(tree_out_dir, "model.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
      