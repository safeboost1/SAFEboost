import json
import math
import os
import time
from pathlib import Path
from typing import Union
import numpy as np
from dataset.data_set import DataList, DataMatrix
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
class BinEdgeManager:
    def __init__(self, max_bin):
        self.max_bin = max_bin
        self.sketch_data = []
        self.pruned_summary = []
        self.final_bin_edges = []
        self.min_val = 0.0
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
        self.set_prune(self.max_bin + 1)
        p_cuts["cut_values"] = []
        self.add_cutpoint(p_cuts, self.max_bin)
        if len(self.pruned_summary) > 0:
            last_candidate = self.pruned_summary[-1].value
        else:
            last_candidate = self.min_val
        final_last = last_candidate + (abs(last_candidate) + 1e-5)
        p_cuts["cut_values"].append(final_last)
        self.final_bin_edges = p_cuts["cut_values"]
class Histogram:
    def __init__(self, context, bin_edges, max_bin=4, base_score=0.5, save_path="./result"):
        self.context = context
        self.max_bin = max_bin
        self.base_score = base_score
        self.d = None
        self.ndata = None
        self.cpp_bin_edges = bin_edges
        self.bin_edges = None
        self.bin_indexes = None
        self.gh = None
        self.y = None
        self.y_hat = None
        self.Fm = None
        self.save_path = save_path
    def F2P(self, x):
        return 1 / (1 + np.exp(-x))
    def build_histogram(self, X, y):
        print("[Histogram] build_histogram start")
        start = time.time()

        num_slots = self.context.num_slots
        self.ndata, self.d = X.shape
        self.n_max = int(X.max().max() + 1)  
        self.bin_edges = []
        self.bin_indexes = []
        for k in range(self.d):
            manager = BinEdgeManager(self.max_bin)
            col = X[:, k]
            edges = np.array(self.cpp_bin_edges[str(k)])  
            ptrs = [0] + list(np.cumsum([len(edges)]))
            X_bin_local = np.array([manager.search_bin(v, 0, ptrs, edges) for v in col], dtype=int)
            X_bin_local = np.clip(X_bin_local, 0, self.max_bin - 1)
            X_bin_local[col >= edges[-1]] = self.max_bin - 1
            tmp_edges = list(edges[:-1])
            if len(tmp_edges) < self.max_bin - 1:
                if len(edges) == 1:
                    tmp_edges = [edges[0]] * (self.max_bin - 1)
                else:
                    tmp_edges += [edges[-2]] * (self.max_bin - 1 - len(edges[:-1]))
            tmp_edges = [edge / self.n_max for edge in tmp_edges]
            self.bin_edges.append(tmp_edges)
            self.bin_indexes.append(X_bin_local)
        F0 = np.log(self.base_score / (1 - self.base_score))
        Fm = np.full(self.ndata, F0)
        y_hat = self.F2P(Fm)  
        residual = y - y_hat  
        B = self.max_bin - 1
        F = self.d
        print(f"B : {B}, F: {F}")
        scaling_coef = ((1 << 18) / (9 / 128 * (self.ndata**4) * B * F)) ** 0.25  
        print("scaling coef : ", scaling_coef)
        g = residual * scaling_coef
        h = (y_hat * (1 - y_hat)) * scaling_coef
        bin_edges_ = np.concatenate(self.bin_edges)  
        bin_edges_ = bin_edges_.tolist()
        self.bin_edges_list = []
        one_ctxt_feature_by_bin = min(
            math.floor(num_slots / (self.max_bin - 1)), self.d
        )  
        if self.d * (self.max_bin - 1) > num_slots:
            n_blocks = int(math.ceil(self.d / one_ctxt_feature_by_bin))
            for i in range(n_blocks):
                start_idx = i * one_ctxt_feature_by_bin * (self.max_bin - 1)
                end_idx = min(start_idx + one_ctxt_feature_by_bin * (self.max_bin - 1), self.d * (self.max_bin - 1))
                block_bin_edges = list(bin_edges_[start_idx:end_idx])
                if len(block_bin_edges) < num_slots:
                    block_bin_edges.extend([0] * (num_slots - len(block_bin_edges)))
                self.bin_edges_list.append(block_bin_edges)
        else:
            block_bin_edges = list(bin_edges_)
            block_bin_edges.extend([0] * (num_slots - len(block_bin_edges)))
            self.bin_edges_list.append(block_bin_edges)
        Fm_list = []
        gh_list = []
        y_list = []
        y_hat_list = []
        if self.ndata > num_slots:
            n_blocks = int(np.ceil(self.ndata / num_slots))
            for i in range(n_blocks):
                start_idx = i * num_slots
                end_idx = min(start_idx + num_slots, self.ndata)
                block_Fm = list(Fm[start_idx:end_idx])
                block_g = list(g[start_idx:end_idx])
                block_h = list(h[start_idx:end_idx])
                block_y = list(y[start_idx:end_idx])
                block_y_hat = list(y_hat[start_idx:end_idx])
                if len(block_g) < num_slots:
                    block_Fm.extend([0] * (num_slots - len(block_Fm)))
                    block_g.extend([0] * (num_slots - len(block_g)))
                    block_h.extend([0] * (num_slots - len(block_h)))
                    block_y.extend([0] * (num_slots - len(block_y)))
                    block_y_hat.extend([0] * (num_slots - len(block_y_hat)))
                block_gh = [complex(r, im) for r, im in zip(block_g, block_h)]
                gh_list.append(block_gh)
                Fm_list.append(block_Fm)
                y_list.append(block_y)
                y_hat_list.append(block_y_hat)
        else:
            block_Fm = list(Fm)
            block_g = list(g)
            block_h = list(h)
            block_y = list(y)
            block_y_hat = list(y_hat)
            block_Fm.extend([0] * (num_slots - len(block_Fm)))
            block_g.extend([0] * (num_slots - len(block_g)))
            block_h.extend([0] * (num_slots - len(block_h)))
            block_y.extend([0] * (num_slots - len(block_y)))
            block_y_hat.extend([0] * (num_slots - len(block_y_hat)))
            block_gh = [complex(r, im) for r, im in zip(block_g, block_h)]
            gh_list.append(block_gh)
            Fm_list.append(block_Fm)
            y_list.append(block_y)
            y_hat_list.append(block_y_hat)
        print("\n=== Fm_list shape ===")
        print(f"Number of blocks: {len(Fm_list)}")
        self.gh = DataList(self.context, False, gh_list, is_complex=True)
        self.bin_edges = DataList(self.context, False, self.bin_edges_list)
        self.Fm = DataList(self.context, False, Fm_list)
        self.y = DataList(self.context, False, y_list)
        self.y_hat = DataList(self.context, False, y_hat_list)
        print(f"[Histogram] build histogram time : {time.time() - start:.4f}s<<<")
    def transform_bin_index(self):

        num_slots = self.context.num_slots

        num_columns = self.max_bin - 1
        cumulative_hist = []
        for arr in self.bin_indexes:
            arr_plus_one = arr + 1
            arr_plus_one[arr_plus_one == self.max_bin] = 0
            one_hot = np.zeros((arr.shape[0], num_columns), dtype=int)
            for i, val in enumerate(arr_plus_one):
                if val > 0:
                    one_hot[i, val - 1 :] = 1
            cumulative_hist.append(one_hot)
        res_list = []  
        for feature_idx, mat in enumerate(cumulative_hist):
            for col_idx in range(mat.shape[1]):
                col_as_row = mat[:, col_idx].tolist()
                res_list.append(col_as_row)
        res_matrix = []
        for row in res_list:
            chunked = []
            for i in range(0, len(row), num_slots):
                chunk = row[i : i + num_slots]
                if len(chunk) < num_slots:
                    chunk.extend([0] * (num_slots - len(chunk)))
                chunked.append(chunk)
            res_matrix.append(chunked)
        print("=== Result (3D) ===")
        d_times_bin_minus1 = len(res_matrix)  
        chunk_count = len(res_matrix[0]) if d_times_bin_minus1 > 0 else 0
        slot_size = len(res_matrix[0][0]) if chunk_count > 0 else 0
        print(f"Result shape: ({d_times_bin_minus1}, {chunk_count}, {slot_size})")
        self.bin_indexes = DataMatrix(self.context, False, res_matrix)
        print("[Histogram] transform bin index end.")
    def encode(self, X, y):
        start = time.time()
        self.build_histogram(X, y)
        self.transform_bin_index()
        print(f"[TIME]Histogram encode Time: {time.time() - start:.4f} s")
    def encrypt(self):
        t0 = time.time()
        self.gh.encrypt()
        self.bin_indexes.encrypt()
        self.y.encrypt()
        self.y_hat.encrypt()
        self.Fm.encrypt()
        self.bin_edges.encrypt()
        print(f"[TIME]Histogram encrypt Time: {time.time() - t0:.4f} s")
    def decrypt(self):
        pass
    def metadata(self):
     
        return {"max_bin": self.max_bin, "base_score": self.base_score, "d": self.d, "ndata": self.ndata}
    def save(self, tree_idx=1) -> None:
        start = time.time()
        path = Path(self.save_path, "enc_data")
        tree_common_path = path / f"tree{tree_idx}" / "common"
        tree_o_path = path / f"tree{tree_idx}" / "o"
        Fm_save_path = tree_common_path / "Fm"
        y_hat_save_path = tree_common_path / "y_hat"
        gh_save_path = tree_o_path / "gh"
        for p in [Fm_save_path, y_hat_save_path, gh_save_path]:
            os.makedirs(p, exist_ok=True)
        self.Fm.save(Fm_save_path)
        self.y_hat.save(y_hat_save_path)
        self.gh.save(gh_save_path)
        if tree_idx == 1:
            y_save_path = path / "label"
            bin_edges_save_path = path / "bin_edges"
            bin_indexes_save_path = path / "bin_indexes"
            for p in [y_save_path, bin_edges_save_path, bin_indexes_save_path]:
                os.makedirs(p, exist_ok=True)
            self.y.save(y_save_path)
            self.bin_edges.save(bin_edges_save_path)
            self.bin_indexes.save(bin_indexes_save_path)
            self.save_metadata()
        print(f"[TIME]Histogram save Time: {time.time() - start:.4f} s")
    def save_metadata(self):
        json_file_path = Path(self.save_path, "Metadata.json")
        metadata_dict = self.metadata()
        if not isinstance(metadata_dict, dict):
            raise ValueError("The metadata() function must return a dict")
        with open(json_file_path, "w", encoding="utf-8") as m_file:
            json.dump(metadata_dict, m_file, indent=4)
        
    def load(self, path: Union[str, Path], encrypted: bool = True) -> None:
        if isinstance(path, str):
            path = Path(path)
        self.gh = DataList.from_path(self.context, encrypted=encrypted, path=path / "gh")
        self.bin_indexes = DataMatrix.from_path(self.context, encrypted=encrypted, path=path / "bin_indexes")
        self.load_metadata()
    def load_metadata(self, metadata_json_path="./result/Metadata.json"):
        if not os.path.exists(metadata_json_path):
            raise FileNotFoundError(f"File not found: {metadata_json_path}")
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)
        required_keys = ["max_bin", "base_score", "d", "ndata"]
        missing_keys = [key for key in required_keys if key not in metadata_dict]
        if missing_keys:
            raise KeyError(f"The {missing_keys} key is missing from the metadata JSON file.")
        self.max_bin = metadata_dict["max_bin"]
        self.base_score = metadata_dict["base_score"]
        self.d = metadata_dict["d"]
        self.ndata = metadata_dict["ndata"]
        print(f"Metadata loaded from {metadata_json_path}")
    @staticmethod
    def from_path(context, path, encrypted=True):

        pass
    def metadata2instance(self, Metadata):
        self.max_bin = Metadata["max_bin"]
        self.d = Metadata["d"]
        self.base_score = Metadata["base_score"]
        self.ndata = Metadata["ndata"]
class Inference_Histogram:
    def __init__(self, hist):
        self.context = hist.context
        self.d = None
        self.ndata = None
        self.max_bin = hist.max_bin
        self.n_max = hist.n_max
        self.bin_indexes = None
        self.save_path = hist.save_path
    def build_histogram(self, X):
        self.ndata, self.d = X.shape
        
        self.bin_indexes = []
        for k in range(self.d):
            col = X[:, k]
            X_one_hot = np.array([self.onehot_encoding(col)])
            self.bin_indexes.append(X_one_hot)
    def onehot_encoding(self, col):
        oh = np.zeros((self.ndata, self.n_max))
        oh[np.arange(self.ndata), col] = 1
        return oh
    def transform_bin_index(self, mode):

        num_slots = self.context.num_slots
        num_columns = self.n_max  
        matrices = []
        tmp_bin_indexes = np.hstack(np.concatenate(self.bin_indexes, axis=0))
        self.bin_indexes = DataMatrix(self.context, False)
        for row in tmp_bin_indexes:
            num_slots = self.context.num_slots
            row_list = row.tolist()
            if len(row_list) <= num_slots:
                split_row = [row.tolist()]
            else:
                cut_idx = math.floor(num_slots / self.n_max)
                split_row = [row_list[i : i + num_slots] for i in range(0, len(row_list), cut_idx)]
            tmp_DataList = DataList(self.context, False, split_row)
            self.bin_indexes.append(tmp_DataList)
    def search_bin(self, value, column_id, ptrs, values):

        beg, end = ptrs[column_id], ptrs[column_id + 1]
        idx = np.searchsorted(values[beg:end], value, side="right") + beg
        if idx == end:
            idx -= 1
        return idx
    def encode(self, X, mode="one"):
        self.build_histogram(X)
        self.transform_bin_index(mode)
    def encrypt(self):
        self.bin_indexes.encrypt()
        # print("[Histogram] Done encrypting histogram.")
    def decrypt(self):
        pass
    def metadata(self):
        return {"n_max": self.n_max, "d": self.d, "ndata": self.ndata}
    def save(self, path=None) -> None:
        if path is None:
            path = Path(self.save_path)
        else:
            path = Path(path)
        bin_indexes_save_path = path / "inf_bin_indexes"
        for path in [bin_indexes_save_path]:
            os.makedirs(path, exist_ok=True)
        self.bin_indexes.save(bin_indexes_save_path)  
        self.save_metadata(path)
    def save_metadata(self, path: Union[str, Path] = None):
        if path is None:
            json_file_path = "./result/Metadata_test.json"
        else:
            json_file_path = path / "Metadata_test.json"
        metadata_dict = self.metadata()
        if not isinstance(metadata_dict, dict):
            raise ValueError("The metadata() function must return a dict")
        with open(json_file_path, "w", encoding="utf-8") as m_file:
            json.dump(metadata_dict, m_file, indent=4)
        
    def load(self, path=None, encrypted: bool = True) -> None:
        if path is None:
            path = Path(self.save_path)
        else:
            path = Path(path)
        if isinstance(self.save_path, str):
            path = Path(self.save_path)
        self.bin_indexes = DataMatrix.from_path(self.context, encrypted=encrypted, path=path / "inf_bin_indexes")
    def load_metadata(self, metadata_json_path="./result/Metadata_test.json"):
        if not os.path.exists(metadata_json_path):
            raise FileNotFoundError(f"{metadata_json_path} cannot find file.")
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)
        required_keys = ["max_bin", "d", "ndata"]
        missing_keys = [key for key in required_keys if key not in metadata_dict]
        if missing_keys:
            raise KeyError(f"The {missing_keys} key is missing from the metadata JSON file.")
        self.n_max = metadata_dict["n_max"]
        self.d = metadata_dict["d"]
        self.ndata = metadata_dict["ndata"]
        
    @staticmethod
    def from_path(context, path, encrypted=True):
        if isinstance(path, str):
            path = Path(path)
        with open(path / "Metadata_test.json", "r") as json_file:
            metadata = json.load(json_file)
        load_path = path / "enc_data" / "o"
        histogram = Histogram(context, metadata["max_bin"], metadata["base_score"])
        histogram.load(load_path, encrypted=encrypted)
        histogram.metadata2instance(metadata)
        return histogram