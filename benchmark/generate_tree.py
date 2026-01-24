import json
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
class Leaf:
    def __init__(self, value, nodeid, depth):
        self.nodeid = nodeid
        self.depth = depth
        self.leaf = float(value)
        self.cover = 1.0  
    def to_dict(self):
        return {
            "nodeid": self.nodeid,
            "depth": self.depth,
            "leaf": self.leaf,
            "cover": self.cover,
        }
class Internal:
    def __init__(self, feature, threshold, left, right, nodeid, depth):
        self.nodeid = nodeid
        self.depth = depth
        self.split = f"f{feature}"
        self.split_condition = float(threshold)
        self.split_bin_idx = 0  
        self.gain = 1.0  
        self.cover = 1.0  
        self.children = [left, right]
    def to_dict(self):
        return {
            "nodeid": self.nodeid,
            "depth": self.depth,
            "split": self.split,
            "split_condition": self.split_condition,
            "split_bin_idx": self.split_bin_idx,
            "gain": self.gain,
            "cover": self.cover,
            "children": [c.to_dict() for c in self.children],
        }
def build_tree(tree):
    node_count = [0]
    return build_tree_rec(tree, 0, node_count), node_count[0]
def build_tree_rec(tree, node_id, node_count):
    left_child = tree.tree_.children_left[node_id]
    right_child = tree.tree_.children_right[node_id]
    is_split_node = left_child != right_child
    if is_split_node:
        left = build_tree_rec(tree, left_child, node_count)
        right = build_tree_rec(tree, right_child, node_count)
        node_count[0] += 1
        return Internal(tree.tree_.threshold[node_id], tree.tree_.feature[node_id], left, right)
    else:
        return Leaf(tree.tree_.value[node_id].argmax())
def generate_balanced_tree_rec(max_depth, depth, bitlength, num_attributes, nodeid_counter):
    if depth < max_depth:
        left = generate_balanced_tree_rec(max_depth, depth + 1, bitlength, num_attributes, nodeid_counter)
        right = generate_balanced_tree_rec(max_depth, depth + 1, bitlength, num_attributes, nodeid_counter)
        nodeid = nodeid_counter[0]
        nodeid_counter[0] += 1
        threshold = random.randint(0, bitlength)
        feature = random.randint(0, num_attributes - 1)
        return Internal(feature, threshold, left, right, nodeid, depth)
    else:
        value = random.uniform(-1, 1)  
        nodeid = nodeid_counter[0]
        nodeid_counter[0] += 1
        return Leaf(value, nodeid, depth)
def generate_tree(max_depth, bitlength, num_attributes, seed=None):
    if seed is not None:
        random.seed(seed)
    nodeid_counter = [0]
    tree = generate_balanced_tree_rec(
        max_depth, depth=0, bitlength=bitlength, num_attributes=num_attributes, nodeid_counter=nodeid_counter
    )
    return tree
def extract_splits(node, path="", splits=None):

    if splits is None:
        splits = []
    if "split" in node:  
        splits.append(
            {
                "node_path": path if path else "-1",
                "depth": node.get("depth", 0),
                "feature": node.get("split"),
                "condition": node.get("split_condition"),
                "split_bin_idx": node.get("split_bin_idx", None),
            }
        )
        for i, child in enumerate(node.get("children", [])):
            extract_splits(child, path + str(i), splits)
    return splits
def save_split_conditions_grouped(tree_dir):

    model_path = os.path.join(tree_dir, "model.json")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No model.json found in {tree_dir!r}")
    with open(model_path, "r") as f:
        tree = json.load(f)
    splits = extract_splits(tree)
    grouped = {}
    for s in splits:
        d = str(s["depth"])
        grouped.setdefault(d, []).append(
            {
                "node_path": s["node_path"],
                "feature": s["feature"],
                "condition": s["condition"],
                "split_bin_idx": s["split_bin_idx"],
            }
        )
    out_path = os.path.join(tree_dir, "splitcond.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(grouped, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved grouped split conditions to {out_path!r}")
def save_tree_to_json(tree, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(tree.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"✅ Tree saved to {save_path}")
if __name__ == "__main__":
    dataset_params = {
        "Breast": {
            "d": 30,
            "n_max": 12,
            "beta": 4
        },
        "Iris": {
            "d": 4,
            "n_max": 10,
            "beta": 4
        },
        "Spam": {
            "d": 57,
            "n_max": 15,
            "beta": 4
        }
    }
    depth_list = [i for i in range(2,9)]
    for dataset_name, params in dataset_params.items():
        d = params["d"]
        n_max = params["n_max"]
        dataset_dir = os.path.join(".", dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        for max_depth in depth_list:
            depth_dir = os.path.join(dataset_dir,f'depth{max_depth}')
            os.makedirs(depth_dir, exist_ok=True)
            random_tree = generate_tree(
                max_depth=max_depth,
                bitlength=n_max,
                num_attributes=d,
                seed=42
            )
            model_filename = "model.json"
            model_path = os.path.join(depth_dir, model_filename)
            with open(model_path, "w") as f:
                json.dump(random_tree.to_dict(), f, indent=2)
            save_split_conditions_grouped(depth_dir)