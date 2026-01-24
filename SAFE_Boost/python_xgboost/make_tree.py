import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from XGBoostClassifier import MyXGBClassifier  
def extract_and_save_split_conditions(tree_dir, n_estimators):

    for i in range(n_estimators):
        model_save_path = f"{tree_dir}/tree{i + 1}/model.json"
        split_cond_path = f"{tree_dir}/tree{i + 1}/splitcond.json"
        with open(model_save_path, "r") as f:
            tree_data = json.load(f)
        depth_splits = defaultdict(list)
        def traverse(node, path="-1"):
            if "leaf" in node:
                return
            depth = node["depth"]
            split_info = {
                "node_path": path,
                "feature": node["split"],
                "condition": node["split_condition"],
                "split_bin_idx": node["split_bin_idx"],
            }
            depth_splits[str(depth)].append(split_info)
            children = node.get("children", [])
            if len(children) == 2:
                left_path = "0" if path == "-1" else path + "0"
                right_path = "1" if path == "-1" else path + "1"
                traverse(children[0], left_path)
                traverse(children[1], right_path)
            else:
                for i, child in enumerate(children):
                    traverse(child, path + str(i))
        traverse(tree_data)
        with open(split_cond_path, "w") as f:
            json.dump(depth_splits, f, indent=4)
        print(f"✅ save done: {split_cond_path}")
def make_save_tree(dataset_name, max_bins, max_depths, n_estimators, cpp_bin_path):
    folds = 1
    base_path = "../data"
    tree_method = "hist"
    base_output_dir = os.path.join("./plain_trees", dataset_name)
    os.makedirs(base_output_dir, exist_ok=True)
    DEFAULT_MAX_BIN = 8
    DEFAULT_MAX_DEPTH = 4
    DEFAULT_N_ESTIMATORS = 15
    combos = []
    if len(max_bins) > 1:
        for mb in max_bins:
            c = (mb, DEFAULT_MAX_DEPTH, DEFAULT_N_ESTIMATORS)
            if c not in combos:
                combos.append(c)
    if len(max_depths) > 1:
        for dp in max_depths:
            c = (DEFAULT_MAX_BIN, dp, DEFAULT_N_ESTIMATORS)
            if c not in combos:
                combos.append(c)
    if len(n_estimators) > 1:
        for ne in n_estimators:
            c = (DEFAULT_MAX_BIN, DEFAULT_MAX_DEPTH, ne)
            if c not in combos:
                combos.append(c)
    if len(max_bins) == 1 and len(max_depths) == 1 and len(n_estimators) == 1:
        combos = [(max_bins[0], max_depths[0], n_estimators[0])]
    
    all_results = []
    for max_bin, depth, n in combos:
        bin_dir = os.path.join(base_output_dir, f"bin{max_bin}")
        os.makedirs(bin_dir, exist_ok=True)
        cutpath = Path(cpp_bin_path, f"bin{max_bin}", "cutpoints.json")
        with open(cutpath, "r") as f:
            bin_edges = json.load(f)
        for fold in range(1, folds + 1):
            train_path = os.path.join(base_path, f"{dataset_name}/train{fold}.csv")
            test_path = os.path.join(base_path, f"{dataset_name}/test{fold}.csv")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            X_train, y_train = train_df.drop("target", axis=1).values, train_df["target"].values
            X_test, y_test = test_df.drop("target", axis=1).values, test_df["target"].values
            model = MyXGBClassifier(
                n_estimators=n,
                max_depth=depth,
                learning_rate=0.5,
                prune_gamma=0.0,
                reg_lambda=1.0,
                base_score=0.5,
                tree_method=tree_method,
                max_bin=max_bin,
                bin_edges=bin_edges,
                hist_method="quantile",
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            all_results.append(
                {
                    "fold": fold,
                    "max_bin": max_bin,
                    "max_depth": depth,
                    "n_estimators": n,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred, average="binary"),
                    "auc": roc_auc_score(y_test, y_prob),
                    "log_loss": log_loss(y_test, y_prob),
                }
            )
            tree_dir = os.path.join(bin_dir, f"depth{depth}", f"n_tree_{n}", f"fold{fold}")
            os.makedirs(tree_dir, exist_ok=True)
            model.save_each_tree(tree_dir)
            extract_and_save_split_conditions(tree_dir, n)
        df = pd.DataFrame(all_results)
        agg_df = df.groupby(["max_bin", "max_depth", "n_estimators"], as_index=False).agg(
            {"accuracy": "mean", "f1_score": "mean", "auc": "mean", "log_loss": "mean"}
        )
        
        print(agg_df)
        output_dir = os.path.join("./plain_trees", dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "xgb_metric_avg.csv")
        agg_df.to_csv(save_path, index=False)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., breast_cancer)")
    parser.add_argument("--max_bins", type=str, required=True, help="Comma-separated list of max_bins (e.g., 4,8,16)")
    parser.add_argument("--depths", type=str, required=True, help="Comma-separated list of max_depths (e.g., 3,4,5)")
    parser.add_argument("--n_estimators", type=str, required=True, help="Number of trees (e.g., 10)")
    parser.add_argument("--bin_path", type=str, required=True, help="Path to C++ binary")
    args = parser.parse_args()
    max_bins = list(map(int, args.max_bins.split(",")))
    max_depths = list(map(int, args.depths.split(",")))
    n_estimators_li = list(map(int, args.n_estimators.split(",")))
    make_save_tree(args.dataset, max_bins, max_depths, n_estimators_li, args.bin_path)