import re
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from XGBoostClassifier import MyXGBClassifier
def test_my_xgb_vs_official_xgboost():
    train_df = pd.read_csv("../../dataset/synthetic_train5.csv")
    test_df = pd.read_csv("../../dataset/synthetic_test5.csv")
    X_train = train_df.drop(["label"], axis=1).values
    y_train = train_df["label"].values
    X_test = test_df.drop(["label"], axis=1).values
    y_test = test_df["label"].values
    tree_methods = ["hist"]
    results = []
    for tm in tree_methods:
        max_bin = 32
        my_model = MyXGBClassifier(
            n_estimators=2,
            max_depth=3,
            learning_rate=0.3,
            prune_gamma=0.0,
            reg_lambda=1.0,
            base_score=0.5,
            tree_method=tm,
            max_bin=max_bin,
        )
        start_fit = time.time()
        my_model.fit(X_train, y_train)
        end_fit = time.time()
        my_train_time = end_fit - start_fit
        start_pred = time.time()
        y_pred_my = my_model.predict(X_test)
        end_pred = time.time()
        my_infer_time = end_pred - start_pred
        acc_my = accuracy_score(y_test, y_pred_my)
        print(f"  -> Training time:  {my_train_time:.4f} sec")
        print(f"  -> Inference time:{my_infer_time:.4f} sec")
        print(f"  -> Accuracy:       {acc_my:.4f}")
        my_model.save_model_trees_to_json("model.json")
        results.append(
            {
                "model": "MyXGB",
                "tree_method": tm,
                "train_time": my_train_time,
                "infer_time": my_infer_time,
                "accuracy": acc_my,
            }
        )
        print(f"\n=== Official XGBoost with tree_method='{tm}' ===")
        xgb_model = xgb.XGBClassifier(
            n_estimators=2,
            max_depth=3,
            learning_rate=0.3,
            reg_lambda=1.0,
            max_bin=max_bin,
            tree_method=tm,
            use_label_encoder=False,  
            eval_metric="logloss",  
        )
        start_fit = time.time()
        xgb_model.fit(X_train, y_train)
        end_fit = time.time()
        xgb_train_time = end_fit - start_fit
        start_pred = time.time()
        y_pred_xgb = xgb_model.predict(X_test)
        end_pred = time.time()
        xgb_infer_time = end_pred - start_pred
        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        print(f"  -> Training time:  {xgb_train_time:.4f} sec")
        print(f"  -> Inference time:{xgb_infer_time:.4f} sec")
        print(f"  -> Accuracy:       {acc_xgb:.4f}")
        booster = xgb_model.get_booster()
        dump_list = booster.get_dump(dump_format="json")
        print("\n=== Official XGBoost Tree Dump ===")
        for i, tree in enumerate(dump_list):
            print(f"Tree {i}:\n{tree}")
        results.append(
            {
                "model": "XGBoost",
                "tree_method": tm,
                "train_time": xgb_train_time,
                "infer_time": xgb_infer_time,
                "accuracy": acc_xgb,
            }
        )
    return results
def param_test():
    train_df = pd.read_csv("../../dataset/breast-cancer-train.csv")
    test_df = pd.read_csv("../../dataset/breast-cancer-test.csv")
    X_train = train_df.drop(["target"], axis=1).values
    y_train = train_df["target"].values
    X_test = test_df.drop(["target"], axis=1).values
    y_test = test_df["target"].values
    tree_methods = ["hist"]  
    max_depth_range = range(3, 4)  
    n_estimators_range = range(1, 2)  
    results = []
    for depth in max_depth_range:
        result_row = {"iteration": []}
        for tm in tree_methods:
            result_row[f"my_xgb_depth_{depth}"] = []
            result_row[f"official_xgb_depth_{depth}"] = []
        for n_estimators in n_estimators_range:
            result_row["iteration"].append(n_estimators)
            for tm in tree_methods:
                my_model = MyXGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=depth,
                    learning_rate=0.3,
                    prune_gamma=0.0,
                    reg_lambda=1.0,
                    base_score=0.5,
                    tree_method=tm,
                    max_bin=4,
                )
                my_model.fit(X_train, y_train)
                y_pred_my = my_model.predict(X_test)
                acc_my = accuracy_score(y_test, y_pred_my)
                print(acc_my)
                result_row[f"my_xgb_depth_{depth}"].append(acc_my)
                xgb_model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=depth,
                    learning_rate=0.3,
                    reg_lambda=1.0,
                    max_bin=4,
                    tree_method=tm,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )
                xgb_model.fit(X_train, y_train)
                y_pred_xgb = xgb_model.predict(X_test)
                acc_xgb = accuracy_score(y_test, y_pred_xgb)
                result_row[f"official_xgb_depth_{depth}"].append(acc_xgb)
        results.append(result_row)
    df_results = pd.DataFrame(results[0])
    for i in range(1, len(results)):
        df_results = df_results.merge(pd.DataFrame(results[i]), on="iteration", how="outer")
    df_results.to_csv("xgb_depth_comparison2.csv", index=False)
    print("✅ CSV save done: xgb_depth_comparison.csv")
def debug_toy_dataset():

    X_small = np.array(
        [[1.0, 2.0], [1.2, 1.8], [2.1, 2.9], [2.5, 1.1], [0.9, 1.7], [3.0, 3.5], [2.4, 2.2], [1.8, 2.1]],
        dtype=np.float32,
    )
    y_small = np.array([0, 0, 1, 0, 0, 1, 1, 1], dtype=np.uint8)
    print("=== Toy Dataset (X_small) ===")
    print(X_small)
    print("Labels (y_small):", y_small)
    my_model = MyXGBClassifier(
        n_estimators=1,
        max_depth=1,
        learning_rate=0.3,
        prune_gamma=0.0,
        reg_lambda=1.0,
        base_score=0.5,
        tree_method="hist",  
        max_bin=4,  
    )
    print("\n[MyXGB] Fitting on toy dataset...")
    my_model.fit(X_small, y_small)  
    y_pred_my = my_model.predict(X_small)
    print("[MyXGB] Predictions on toy data:", y_pred_my)
    accuracy_my = accuracy_score(y_small, y_pred_my)
    print(f"[MyXGB] Accuracy: {accuracy_my:.4f}")
    my_model.save_model_trees_to_json("model.json")
    xgb_model = xgb.XGBClassifier(
        n_estimators=2,
        max_depth=2,
        learning_rate=0.3,
        reg_lambda=1.0,
        max_bin=4,
        tree_method="hist",
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=1,  
    )
    print("\n[Official XGBoost] Fitting on toy dataset...")
    xgb_model.fit(X_small, y_small)
    y_pred_xgb = xgb_model.predict(X_small)
    print("[XGBoost] Predictions on toy data:", y_pred_xgb)
    accuracy_xgb = accuracy_score(y_small, y_pred_xgb)
    print(f"[XGBoost] Accuracy: {accuracy_xgb:.4f}")
    booster = xgb_model.get_booster()
    dump_list = booster.get_dump(dump_format="json")
    print("\n=== Official XGBoost Tree Dump ===")
    for i, tree in enumerate(dump_list):
        print(f"Tree {i}:\n{tree}")
log_data = """
[CALL] AddCutPoint 
[DEBUG] summary.size: 5, max_bin: 4
[DEBUG] Sketch Data - Index: 0, Value: 7.691, rmin: 0, rmax: 1, weight: 1
[DEBUG] Sketch Data - Index: 1, Value: 11.71, rmin: 114, rmax: 117, weight: 3
[DEBUG] Sketch Data - Index: 2, Value: 13.43, rmin: 232, rmax: 234, weight: 1
[DEBUG] Sketch Data - Index: 3, Value: 15.5, rmin: 335, rmax: 337, weight: 1
[DEBUG] Sketch Data - Index: 4, Value: 28.11, rmin: 454, rmax: 455, weight: 1
[CALL] AddCutPoint 
[DEBUG] summary.size: 5, max_bin: 4
[DEBUG] Sketch Data - Index: 0, Value: 9.71, rmin: 0, rmax: 1, weight: 1
[DEBUG] Sketch Data - Index: 1, Value: 16.21, rmin: 115, rmax: 117, weight: 2
[DEBUG] Sketch Data - Index: 2, Value: 18.59, rmin: 220, rmax: 223, weight: 2
[DEBUG] Sketch Data - Index: 3, Value: 21.56, rmin: 337, rmax: 339, weight: 1
[DEBUG] Sketch Data - Index: 4, Value: 39.28, rmin: 454, rmax: 455, weight: 1
"""
def extract_values_exclude_index_0(log_text):
    cutpoints = log_text.split("[CALL] AddCutPoint")[1:]  
    values_by_cutpoint = []
    for cut_idx, cutpoint in enumerate(cutpoints):
        values = re.findall(r"Index: (\d+), Value: ([\d\.]+)", cutpoint)
        filtered_values = [float(value) for idx, value in values if int(idx) > 0]  
        values_by_cutpoint.append((cut_idx, filtered_values))
    return values_by_cutpoint
if __name__ == "__main__":
    final_results = test_my_xgb_vs_official_xgboost()
    print("\n=== Final Comparison Results ===")
    for res in final_results:
        print(res)