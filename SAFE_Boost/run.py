import os
os.environ["HEAAN_TYPE"] = "pi"
import warnings
warnings.filterwarnings("ignore")
import json
import os
import subprocess
from pathlib import Path
import HE_codes.utils as utils
import heaan_stat
import pandas as pd
from HE_codes.histogram import Histogram,Inference_Histogram
from HE_codes.xgboost import MyXGBClassifier as HEboost
from python_xgboost.XGBoostClassifier import MyXGBClassifier as plainboost
from sklearn.metrics import accuracy_score
from collections import defaultdict
generate_keys = True
preset = "FGb"
print(preset)
param = heaan_stat.HEParameter(preset)
res_path = "./result"
if not os.path.exists(res_path):
    os.mkdir(res_path)
key_dir_path = Path(f"{res_path}/keys/{preset}")
if os.environ["HEAAN_TYPE"] == "pi":
    use_gpu = False
    key_dir_path = key_dir_path / "pi"
else:
    use_gpu = True
if os.path.isdir(key_dir_path):
    generate_keys = False
from heaan_stat import Context
context = Context(
    key_dir_path='./keys',
    generate_keys=False,  
)

def run_plainxgb(X_train, y_train, X_test, y_test, max_bin, depth, n_estimators, bin_edges):
    my_model = plainboost(
        n_estimators=n_estimators,
        max_depth=depth,
        learning_rate=0.3,
        max_bin=max_bin,  
        bin_edges=bin_edges,
    )
    print("\n[plainXGB] Fitting on  dataset...")
    my_model.fit(X_train, y_train)  
    y_pred_my = my_model.predict(X_test)
    print("[plainXGB] Predictions on  data:", y_pred_my)
    accuracy_my = accuracy_score(y_test, y_pred_my)
    print(f"[plainXGB] Accuracy: {accuracy_my:.4f}")
    my_model.save_each_tree()
def run_HExgb(X_train, y_train, X_test, y_test, max_bin, depth, n_estimators, bin_edges):
    hist = Histogram(
        context,
        bin_edges=bin_edges,
        max_bin=max_bin,
        base_score=0.5,
    )
    hist.encode(X_train, y_train)
    hist.encrypt()
    hist.save()
    model = HEboost(
        context,
        n_estimators=n_estimators,
        max_depth=depth,
        learning_rate=0.3,  
        max_bin=max_bin,
    )
    model.fit(hist)
    inf_hist = Inference_Histogram(hist)
    inf_hist.encode(X_test)
    inf_hist.encrypt()
    model.predict(inf_hist)
    y_pred, _ = utils.print_inferec_res(context, model)
    acc, is_correct = utils.accuracy(y_pred, y_test)
if __name__ == "__main__":
    data_base_path = "../data"
    dataset_name = "iris"
    method = "sturges"
    max_bin = 8
    depth = 3
    ntree = 3
    train_path = os.path.join(data_base_path, dataset_name, method, "train.csv")  
    test_path = os.path.join(data_base_path, dataset_name, method, "test.csv")  
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train, y_train = train_df.drop("target", axis=1).values, train_df["target"].values
    X_test, y_test = test_df.drop("target", axis=1).values, test_df["target"].values
    bin_path = os.path.join("./processed_bin", dataset_name, method, f"bin{max_bin}", "cutpoints.json")  
    with open(bin_path, "r") as f:
        bin_edges = json.load(f)
    run_HExgb(X_train, y_train, X_test, y_test, max_bin, depth, ntree, bin_edges)
    run_plainxgb(X_train, y_train, X_test, y_test, max_bin, depth, ntree, bin_edges)