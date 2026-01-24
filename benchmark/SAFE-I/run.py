import os
os.environ["HEAAN_TYPE"] = "pi"
print(os.environ["HEAAN_TYPE"])
import warnings
warnings.filterwarnings("ignore")
import csv
import json
import os
import numpy as np
from pathlib import Path
import heaan_stat
import utils
from inference import MyXGBClassifier
from heaan_stat import Context
generate_keys = False
preset = "FGb"
print(preset)
param = heaan_stat.HEParameter(preset)
use_gpu = False
key_path = "./Keys"
if not os.path.exists(key_path):
    os.mkdir(key_path)
context = Context(
    parameter=param,
    key_dir_path='./keys',
    generate_keys=False,  
)
def predict_from_tree(model_path, input_vector):

    json_path = os.path.join(model_path, "model.json")
    with open(json_path, "r") as f:
        node = json.load(f)
    while "leaf" not in node:
        feature_idx = int(node["split"][1:])
        threshold = node["split_condition"]
        if input_vector[feature_idx] < threshold:
            node = node["children"][0]  
        else:
            node = node["children"][1]  
    return node["leaf"]
def save_result_to_csv(filename, d, nbits, nt, dep, avg_time, avg_acc):
    file_exists = os.path.exists(filename)
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["dimension", "nbits", "nt", "depth", "avg_eval_time", "avg_accuracy"])
        writer.writerow([d, nbits, nt, dep, avg_time, avg_acc])
def run_safe(context):
    dataset = {"Breast": (30, 12), "Iris": (4, 10), "Spam": (57, 15)}
    ntree = [1, 3, 5, 7, 5, 9,11,13, 15]
    os.makedirs("exp", exist_ok=True)
    for data in dataset.keys():
        d, n_max = dataset[data]
        result_file = os.path.join("exp", f"{data}_result.csv")
        for nt in ntree:
            for dep in range(2, 9):
                all_times = []
                all_accuracies = []
                model_path = os.path.join(f"../{data}", f"depth{dep}")
                model = MyXGBClassifier(context=context, n_estimators=nt, max_depth=dep, d=d, n_max=n_max)
                for _ in range(11):
                    input = model.generate_random_input()
                    print('input:',input)
                    plain_pred = predict_from_tree(model_path=model_path, input_vector=input)
                    print("plain_predict:", plain_pred)
                    ct_predict,eval_time = model.predict(model_path=model_path, input=input)
                    ct_check = utils.check_ct_pred(context, ct_pred=ct_predict)
                    print("ct_check:", ct_check)
                    check_list = [1 if ct_check[i] == plain_pred else 0 for i in range(len(ct_check))]
                    num_correct = sum(check_list)
                    total = len(check_list)
                    accuracy = num_correct / total * 100
                    print(f"✔ num_correct: {num_correct} / {total}")
                    print(f"✅ acc: {accuracy:.2f}%")
                    if _ != 0:
                        all_times.append(eval_time)
                        all_accuracies.append(accuracy)
                avg_time = np.mean(all_times)
                avg_accuracy = np.mean(all_accuracies)
                save_result_to_csv(result_file, d, n_max, nt, dep, avg_time, avg_accuracy)
if __name__ == "__main__":
    run_safe(context)