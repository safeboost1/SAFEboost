import os
os.environ["HEAAN_TYPE"] = "pi"
print(os.environ["HEAAN_TYPE"])
import warnings
warnings.filterwarnings("ignore")
import csv
import os
import numpy as np
from pathlib import Path
import heaan_stat
from heaan_stat import Context
import utils 
from client.client import Client
from server.server import Server
generate_keys = False
preset = "FGb"
param = heaan_stat.HEParameter(preset)
use_gpu = False
key_path = "./Keys"
if not os.path.exists(key_path):
    os.mkdir(key_path)
key_dir_path = Path(f"{key_path}/{preset}")
if os.environ["HEAAN_TYPE"] == "pi":
    key_dir_path = key_dir_path / "pi"
    use_gpu = False
if not os.path.isdir(key_dir_path):
    generate_keys = True
context = Context(
    parameter=param,
    key_dir_path='./keys',
    generate_keys=True,  
)
def save_result_to_csv(filename, d, nbits, nt, dep, avg_time, avg_acc):
    file_exists = os.path.exists(filename)
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["dimension", "nbits", "nt", "depth", "avg_eval_time", "avg_accuracy"])
        writer.writerow([d, nbits, nt, dep, avg_time, avg_acc])
def run_levelup(context):
    dataset = {"Breast": (30, 4), "Iris": (4, 4), "Spam": (57, 4)}
    ntree = [1, 3, 5, 7, 5, 9,11,13, 15]
    os.makedirs("exp", exist_ok=True)
    for data in dataset.keys():
        d, n_bits = dataset[data]
        result_file = os.path.join("exp", f"{data}_result.csv")
        for nt in ntree:
            for dep in range(2, 9):
                params = {"n_estimators": nt, "max_depth": dep, "n_bits": n_bits, "d": d}
                all_times = []
                all_accuracies = []
                model_path = os.path.join(f"../{data}", f"depth{dep}")
                for _ in range(11):
                    clf = Server(context, **params)
                    client = Client(context, **params)
                    plain_input, ct_input = client.make_input()
                    plain_pred = utils.evaluate_model(model_path, plain_input)
                    print('plain_pred:',plain_pred)
                   
                    ct_pred, eval_time = clf.eval_tree(ct_input, model_path)
                    print("eval_time:", eval_time)
                    ct_check = utils.check_ct_pred(ct_pred, clf)
                    print("ct_check:", ct_check)
                    check_list = [1 if ct_check[i] == plain_pred else 0 for i in range(len(ct_check))]
                    num_correct = sum(check_list)
                    total = len(check_list)
                    accuracy = num_correct / total * 100
                    print(f"✔ num_correct: {num_correct} / {total}")
                    print(f"✅ accuracy: {accuracy:.2f}%")
                    if _ != 0:
                        all_times.append(eval_time)
                        all_accuracies.append(accuracy)
                avg_time = np.mean(all_times)
                avg_accuracy = np.mean(all_accuracies)
                save_result_to_csv(result_file, d, n_bits, nt, dep, avg_time, avg_accuracy)
if __name__ == "__main__":
    run_levelup(context)