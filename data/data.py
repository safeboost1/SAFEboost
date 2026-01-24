import json
import os
import kagglehub
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import subprocess
def get_bins(data: np.ndarray, method: str) -> np.ndarray:
    """Compute bin edges using numpy's histogram_bin_edges."""
    return np.histogram_bin_edges(data, bins=method)
def assign_bins(data: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Assign bin indices (0-based) to data using numpy digitize."""
    return np.digitize(data, bins=bin_edges, right=False) - 1
def process_dataset(is_validset: bool = True):  
    methods = ["sturges"]
    data_list = ["processed_bank_marketing6", "iris", "breast-cancer", "spam"]
    for name in data_list:
        input_file = f"./{name}.csv"
        if not os.path.exists(input_file):
            print(f"⚠️ File not found: {input_file}")
            continue
        df = pd.read_csv(input_file)
        df.dropna(subset=["target"], inplace=True)
        print(df["target"].unique(), df["target"].dtype)
        if "target" not in df.columns:
            print(f"⚠️ 'The 'target' column does not exist: {name}.csv")
            continue
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "target"]
        df_scaled = df.copy()
        plain_dir = os.path.join(name, "plain")
        os.makedirs(plain_dir, exist_ok=True)
        df_plain = df_scaled.copy()
        unique_vals = df_plain["target"].dropna().unique()
        if len(unique_vals) == 2:
            if name == "steel":
                df_plain["target"] = df["target"] - 1
            else:
                df_plain["target"] = df_plain["target"].astype(int)
        else:
            df_plain["target"] = (df_plain["target"] == 2).astype(int)
        df_plain = df_plain.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df_plain)
        if is_validset:
            n_train = int(0.6 * n)
            n_test = int(0.2 * n)
            train_df = df_plain.iloc[:n_train]
            test_df = df_plain.iloc[n_train : n_train + n_test]
            valid_df = df_plain.iloc[n_train + n_test :]
            train_df.to_csv(os.path.join(plain_dir, "train.csv"), index=False)
            test_df.to_csv(os.path.join(plain_dir, "test.csv"), index=False)
            valid_df.to_csv(os.path.join(plain_dir, "valid.csv"), index=False)
            print(f"✅ [{name}/plain] 6:2:2 split save done")
            print(f"   ├─ train: {plain_dir}/train.csv")
            print(f"   ├─ test : {plain_dir}/test.csv")
            print(f"   └─ valid: {plain_dir}/valid.csv\n")
        else:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for fold, (train_idx, test_idx) in enumerate(skf.split(df_plain, df_plain["target"]), start=1):
                train_df = df_plain.iloc[train_idx]
                test_df = df_plain.iloc[test_idx]
                train_df.to_csv(os.path.join(plain_dir, f"train{fold}.csv"), index=False)
                test_df.to_csv(os.path.join(plain_dir, f"test{fold}.csv"), index=False)
                print(f"✅ [{name}/plain] Fold {fold} save done")
                print(f"   ├─ train{fold}.csv")
                print(f"   └─ test{fold}.csv\n")
        if is_validset:
            print(" valid:", valid_df["target"].value_counts().to_dict())
        for method in methods:
            max_vals = {}
            binned = pd.DataFrame()
            for col in numeric_cols:
                arr = df_scaled[col].values
                edges = get_bins(arr, method)
                binned[col] = assign_bins(arr, edges)
                max_vals[col] = int(binned[col].max())
            unique_vals = df["target"].unique()
            if len(unique_vals) == 2:
                binned["target"] = df["target"] - 1 if name == "steel" else df["target"]
            else:
                binned["target"] = (df["target"] == 2).astype(int)
            binned = binned.sample(frac=1, random_state=42).reset_index(drop=True)
            n = len(binned)
            method_dir = os.path.join(name, method)
            os.makedirs(method_dir, exist_ok=True)
            if is_validset:
                n_train = int(0.6 * n)
                n_test = int(0.2 * n)
                train_df = binned.iloc[:n_train]
                test_df = binned.iloc[n_train : n_train + n_test]
                valid_df = binned.iloc[n_train + n_test :]
                train_df.to_csv(os.path.join(method_dir, "train.csv"), index=False)
                test_df.to_csv(os.path.join(method_dir, "test.csv"), index=False)
                valid_df.to_csv(os.path.join(method_dir, "valid.csv"), index=False)
                print(f"✅ [{name}/{method}] 6:2:2 split save done")
                print(f"   ├─ train: {method_dir}/train.csv")
                print(f"   ├─ test : {method_dir}/test.csv")
                print(f"   └─ valid: {method_dir}/valid.csv")
            else:
                skf_b = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                fold_b = 0
                for train_idx, test_idx in skf_b.split(binned, binned["target"]):
                    fold_b += 1
                    train_df = binned.iloc[train_idx].reset_index(drop=True)
                    test_df = binned.iloc[test_idx].reset_index(drop=True)
                    train_path = os.path.join(method_dir, f"train{fold_b}.csv")
                    test_path = os.path.join(method_dir, f"test{fold_b}.csv")
                    train_df.to_csv(train_path, index=False)
                    test_df.to_csv(test_path, index=False)
                    print(f"✅ [{name}/{method}] Fold {fold_b} save done (train{fold_b}.csv / test{fold_b}.csv)")
                    print(f"   ├─ train Class distribution: {train_df['target'].value_counts().to_dict()}")
                    print(f"   └─ test  Class distribution: {test_df['target'].value_counts().to_dict()}")
            json_file = os.path.join(method_dir, "max_values.json")
            with open(json_file, "w") as f:
                json.dump({"global_max": max(max_vals.values())}, f, indent=2)
            print(f"   └─ max_values: {json_file} (ex: {list(max_vals.items())[:3]})\n")
def save_dataframe(df, name):
    path = f"./{name}.csv"
    df.to_csv(path, index=False)
    print(f"✅ save: {path}")
def load_and_save_ucirepo():
    """
    make credit card, bank marketing csv
    """
    """
    default_of_credit_card_clients = fetch_ucirepo(id=350)
    X_credit = default_of_credit_card_clients.data.features
    y_credit = default_of_credit_card_clients.data.targets
    y_credit.columns = ["target"]
    categorical_credit = ["X2", "X3", "X4"]
    X_credit_encoded = pd.get_dummies(X_credit, columns=categorical_credit, drop_first=True)
    X_credit_encoded = X_credit_encoded.astype(int)
    df_credit = pd.concat([X_credit_encoded, y_credit], axis=1)
    df_credit.to_csv("default_of_credit_card.csv", index=False)
    bank_marketing = fetch_ucirepo(id=222)
    X_bank = bank_marketing.data.features
    y_bank = bank_marketing.data.targets
    y_bank.columns = ["target"]
    y_bank["target"] = (y_bank["target"] == "yes").astype(int)
    categorical_bank = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
    X_bank_encoded = pd.get_dummies(X_bank, columns=categorical_bank, drop_first=True)
    X_bank_encoded = X_bank_encoded.astype(int)
    df_bank = pd.concat([X_bank_encoded, y_bank], axis=1)
    df_bank.to_csv("bank_marketing.csv", index=False)
    """
    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets
    y.columns = ["target"]
    df = pd.concat([X, y], axis=1)
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    df["target"] = df["target"].map({"<=50K": 0, ">50K": 1})
    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != "target"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    bool_cols = df.select_dtypes(include="bool").columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)
    cols = ["target"] + [c for c in df.columns if c != "target"]
    df = df[cols]
    csv_path = "adult.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved cleaned dataset to {csv_path} — shape: {df.shape}")
def load_and_save_arff(filepath, name):
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode("utf-8")
    if "class" in df.columns:
        df = df.rename(columns={"class": "target"})
    if "Class" in df.columns:
        df = df.rename(columns={"Class": "target"})
    save_dataframe(df, name)
def load_and_save_iris():
    data = load_iris(as_frame=True)
    save_dataframe(data.frame, "iris")
def load_and_save_breast():
    data = load_breast_cancer(as_frame=True)
    save_dataframe(data.frame, "breast-cancer")
def load_data():
    load_and_save_iris()
    load_and_save_breast()
    download_with_wget(
        "https://www.openml.org/data/download/44/dataset_44_spambase.arff",
        "datasets/spam.arff"
    )
    spam_arff = "./spam.arff"
    if os.path.exists(spam_arff):
        load_and_save_arff(spam_arff, "spam")
    else:
        print("❌ spam.arff no file.")
def download_with_wget(url, output_path):
    try:
        subprocess.run(["wget", "-c", url, "-O", output_path], check=True)
        print(f"✅ download done: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ download fail: {e}")
def hqsfl_gen_synthetic_datasets(random_seed=42):

    datasets = [
        ("hqsfl_syn3_80F.csv", 150_000, 80),
        ("hqsfl_syn3_100F.csv", 150_000, 100),
    ]
    for filename, n_samples, n_features in datasets:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=min(n_features, 2),
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=random_seed,
        )
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
        df["target"] = y
        df.to_csv(filename, index=False)
        print(f"✅ Saved {filename} (shape: {df.shape})")
def generate_uniform_datasets(random_seed=42):

    datasets = [
        ("secureXGB_syn_n30m50.csv", 30_000, 50),
        ("secureXGB_syn_n70m50.csv", 70_000, 50),
        ("secureXGB_syn_n50m30.csv", 50_000, 30),
        ("secureXGB_syn_n50m70.csv", 50_000, 70),
    ]
    np.random.seed(random_seed)
    for filename, n_samples, n_features in datasets:
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, size=n_samples)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
        df["target"] = y
        df.to_csv(filename, index=False)
        print(f"✅ Saved {filename} (shape: {df.shape})")
def load_and_save_credit_card():
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    print("Path to dataset files:", path)
def pre_process_credit_card_data():
    df = pd.read_csv("creditcard.csv")
    df.drop(["Time"], axis=1, inplace=True)
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df.drop(columns=["Amount"], inplace=True)
    df.rename(columns={"Class": "target"}, inplace=True)
    df.to_csv("credit_card_fraud.csv", index=False)
    print("✅ Preprocessed credit card data saved as 'credit_card_preprocessed.csv'")
if __name__ == "__main__":
    load_data()
    process_dataset(is_validset=True)  