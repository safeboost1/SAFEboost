
#  SAFE-Boost: Privacy-Preserving XGBoost using Homomorphic Encryption

**SAFE-Boost** is a framework that performs XGBoost training and inference while preserving data privacy using Homomorphic Encryption (HE). This project is built upon the **HEaaN.Stat** library.

---

##  Installation & Setup

### Prerequisite: HEaaN.Stat (Docker)

This project requires the **HEaaN.Stat** environment for encrypted computations. Please visit the link below to set up the environment in your Docker container:

* **HEaaN.Stat Installation:** [https://iheaan.com/?target=heaan-stat](https://iheaan.com/?target=heaan-stat)

### Clone the Repository

Clone this repository inside your configured Docker environment:

```bash
git clone https://github.com/safeboost1/SAFEboost.git
cd SAFEboost

```

---

##  Quick Start: Training & Inference



Run `run.py` in the `SAFE_Boost` directory to perform the entire process from training to inference under homomorphic encryption.

```bash
cd SAFE_Boost
pip install -r requirements.txt # Install required dependencies
python run.py

```

To change the dataset or adjust hyperparameters, modify the following configuration in SAFE_Boost/run.py

```python
dataset_name = "iris"  # Supported: "iris", "default_of_credit_card", "breast-cancer"
max_bin = 8            # Max bins (Choose from [4, 6, 8, 10, 12])
depth = 3              # Tree depth
ntree = 3              # Number of trees

```
If you wish to use other datasets, please reach out via the email address listed in the Contact section.

* **Functionality:** Executes HE-based Training and Inference.
* **Output:** You can verify the accuracy of the encrypted computation results alongside the plaintext model for comparison.

---

##  Benchmark: SAFE-I vs Levelup (RCC)

The `benchmark` directory allows you to compare **SAFE-I (SAFE-Boost's Inference)** and **Levelup’s RCC (Reduced Complexity Comparison)** methods in an encrypted environment.

### 1. Pre-step: Generate Tree

Before starting the benchmark, you must generate the tree models:

```bash
python generate_tree.py

```

###  Running Benchmarks

Navigate to each folder (`SAFE-I`, `levelup`) and run the benchmarks for the **Iris, Breast-Cancer, and Spambase** datasets.

* **Run SAFE-I:** `cd benchmark/SAFE-I && python run.py`
* **Run Levelup:** `cd benchmark/levelup && python run.py`

All results (performance metrics and logs) are saved in the `exp` directory within each respective folder.

---

##  Troubleshooting: Key Generation

If you encounter the following error:

> `heaan_stat.exceptions.SDKHEError: [HEaaN.Stat SDK] Secret key does not exist`

Please modify the `generate_keys` argument in the `Context` configuration:

* **If keys do not exist (First run):** Set `generate_keys=True`
* **If keys already exist:** Keep `generate_keys=False`

```python
context = Context(
    parameter=param,
    key_dir_path='./keys',
    generate_keys=True,  # Change to True if keys are missing
)

```
---
##  Execution Modes: Simulation vs. Real HE
Since Homomorphic Encryption (HE) is computationally intensive and this version of the library is CPU-bound, we provide a Simulation Mode for logic verification. You can toggle this via an environment variable in run.py.

Simulation Mode (pi): This is the default setting. It mimics HE logic using plaintext, allowing you to verify the algorithm flow quickly.

Real HE Mode (real): This performs actual encrypted computations. It is significantly slower and requires more memory/CPU resources.

```python
import os
os.environ["HEAAN_TYPE"] = "pi"  # Change to "real" for actual encryption

```
---

## Notice (Commercial Use & Performance)

* **Public Library Version:** Due to commercial security policies and licensing restrictions, this repository contains the **public version** of the library.
* **Performance Note:** Please note that the performance may differ from the results reported in official papers or commercial environments depending on the library version and hardware configuration used.

---

## Contact

For inquiries regarding performance on additional datasets, detailed implementation details, or collaboration, please contact us via email.

* **Email:** safeboost1@gmail.com
