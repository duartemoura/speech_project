# run_all_experiments.py
# This script runs all the experiment YAMLs you've created

import subprocess

# List of experiment config files
experiments = [
    "train.yaml",
    "train_exp1_features.yaml",
    "train_exp2_norm.yaml",
    "train_exp3_noaugment.yaml",
    "train_exp4_seed42.yaml",
    "train_exp5_embed_small.yaml",
    "train_exp6_classifier_deep.yaml"
]

for config in experiments:
    print(f"\n🔧 Running: {config}")
    try:
        subprocess.run(["python", "train.py", config], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {config}: {e}")
