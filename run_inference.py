#!/usr/bin/env python3
"""Script to run inference across all trained models and data files.

This script will:
1. Find all trained models in the results directory
2. Process all audio files in the data directory
3. Run speaker verification between all pairs of files
4. Save results to a CSV file

Usage:
    python run_inference.py
"""
import os
import glob
import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from speechbrain.inference.speaker import SpeakerRecognition
from hyperpyyaml import load_hyperpyyaml

def get_model_configs():
    """Get all model configurations from the results directory."""
    model_dirs = glob.glob("results/speaker_id/*/")
    configs = []
    
    for model_dir in model_dirs:
        # Check if this is a valid model directory
        if not os.path.exists(os.path.join(model_dir, "save")):
            continue
            
        # Try to find the training config
        train_yaml = glob.glob(os.path.join(model_dir, "*.yaml"))
        if not train_yaml:
            continue
            
        configs.append({
            "model_dir": model_dir,
            "config_file": train_yaml[0],
            "save_dir": os.path.join(model_dir, "save")
        })
    
    return configs

def get_audio_files():
    """Get all audio files from the data directory."""
    audio_files = []
    
    # Search in LibriSpeech directory
    librispeech_dir = "data/LibriSpeech"
    if os.path.exists(librispeech_dir):
        audio_files.extend(glob.glob(os.path.join(librispeech_dir, "**/*.flac"), recursive=True))
    
    return audio_files

def load_model(model_config):
    """Load a model with its configuration."""
    try:
        # Load the training config to get model parameters
        with open(model_config["config_file"], encoding="utf-8") as fin:
            hparams = load_hyperpyyaml(fin)
        
        # Create a temporary inference config
        inference_config = {
            "n_mels": hparams.get("n_mels", 23),
            "emb_dim": hparams.get("emb_dim", 512),
            "n_classes": hparams.get("n_classes", 28),
            "embedding_model": hparams.get("embedding_model", None),
            "classifier": hparams.get("classifier", None)
        }
        
        # Save temporary inference config in the model's save directory
        temp_inference_yaml = os.path.join(model_config["save_dir"], "temp_inference.yaml")
        with open(temp_inference_yaml, "w") as f:
            f.write(f"""# Feature parameters
n_mels: {inference_config['n_mels']}
emb_dim: {inference_config['emb_dim']}
n_classes: {inference_config['n_classes']}

# Feature extraction
compute_features: !new:speechbrain.lobes.features.Fbank
    n_mels: !ref <n_mels>

# Mean and std normalization
mean_var_norm: !new:speechbrain.processing.features.InputNormalization
    norm_type: sentence
    std_norm: False

# Models
embedding_model: !new:custom_model.Xvector
    in_channels: !ref <n_mels>
    activation: !name:torch.nn.LeakyReLU
    tdnn_blocks: 5
    tdnn_channels: [512, 512, 512, 512, 1500]
    tdnn_kernel_sizes: [5, 3, 3, 1, 1]
    tdnn_dilations: [1, 2, 3, 1, 1]
    lin_neurons: !ref <emb_dim>

classifier: !new:custom_model.Classifier
    input_shape: [null, null, !ref <emb_dim>]
    activation: !name:torch.nn.LeakyReLU
    lin_blocks: 1
    lin_neurons: !ref <emb_dim>
    out_neurons: !ref <n_classes>

# Modules
modules:
    compute_features: !ref <compute_features>
    embedding_model: !ref <embedding_model>
    classifier: !ref <classifier>
    mean_var_norm: !ref <mean_var_norm>

# Pretrainer
pretrainer: !new:speechbrain.utils.parameter_transfer.Pretrainer
    loadables:
        embedding_model: !ref <embedding_model>
        classifier: !ref <classifier>
        normalizer: !ref <mean_var_norm>
""")
        
        # Initialize the speaker recognition system
        verifier = SpeakerRecognition.from_hparams(
            source=model_config["save_dir"],  # Use the model's save directory as source
            hparams_file="temp_inference.yaml",  # Use relative path
            savedir=model_config["save_dir"]  # Use the model's save directory
        )
        
        return verifier, temp_inference_yaml
        
    except Exception as e:
        print(f"Error loading model {model_config['model_dir']}: {str(e)}")
        return None, None

def run_inference():
    """Run inference across all models and audio files."""
    # Get all model configurations
    model_configs = get_model_configs()
    print(f"Found {len(model_configs)} models")
    
    # Get all audio files
    audio_files = get_audio_files()
    print(f"Found {len(audio_files)} audio files")
    
    # Create results directory
    results_dir = "inference_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Process each model
    for model_config in model_configs:
        print(f"\nProcessing model: {model_config['model_dir']}")
        
        # Load model
        verifier, temp_yaml = load_model(model_config)
        if verifier is None:
            continue
            
        try:
            # Create results dataframe
            results = []
            
            # Process all pairs of audio files
            for i, file1 in enumerate(tqdm(audio_files)):
                for file2 in audio_files[i+1:]:
                    try:
                        # Get speaker IDs from filenames
                        spk1 = os.path.basename(file1).split("-")[0]
                        spk2 = os.path.basename(file2).split("-")[0]
                        
                        # Run verification
                        score, prediction = verifier.verify_files(file1, file2)
                        
                        # Store results
                        results.append({
                            "model": os.path.basename(model_config["model_dir"]),
                            "file1": file1,
                            "file2": file2,
                            "speaker1": spk1,
                            "speaker2": spk2,
                            "same_speaker": spk1 == spk2,
                            "score": score,
                            "prediction": prediction,
                            "correct": (spk1 == spk2) == prediction
                        })
                    except Exception as e:
                        print(f"Error processing {file1} vs {file2}: {str(e)}")
            
            # Save results
            if results:
                df = pd.DataFrame(results)
                output_file = os.path.join(results_dir, f"{os.path.basename(model_config['model_dir'])}_results.csv")
                df.to_csv(output_file, index=False)
                print(f"Saved results to {output_file}")
                
                # Calculate and print accuracy
                accuracy = df["correct"].mean() * 100
                print(f"Accuracy: {accuracy:.2f}%")
            
        finally:
            # Clean up temporary files
            if temp_yaml and os.path.exists(temp_yaml):
                os.remove(temp_yaml)

if __name__ == "__main__":
    run_inference() 