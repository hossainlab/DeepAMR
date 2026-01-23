# src/ml/download_models.py
from transformers import AutoTokenizer, AutoModel
import torch

def download_esm2_model():
    """Download ESM-2 model for protein sequences"""
    print("Downloading ESM-2 model...")
    
    # Start with smaller model for testing
    model_name = "facebook/esm2_t12_35M_UR50D"  # 35M parameters
    # Later upgrade to: "facebook/esm2_t33_650M_UR50D"  # 650M parameters
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Save locally
    tokenizer.save_pretrained("models/pretrained/esm2")
    model.save_pretrained("models/pretrained/esm2")
    print("ESM-2 downloaded successfully!")

def download_dnabert2_model():
    """Download DNABERT-2 for DNA sequences"""
    print("Downloading DNABERT-2 model...")
    
    model_name = "zhihan1996/DNABERT-2-117M"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    tokenizer.save_pretrained("models/pretrained/dnabert2")
    model.save_pretrained("models/pretrained/dnabert2")
    print("DNABERT-2 downloaded successfully!")

if __name__ == "__main__":
    download_esm2_model()
    download_dnabert2_model()