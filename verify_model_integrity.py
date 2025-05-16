#!/usr/bin/env python
"""
Model Integrity Verification Script

This script verifies that PyTorch model files in the repository are valid
and can be loaded properly. It helps catch issues before deployment.
"""

import os
import sys
import torch
from pathlib import Path

def verify_model_file(file_path):
    """Verify that a model file is valid and can be loaded by PyTorch."""
    print(f"Verifying model file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File does not exist: {file_path}")
        return False
        
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size:,} bytes")
    
    if file_size == 0:
        print(f"❌ ERROR: File is empty (0 bytes)")
        return False
    
    # Minimum expected size for a real model (10KB)
    if file_size < 10 * 1024:
        print(f"⚠️ WARNING: File is suspiciously small ({file_size} bytes)")
        
    try:
        # Try loading the model
        checkpoint = torch.load(file_path, map_location="cpu")
        
        # Verify it has expected keys for a GAN model
        required_keys = ['G', 'D', 'epoch']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            print(f"❌ ERROR: Model missing required keys: {missing_keys}")
            return False
            
        # Print all keys in the checkpoint
        print(f"Model keys: {list(checkpoint.keys())}")
        
        # Check if G is a state dict
        if not isinstance(checkpoint['G'], dict):
            print("❌ ERROR: 'G' is not a state dictionary")
            return False
            
        print(f"✅ Model verification successful!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load model: {str(e)}")
        return False

def main():
    """Main function to verify all model files."""
    repo_root = Path(__file__).parent
    model_files = [
        repo_root / 'backend' / 'data' / 'gan_full_model.pth',
        repo_root / 'backend' / 'data' / 'gan_checkpoint.pth',
    ]
    
    print(f"Checking {len(model_files)} model files...")
    
    # Make sure the data directory exists
    data_dir = repo_root / 'backend' / 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Created directory: {data_dir}")
        
    success = True
    for model_file in model_files:
        print(f"\n{'=' * 50}")
        if not verify_model_file(model_file):
            success = False
    
    if not success:
        print("\n❌ One or more model verification checks failed!")
        print("   Run backend/check_model.sh to create valid model files.")
        sys.exit(1)
    else:
        print("\n✅ All model verification checks passed!")
        
if __name__ == "__main__":
    main()
