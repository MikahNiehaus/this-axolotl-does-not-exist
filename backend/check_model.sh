#!/bin/bash
# Check if the GAN checkpoint exists and create a dummy one if not

DATA_DIR="data"
CHECKPOINT_PATH="$DATA_DIR/gan_checkpoint.pth"
SAMPLE_DIR="$DATA_DIR/gan_samples"

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$SAMPLE_DIR"

# Check if the checkpoint file exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "GAN checkpoint not found. Setting up a dummy model for testing."
    # This is just a placeholder to ensure the app doesn't crash in test environments
    # In a real deployment, you should have your model file included in your repository
    python -c "import torch; torch.save({'G': {}, 'D': {}, 'epoch': 0}, '$CHECKPOINT_PATH')"
fi

echo "Model check complete."
