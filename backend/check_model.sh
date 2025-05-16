#!/bin/bash
# Check if the GAN checkpoint exists and create a dummy one if not

DATA_DIR="data"
CHECKPOINT_PATH="$DATA_DIR/gan_checkpoint.pth"
SAMPLE_DIR="$DATA_DIR/gan_samples"

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$SAMPLE_DIR"

# Record model hash for verification
MODEL_HASH=""
if [ -f "$CHECKPOINT_PATH" ]; then
    if command -v md5sum > /dev/null 2>&1; then
        MODEL_HASH=$(md5sum "$CHECKPOINT_PATH" | cut -d ' ' -f 1)
    elif command -v md5 > /dev/null 2>&1; then
        MODEL_HASH=$(md5 -q "$CHECKPOINT_PATH")
    fi
    echo "Model found with hash: $MODEL_HASH"
fi

# Check both full model and checkpoint file
FULL_MODEL_PATH="$DATA_DIR/gan_full_model.pth"

create_viable_model() {
    local target_file=$1
    echo "Creating a minimal viable model file at $target_file"
    
    # This creates a real but tiny GAN model that can actually be loaded
    # Much better than just an empty dict which causes "invalid load key" errors
    python -c "
import torch
import torch.nn as nn

# Create minimal but loadable generator
class SimpleGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 3*32*32),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x).view(-1, 3, 32, 32)

# Create minimal discriminator
class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*32*32, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Create and save dummy model
G = SimpleGenerator()
D = SimpleDiscriminator()
torch.save({
    'G': G.state_dict(),
    'D': D.state_dict(),
    'epoch': 1,
    'img_size': 32
}, '$target_file')
"
}

if [ ! -f "$FULL_MODEL_PATH" ]; then
    echo "GAN full model not found. Setting up a minimal viable model."
    create_viable_model "$FULL_MODEL_PATH"
    
    # Also create checkpoint if it doesn't exist
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "Also creating checkpoint model"
        cp "$FULL_MODEL_PATH" "$CHECKPOINT_PATH"
    fi
elif [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Full model exists but checkpoint doesn't. Setting up checkpoint."
    cp "$FULL_MODEL_PATH" "$CHECKPOINT_PATH"
else
    echo "Using existing model files:"
    echo "- Full model: $FULL_MODEL_PATH"
    echo "- Checkpoint: $CHECKPOINT_PATH"
fi

echo "Model check complete."
