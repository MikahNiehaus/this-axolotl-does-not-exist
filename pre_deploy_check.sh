#!/bin/bash
# Pre-deployment check script for Railway
# Run this script before deploying to Railway to ensure
# model files are valid and properly set up

set -e

echo "=================================================="
echo "  Axolotl AI App Pre-Deployment Verification"
echo "=================================================="

# Check if we're in the right directory
if [ ! -d "backend" ]; then
  echo "❌ Error: Please run this script from the project root directory"
  exit 1
fi

# Check Python is installed
if ! command -v python &> /dev/null; then
  echo "❌ Error: Python not found. Please install Python 3.6+"
  exit 1
fi

# Check Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
  echo "⚠️ Warning: Git LFS not found. Large model files might not be tracked properly."
  echo "   Install Git LFS: https://git-lfs.github.com/"
else
  echo "✅ Git LFS is installed"
  
  # Verify Git LFS configuration
  echo -e "\nGit LFS tracked patterns:"
  git lfs track
  
  echo -e "\nGit LFS tracked files:"
  git lfs ls-files
fi

# Verify model files
echo -e "\n✨ Verifying model integrity..."
python verify_model_integrity.py

if [ $? -ne 0 ]; then
  echo "❌ Model verification failed. Please fix model issues before deploying."
  exit 1
fi

# Check model file sizes
echo -e "\n📊 Model file sizes:"
ls -lh backend/data/gan_*.pth

# Check .gitattributes config
if grep -q "pth" .gitattributes; then
  echo "✅ .gitattributes has model file configuration"
else
  echo "⚠️ Warning: .gitattributes might be missing model file configurations"
  echo "   Run: git lfs track \"backend/data/*.pth\""
fi

# Check .gitignore exceptions
if grep -q "!backend/data/gan_" .gitignore; then
  echo "✅ .gitignore has model file exceptions"
else
  echo "⚠️ Warning: .gitignore might be blocking model files"
fi

# All checks passed
echo -e "\n✅ Pre-deployment checks completed successfully!"
echo "   You're now ready to deploy to Railway"
echo "   Railway deployment command: cd backend && railway up"
echo "=================================================="
