# Deploying to Railway

This guide provides instructions on how to deploy the Axolotl GAN application to Railway.

## Prerequisites

1. Make sur4. Check that the server can bind to the PORT environment variable provided by Railway

## Setting Up Git LFS for Model Files

Git LFS (Large File Storage) is recommended for managing large model files in your repository. This ensures that model files are properly tracked and transferred during deployment.

### Installation and Setup

1. **Install Git LFS**:
   ```bash
   # For macOS (using Homebrew)
   brew install git-lfs
   
   # For Ubuntu/Debian
   sudo apt-get install git-lfs
   
   # For Windows (via Chocolatey)
   choco install git-lfs
   
   # Initialize Git LFS
   git lfs install
   ```

2. **Track Model Files**:
   ```bash
   # Track all PyTorch models
   git lfs track "backend/data/*.pth"
   git lfs track "backend/data/*.pt"
   
   # Commit the .gitattributes file
   git add .gitattributes
   git commit -m "Configure Git LFS for model files"
   ```

3. **Add Models to Repository**:
   ```bash
   # Add model files
   git add backend/data/gan_checkpoint.pth backend/data/gan_full_model.pth
   git commit -m "Add GAN model files"
   
   # Push changes including LFS objects
   git push
   ```

### Verifying Git LFS Setup

To check if Git LFS is properly tracking your model files:

```bash
# List all tracked patterns
git lfs track

# Check which files are being managed by Git LFS
git lfs ls-files
```

### Railway Integration

When deploying to Railway with Git LFS:

1. Make sure your Railway project is linked to the GitHub repository
2. Railway should automatically fetch LFS objects during deployment
3. If issues persist, you may need to manually upload model files using SFTP or through the Railway dashboard you have a Railway account (create one at [railway.app](https://railway.app) if needed)
2. Install the Railway CLI:
   ```bash
   npm i -g @railway/cli
   ```
3. Login to Railway:
   ```bash
   railway login
   ```

## Deployment Steps

### 1. Initialize the Railway Project

Navigate to your backend directory:

```bash
cd backend
```

Link to an existing project or create a new one:

```bash
railway init
```

### 2. Deploy to Railway

Push your application to Railway:

```bash
railway up
```

### 3. Verify Deployment

Once deployed, you can verify your application is working:

```bash
railway open
```

## Environment Variables

If needed, you can set environment variables using the Railway Dashboard or CLI:

```bash
railway variables set KEY=VALUE
```

## Logs and Monitoring

To view logs of your deployment:

```bash
railway logs
```

## Configuring Your Frontend

Update your frontend to point to your new Railway API endpoint:

1. In your frontend's `vite.config.js`, update the proxy target to your Railway URL
2. Or set an environment variable in your frontend deployment for the API URL

## Multiple Services

If you want to deploy both frontend and backend to Railway:

1. Create separate Railway services for frontend and backend
2. Configure them to communicate with each other

## Important Notes

- The Axolotl GAN application requires two model files to operate properly:
  - `gan_checkpoint.pth`: A checkpoint model for continued training
  - `gan_full_model.pth`: The primary model used for inference
- The application is configured to use CPU if GPU is not available, so it will work on Railway's standard instances.
- The `check_model.sh` script ensures the application won't crash if the model files are missing, but using actual trained models is highly recommended for best results.

## Model File Management

### Verifying Model Files Before Deployment

Always validate your model files before deploying to Railway:

```bash
# Run the verification script from the project root
python verify_model_integrity.py
```

This script checks that:
- Model files exist and are not empty
- Files are a proper size (not just a few bytes)
- PyTorch can successfully load the model files
- Models contain the required keys (G, D, epoch)

### Handling Model File Issues

If you encounter the "invalid load key, 'v'" error or other model loading issues:

1. **Local Verification**: Make sure your local model files are valid:
   ```bash
   cd backend
   ./check_model.sh
   ```

2. **File Size Check**: Ensure model files aren't just placeholders:
   ```bash
   ls -lh data/gan_*.pth
   ```
   Proper model files should be at least several MB in size.

3. **Git LFS Setup**: For large model files, consider using Git LFS:
   ```bash
   # Install Git LFS
   git lfs install
   
   # Track model files
   git lfs track "backend/data/*.pth"
   
   # Make sure .gitattributes is committed
   git add .gitattributes
   
   # Add and commit your model files
   git add backend/data/*.pth
   git commit -m "Add model files via Git LFS"
   ```

## Troubleshooting

If your deployment fails:

1. Check Railway logs: `railway logs`
2. Look for specific errors related to model loading:
   - "invalid load key" typically means corrupted or placeholder model files
   - File size errors mean the model files were not properly uploaded to Railway
3. Run the model verification script locally before deploying
4. If model files are corrupted in deployment, try:
   ```bash
   # Force Railway to rebuild with fresh files
   railway up --detach --force
   ```
5. Verify that all dependencies are properly listed in `requirements.txt`
6. Check that the server can bind to the PORT environment variable provided by Railway
