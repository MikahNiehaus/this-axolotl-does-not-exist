# Axolotl AI App

This project is an AI application that generates images of axolotls using PyTorch diffusion models. The application utilizes a progressive resolution neural network model to create realistic images of axolotls, similar to the concept of "This Person Does Not Exist."

## Features

- **Image Scraping**: Automated collection of axolotl images from web sources with duplicate detection
- **Data Preprocessing**: Filtering and splitting the dataset into training and testing sets
- **Progressive Resolution Diffusion Model**: Neural network that can be trained at multiple resolutions
- **Dynamic Resolution Scaling**: Ability to increase resolution as training progresses
- **Memory-Optimized Training**: CUDA optimizations for efficient GPU usage
- **Error Recovery**: Automatic fallback to CPU when GPU errors occur
- **Railway Deployment Ready**: Configuration for easy deployment to Railway

## Project Structure

The project is organized into two main directories: `backend` and `frontend`.

### Backend

The backend is built using Python and PyTorch. It handles image generation using diffusion models.

- **app.py**: The main entry point for the backend application.
- **requirements.txt**: Lists the Python dependencies required for the backend.
- **train_diffusion.py**: The main diffusion model implementation with progressive resolution scaling.
- **scrape_axolotl_images.py**: Script for collecting axolotl images from search engines.
- **split_train_test.py**: Script for preprocessing data and creating train/test splits.
- **generate_axolotl.py**: Helper script for batch generation of axolotl images.
- **data/**: Directory containing image data and trained models.

### Frontend

The frontend provides the user interface for interacting with the backend.

- **public/index.html**: The main HTML file for the web application.
- **src/App.jsx**: The main component of the React application.
- **src/components/**: React components for the user interface.
- **package.json**: Configuration file for npm, listing dependencies and scripts.

## Setup Instructions

### Backend

1. Navigate to the `backend` directory.
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the image scraper to collect axolotl images:
   ```
   python scrape_axolotl_images.py
   ```
4. Split data into training and testing sets:
   ```
   python split_train_test.py
   ```
5. Train the diffusion model:
   ```
   python train_diffusion.py
   ```
6. Generate images using the trained model:
   ```
   python generate_axolotl.py --count 5
   ```
   
### Advanced Usage

#### Training with Progressive Resolution

Train with increasing resolution (starts low, gradually increases):
```
python backend/train_diffusion.py --resolution 2.0
```

Start fresh training (ignore checkpoints):
```
python backend/train_diffusion.py --fresh
```

#### Generating High-Quality Images

Generate a high-quality image:
```
python backend/train_diffusion.py --mode sample --steps 200 --resolution 1.5 --upscale 2.0
```

Use CPU for more stable generation:
```
python backend/generate_axolotl.py --cpu
```

#### Scaling an Existing Model

Convert a trained model to a higher resolution:
```
python backend/train_diffusion.py --mode scale --model data/best_diffusion_model.pth --target-scale 2.0 --output-model data/high_res_model.pth
```

### Memory Issues Troubleshooting

If you encounter CUDA memory errors:
1. Reduce resolution: `--resolution 0.75`
2. Reduce steps: `--steps 50` 
3. Force CPU usage: `--cpu`

### Frontend

1. Navigate to the `frontend` directory.
2. Install the required npm packages:
   ```
   npm install
   ```
3. Start the React application:
   ```
   npm start
   ```

## Usage

Once both the backend and frontend are running, you can access the application in your web browser. The frontend will allow you to request and view generated axolotl images.

## Deployment

### Railway Deployment

The backend of this application is configured for deployment on Railway. The necessary configuration files are included:

- `Procfile`: Specifies the command to start the application
- `railway.json`: Contains Railway-specific configuration
- `start.sh`: Script for starting the application
- `check_model.sh`: Ensures the model file exists

For detailed deployment instructions, see the [RAILWAY_DEPLOYMENT.md](RAILWAY_DEPLOYMENT.md) guide.

### Model Files

The GAN model checkpoint (`gan_checkpoint.pth`) is included in the repository and won't be ignored by git, ensuring it's available when deploying to Railway.

#### Tracking Model File Changes

The model files are configured to be properly tracked by Git even when they change. We use:

1. `.gitattributes` file to properly handle binary model files
2. Git hooks to detect model file changes automatically
3. Automated Git pushes during training (see below)

The checkpoint file (`gan_checkpoint.pth`) is updated every 100 epochs during training to capture model improvements regularly.

#### Automated Git Model Versioning

The training process now automatically commits and pushes model changes to Git:

- **Local checkpoint saves**: Every 100 epochs  
- **Git pushes to main branch**: Every 1000 epochs

This ensures your model improvements are automatically versioned and backed up to the remote repository without manual intervention. The system uses the `GitModelHandler` class in `backend/models/git_model_handler.py` to manage this process.

To test the Git integration without running a full training cycle:

```powershell
# Test the Git model handler with the default model file
.\test_git_model.ps1

# Test with a different model file
.\test_git_model.ps1 -ModelPath "data/another_model.pth"

# Test with a different branch
.\test_git_model.ps1 -Branch "development"
```

If you make changes to your model files and want to ensure they're tracked:

```powershell
# Run this PowerShell script to ensure model changes are tracked
.\track_models.ps1
```

For best results with large model files, consider installing [Git LFS](https://git-lfs.github.com/).

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.#   t h i s - a x o l o t l - d o e s - n o t - e x i s t 
 
 