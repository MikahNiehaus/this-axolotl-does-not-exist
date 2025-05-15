# Deploying to Railway

This guide provides instructions on how to deploy the Axolotl GAN application to Railway.

## Prerequisites

1. Make sure you have a Railway account (create one at [railway.app](https://railway.app) if needed)
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

- The Axolotl GAN application uses a pre-trained model file (`gan_checkpoint.pth`) that is included in the repository.
- The application is configured to use CPU if GPU is not available, so it will work on Railway's standard instances.
- The `check_model.sh` script ensures the application won't crash if the model file is missing, but it's best to include the actual trained model.

## Troubleshooting

If your deployment fails:

1. Check Railway logs: `railway logs`
2. Ensure your model file is correctly included in git
3. Verify that all dependencies are properly listed in `requirements.txt`
4. Check that the server can bind to the PORT environment variable provided by Railway
