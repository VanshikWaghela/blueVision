# Deployment Guide

This guide explains how to deploy the Blueprint Symbol Detector to GitHub and Hugging Face Spaces.

## 1. Preparing for GitHub

### 1.1. Initialize Git and Git LFS

First, initialize a Git repository and set up Git LFS for handling large model files:

```bash
# Install Git LFS if not already installed
# On macOS: brew install git-lfs
# On Ubuntu: sudo apt-get install git-lfs

# Initialize Git repository
git init

# Setup Git LFS
git lfs install
```

The `.gitattributes` file is already configured to use Git LFS for model files.

### 1.2. Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Follow GitHub's instructions to push your existing repository

### 1.3. Push to GitHub

```bash
# Add all files
git add .

# Commit
git commit -m "Initial commit"

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/blueprint-detector.git

# Push
git push -u origin main
```

## 2. Deploying to Hugging Face Spaces

### 2.1. Prepare Files for Hugging Face

1. Make sure these files are in your repository root:
   - `app.py` - Gradio interface
   - `requirements-huggingface.txt` - Dependencies
   - `README-huggingface.md` - Documentation for the Space

2. Include at least one sample image in `data/images/` for the examples to work

### 2.2. Create a New Hugging Face Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Fill in the form:
   - Owner: Your username
   - Space name: `blueprint-detector` (or your preferred name)
   - License: Choose an appropriate license
   - SDK: Choose "Gradio"
   - Hardware: CPU (or GPU if needed)
   - Visibility: Public or Private

### 2.3. Clone Space and Push Files

```bash
# Clone your new space
git clone https://huggingface.co/spaces/YOUR_USERNAME/blueprint-detector

# Copy files from your project to the space directory
cp -r app.py app/ requirements-huggingface.txt README-huggingface.md data/ blueprint-detector/

# Enter the space directory
cd blueprint-detector

# Rename files
mv README-huggingface.md README.md
mv requirements-huggingface.txt requirements.txt

# Add the files
git add .
git commit -m "Initial space setup"
git push
```

### 2.4. Setting Up Space Runtime

After pushing, go to the "Settings" tab of your Space on Hugging Face and:

1. Under "Repository", enable "Always use the latest revision"
2. Under "Variables & Secrets", add any required secrets
3. Under "Hardware", ensure you're using at least a "CPU-basic" for this application

### 2.5. Troubleshooting

- If facing memory issues, consider reducing the model size or using a different hardware tier
- Check Space logs for any errors during startup
- Make sure all paths in `app.py` correctly reference files in the Space structure

## 3. Updating Your Deployment

### 3.1. Updating GitHub

```bash
git add .
git commit -m "Update with improvements"
git push
```

### 3.2. Updating Hugging Face Space

```bash
# Update files in your Space directory
cp -r updated_files/* blueprint-detector/

# Commit and push
cd blueprint-detector
git add .
git commit -m "Update Space with latest improvements"
git push
```

You can also configure GitHub Actions to automatically deploy to Hugging Face Spaces when you push to your GitHub repository. 