# 🚀 RAG Pipeline Deployment Guide

This guide covers deploying the RAG Pipeline to Hugging Face Spaces and Docker.

## Prerequisites

- Git installed and repository initialized
- Hugging Face account (for Spaces deployment)
- Docker installed (for local Docker testing)

---

## Option 1: Deploy to Hugging Face Spaces (Recommended)

### Recommended for:
- Free hosting
- Automatic HTTPS
- Easy sharing
- No server management

### Steps:

1. **Create a Space on Hugging Face**
   - Visit [huggingface.co/new-space](https://huggingface.co/new-space)
   - Enter space name: `RAG-Pipeline-for-LLMs`
   - Select **Docker** as SDK
   - License: Choose as appropriate
   - Click "Create space"

2. **Add Hugging Face remote to your repo**
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/RAG-Pipeline-for-LLMs
   ```

3. **Push to Hugging Face**
   ```bash
   git push hf main
   ```
   The space will automatically build and deploy. Check the logs in the Hugging Face Spaces interface.

4. **Access your deployment**
   - URL: `https://huggingface.co/spaces/YOUR_USERNAME/RAG-Pipeline-for-LLMs`

### Troubleshooting HF Spaces:
- Check the **Logs** tab for build errors
- Ensure all required files are committed to git
- Verify Docker builds successfully locally first

---

## Option 2: Deploy with Docker Locally

### Recommended for:
- Testing before cloud deployment
- Custom server deployment
- Development environments

### Build & Run:

```bash
# Build the Docker image
docker build -t rag-pipeline:latest .

# Run the container
docker run -p 7860:7860 rag-pipeline:latest
```

### Access:
- Open [http://localhost:7860](http://localhost:7860)

### Push to Docker Registry (optional):

```bash
# Tag for Docker Hub (replace YOUR_USERNAME)
docker tag rag-pipeline:latest YOUR_USERNAME/rag-pipeline:latest

# Login to Docker Hub
docker login

# Push image
docker push YOUR_USERNAME/rag-pipeline:latest
```

---

## Option 3: Deploy to Other Cloud Platforms

### AWS (Elastic Beanstalk, ECS, Lambda)
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker tag rag-pipeline:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/rag-pipeline:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/rag-pipeline:latest
```

### Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/rag-pipeline
gcloud run deploy rag-pipeline --image gcr.io/YOUR_PROJECT_ID/rag-pipeline --platform managed --port 7860
```

### Azure Container Instances
```bash
az acr build --registry YOUR_REGISTRY --image rag-pipeline:latest .
az container create --resource-group YOUR_RG --name rag-pipeline --image YOUR_REGISTRY.azurecr.io/rag-pipeline:latest --ports 7860
```

---

## Environment Variables (Optional)

The app respects these environment variables:

```bash
FLASK_HOST=0.0.0.0          # Default
FLASK_PORT=7860              # Default
FLASK_DEBUG=False            # Set True for dev only
```

Example with Docker:
```bash
docker run -p 7860:7860 \
  -e FLASK_DEBUG=False \
  rag-pipeline:latest
```

---

## Pre-Deployment Checklist

- [ ] All code is committed to git
- [ ] `requirements.txt` is up-to-date
- [ ] Docker builds successfully: `docker build -t rag-pipeline .`
- [ ] App runs on port 7860 (check `config.py`)
- [ ] `.dockerignore` excludes unnecessary files
- [ ] No hardcoded API keys or secrets (use env vars if needed)
- [ ] Git `.gitignore` is properly configured

---

## Monitoring & Scaling

### Hugging Face Spaces:
- View logs and resource usage in the Space settings
- No autoscaling; one instance per space

### Docker/Cloud:
- Implement logging in `app.py`
- Set up monitoring with your cloud provider
- Configure auto-scaling if needed

---

## Troubleshooting

### "Port already in use"
```bash
docker run -p 8080:7860 rag-pipeline:latest  # Map to different port
```

### "Module not found" errors
- Ensure all imports in code match `requirements.txt`
- Rebuild Docker image: `docker build --no-cache -t rag-pipeline .`

### Long startup time
- First build of ML models (FAISS, transformers) takes time
- Subsequent runs cache models; consider persisting a volume for model cache

### Model download failures
```bash
# Pre-cache models locally and add to Docker
docker run -v ~/.cache/huggingface:/root/.cache/huggingface rag-pipeline:latest
```

---

## Performance Tips

1. **Model Caching**: Use Docker volumes to persist model cache between deployments
2. **Resource Allocation**: Allocate sufficient RAM for FAISS indexes and transformer models (minimum 4GB recommended)
3. **Load Testing**: Test with `wrk` or `Apache Bench` before production

---

## Support

- Documentation: Check [README.md](README.md)
- Issues: Report in repository issues
- Hugging Face Support: [huggingface.co/support](https://huggingface.co/support)
