# Production Deployment Guide — RetinaX AI Platform

## 1. Local Container Deployment with Docker Compose

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### Build and Run
```bash
# 1. Build and start the container in detached mode
docker-compose up --build -d

# 2. View running containers
docker-compose ps

# 3. Check live container logs
docker-compose logs -f
```

Access the application in your browser:
- **Web Dashboard**: `http://localhost:8000/`
- **Interactive OpenAPI Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

To stop the container:
```bash
docker-compose down
```

---

## 2. Cloud Deployment on Render.com

1. Create a GitHub repository and push your project code.
2. Log into [Render Dashboard](https://dashboard.render.com/) and click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Select environment: **Docker**.
5. Set Build & Runtime Settings:
   - **Name**: `retinax-ai-platform`
   - **Region**: Oregon (US West) or closest region
   - **Instance Type**: Starter or Standard (At least 2GB RAM for TensorFlow model loading)
6. Click **Deploy Web Service**. Render will automatically detect the `Dockerfile`, build the image, and assign an HTTPS URL (e.g., `https://retinax-ai.onrender.com`).

---

## 3. Cloud Deployment on Google Cloud Run (Serverless)

```bash
# 1. Authenticate with GCP
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID

# 2. Build image using Google Cloud Build
gcloud builds submit --tag gcr.io/YOUR_GCP_PROJECT_ID/retinax-ai:latest

# 3. Deploy to Cloud Run (Auto-scaling 0 to N instances)
gcloud run deploy retinax-ai \
    --image gcr.io/YOUR_GCP_PROJECT_ID/retinax-ai:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2
```

---

## 4. Environment Health Auditing

Verify container health status:
```bash
curl http://localhost:8000/health
```

Expected Response:
```json
{
  "status": "healthy",
  "project_name": "Diabetic Retinopathy Explainable AI Platform",
  "version": "1.0.0",
  "model_loaded": true,
  "model_architecture": "EfficientNet-B0 (Transfer Learning)"
}
```
