#!/bin/bash
# deployment/aws/deploy.sh

# Configuration
AWS_REGION="us-east-1"
ECR_REPO="your-account-id.dkr.ecr.us-east-1.amazonaws.com/marketing-finance"
ECS_CLUSTER="production-cluster"
ECS_SERVICE="marketing-finance-service"

echo "🚀 Deploying to AWS..."

# Step 1: Build Docker image
echo "Building Docker image..."
docker build -t marketing-finance-api:latest ../../

# Step 2: Tag for ECR
echo "Tagging image for ECR..."
docker tag marketing-finance-api:latest $ECR_REPO:latest

# Step 3: Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPO

# Step 4: Push to ECR
echo "Pushing image to ECR..."
docker push $ECR_REPO:latest

# Step 5: Update ECS service
echo "Updating ECS service..."
aws ecs update-service \
    --cluster $ECS_CLUSTER \
    --service $ECS_SERVICE \
    --force-new-deployment \
    --region $AWS_REGION

echo "✅ Deployment complete!"
