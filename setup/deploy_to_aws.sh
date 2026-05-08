#!/bin/bash
# deploy_to_aws.sh
# Run from Mac. Launches EC2 p3.2xlarge, transfers code, runs bootstrap.
# Usage: bash deploy_to_aws.sh [--spot] [--key KEY_NAME] [--region REGION]
#
# Prerequisites:
#   1. AWS CLI installed and configured: aws configure
#   2. EC2 key pair exists: aws ec2 describe-key-pairs
#   3. GEMS data uploaded to S3 (one-time):
#      aws s3 sync /path/to/GEMS_DATA s3://YOUR_BUCKET/gems_data

set -e

# ── Config (edit these) ───────────────────────────────────────────────
KEY_NAME="${KEY_NAME:-my-ec2-key}"           # your EC2 key pair name
REGION="${REGION:-us-east-1}"
S3_DATA_BUCKET="${S3_DATA_BUCKET:-}"         # e.g. s3://my-bucket/gems_data
INSTANCE_TYPE="p3.2xlarge"                  # V100, best for float64
USE_SPOT=false

LOCAL_PROJECT="/Users/joonwonlee/Documents/GEMS_TCO-1"
REMOTE_HOME="/home/ec2-user/gems_tco"
SSH_USER="ec2-user"

# ── Parse args ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --spot)       USE_SPOT=true ;;
        --key)        KEY_NAME="$2"; shift ;;
        --region)     REGION="$2"; shift ;;
        --s3)         S3_DATA_BUCKET="$2"; shift ;;
        --instance)   INSTANCE_TYPE="$2"; shift ;;
    esac
    shift
done

echo "================================================================"
echo "GEMS TCO — Deploy to AWS"
echo "Instance:  ${INSTANCE_TYPE}  (Spot: ${USE_SPOT})"
echo "Region:    ${REGION}"
echo "Key:       ${KEY_NAME}"
echo "S3 data:   ${S3_DATA_BUCKET:-'(not set — skip data sync)'}"
echo "================================================================"

# ── Find Deep Learning AMI ────────────────────────────────────────────
echo "[1/6] Finding latest Deep Learning AMI..."
AMI_ID=$(aws ec2 describe-images \
    --region ${REGION} \
    --owners amazon \
    --filters \
        "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 20.04)*" \
        "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text)

if [ -z "${AMI_ID}" ] || [ "${AMI_ID}" = "None" ]; then
    # Fallback: PyTorch DLAMI
    AMI_ID=$(aws ec2 describe-images \
        --region ${REGION} \
        --owners amazon \
        --filters \
            "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu 20.04*" \
            "Name=state,Values=available" \
        --query "sort_by(Images, &CreationDate)[-1].ImageId" \
        --output text)
fi
echo "   AMI: ${AMI_ID}"

# ── Launch instance ───────────────────────────────────────────────────
echo "[2/6] Launching ${INSTANCE_TYPE}..."

LAUNCH_SPEC='{
    "ImageId": "'"${AMI_ID}"'",
    "InstanceType": "'"${INSTANCE_TYPE}"'",
    "KeyName": "'"${KEY_NAME}"'",
    "BlockDeviceMappings": [{
        "DeviceName": "/dev/sda1",
        "Ebs": {"VolumeSize": 100, "VolumeType": "gp3", "DeleteOnTermination": true}
    }],
    "TagSpecifications": [{
        "ResourceType": "instance",
        "Tags": [{"Key": "Name", "Value": "gems-tco-sim"}]
    }]
}'

if [ "${USE_SPOT}" = true ]; then
    INSTANCE_ID=$(aws ec2 run-instances \
        --region ${REGION} \
        --cli-input-json "${LAUNCH_SPEC}" \
        --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
        --query "Instances[0].InstanceId" --output text)
else
    INSTANCE_ID=$(aws ec2 run-instances \
        --region ${REGION} \
        --cli-input-json "${LAUNCH_SPEC}" \
        --query "Instances[0].InstanceId" --output text)
fi

echo "   Instance ID: ${INSTANCE_ID}"

# ── Wait for instance ─────────────────────────────────────────────────
echo "[3/6] Waiting for instance to be running..."
aws ec2 wait instance-running --region ${REGION} --instance-ids ${INSTANCE_ID}
sleep 30  # wait for SSH to be ready

PUBLIC_IP=$(aws ec2 describe-instances \
    --region ${REGION} \
    --instance-ids ${INSTANCE_ID} \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text)
echo "   Public IP: ${PUBLIC_IP}"
echo "   SSH: ssh -i ~/.ssh/${KEY_NAME}.pem ${SSH_USER}@${PUBLIC_IP}"

# Save connection info
echo "${INSTANCE_ID} ${PUBLIC_IP}" > /tmp/gems_aws_instance.txt
echo "   Saved to /tmp/gems_aws_instance.txt"

SSH_CMD="ssh -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no ${SSH_USER}@${PUBLIC_IP}"

# Wait for SSH
echo "   Waiting for SSH to be available..."
for i in {1..20}; do
    if ${SSH_CMD} "echo ok" &>/dev/null; then break; fi
    sleep 10
done

# ── Transfer code ─────────────────────────────────────────────────────
echo "[4/6] Transferring code..."

# Create remote directories
${SSH_CMD} "mkdir -p ${REMOTE_HOME}/src/GEMS_TCO/cpp_src ${REMOTE_HOME}/setup"

# Transfer GEMS_TCO package (src only, no .so files - will be recompiled)
rsync -az --exclude="*.so" --exclude="*.pyd" --exclude="__pycache__" \
    -e "ssh -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no" \
    ${LOCAL_PROJECT}/src/GEMS_TCO/ \
    ${SSH_USER}@${PUBLIC_IP}:${REMOTE_HOME}/src/GEMS_TCO/

# Transfer simulation scripts
scp -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no \
    ${LOCAL_PROJECT}/Exercises/st_model/day/simulation/sim_vecchia_irregular_hybrid_compare_050226.py \
    ${SSH_USER}@${PUBLIC_IP}:${REMOTE_HOME}/exercise_25/st_model/

# Transfer bootstrap + run scripts
scp -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no \
    ${LOCAL_PROJECT}/setup/aws_bootstrap.sh \
    ${LOCAL_PROJECT}/setup/run_sim_aws.sh \
    ${SSH_USER}@${PUBLIC_IP}:${REMOTE_HOME}/

echo "   Code transferred."

# ── Run bootstrap ─────────────────────────────────────────────────────
echo "[5/6] Running bootstrap on EC2 (this takes ~5 min)..."
${SSH_CMD} "bash ${REMOTE_HOME}/aws_bootstrap.sh ${S3_DATA_BUCKET} 2>&1 | tee ${REMOTE_HOME}/bootstrap.log"

# ── Launch simulation in tmux ─────────────────────────────────────────
echo "[6/6] Starting simulation in tmux..."
${SSH_CMD} "tmux new-session -d -s sim 'conda run -n faiss_env bash ${REMOTE_HOME}/run_sim_aws.sh 2>&1 | tee ${REMOTE_HOME}/exercise_output/sim_run.log'"

echo ""
echo "================================================================"
echo "Deployment complete!"
echo ""
echo "Monitor:  ${SSH_CMD} \"tmux attach -t sim\""
echo "Log:      ${SSH_CMD} \"tail -f ${REMOTE_HOME}/exercise_output/sim_run.log\""
echo ""
echo "When done, fetch results:"
echo "  bash ${LOCAL_PROJECT}/setup/fetch_results.sh ${PUBLIC_IP} ~/.ssh/${KEY_NAME}.pem"
echo ""
echo "IMPORTANT — Terminate instance when done (stops billing):"
echo "  aws ec2 terminate-instances --region ${REGION} --instance-ids ${INSTANCE_ID}"
echo "================================================================"
