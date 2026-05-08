#!/bin/bash
# fetch_results.sh
# Pull simulation outputs from EC2 back to Mac.
# Usage: bash fetch_results.sh PUBLIC_IP KEY_PATH
# Example: bash fetch_results.sh 54.12.34.56 ~/.ssh/my-ec2-key.pem

PUBLIC_IP=${1:-$(cut -d' ' -f2 /tmp/gems_aws_instance.txt 2>/dev/null)}
KEY_PATH=${2:-~/.ssh/my-ec2-key.pem}
SSH_USER="ec2-user"
REMOTE_OUT="/home/ec2-user/gems_tco/exercise_output/estimates/day"
LOCAL_OUT="/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/estimates/save/day"

if [ -z "${PUBLIC_IP}" ]; then
    echo "Usage: bash fetch_results.sh PUBLIC_IP [KEY_PATH]"
    exit 1
fi

echo "Fetching results from ${PUBLIC_IP}..."
mkdir -p ${LOCAL_OUT}

rsync -az --progress \
    -e "ssh -i ${KEY_PATH} -o StrictHostKeyChecking=no" \
    ${SSH_USER}@${PUBLIC_IP}:${REMOTE_OUT}/ \
    ${LOCAL_OUT}/

echo "Done. Results saved to ${LOCAL_OUT}"
ls -lh ${LOCAL_OUT}/*.csv 2>/dev/null || echo "(no CSV files yet)"
