#!/bin/bash

# First, capture the output of the commands in a variable
nccl_dir=$(find / -name libnccl.so.2 -print0 2>/dev/null | xargs -0 dirname)

# Now, check if nccl_dir is non-empty before updating LD_LIBRARY_PATH
if [[ -n $nccl_dir ]]; then
    export LD_LIBRARY_PATH="$nccl_dir:$LD_LIBRARY_PATH"
else
    echo "libnccl.so.2 not found"
fi

# Run the inference code
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-chat-hf

# Instructions for running the inference code
# Ensure that you have the necessary dependencies installed
# You can install them using the following command:
# pip install -r requirements.txt

# To run the inference code, use the following command:
# bash inference/deploy_local.sh
