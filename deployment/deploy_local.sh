#!/bin/bash

# First, capture the output of the commands in a variable
nccl_dir=$(find / -name libnccl.so.2 -print0 2>/dev/null | xargs -0 dirname)

# Now, check if nccl_dir is non-empty before updating LD_LIBRARY_PATH
if [[ -n $nccl_dir ]]; then
    export LD_LIBRARY_PATH="$nccl_dir:$LD_LIBRARY_PATH"
else
    echo "libnccl.so.2 not found"
fi

python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-chat-hf
