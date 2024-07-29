# Inference

## Benchmarking Online LLM API Services
There's no obvious standards in benchmarking LLM API Services, where inference optimization packages mostly focuses on benchmarking model serving workers instead of the end to end model serving services. This benchmark aims to benchmark LLM APIs from an user's perspective for capacity planning and scaling needs.

## Running the Inference Code
To run the inference code, follow these steps:
1. Ensure that you have the necessary dependencies installed. You can install them using the following command:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the `deploy_local.sh` script to deploy the model locally:
   ```bash
   bash inference/deploy_local.sh
   ```

## Examples of Using the Inference Code
Here are some examples of how to use the inference code:

### Example 1: Generating a Response
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Generate a response
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Example 2: Benchmarking the Model
```bash
python inference/bench.py --api-url http://localhost:8000 --model meta-llama/Llama-2-7b-chat-hf --dataset inference/sonnet.txt --num-prompts 10 --request-rate 60
```

## References
- [Anyscale: Reproducible Performance Metrics for LLM inference](https://www.anyscale.com/blog/reproducible-performance-metrics-for-llm-inference)
- [Anyscale: LLMPerf](https://github.com/ray-project/llmperf)
- [Anyscale: Continuous Batching](https://github.com/anyscale/llm-continuous-batching-benchmarks/blob/master/benchmark_throughput.py)
- [FastChat: Test Throughput](https://github.com/lm-sys/FastChat/blob/5dbc4f30ab36f17b8e004246e53f1e13fce4a01c/fastchat/serve/test_throughput.py)
- [vLLM: Benchmark Serving](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py)
