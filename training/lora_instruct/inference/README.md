# Inference


## Benchmarking Online LLM API Services
There's no obvious standards in benchmarking LLM API Services, where inference optimization packages mostly focuses on benchmarking model serving workers instead of the end to end model serving services. This benchmark aims to benchmark LLM APIs from an user's perspective for capacity planning and scaling needs.


## References
- [Anyscale: Reproducible Performance Metrics for LLM inference](https://www.anyscale.com/blog/reproducible-performance-metrics-for-llm-inference)
- [Anyscale: LLMPerf](https://github.com/ray-project/llmperf)
- [Anyscale: Continuous Batching](https://github.com/anyscale/llm-continuous-batching-benchmarks/blob/master/benchmark_throughput.py)
- [FastChat: Test Throughput](https://github.com/lm-sys/FastChat/blob/5dbc4f30ab36f17b8e004246e53f1e13fce4a01c/fastchat/serve/test_throughput.py)
- [vLLM: Benchmark Serving](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py)
