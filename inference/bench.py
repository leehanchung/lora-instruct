#!/usr/bin/env python3
import argparse
import asyncio
import datetime
import json
import os
import re
import time
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import NamedTuple

import httpx
import numpy as np
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from num2words import num2words

load_dotenv()

DEFAULT_MAX_TOKENS = 4096

class RequestLatency(NamedTuple):
    input_len: int
    output_len: int
    latency: float
    ttft: float


REQUEST_LATENCY: list[RequestLatency] = []


class RequestResult(NamedTuple):
    """Results of a single chat LLM post request.

    Args:
        NamedTuple (_type_): _description_
    """
    valid: str
    ttft: float
    total_time: float
    tokens_in: int
    tokens_out: int
    cause: str
    id: str


class InputPrompt(NamedTuple):
    messages: dict
    random_key: int
    tokens_in: int


def generate_single_prompt(
    *,
    tokenizer: str,
    sample_sentences: list[str],
    recite_lines: int = 10,
    random_key_digits: int = 3,
    min_sample_lines: int = 100,
    max_sample_lines: int = 150,
) -> InputPrompt:
    """Generate inference benchmarking prompts. In order to prevent endpoint
    caching from interfering with the benchmarking, we introduces two
    stochastic mechanisms.

    1. Random key. We randomly generate a numeric key, convert it into words,
    form, and insert it into our prompt. We then instruct the model to convert
    the words into numeric form.

    2. Recitations. We randomly sample without replacement a number of lines
    from sentence banks. We then instruct ask the model to "recite_lines"
    number of sentences. This also enables us to have fine grained control
    on both input length and output length.

    Args:
        sample_sentences (list[str]): sentence bank.
        recite_lines (int, optional): number of lines to recite. Defaults 10.
        random_key_digits (int, optional): digits for random key. Defaults 3.
        min_sample_lines (int, optional): min lines to sample. Defaults 20.
        max_sample_lines (int, optional): max lines to sample, Defaults 100.

    Raises:
        ValueError: when min and/or max sample lines are not valid.

    Returns:
        InputPrompt: input prompt, including messages, random key, and tokens in.
    """
    # tokenizer = get_tokenizer(tokenizer, trust_remote_code=args.trust_remote_code)
    tokenizer = tiktoken.encoding_for_model(tokenizer)
    if sample_sentences is None:
        sample_sentences = []

    # Generate a random key to avoid collisions.
    random_key: int = np.random.randint(10**(random_key_digits - 1), 10**random_key_digits)
    random_key_words: str = num2words(random_key)

    diff = max_sample_lines - min_sample_lines
    if max_sample_lines < min_sample_lines or len(sample_sentences) < diff:
        raise ValueError(
            f"Error: min sample lines: {min_sample_lines} must be less than "
            f"max sample lines: {max_sample_lines}. Their difference must be "
            f"less than len(sample_sentences): {len(sample_sentences)}"
        )

    random_num_lines: int = np.random.randint(min_sample_lines, max_sample_lines + 1)
    random_picked_lines: str = "\n".join(np.random.choice(sample_sentences, random_num_lines, replace=False))

    system_prompt: str = (
        "You are a helpful assistant providing concise and correct answers."
    )
    user_prompt = (
        f"Please convert the sequences words {random_key_words} into a numeric"
        f" number.\nPrint the converted number first. Then select {recite_lines}"
        f" lines from the following list:\n{random_picked_lines}"
    )
    tokens_in = len(tokenizer.encode(system_prompt + user_prompt)) + 4
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return InputPrompt(messages=messages, random_key=random_key, tokens_in=tokens_in)


def generate_prompts(*, num_prompts: int, **kwargs) -> list[InputPrompt]:

    # TODO: Using concurrent futures generates tons of duplicate prompts due to serialization.
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     tasks = [executor.submit(generate_single_prompt, **kwargs) for _ in range(num_prompts)]
    #     results = [task.result() for task in concurrent.futures.as_completed(tasks)]
    return [generate_single_prompt(**kwargs) for _ in range(num_prompts)]


def get_headers(*, auth_token: str | None = None, x_api_key: str | None = None) -> dict:
    headers = {
        "Content-Type": "application/json",
    }
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    if x_api_key:
        headers["x-api-key"] = x_api_key
    return headers


async def delay_requests_poisson(
    input_prompts: list[InputPrompt],
    request_rate: float,
) -> AsyncGenerator[InputPrompt, None]:
    """Forms requests into a Poisson process using the provided request_rate by
    adding an async sleep timer. The request_rate is measured as requests per
    second.

    Args:
        input_prompts (list[tuple[str, int, int]]): _description_
        request_rate (int): request rate, measured in request per minute.

    Returns:
        AsyncGenerator[tuple[str, int, int], None]: _description_

    Yields:
        Iterator[AsyncGenerator[tuple[str, int, int], None]]: _description_
    """
    requests_per_second = request_rate / 60.0
    input_prompts = iter(input_prompts)
    for request in input_prompts:
        yield request

        if request_rate == float("inf"):
            continue

        interval = np.random.exponential(1.0 / requests_per_second)
        await asyncio.sleep(interval)

async def parse_openai_compat_sse_chunks(
    response: httpx.Response,
) -> AsyncGenerator[dict]:
    """Parse OpenAI 1.0+ compatible infernece services response SSE chunks.

    Args:
        response (httpx.Response): server side event response

    Returns:
        AsyncGenerator[dict]: json data chunks
    """
    async for line in response.aiter_lines():
        line = line.strip()
        if line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]":
                break
            yield json.loads(data)

async def post_chat_completion_stream(
    api_url: str,
    payload: dict,
    headers: dict,
    input_len: int,
    random_key: int,
) -> tuple[str, int, int, str]:
    request_start_time = time.perf_counter()
    async with httpx.AsyncClient() as client:
        chunks = []
        output_len = 0
        time_to_first_token = None

        response = await client.post(api_url, headers=headers, json=payload)
        response.raise_for_status()

        # Stream the response
        async for chunk in parse_openai_compat_sse_chunks(response):
            if chunk["choices"]:
                request_id = chunk['id']
                output_len += 1
                delta_content = chunk["choices"][0]["delta"].get("content")
                if delta_content:
                    if time_to_first_token is None:
                        time_to_first_token = time.perf_counter() - request_start_time
                    # print(content, end='', flush=True)
                    chunks.append(delta_content)

        content = "".join(chunks)

    request_end_time = time.perf_counter()
    total_time = request_end_time - request_start_time

    # TODO: Move this away from using global
    REQUEST_LATENCY.append(
        RequestLatency(
            input_len=input_len,
            output_len=output_len,
            latency=total_time,
            ttft=time_to_first_token,
        )
    )

    return request_id, time_to_first_token, total_time, content, random_key


async def benchmark(
    api_url: str,
    model: str,
    headers: dict,
    input_prompts: list[InputPrompt],
    request_rate: float,
) -> None:
    tokenizer = tiktoken.encoding_for_model(model)
    tasks: list[asyncio.Task] = []
    results = []

    # Build async API request tasks
    async for request in delay_requests_poisson(input_prompts, request_rate):
        payload = {
            "model": model,
            "messages": request.messages,
            "temperature": 0.0,
            "stream": True,
        }
        task = asyncio.create_task(
            post_chat_completion_stream(
                api_url=api_url,
                payload=payload,
                headers=headers,
                input_len=request.tokens_in,
                random_key=request.random_key,
            )
        )
        tasks.append(task)
    print(f"Number of tasks: {len(tasks)}", flush=True)

    for task in asyncio.as_completed(tasks):
        try:
            request_id, time_to_first_token, total_time, content, random_key = await task
        except Exception as e:
            print(e, flush=True)
            results.append(
                RequestResult(valid='Exception', ttft=-1, total_time=-1, tokens_in=-1, tokens_out=-1, cause=str(e), id='',)
            )

        tokens_out = len(tokenizer.encode(content))
        # Find the random key
        nums = re.findall(r"\d+", content)

        if len(nums) > 0:
            return_nums: int = int(nums[0])
            valid: str = "OK"
            cause: str = ""
            if return_nums != random_key:
                print(f"Error: expected {random_key}, got {return_nums}", flush=True)
                valid = "Mismatch"
                cause = f"Expected {random_key}, got {return_nums}.\nOutput:\n{content}"
        else:
            print(f"Error: no number found in {content}", flush=True)
            valid:str = "Mismatch"
            cause = f"Random key not found. Input = {random_key}.\nOutput:\n{content}"

        results.append(
            RequestResult(valid=valid, ttft=time_to_first_token, total_time=total_time, tokens_in=request.tokens_in, tokens_out=tokens_out, cause=cause, id=request_id,)
        )

    return results



def result_analytics(results: list[RequestResult]) -> None:
    args_dict = {}
    df = pd.DataFrame(
        results,
        columns=[
            'valid',
            'ttft',
            'total_time',
            'tokens_in',
            'tokens_out',
            'cause',
            'id',
        ]
    )
    ts = int(time.time())
    filename = f"service-{ts}_raw.json"
    df.to_json(filename)
    print(f"Results saved to {filename}")

    print('Validity results:')
    print(df['valid'].value_counts())

    value_counts = df['valid'].value_counts()
    args_dict['num_valid'] = int(value_counts.get("OK", 0))
    args_dict['num_mismatch'] = int(value_counts.get("Mismatch", 0))
    args_dict['num_exceptions'] = int(value_counts.get("Exception", 0))
    args_dict['valid_rate'] = args_dict['num_valid'] / len(df)
    args_dict['mismatch_rate'] = args_dict['num_mismatch'] / len(df)
    args_dict['exception_rate'] = args_dict['num_exceptions'] / len(df)

    cdf = df[df.valid != 'Exception'].copy()
    print(f'Clean DF is {len(cdf)}')
    if len(cdf) > 0:
        cdf['total_tokens_per_sec'] = (cdf['tokens_in'] + cdf['tokens_out']) / cdf['total_time']
        cdf['out_tokens_per_sec'] = cdf['tokens_out'] / cdf['total_time']
        cdf['inter_token_deloy'] = cdf['total_time'] / cdf['tokens_out']
        mean_e2e = cdf['total_time'].mean()
        mean_tokens_in = cdf['tokens_in'].mean()
        mean_tokens_out = cdf['tokens_out'].mean()
        mean_ttft = cdf['ttft'].mean()
        max_ttft = cdf['ttft'].max()
        gt_3_ttft = len(cdf[cdf['ttft'] > 3])

        print(f'Mean End-to-end time: {mean_e2e*1000.0:.0f} ms')
        print(f'Mean TTFT: {mean_ttft*1000.0:.0f} ms', end=" ")
        print(f'mean_tokens_in: {mean_tokens_in:.0f}', end=" ")
        print(f'mean_tokens_out: {mean_tokens_out:.0f}')
        print(f'Max TTFT: {max_ttft*1000.0:.0f} ms')
        print(f'TTFT > 3s: {gt_3_ttft*100:.2f}%')

        print(f"ITL (out)': {cdf['inter_token_deloy'].mean()*1000.0:.0f} ms/token")
        print(f"mean tokens/s output (out): {cdf['out_tokens_per_sec'].mean():.0f} tokens/s")

        args_dict['end_timestamp'] = datetime.datetime.fromtimestamp(ts).isoformat()
        args_dict['total_time'] = float(cdf.total_time.mean())
        args_dict['mean_ttft'] = int(f'{mean_ttft*1000.0:.0f}')
        args_dict['max_ttft'] = int(f'{max_ttft*1000.0:.0f}')
        args_dict['mean_tokens_in'] = mean_tokens_in
        args_dict['mean_tokens_out'] = mean_tokens_out
        args_dict['total_tokens_per_sec'] = float(cdf['total_tokens_per_sec'].mean())
        args_dict['out_tokens_per_sec'] = float(cdf['out_tokens_per_sec'].mean())
        args_dict['inter_token_deloy'] = float(cdf['inter_token_deloy'].mean())

    def error_analysis(df: pd.DataFrame) -> None:
        exceptions = df[df.valid == 'Exception']
        exceptions_by_cause = defaultdict(int)
        for cause in exceptions['cause']:
            exceptions_by_cause[cause] += 1
        print('Exceptions by cause:')
        for cause, count in exceptions_by_cause.items():
            print(f' - {cause}: {count}')

    error_analysis(df)
    args_dict['raw_output'] = filename
    benchmark_result = f"benchmark-{ts}.json"

    with open(benchmark_result, 'w') as f:
        f.write(json.dumps(args_dict, indent=4))


def get_sentences_bank(*, filename: str) -> list[str]:
    """Reads the sentences bank from the given file.

    Args:
        filename (str): _description_

    Returns:
        list[str]: _description_
    """
    with open(filename) as f:
        return f.readlines()


def main(args: argparse.Namespace):

    # Read from sentences bank
    print("Loading sentence bank...")
    sample_sentences = get_sentences_bank(filename=args.dataset)


    print("Generating prompts...")
    # TODO: set proper tokenizer to support different models.
    # tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    input_prompts = generate_prompts(
        num_prompts=args.num_prompts,
        tokenizer="gpt-4",
        sample_sentences=sample_sentences
    )

    request_rate = args.request_rate
    headers = get_headers(auth_token=os.getenv("OPENAI_API_KEY"))
    print("Starting benchmarking...")
    benchmark_start_time = time.perf_counter()
    results = asyncio.run(
        benchmark(args.api_url, args.model, headers, input_prompts, request_rate)
    )
    print("\n\nBenchmarking finished.")
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time

    result_analytics(results)

    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {len(REQUEST_LATENCY) / benchmark_time:.2f} requests/s")
    print(f"Throughput: {len(REQUEST_LATENCY) / benchmark_time * 60:.2f} RPM")

    # Compute the latency statistics.
    avg_ttft = np.mean([ttft for _, _, _, ttft in REQUEST_LATENCY])
    print(f"Average time to first token: {avg_ttft:.2f} s")
    avg_latency = np.mean([latency for _, _, latency, _ in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean(
        [
            latency / (input_len + output_len)
            for input_len, output_len, latency, _ in REQUEST_LATENCY
        ]
    )
    print(f"Average latency per token: {avg_per_token_latency*1000:.2f} ms")
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency, _ in REQUEST_LATENCY]
    )
    print(
        "Average latency per output token: "
        f"{avg_per_output_token_latency*1000:.2f} ms"
    )


def setup_parser():
    parser = argparse.ArgumentParser(
        description="LLM online serving endpoints benchmarking and load testing."
    )

    # API configuration
    # For local testing, localhost:8000
    parser.add_argument(
        "--api-url",
        "-a",
        type=str,
        default="https://api.openai.com/v1/chat/completions",
        help="Base URL for the LLM API endpoint",
    )
    parser.add_argument(
        "--model", "-m", type=str, default="gpt-3.5-turbo", help="Model to benchmark"
    )

    # Prompt configuration
    parser.add_argument(
        "--dataset", "-d", type=str, default="sonnet.txt", help="Sentences bank file."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max tokens for the response",
    )

    # Benchmark configuration
    group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--num-prompts", type=int, default=60, help="Number of prompts to process."
    )
    group.add_argument(
        "--request-rate",
        type=float,
        default=60, # float("inf"),
        help="Number of requests per minute (RPM). If this is inf, then all "
        "requests are sent at time 0. Otherwise, we use Poisson process to "
        "simulate the request arrival times.",
    )
    # group.add_argument(
    #     "--num-requests",
    #     "-n",
    #     type=int,
    #     help="Number of concurrent requests to make",
    # )

    return parser.parse_args()


if __name__ == "__main__":
    args = setup_parser()
    main(args)
