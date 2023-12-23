# Benchmarking online serving throughput for LLM API endpoints.
import argparse
import asyncio
import datetime
import json
import os
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


DEFAULT_PROMPT = """Please act as an expert financial analyst when you answer the questions and pay special attention to the financial statements.  Operating margin is also known as op margin and is calculated by dividing operating income by revenue.
Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" field in your answer, with the format "SOURCES: <source1>, <source2>, <source3>, ...".

QUESTION: What's the operating margin?
=========
Content: Three Months Ended
Six Months Ended
July 30, April 30, July 31, July 30, July 31,
2023 2023 2022 2023 2022
GAAP gross profit $ 9,462 $ 4,648 $ 2,915 $ 14,110 $ 8,346
GAAP gross margin 70.1% 64.6% 43.5% 68.2% 55.7%
Acquisition-related and other costs (A) 119 119 121 239 214
Stock-based compensation expense (B) 31 27 38 58 76
IP-related costs 2 8 — 10 —
Non-GAAP gross profit $ 9,614 $ 4,802 $ 3,074 $ 14,417 $ 8,636
Non-GAAP gross margin 71.2% 66.8% 45.9% 69.7% 57.6%
GAAP operating expenses $ 2,662 $ 2,508 $ 2,416 $ 5,169 $ 5,979
Stock-based compensation expense (B) (811) (708) (611) (1,518) (1,151)
Acquisition-related and other costs (A) (18) (54) (54) (72) (110)
Acquisition termination cost — — — — (1,353)
Legal settlement costs — — — — (7)
Contributions — — (2) — (2)
Other (C) 5 4 — 10 —
Non-GAAP operating expenses $ 1,838 $ 1,750 $ 1,749 $ 3,589 $ 3,356
GAAP operating income $ 6,800 $ 2,140 $ 499 $ 8,941 $ 2,367
Total impact of non-GAAP adjustments to
operating income 976 912 826 1,887 2,913
Non-GAAP operating income $ 7,776 $ 3,052 $ 1,325 $ 10,828 $ 5,280
GAAP other income (expense), net $ 181 $ 69 $ (24) $ 249 $ (87)
(Gains) losses from non-affiliated
investments
(62) 14 7 (46) 24
Interest expense related to amortization of
debt discount 1 1 1 2 2
Non-GAAP other income (expense), net $ 120 $ 84 $ (16) $ 205 $ (61)
GAAP net income $ 6,188 $ 2,043 $ 656 $ 8,232 $ 2,274
Total pre-tax impact of non-GAAP
adjustments 915 927 833 1,843 2,940
Income tax impact of non-GAAP
adjustments (D) (363) (257) (197) (622) (478)
Non-GAAP net income $ 6,740 $ 2,713 $ 1,292 $ 9,453 $ 4,736
Source: source_8

Content: Outlook
($ in millions)
GAAP gross margin 71.5%
Impact of stock-based compensation expense, acquisition-related costs, and other costs 1.0%
Non-GAAP gross margin 72.5%
GAAP operating expenses $ 2,950
Stock-based compensation expense, acquisition-related costs, and other costs (950)
Non-GAAP operating expenses $ 2,000
About NVIDIA
Since its founding in 1993, NVIDIA (NASDAQ: NVDA) has been a pioneer in accelerated computing. The company’s
invention of the GPU in 1999 sparked the growth of the PC gaming market, redefined computer graphics, ignited the era of
modern AI and is fueling industrial digitalization across markets. NVIDIA is now a full-stack computing company with data-
center-scale offerings that are reshaping industry. More information at https://nvidianews.nvidia.com/.
Certain statements in this press release including, but not limited to, statements as to: companies worldwide transitioning
from general-purpose to accelerated computing and generative AI; NVIDIA GPUs running CUDA AI software stack making
up the computing infrastructure of generative AI; the race to adopt generative AI; NVIDIA’s plans to continue share
repurchases; NVIDIA’s next quarterly cash dividend; NVIDIA’s financial outlook and expected tax rates for the third quarter of
fiscal 2024; the benefits, impact, performance, features and availability of our products and technologies, including the
NVIDIA GH200 Grace Hopper Superchip, NVIDIA L40S GPU, NVIDIA OVX, NVIDIA AI Enterprise, BlueField DPUs, NVIDIA
MGX, NVIDIA Omniverse, NVIDIA Spectrum-X, NVIDIA RTX workstations, NVIDIA RTX 6000 Ada GPU, NVIDIA Omniverse
Enterprise software, NVIDIA H100 Tensor Core GPU, NVIDIA DGX Cloud AI, NVIDIA AI Workbench, NVIDIA AI Enterprise
4.0, the GeForce RTX 4060 family, NVIDIA Ada Lovelace, DLSS, NVIDIA Avatar Cloud Engine, NVIDIA’s RTX Remix,
NVIDIA RTX 5000, RTX 4500 and RTX 4000, and NVIDIA DRIVE Orin; and the benefits and impact of NVIDIA’s
partnerships with ServiceNow, Accenture, VMware, Snowflake, WPP, SoftBank, Hugging Face, and MediaTek, and
NVIDIA’s Alliance for OpenUSD with Pixar, Adobe, Apple and Autodesk are forward-looking statements that are subject to
risks and uncertainties that could cause results to be materially different than expectations. Important factors that could cause
actual results to differ materially include: global economic conditions; our reliance on third parties to manufacture, assemble,
package and test our products; the impact of technological development and competition; development of new products and
technologies or enhancements to our existing product and technologies; market acceptance of our products or our partners’
products; design, manufacturing or software defects; changes in consumer preferences or demands; changes in industry
standards and interfaces; unexpected loss of performance of our products or technologies when integrated into systems; as
Source: source_10

Content: different from non-GAAP measures used by other companies.
NVIDIA CORPORATION
CONDENSED CONSOLIDATED STATEMENTS OF INCOME
(In millions, except per share data)
(Unaudited)
Three Months Ended Six Months Ended
July 30, July 31, July 30, July 31,
2023 2022 2023 2022
Revenue $ 13,507 $ 6,704 $ 20,699 $ 14,992
Cost of revenue 4,045 3,789 6,589 6,646
Gross profit 9,462 2,915 14,110 8,346
Operating expenses
Research and development 2,040 1,824 3,916 3,443
Sales, general and administrative 622 592 1,253 1,183
Acquisition termination cost — — — 1,353
Total operating expenses 2,662 2,416 5,169 5,979
Source: source_4

Content: Operating income 6,800 499 8,941 2,367
Interest income 187 46 338 64
Interest expense (65) (65) (131) (132)
Other, net 59 (5) 42 (19)
Other income (expense), net 181 (24) 249 (87)
Income before income tax 6,981 475 9,190 2,280
Income tax expense (benefit) 793 (181) 958 6
Net income $ 6,188 $ 656 $ 8,232 $ 2,274
Net income per share:
Basic $ 2.50 $ 0.26 $ 3.33 $ 0.91
Diluted $ 2.48 $ 0.26 $ 3.30 $ 0.90
Weighted average shares used in per share computation:
Basic 2,473 2,495 2,472 2,500
Diluted 2,499 2,516 2,495 2,526
NVIDIA CORPORATION
CONDENSED CONSOLIDATED BALANCE SHEETS
(In millions)
(Unaudited)
July 30, 2023 January 29, 2023
ASSETS
Current assets:
Cash, cash equivalents and marketable securities $ 16,023 $ 13,296
Accounts receivable, net 7,066 3,827
Inventories 4,319 5,159
Prepaid expenses and other current assets 1,389 791
Total current assets 28,797 23,073
Property and equipment, net 3,799 3,807
Operating lease assets 1,235 1,038
Goodwill 4,430 4,372
Intangible assets, net 1,395 1,676
Deferred income tax assets
5,398 3,396
Other assets 4,501 3,820
Total assets $ 49,555 $ 41,182
LIABILITIES AND SHAREHOLDERS' EQUITY
Source: source_5
=========
FINAL ANSWER:"""
DEFAULT_MAX_TOKENS = 4096

# (input_len, output_len, latency, ttft)

class RequestLatency(NamedTuple):
    input_len: int
    output_len: int
    latency: float
    ttft: float


REQUEST_LATENCY: list[RequestLatency] = []


class RequestResult(NamedTuple):
    valid: str
    ttft: float
    total_time: float
    tokens_in: int
    tokens_out: int
    cause: str
    id: str


class Prompt(NamedTuple):
    system_prompt: str
    user_prompt: str
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
) -> Prompt:
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
        recite_lines (int, optional): _description_. Defaults to 1.
        random_key_digits (int, optional): _description_. Defaults to 3.
        min_sample_lines (int, optional): _description_. Defaults to 20.
        max_sample_lines (int, optional): _description_. Defaults to 100.
        sample_sentences (list[str], optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        tuple[str, int]: _description_
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
        "You are a helpful assistant that responds with the correct answer in "
        "the most concise possible way."
    )

    user_prompt = (
        "Please help convert the following sequences words into a numeric "
        f"number: {random_key_words}.\nPrint the converted number first. Then "
        f"pick {recite_lines} lines from pick from the following lines:\n "
        f"{random_picked_lines}"
    )
    tokens_in = len(tokenizer.encode(system_prompt + user_prompt)) + 4
    return system_prompt, user_prompt, random_key, tokens_in


def generate_prompts(*, num_prompts: int, **kwargs) -> list[Prompt]:

    # NOTE: This generates tons of duplicate prompts due to serialization.
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
    input_requests: list[tuple[str, int, int]],
    request_rate: int,
) -> AsyncGenerator[tuple[str, int, int], None]:
    """Forms requests into a Poisson process using the provided request_rate by
    adding an async sleep timer. The request_rate is measured as requests per
    second.

    Args:
        input_requests (list[tuple[str, int, int]]): _description_
        request_rate (int): request rate, measured in request per minute.

    Returns:
        AsyncGenerator[tuple[str, int, int], None]: _description_

    Yields:
        Iterator[AsyncGenerator[tuple[str, int, int], None]]: _description_
    """
    requests_per_second = request_rate / 60.0
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            continue

        interval = np.random.exponential(1.0 / requests_per_second)
        await asyncio.sleep(interval)


async def post_chat_completion_stream(
    api_url: str,
    model: str,
    headers: dict,
    prompt: str,
    input_len: int,
    output_len: int,
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{prompt}"}],
        "temperature": 0.0,
        "stream": True,
    }
    timeout = httpx.Timeout(3 * 3600)
    request_start_time = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            response = await client.post(api_url, headers=headers, json=payload)

            chunks = []
            output_len = 0
            time_to_first_token = None

            # Stream the response
            async for line in response.aiter_lines():
                if not line and not time_to_first_token:
                    time_to_first_token = time.perf_counter() - request_start_time

                line = line.strip()
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    data = json.loads(data)
                    if data["choices"]:
                        output_len += 1
                        content = data["choices"][0]["delta"].get("content")
                        if content:
                            chunks.append(content)

            output = "".join(chunks)

            # Re-send the request if it failed.
            if "error" not in output:
                break

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    # REQUEST_LATENCY.append((input_len, output_len, request_latency, time_to_first_token))
    REQUEST_LATENCY.append(
        RequestLatency(
            input_len=input_len,
            output_len=output_len,
            latency=request_latency,
            ttft=time_to_first_token,
        )
    )
    # valid: str
    # ttft: float
    # total_time: float
    # tokens_in: int
    # tokens_out: int
    # cause: str
    # id: str
    # return RequestResult(

async def benchmark(
    api_url: str,
    model: str,
    headers: dict,
    input_requests: list[tuple[str, int, int]],
    request_rate: float,
) -> None:
    tasks: list[asyncio.Task] = []
    async for request in delay_requests_poisson(input_requests, request_rate):
        prompt, input_len, output_len = request
        task = asyncio.create_task(
            post_chat_completion_stream(api_url, args.model, headers, prompt, input_len, output_len)
        )
        tasks.append(task)
    await asyncio.gather(*tasks)

def result_analytics(results: list[tuple[int, int, float, float]]) -> None:
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
    print("Results saved to {filename}")

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
        print(f'Mean TTFT: {mean_ttft*1000.0:.0f} ms', end="")
        print(f'mean_tokens_in: {mean_tokens_in:.0f}', end="")
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
        f.write(json.dump(args_dict, f, indent=4))


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
    sample_sentences = get_sentences_bank(filename=args.dataset)

    # tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    # input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    generate_prompts(num_prompts=args.num_prompts, tokenizer="gpt-4", sample_sentences=sample_sentences)


    num_prompts = args.num_prompts
    enc = tiktoken.encoding_for_model("gpt-4")
    input_len = len(enc.encode(DEFAULT_PROMPT))

    max_len = 4000
    # requests: prompt, input_len, max_len
    input_requests = [(DEFAULT_PROMPT, input_len, max_len)] * num_prompts

    request_rate = args.request_rate

    headers = get_headers(auth_token=os.getenv("OPENAI_API_KEY"))
    benchmark_start_time = time.perf_counter()
    asyncio.run(
        benchmark(args.api_url, args.model, headers, input_requests, request_rate)
    )
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time

    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {num_prompts / benchmark_time:.2f} requests/s")
    print(f"Throughput: {num_prompts / benchmark_time * 60:.2f} RPM")

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
        "--num-prompts", type=int, default=100, help="Number of prompts to process."
    )
    group.add_argument(
        "--request-rate",
        type=int,
        default=50,
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
