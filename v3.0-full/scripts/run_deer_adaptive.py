#!/usr/bin/env python3
"""
Per-Dataset Adaptive DEER Optimizer v5
- Per-dataset think_budget based on baseline p90
- Fallback to baseline (no stop) when max_steps reached
- 256 concurrency
- Tests multiple configs on 125 regression cases
"""
import asyncio, json, math, os, sys, time
from datetime import datetime
from collections import Counter
import httpx

API = "http://127.0.0.1:8000"
MODEL = "/root/models/Qwen/Qwen3-32B"
MAX_TOKENS = 32768
TEMPERATURE = 0.6
CONCURRENCY = 256
BASELINE_DIR = "/root/benchmarks/results/v3.0-full"
RESULTS_DIR = "/root/benchmarks/results/v3.0-full/iter_v1"

sys.path.insert(0, os.path.dirname(__file__))
from run_full_mixed import (
    parse_thinking, apply_prompt, api_call, _stream_simple, geometric_mean,
    DATASET_CONFIG,
)

DATASET_CONFIGS = {
    "gsm8k": {
        "think_budget": 2000,
        "max_judge_steps": 8,
        "threshold": 0.98,
        "min_think_tokens": 600,
    },
    "math500": {
        "think_budget": 5500,
        "max_judge_steps": 15,
        "threshold": 0.98,
        "min_think_tokens": 1200,
    },
    "gpqa": {
        "think_budget": 900,
        "max_judge_steps": 5,
        "threshold": 0.95,
        "min_think_tokens": 300,
    },
    "amc": {
        "think_budget": 10000,
        "max_judge_steps": 25,
        "threshold": 0.98,
        "min_think_tokens": 2000,
    },
    "aime": {
        "think_budget": 15000,
        "max_judge_steps": 30,
        "threshold": 0.98,
        "min_think_tokens": 3000,
    },
}

PROB_CHECK_TOKENS = 10


async def adaptive_deer(question, dataset_key, ds_config):
    question_prompted = apply_prompt(question, dataset_key)
    think_budget = min(ds_config["think_budget"] * 4, 60000)
    max_steps = ds_config["max_judge_steps"]
    threshold = ds_config["threshold"]
    min_think = ds_config["min_think_tokens"]
    temp = TEMPERATURE

    thinking = ""
    total_completion_tokens = 0
    prompt_tokens = 0
    judge_step = 0
    natural_end = False
    deer_exited = False
    fell_back = False

    t0 = time.perf_counter()
    ttft = None
    think_start = think_end = None

    while judge_step < max_steps:
        if not thinking:
            messages = [{"role": "user", "content": question_prompted}]
        else:
            messages = [
                {"role": "user", "content": question_prompted},
                {"role": "assistant", "content": thinking},
            ]

        payload = {
            "model": MODEL, "messages": messages,
            "max_tokens": think_budget, "temperature": temp,
            "stream": True, "stream_options": {"include_usage": True},
            "stop": ["Wait"],
        }

        chunk = ""
        chunk_usage = {}
        stop_reason = None

        async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=30.0)) as client:
            async with client.stream("POST", f"{API}/v1/chat/completions",
                                      json=payload, headers={"Content-Type": "application/json"}) as resp:
                async for line in resp.aiter_lines():
                    if not line.strip() or not line.startswith("data: "):
                        continue
                    ds = line[6:]
                    if ds.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(ds)
                    except Exception:
                        continue
                    if data.get("usage"):
                        chunk_usage = data["usage"]
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        c = delta.get("content", "")
                        if c:
                            chunk += c
                        if choices[0].get("finish_reason"):
                            stop_reason = choices[0]["finish_reason"]

        if ttft is None:
            ttft = time.perf_counter() - t0
        if think_start is None:
            think_start = time.perf_counter() - t0

        total_completion_tokens += chunk_usage.get("completion_tokens", 0)
        prompt_tokens = max(prompt_tokens, chunk_usage.get("prompt_tokens", 0))

        if "</think" in chunk:
            thinking += chunk
            think_end = time.perf_counter() - t0
            natural_end = True
            break

        thinking += chunk

        if stop_reason == "stop":
            thinking += "Wait"
        elif stop_reason != "length":
            think_end = time.perf_counter() - t0
            natural_end = True
            break

        judge_step += 1
        if judge_step >= max_steps:
            think_end = time.perf_counter() - t0
            break

        thinking_body = thinking
        idx_t = thinking_body.find("<think")
        if idx_t >= 0:
            gt = thinking_body.find(">", idx_t)
            if gt >= 0:
                thinking_body = thinking_body[gt + 1:]

        est_think = max(1, len(thinking_body) // 4)
        if est_think < min_think:
            thinking += "Wait"
            judge_step += 1
            if judge_step >= max_steps:
                think_end = time.perf_counter() - t0
                break
            continue

        prob_messages = [
            {"role": "user", "content": question_prompted},
            {"role": "assistant", "content": thinking_body + "\n</think >\n\n**Final Answer**\n\\boxed"},
        ]
        data = await api_call(prob_messages, PROB_CHECK_TOKENS, 0.0, logprobs=True)

        choice = data["choices"][0]
        token_probs = []
        logprobs_data = choice.get("logprobs") or {}
        if logprobs_data.get("content"):
            for t in logprobs_data["content"]:
                prob = math.exp(t["logprob"])
                token_probs.append(prob)

        confidence = geometric_mean(token_probs) if token_probs else 0.0

        if confidence >= threshold:
            deer_exited = True
            think_end = time.perf_counter() - t0
            break
        else:
            thinking += "Wait"

    # KEY CHANGE: if max_steps reached and not natural/deer_exit, FALLBACK to baseline
    if not natural_end and not deer_exited:
        fell_back = True
        messages = [
            {"role": "user", "content": question_prompted},
            {"role": "assistant", "content": thinking + "Wait"},
        ]
        payload = {
            "model": MODEL, "messages": messages,
            "max_tokens": min(MAX_TOKENS, 60000), "temperature": temp,
            "stream": True, "stream_options": {"include_usage": True},
        }
        chunk = ""
        chunk_usage = {}
        async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=30.0)) as client:
            async with client.stream("POST", f"{API}/v1/chat/completions",
                                      json=payload, headers={"Content-Type": "application/json"}) as resp:
                async for line in resp.aiter_lines():
                    if not line.strip() or not line.startswith("data: "):
                        continue
                    ds = line[6:]
                    if ds.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(ds)
                    except Exception:
                        continue
                    if data.get("usage"):
                        chunk_usage = data["usage"]
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        c = delta.get("content", "")
                        if c:
                            chunk += c

        thinking += "Wait" + chunk
        total_completion_tokens += chunk_usage.get("completion_tokens", 0)
        prompt_tokens = max(prompt_tokens, chunk_usage.get("prompt_tokens", 0))

        if "</think" in chunk:
            natural_end = True
            think_end = time.perf_counter() - t0
        else:
            think_end = time.perf_counter() - t0

    if think_start and not think_end:
        think_end = time.perf_counter() - t0
    think_time = (think_end - think_start) if think_start and think_end else 0

    if natural_end:
        full_text = thinking
        reasoning, answer_content = parse_thinking(full_text)
    else:
        thinking_body = thinking
        idx_t = thinking_body.find("<think")
        if idx_t >= 0:
            gt = thinking_body.find(">", idx_t)
            if gt >= 0:
                thinking_body = thinking_body[gt + 1:]

        ans_messages = [
            {"role": "user", "content": question_prompted},
            {"role": "assistant", "content": "<think >\n" + thinking_body + "\n</think >"},
            {"role": "user", "content": "/no_think\nBased ONLY on the reasoning above, output your final answer. Do NOT re-derive. Use the format: \\boxed{answer} or #### answer"},
        ]
        answer_text, ans_usage = await _stream_simple(ans_messages, 512, temp)
        total_completion_tokens += ans_usage.get("completion_tokens", 0)
        prompt_tokens = max(prompt_tokens, ans_usage.get("prompt_tokens", 0))
        full_text = thinking + "\n</think >\n" + answer_text
        reasoning, answer_content = parse_thinking(full_text)

    total_time = time.perf_counter() - t0

    stop_label = "natural" if natural_end else ("deer_exit" if deer_exited else "fallback" if fell_back else "max_steps")

    return {
        "total_time": round(total_time, 2),
        "ttft": round(ttft, 2) if ttft else round(total_time, 2),
        "thinking_time": round(think_time, 2),
        "completion_tokens": total_completion_tokens,
        "prompt_tokens": prompt_tokens,
        "thinking_tokens_est": max(1, len(reasoning) // 4),
        "stop_reason": stop_label,
        "answer_content": answer_content,
        "full_text": full_text,
        "deer_judge_steps": judge_step,
        "early_stopped": deer_exited,
        "fell_back": fell_back,
    }


async def judge_answer(question, ground_truth, model_answer, qtype):
    JUDGE_SYSTEM = """你是一个严格的答案评判裁判模型。判断被评测模型的输出是否与标准答案一致。
【评判规则】
1. 数学题：数值在数学上等价或非常接近（误差<1%）则正确
2. 选择题：选项一致则正确
3. 只看最终答案，不看推理过程
4. 只输出一行JSON：{"correct": true} 或 {"correct": false}"""
    JUDGE_USER = """/no_think
【题目类型】{qtype}
【标准答案】{ground_truth}
【被评测模型的输出】
{model_output}
请判断模型输出是否正确。只输出一行JSON。"""
    import re
    prompt = JUDGE_USER.format(qtype=qtype, ground_truth=ground_truth[:500], model_output=model_answer[:2000])
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
        "max_tokens": 128, "temperature": 0, "stream": False,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        resp = await client.post(f"{API}/v1/chat/completions",
                                  headers={"Content-Type": "application/json"}, json=payload)
        content = resp.json()["choices"][0]["message"]["content"]
    json_match = re.search(r'\{[^{}]*"correct"\s*:\s*(true|false)[^{}]*\}', content, re.IGNORECASE)
    if json_match:
        return json.loads(json_match.group()).get("correct")
    if re.search(r'"correct"\s*:\s*true', content, re.IGNORECASE):
        return True
    return False


async def run_test(regression_cases, config_name, dataset_configs):
    sem = asyncio.Semaphore(CONCURRENCY)
    results = []
    completed = [0]
    total = len(regression_cases)
    t0 = time.time()

    async def run_one(case):
        async with sem:
            ds = case["dataset"]
            q = case["question"]
            gt = case["ground_truth"]
            qtype = DATASET_CONFIG[ds]["type"]
            try:
                r = await adaptive_deer(q, ds, dataset_configs[ds])
                correct = await judge_answer(q, gt, r.get("answer_content", ""), qtype)
                r["judge_correct"] = correct
                r["id"] = case["id"]
                r["dataset"] = ds
                results.append(r)
                completed[0] += 1
                if completed[0] % 5 == 0 or completed[0] == total:
                    fixed_so_far = sum(1 for x in results if x.get("judge_correct"))
                    elapsed = time.time() - t0
                    print(f"  [{completed[0]}/{total}] fixed={fixed_so_far} elapsed={elapsed/60:.1f}min "
                          f"{ds}/{case['id']} stop={r.get('stop_reason')} steps={r.get('deer_judge_steps')} "
                          f"E2E={r['total_time']:.1f}s", flush=True)
                return case["id"], correct, r
            except Exception as e:
                completed[0] += 1
                print(f"  ERROR {ds}/{case['id']}: {e}", flush=True)
                return case["id"], None, None

    tasks = [run_one(c) for c in regression_cases]
    gather_results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0

    fixed = sum(1 for _, c, _ in gather_results if c is True)
    still_wrong = sum(1 for _, c, _ in gather_results if c is False)
    errors = sum(1 for _, c, _ in gather_results if c is None)

    print(f"\n{'='*60}", flush=True)
    print(f"CONFIG: {config_name}", flush=True)
    print(f"Cases: {len(regression_cases)} | Fixed: {fixed} | Still wrong: {still_wrong} | Errors: {errors}", flush=True)
    print(f"Elapsed: {elapsed/60:.1f}min", flush=True)

    stop_counts = Counter(r.get("stop_reason") for r in results if r)
    print(f"Stop reasons: {dict(stop_counts)}", flush=True)

    fallback_count = sum(1 for r in results if r and r.get("fell_back"))
    print(f"Fallback to baseline: {fallback_count}", flush=True)

    ds_breakdown = {}
    for r in results:
        if not r:
            continue
        ds = r["dataset"]
        if ds not in ds_breakdown:
            ds_breakdown[ds] = {"total": 0, "correct": 0}
        ds_breakdown[ds]["total"] += 1
        if r.get("judge_correct"):
            ds_breakdown[ds]["correct"] += 1
    print(f"Per-dataset:", flush=True)
    for ds, v in ds_breakdown.items():
        print(f"  {ds}: {v['correct']}/{v['total']}", flush=True)

    remaining_cases = []
    result_map = {r["id"]: r for r in results if r}
    for case in regression_cases:
        r = result_map.get(case["id"])
        if r and not r.get("judge_correct"):
            remaining_cases.append(case)

    return results, remaining_cases


async def main():
    reg_path = os.path.join(RESULTS_DIR, "regressions.json")
    regression_cases = json.load(open(reg_path))
    print(f"Loaded {len(regression_cases)} regression cases", flush=True)

    configs = [
        {
            "name": "v5-per-dataset-adaptive",
            "dataset_configs": DATASET_CONFIGS,
        },
    ]

    best_fixed = 0
    best_results = None
    best_name = ""
    best_config = None

    for trial in configs:
        name = trial["name"]
        ds_configs = trial["dataset_configs"]
        print(f"\n{'='*60}", flush=True)
        print(f">>> Testing: {name}", flush=True)
        for ds, cfg in ds_configs.items():
            print(f"  {ds}: budget={cfg['think_budget']}tok steps={cfg['max_judge_steps']} "
                  f"thresh={cfg['threshold']} min_think={cfg['min_think_tokens']}", flush=True)

        results, remaining = await run_test(regression_cases, name, ds_configs)
        fixed = len(regression_cases) - len(remaining)
        if fixed > best_fixed:
            best_fixed = fixed
            best_results = results
            best_name = name
            best_config = ds_configs
        print(f"  Fixed: {fixed}/{len(regression_cases)} | Remaining: {len(remaining)}", flush=True)
        if not remaining:
            print("ALL FIXED!", flush=True)
            break

    print(f"\n{'='*60}", flush=True)
    print(f"BEST: {best_name}", flush=True)
    print(f"Fixed: {best_fixed}/{len(regression_cases)}", flush=True)
    for ds, cfg in best_config.items():
        print(f"  {ds}: {json.dumps(cfg)}", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(RESULTS_DIR, f"adaptive_v5_{ts}.json")
    json.dump({
        "best_name": best_name,
        "best_config": {ds: cfg for ds, cfg in best_config.items()},
        "fixed": best_fixed,
        "total": len(regression_cases),
        "results": best_results,
    }, open(save_path, "w"), ensure_ascii=False, indent=2)
    print(f"Saved: {save_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
