#!/usr/bin/env python3
"""
DEER Error Regression Optimizer
- Run DEER ONLY on regression cases with different params
- Compare results with baseline
- Iterate until no regressions
"""
import asyncio, json, math, os, sys, time, statistics
from datetime import datetime
import httpx

API = "http://127.0.0.1:8000"
MODEL = "/root/models/Qwen/Qwen3-32B"
BASELINE_DIR = "/root/benchmarks/results/v3.0-full"
MAX_TOKENS = 32768
TEMPERATURE = 0.6
CONCURRENCY = 256

sys.path.insert(0, os.path.dirname(__file__))
from run_full_mixed import (
    stream_request, deer_inference, parse_thinking, apply_prompt,
    api_call, _stream_simple, geometric_mean,
    DATASET_CONFIG, load_checkpoint, save_results
)

def build_deer_inference(params):
    threshold = params["threshold"]
    prob_tokens = params["prob_check_tokens"]
    think_ratio = params["think_ratio"]
    max_steps = params["max_judge_steps"]
    temp = params["temperature"]
    min_think_tokens = params.get("min_think_tokens", 0)
    no_stop_wait = params.get("no_stop_wait", False)

    async def _deer(question, dataset_key):
        question = apply_prompt(question, dataset_key)
        think_budget = int(think_ratio * MAX_TOKENS)
        thinking = ""
        total_completion_tokens = 0
        prompt_tokens = 0
        judge_step = 0
        natural_end = False
        deer_exited = False
        t0 = time.perf_counter()
        ttft = None
        think_start = think_end = None

        while judge_step < max_steps:
            if not thinking:
                messages = [{"role": "user", "content": question}]
            else:
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": thinking},
                ]
            payload = {
                "model": MODEL, "messages": messages,
                "max_tokens": think_budget, "temperature": temp,
                "stream": True, "stream_options": {"include_usage": True},
            }
            if not no_stop_wait:
                payload["stop"] = ["Wait"]
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
            if est_think < min_think_tokens:
                thinking += "Wait"
                judge_step += 1
                if judge_step >= max_steps:
                    think_end = time.perf_counter() - t0
                    break
                continue

            prob_messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": thinking_body + "\n</think >\n\n**Final Answer**\n\\boxed"},
            ]
            data = await api_call(prob_messages, prob_tokens, 0.0, logprobs=True)
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

        if think_start and not think_end:
            think_end = time.perf_counter() - t0

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
                {"role": "user", "content": question},
                {"role": "assistant", "content": "<think >\n" + thinking_body + "\n</think >"},
                {"role": "user", "content": "/no_think\nBased ONLY on the reasoning above, output your final answer. Do NOT re-derive. Use the format: \\boxed{answer} or #### answer"},
            ]
            answer_text, ans_usage = await _stream_simple(ans_messages, 512, temp)
            total_completion_tokens += ans_usage.get("completion_tokens", 0)
            prompt_tokens = max(prompt_tokens, ans_usage.get("prompt_tokens", 0))
            full_text = thinking + "\n</think >\n" + answer_text
            reasoning, answer_content = parse_thinking(full_text)

        total_time = time.perf_counter() - t0
        return {
            "total_time": round(total_time, 2),
            "ttft": round(ttft, 2) if ttft else round(total_time, 2),
            "completion_tokens": total_completion_tokens,
            "prompt_tokens": prompt_tokens,
            "thinking_tokens_est": max(1, len(reasoning) // 4),
            "stop_reason": "natural" if natural_end else ("deer_exit" if deer_exited else "max_steps"),
            "answer_content": answer_content,
            "full_text": full_text,
            "deer_judge_steps": judge_step,
            "early_stopped": deer_exited,
        }
    return _deer


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


async def run_iteration(regression_cases, params, iter_name):
    deer_fn = build_deer_inference(params)
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
                r = await deer_fn(q, ds)
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
                          f"{ds}/{case['id']} stop={r.get('stop_reason')} E2E={r['total_time']:.1f}s", flush=True)
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
    print(f"ITERATION: {iter_name}", flush=True)
    print(f"Params: {json.dumps(params, indent=2)}", flush=True)
    print(f"Cases: {len(regression_cases)} | Fixed: {fixed} | Still wrong: {still_wrong} | Errors: {errors}", flush=True)
    print(f"Elapsed: {elapsed/60:.1f}min", flush=True)

    stop_counts = {}
    for r in results:
        if r:
            k = r.get("stop_reason", "?")
            stop_counts[k] = stop_counts.get(k, 0) + 1
    print(f"Stop reasons: {stop_counts}", flush=True)

    remaining_cases = []
    result_map = {r["id"]: r for r in results if r}
    for case in regression_cases:
        r = result_map.get(case["id"])
        if r and not r.get("judge_correct"):
            remaining_cases.append(case)

    return results, remaining_cases


async def main():
    reg_path = "/root/benchmarks/results/v3.0-full/iter_v1/regressions.json"
    regression_cases = json.load(open(reg_path))
    print(f"Loaded {len(regression_cases)} regression cases", flush=True)

    param_grid = [
        {
            "name": "v4a: threshold=0.95, steps=20, think_ratio=0.8, min_think=500",
            "params": {
                "threshold": 0.95, "prob_check_tokens": 10,
                "think_ratio": 0.8, "max_judge_steps": 20,
                "temperature": 0.6, "min_think_tokens": 500,
            }
        },
        {
            "name": "v4b: threshold=0.98, steps=20, think_ratio=0.8, min_think=800",
            "params": {
                "threshold": 0.98, "prob_check_tokens": 10,
                "think_ratio": 0.8, "max_judge_steps": 20,
                "temperature": 0.6, "min_think_tokens": 800,
            }
        },
        {
            "name": "v4c: threshold=0.95, steps=30, think_ratio=0.9, min_think=500",
            "params": {
                "threshold": 0.95, "prob_check_tokens": 10,
                "think_ratio": 0.9, "max_judge_steps": 30,
                "temperature": 0.6, "min_think_tokens": 500,
            }
        },
    ]

    best_fixed = 0
    best_params = None
    best_results = None
    best_name = ""

    for trial in param_grid:
        name = trial["name"]
        params = trial["params"]
        print(f"\n>>> Testing: {name}", flush=True)
        results, remaining = await run_iteration(regression_cases, params, name)
        fixed = len(regression_cases) - len(remaining)
        if fixed > best_fixed:
            best_fixed = fixed
            best_params = params
            best_results = results
            best_name = name
        print(f"  Fixed: {fixed}/{len(regression_cases)} | Remaining: {len(remaining)}", flush=True)
        if not remaining:
            print("ALL FIXED!", flush=True)
            break

    print(f"\n{'='*60}", flush=True)
    print(f"BEST: {best_name}", flush=True)
    print(f"Fixed: {best_fixed}/{len(regression_cases)}", flush=True)
    print(f"Params: {json.dumps(best_params, indent=2)}", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"/root/benchmarks/results/v3.0-full/iter_v1/optim_v4_{ts}.json"
    json.dump({
        "best_name": best_name,
        "best_params": best_params,
        "fixed": best_fixed,
        "total": len(regression_cases),
        "results": best_results,
    }, open(save_path, "w"), ensure_ascii=False, indent=2)
    print(f"Saved: {save_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
