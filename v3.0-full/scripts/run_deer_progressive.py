#!/usr/bin/env python3
"""
Progressive Budget DEER (方案 P)
- Per-dataset budget levels: 0.5x → 0.8x → full
- Single API call per level, no stop, no confidence check
- If finish_reason=length → extend budget
- If finish_reason=stop (natural) → done
- 256 concurrency
"""
import asyncio, json, math, os, sys, time
from datetime import datetime
from collections import Counter
import httpx

API = "http://127.0.0.1:8000"
MODEL = "/root/models/Qwen/Qwen3-32B"
MAX_TOKENS = 60000
TEMPERATURE = 0.6
CONCURRENCY = 256
RESULTS_DIR = "/root/benchmarks/results/v3.0-full/iter_v1"

sys.path.insert(0, os.path.dirname(__file__))
from run_full_mixed import (
    parse_thinking, apply_prompt, _stream_simple, DATASET_CONFIG,
)

BUDGET_LEVELS = {
    "gsm8k":  [0.50, 0.80, 1.00],
    "math500": [0.50, 0.80, 1.00],
    "gpqa":   [0.50, 0.80, 1.00],
    "amc":    [0.50, 0.80, 1.00],
    "aime":   [0.50, 0.80, 1.00],
}

BASELINE_AVG_THINK = {
    "gsm8k": 1144,
    "math500": 2799,
    "gpqa": 651,
    "amc": 4240,
    "aime": 7986,
}


async def progressive_deer(question, dataset_key, budget_levels):
    question_prompted = apply_prompt(question, dataset_key)
    avg_think = BASELINE_AVG_THINK[dataset_key]
    temp = TEMPERATURE

    thinking = ""
    total_completion_tokens = 0
    prompt_tokens = 0
    final_budget_level = 0
    natural_end = False

    t0 = time.perf_counter()
    ttft = None

    for level_idx, budget_mult in enumerate(budget_levels):
        budget_tokens = int(avg_think * budget_mult) * 4
        budget_tokens = min(budget_tokens, MAX_TOKENS)

        if not thinking:
            messages = [{"role": "user", "content": question_prompted}]
        else:
            messages = [
                {"role": "user", "content": question_prompted},
                {"role": "assistant", "content": thinking},
            ]

        payload = {
            "model": MODEL, "messages": messages,
            "max_tokens": budget_tokens, "temperature": temp,
            "stream": True, "stream_options": {"include_usage": True},
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

        total_completion_tokens += chunk_usage.get("completion_tokens", 0)
        prompt_tokens = max(prompt_tokens, chunk_usage.get("prompt_tokens", 0))

        thinking += chunk
        final_budget_level = level_idx

        if stop_reason == "stop":
            natural_end = True
            break
        elif stop_reason == "length":
            if level_idx < len(budget_levels) - 1:
                continue
            else:
                break
        else:
            natural_end = True
            break

    total_time = time.perf_counter() - t0

    if natural_end and "</think" in thinking:
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
        answer_text, ans_usage = await _stream_simple(ans_messages, 256, temp)
        total_completion_tokens += ans_usage.get("completion_tokens", 0)
        prompt_tokens = max(prompt_tokens, ans_usage.get("prompt_tokens", 0))
        full_text = thinking + "\n</think >\n" + answer_text
        reasoning, answer_content = parse_thinking(full_text)

    budget_label = f"L{final_budget_level+1}({budget_levels[final_budget_level]}x)"

    return {
        "total_time": round(total_time, 2),
        "ttft": round(ttft, 2) if ttft else round(total_time, 2),
        "completion_tokens": total_completion_tokens,
        "prompt_tokens": prompt_tokens,
        "thinking_tokens_est": max(1, len(reasoning) // 4),
        "stop_reason": "natural" if natural_end else f"truncated_L{final_budget_level+1}",
        "budget_level": final_budget_level,
        "budget_label": budget_label,
        "answer_content": answer_content,
        "full_text": full_text,
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


async def run_test(regression_cases, config_name, budget_levels_map):
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
                r = await progressive_deer(q, ds, budget_levels_map[ds])
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
                          f"{ds}/{case['id']} stop={r.get('stop_reason')} budget={r.get('budget_label')} "
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

    level_counts = Counter(r.get("budget_label") for r in results if r)
    print(f"Budget levels: {dict(level_counts)}", flush=True)

    ds_breakdown = {}
    for r in results:
        if not r:
            continue
        ds = r["dataset"]
        if ds not in ds_breakdown:
            ds_breakdown[ds] = {"total": 0, "correct": 0, "level_dist": Counter()}
        ds_breakdown[ds]["total"] += 1
        ds_breakdown[ds]["level_dist"][r.get("budget_label", "?")] += 1
        if r.get("judge_correct"):
            ds_breakdown[ds]["correct"] += 1
    print(f"Per-dataset:", flush=True)
    for ds, v in ds_breakdown.items():
        print(f"  {ds}: {v['correct']}/{v['total']} levels={dict(v['level_dist'])}", flush=True)

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
            "name": "P1: progressive [0.5x, 0.8x, 1.0x]",
            "budget_levels": BUDGET_LEVELS,
        },
    ]

    for trial in configs:
        name = trial["name"]
        bl = trial["budget_levels"]
        print(f"\n{'='*60}", flush=True)
        print(f">>> Testing: {name}", flush=True)
        for ds in ["gsm8k", "math500", "gpqa", "amc", "aime"]:
            print(f"  {ds}: levels={bl[ds]} avg_think={BASELINE_AVG_THINK[ds]}", flush=True)

        results, remaining = await run_test(regression_cases, name, bl)
        fixed = len(regression_cases) - len(remaining)
        print(f"  Fixed: {fixed}/{len(regression_cases)} | Remaining: {len(remaining)}", flush=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(RESULTS_DIR, f"progressive_P1_{ts}.json")
        json.dump({
            "config_name": name,
            "budget_levels": bl,
            "fixed": fixed,
            "total": len(regression_cases),
            "results": results,
        }, open(save_path, "w"), ensure_ascii=False, indent=2)
        print(f"Saved: {save_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
