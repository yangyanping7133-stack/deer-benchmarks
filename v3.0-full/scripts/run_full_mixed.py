#!/usr/bin/env python3
"""
Full-scale Mixed Benchmark Runner v3.0
- 256 concurrency, all datasets mixed and interleaved
- Results saved per-dataset
- Progress log with vLLM status, KV cache, throughput
- Self-validation before run
"""
import asyncio, json, math, os, random, re, sys, time, statistics, subprocess, threading
from datetime import datetime
import httpx

API = "http://127.0.0.1:8000"
MODEL = "/root/models/Qwen/Qwen3-32B"
DATA_DIR = "/root/benchmarks/data/CRC-QAD/v2.0-6sets"
OUT_DIR = "/root/benchmarks/results/v3.0-full"
LOG_FILE = os.path.join(OUT_DIR, "progress.log")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_TOKENS = 32768
TEMPERATURE = 0.6
CONCURRENCY = 256

DEER_PARAMS = {
    "threshold": 0.95,
    "prob_check_tokens": 10,
    "think_ratio": 0.8,
    "max_judge_steps": 4,
    "temperature": 0.6,
    "min_think_tokens": 1000,
}

DATASET_CONFIG = {
    "gsm8k":         {"file": "gsm8k.json",         "type": "math", "prompt_suffix": "\nPlease solve this step by step."},
    "math500":       {"file": "math500.json",        "type": "math", "prompt_suffix": ""},
    "amc":           {"file": "amc.json",            "type": "math", "prompt_suffix": ""},
    "gpqa":          {"file": "gpqa.json",           "type": "mc",   "prompt_suffix": ""},
    "aime":          {"file": "aime.json",           "type": "math", "prompt_suffix": ""},
}

_progress = {"done": 0, "total": 0, "method": "", "start": time.time(),
             "ds_done": {}}  # per-dataset done count


# ========== vLLM / NPU monitoring ==========

def _get_vllm_status():
    running = waiting = 0
    kv_pct = 0.0
    try:
        import urllib.request
        resp = urllib.request.urlopen(f"{API}/metrics", timeout=5)
        for line in resp.read().decode().split("\n"):
            s = line.strip()
            if s.startswith("vllm:num_requests_running{"):
                running = int(float(s.split()[-1]))
            elif s.startswith("vllm:num_requests_waiting{"):
                waiting = int(float(s.split()[-1]))
            elif s.startswith("vllm:kv_cache_usage_perc{"):
                kv_pct = float(s.split()[-1])
    except Exception:
        pass
    return running, waiting, kv_pct


def _get_npu_hbm():
    try:
        out = subprocess.check_output(["npu-smi", "info"], timeout=10, text=True)
        used = total = 0
        for u, t in re.findall(r'(\d+)\s*/\s*(\d+)\s*', out):
            u, t = int(u), int(t)
            if t == 32768:
                used += u
                total += t
        if total > 0:
            return used, total
    except Exception:
        pass
    return 0, 0


# ========== Inference ==========

def geometric_mean(probs):
    if not probs:
        return 0.0
    return math.exp(sum(math.log(max(p, 1e-10)) for p in probs) / len(probs))


def parse_thinking(text):
    reasoning, content = "", text
    idx = text.find("<think")
    if idx >= 0:
        gt = text.find(">", idx)
        if gt >= 0:
            end = text.find("</think", gt)
            if end >= 0:
                close = text.find(">", end)
                reasoning = text[gt + 1:end].strip()
                content = text[close + 1:].strip() if close >= 0 else ""
            else:
                reasoning = text[gt + 1:].strip()
                content = ""
    return reasoning, content


def apply_prompt(question, dataset_key):
    cfg = DATASET_CONFIG.get(dataset_key, {})
    suffix = cfg.get("prompt_suffix", "")
    return question + suffix if suffix else question


async def stream_request(messages):
    payload = {
        "model": MODEL, "messages": messages,
        "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
        "stream": True, "stream_options": {"include_usage": True},
    }
    full_text = ""
    ttft = None
    t0 = time.perf_counter()
    usage = {}
    think_start = think_end = None
    in_think = False
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
                    usage = data["usage"]
                choices = data.get("choices", [])
                if not choices:
                    continue
                ch = choices[0]
                delta = ch.get("delta", {})
                c = delta.get("content", "")
                if c:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    full_text += c
                    if "<think" in full_text and not in_think:
                        gt = full_text.find(">", full_text.find("<think"))
                        if gt >= 0:
                            in_think = True
                            think_start = time.perf_counter() - t0
                    if "</think" in full_text and in_think:
                        in_think = False
                        think_end = time.perf_counter() - t0
                if ch.get("finish_reason"):
                    stop_reason = ch["finish_reason"]

    total_time = time.perf_counter() - t0
    if ttft is None:
        ttft = total_time
    reasoning, answer_content = parse_thinking(full_text)
    think_time = (think_end - think_start) if think_start and think_end else 0

    return {
        "total_time": round(total_time, 2),
        "ttft": round(ttft, 2),
        "thinking_time": round(think_time, 2),
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "thinking_tokens_est": max(1, len(reasoning) // 4),
        "stop_reason": stop_reason,
        "answer_content": answer_content,
        "full_text": full_text,
    }


async def api_call(messages, max_tokens, temperature=0.0, logprobs=False):
    payload = {
        "model": MODEL, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
        "stream": False,
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 1
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        resp = await client.post(
            f"{API}/v1/chat/completions",
            json=payload, headers={"Content-Type": "application/json"},
        )
        return resp.json()


async def _stream_simple(messages, max_tokens, temperature):
    payload = {
        "model": MODEL, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
        "stream": True, "stream_options": {"include_usage": True},
    }
    full_text = ""
    usage = {}
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
                    usage = data["usage"]
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    c = delta.get("content", "")
                    if c:
                        full_text += c
    return full_text, usage


async def deer_inference(question, dataset_key):
    question = apply_prompt(question, dataset_key)
    threshold = DEER_PARAMS["threshold"]
    prob_tokens = DEER_PARAMS["prob_check_tokens"]
    think_budget = int(DEER_PARAMS["think_ratio"] * MAX_TOKENS)
    max_steps = DEER_PARAMS["max_judge_steps"]
    temp = DEER_PARAMS["temperature"]
    min_think_tokens = DEER_PARAMS.get("min_think_tokens", 0)

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
        "thinking_time": round(think_time, 2),
        "completion_tokens": total_completion_tokens,
        "prompt_tokens": prompt_tokens,
        "thinking_tokens_est": max(1, len(reasoning) // 4),
        "stop_reason": "natural" if natural_end else ("deer_exit" if deer_exited else "max_steps"),
        "answer_content": answer_content,
        "full_text": full_text,
        "deer_judge_steps": judge_step,
        "early_stopped": deer_exited,
    }


# ========== Checkpoint ==========

def load_checkpoint(save_path):
    if os.path.exists(save_path):
        try:
            results = json.load(open(save_path))
            done_ids = {r.get("id", r.get("index")) for r in results if r and "error" not in r}
            return results, done_ids
        except Exception:
            pass
    return [], set()


def save_results(save_path, results_map):
    sorted_r = sorted(results_map.values(), key=lambda x: x.get("index", 999))
    with open(save_path, "w") as f:
        json.dump(sorted_r, f, ensure_ascii=False, indent=2)


# ========== Self-validation ==========

def self_validate(all_samples, datasets):
    errors = []
    ds_counts = {}
    for ds in datasets:
        ds_counts[ds] = 0
    for s in all_samples:
        ds = s["_dataset"]
        ds_counts[ds] = ds_counts.get(ds, 0) + 1
        if "question" not in s:
            errors.append(f"sample id={s.get('id','?')} missing 'question'")
        if "answer" not in s:
            errors.append(f"sample id={s.get('id','?')} missing 'answer'")
        if ds not in DATASET_CONFIG:
            errors.append(f"sample id={s.get('id','?')} unknown dataset '{ds}'")

    expected = {"gsm8k": 500, "math500": 500, "gpqa": 200, "amc": 40, "aime": 30}
    for ds in datasets:
        exp = expected.get(ds, 0)
        got = ds_counts.get(ds, 0)
        if exp > 0 and got != exp:
            errors.append(f"{ds}: expected {exp} samples, got {got}")

    if len(all_samples) != len(set(s.get("id", i) for i, s in enumerate(all_samples))):
        errors.append("duplicate IDs detected across datasets")

    return errors, ds_counts


# ========== Main mixed runner ==========

async def run_full_mixed(method="baseline"):
    datasets = list(DATASET_CONFIG.keys())
    all_samples = []

    for ds in datasets:
        cfg = DATASET_CONFIG[ds]
        data_path = os.path.join(DATA_DIR, cfg["file"])
        samples = json.load(open(data_path))
        for i, s in enumerate(samples):
            s["_dataset"] = ds
            s["_qtype"] = cfg["type"]
            if "id" not in s:
                s["id"] = f"{ds}-{i:04d}"
            s["_orig_index"] = i
        all_samples.extend(samples)

    errors, ds_counts = self_validate(all_samples, datasets)
    if errors:
        print("=== SELF-VALIDATION FAILED ===", flush=True)
        for e in errors:
            print(f"  ERROR: {e}", flush=True)
        sys.exit(1)

    print("=== SELF-VALIDATION PASSED ===", flush=True)
    for ds in datasets:
        print(f"  {ds}: {ds_counts.get(ds, 0)} samples", flush=True)
    print(f"  Total: {len(all_samples)} samples", flush=True)

    random.seed(42)
    random.shuffle(all_samples)

    results_by_ds = {}
    pending_by_ds = {}
    for ds in datasets:
        save_path = os.path.join(OUT_DIR, f"{ds}_{method}.json")
        existing, done_ids = load_checkpoint(save_path)
        results_by_ds[ds] = {
            "map": {r.get("id", r.get("index")): r for r in existing if r},
            "save_path": save_path,
            "done": len(done_ids),
            "total": ds_counts.get(ds, 0),
        }
        pending_by_ds[ds] = []

    pending = []
    for s in all_samples:
        ds = s["_dataset"]
        sid = s["id"]
        if sid not in results_by_ds[ds]["map"]:
            pending.append(s)
            pending_by_ds[ds].append(s)

    total_pending = len(pending)
    total_all = len(all_samples)
    already_done = total_all - total_pending

    print(f"\n{'='*70}", flush=True)
    print(f"FULL MIXED | Method: {method} | Concurrency: {CONCURRENCY}", flush=True)
    print(f"Total: {total_all} | Already done: {already_done} | Pending: {total_pending}", flush=True)
    for ds in datasets:
        info = results_by_ds[ds]
        p = len(pending_by_ds[ds])
        print(f"  {ds}: {info['total']} total, {info['done']} done, {p} pending", flush=True)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    if not pending:
        print("All done.", flush=True)
        return

    _progress["total"] = total_all
    _progress["done"] = already_done
    _progress["method"] = method
    _progress["start"] = time.time()
    _progress["ds_done"] = {ds: results_by_ds[ds]["done"] for ds in datasets}

    sem = asyncio.Semaphore(CONCURRENCY)

    LOG_INTERVAL = 60
    log_stop = threading.Event()

    def _log_progress():
        while True:
            done = _progress["done"]
            total = _progress["total"]
            elapsed = time.time() - _progress["start"]
            eta = (elapsed / max(done - already_done, 1)) * (total - done) if (done - already_done) > 0 else 0
            throughput = (done - already_done) / (elapsed / 60) if elapsed > 0 else 0
            vllm_run, vllm_wait, kv_pct = _get_vllm_status()
            used_hbm, total_hbm = _get_npu_hbm()
            hbm_pct = used_hbm * 100 // total_hbm if total_hbm > 0 else 0

            ds_summary = " ".join(f"{ds}:{_progress['ds_done'].get(ds,0)}/{results_by_ds[ds]['total']}" for ds in datasets)
            line = (f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"{method} progress: {done}/{total} ({done*100//max(total,1)}%) "
                    f"elapsed:{elapsed/60:.1f}min ETA:{eta/60:.1f}min TP:{throughput:.1f}req/min "
                    f"vLLM:{vllm_run}run/{vllm_wait}wait KV:{kv_pct*100:.2f}% HBM:{hbm_pct}% "
                    f"| {ds_summary}\n")
            with open(LOG_FILE, "a") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            print(f"  [LOG] {done}/{total} ({done*100//max(total,1)}%) ETA:{eta/60:.1f}min "
                  f"TP:{throughput:.1f}req/min | vLLM:{vllm_run}/{vllm_wait} KV:{kv_pct*100:.2f}% "
                  f"| {ds_summary}", flush=True)
            if log_stop.wait(LOG_INTERVAL):
                break

    log_thread = threading.Thread(target=_log_progress, daemon=True)
    log_thread.start()

    completed = 0

    async def _run_one(sample):
        nonlocal completed
        async with sem:
            ds = sample["_dataset"]
            qtype = sample["_qtype"]
            q = sample["question"]
            gt = sample.get("answer", "")
            sid = sample["id"]
            orig_idx = sample.get("_orig_index", 0)

            try:
                if method == "deer":
                    r = await deer_inference(q, ds)
                else:
                    q_prompted = apply_prompt(q, ds)
                    msgs = [{"role": "user", "content": q_prompted}]
                    r = await stream_request(msgs)

                r["index"] = orig_idx
                r["id"] = sid
                r["dataset"] = ds
                r["question"] = q[:200]
                r["ground_truth"] = gt
                r["method"] = method
                r["qtype"] = qtype
            except Exception as e:
                print(f"  ERROR [{ds}/{sid}]: {e}", flush=True)
                return

            results_by_ds[ds]["map"][sid] = r
            save_results(results_by_ds[ds]["save_path"], results_by_ds[ds]["map"])
            completed += 1
            _progress["done"] += 1
            _progress["ds_done"][ds] = _progress["ds_done"].get(ds, 0) + 1

            if completed % 50 == 0 or completed <= 10 or completed == total_pending:
                print(f"  [{completed}/{total_pending}] {ds}/{sid} E2E={r['total_time']:.1f}s "
                      f"tok={r['completion_tokens']}", flush=True)

    tasks = [_run_one(s) for s in pending]
    await asyncio.gather(*tasks)

    log_stop.set()
    log_thread.join(timeout=5)

    print(f"\n{'='*70}", flush=True)
    print(f"DONE | {method} | {datetime.now().strftime('%H:%M:%S')}", flush=True)
    for ds in datasets:
        info = results_by_ds[ds]
        valid = [r for r in info["map"].values() if "error" not in r]
        if valid:
            avg_e2e = statistics.mean(r["total_time"] for r in valid)
            avg_tok = statistics.mean(r["completion_tokens"] for r in valid)
            print(f"  {ds}: {len(valid)} done, avg E2E={avg_e2e:.1f}s, avg tok={avg_tok:.0f}", flush=True)


async def run_judge_all():
    datasets = list(DATASET_CONFIG.keys())

    JUDGE_SYSTEM = """你是一个严格的答案评判裁判模型。判断被评测模型的输出是否与标准答案一致。

【评判规则】
1. 数学题：数值在数学上等价或非常接近（误差<1%）则正确
2. 选择题：选项一致则正确
3. 代码题：核心算法逻辑正确则正确
4. 只看最终答案，不看推理过程
5. 只输出一行JSON：{"correct": true} 或 {"correct": false}"""

    JUDGE_USER = """/no_think
【题目类型】{qtype}
【标准答案】{ground_truth}
【被评测模型的输出】
{model_output}

请判断模型输出是否正确。只输出一行JSON。"""

    def parse_judge(content):
        answer = content
        te = content.find("</think")
        if te >= 0:
            cg = content.find(">", te)
            if cg >= 0:
                answer = content[cg + 1:].strip()
        json_match = re.search(r'\{[^{}]*"correct"\s*:\s*(true|false)[^{}]*\}', answer, re.IGNORECASE)
        if json_match:
            return json.loads(json_match.group()).get("correct")
        if re.search(r'"correct"\s*:\s*true', answer, re.IGNORECASE):
            return True
        if re.search(r'"correct"\s*:\s*false', answer, re.IGNORECASE):
            return False
        return None

    for method in ["baseline", "deer"]:
        for ds in datasets:
            in_path = os.path.join(OUT_DIR, f"{ds}_{method}.json")
            save_path = os.path.join(OUT_DIR, f"{ds}_{method}_judged.json")
            if not os.path.exists(in_path):
                continue

            results = json.load(open(in_path))
            if os.path.exists(save_path):
                judged_existing = json.load(open(save_path))
                judged_map = {r.get("id", r.get("index")): r for r in judged_existing}
            else:
                judged_map = {r.get("id", r.get("index")): r for r in results}

            to_judge = [r for r in results if "error" not in r and r.get("id") not in {k for k, v in judged_map.items() if v.get("judge_correct") is not None}]

            if not to_judge:
                already = sum(1 for r in judged_map.values() if r.get("judge_correct") is not None)
                print(f"  Judge {ds}/{method}: {already} already judged, skip", flush=True)
                continue

            qtype = DATASET_CONFIG[ds]["type"]
            sem = asyncio.Semaphore(32)

            async def judge_one(r):
                async with sem:
                    prompt = JUDGE_USER.format(
                        qtype=qtype,
                        ground_truth=r.get("ground_truth", "")[:500],
                        model_output=(r.get("answer_content", "") or "")[:2000],
                    )
                    payload = {
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 128, "temperature": 0, "stream": False,
                    }
                    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                        resp = await client.post(f"{API}/v1/chat/completions",
                                                  headers={"Content-Type": "application/json"}, json=payload)
                        data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    correct = parse_judge(content)
                    return r.get("id"), correct, content[:200]

            print(f"  Judging {ds}/{method}: {len(to_judge)} samples...", flush=True)
            judged = await asyncio.gather(*[judge_one(r) for r in to_judge], return_exceptions=True)

            for item in judged:
                if isinstance(item, Exception):
                    continue
                sid, correct, raw = item
                if sid in judged_map:
                    judged_map[sid]["judge_correct"] = correct
                    judged_map[sid]["judge_raw"] = raw

            all_results = sorted(judged_map.values(), key=lambda x: x.get("index", 999))
            with open(save_path, "w") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

            valid = [r for r in all_results if r.get("judge_correct") is not None]
            correct_count = sum(1 for r in valid if r["judge_correct"])
            acc = correct_count / len(valid) * 100 if valid else 0
            print(f"  {ds}/{method}: {correct_count}/{len(valid)} = {acc:.0f}%", flush=True)


def generate_report():
    datasets = list(DATASET_CONFIG.keys())
    lines = []
    lines.append("# DEER CoT Compression Full Benchmark Report v3.0")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Environment")
    lines.append("- Model: Qwen3-32B (bfloat16)")
    lines.append("- Framework: vLLM-Ascend (8x Ascend 910B4)")
    lines.append(f"- Concurrency: {CONCURRENCY}")
    lines.append(f"- Temperature: {TEMPERATURE}, Max Tokens: {MAX_TOKENS}")
    lines.append("")
    lines.append("## DEER Parameters")
    for k, v in DEER_PARAMS.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Comparison: Baseline vs DEER\n")

    header = "| Dataset | Samples | BL Acc | DEER Acc | BL E2E(s) | DEER E2E(s) | Speedup | BL Tok | DEER Tok | Tok Red | Acc Diff |"
    sep = "|---------|---------|--------|----------|-----------|-------------|---------|--------|----------|---------|----------|"
    lines.append(header)
    lines.append(sep)

    speedups = []
    acc_diffs = []

    for ds in datasets:
        bl_path = os.path.join(OUT_DIR, f"{ds}_baseline_judged.json")
        dr_path = os.path.join(OUT_DIR, f"{ds}_deer_judged.json")
        bl_path2 = os.path.join(OUT_DIR, f"{ds}_baseline.json")
        dr_path2 = os.path.join(OUT_DIR, f"{ds}_deer.json")

        bl_data = json.load(open(bl_path if os.path.exists(bl_path) else bl_path2))
        dr_data = json.load(open(dr_path if os.path.exists(dr_path) else dr_path2))

        bl_valid = [r for r in bl_data if "error" not in r]
        dr_valid = [r for r in dr_data if "error" not in r]

        bl_judged = [r for r in bl_valid if r.get("judge_correct") is not None]
        dr_judged = [r for r in dr_valid if r.get("judge_correct") is not None]

        bl_correct = sum(1 for r in bl_judged if r["judge_correct"]) if bl_judged else 0
        dr_correct = sum(1 for r in dr_judged if r["judge_correct"]) if dr_judged else 0

        bl_acc = f"{bl_correct}/{len(bl_judged)}" if bl_judged else "N/A"
        dr_acc = f"{dr_correct}/{len(dr_judged)}" if dr_judged else "N/A"

        bl_e2e = statistics.mean(r["total_time"] for r in bl_valid) if bl_valid else 0
        dr_e2e = statistics.mean(r["total_time"] for r in dr_valid) if dr_valid else 0
        bl_tok = statistics.mean(r["completion_tokens"] for r in bl_valid) if bl_valid else 0
        dr_tok = statistics.mean(r["completion_tokens"] for r in dr_valid) if dr_valid else 0

        speedup = (1 - dr_e2e / bl_e2e) * 100 if bl_e2e > 0 else 0
        tok_red = (1 - dr_tok / bl_tok) * 100 if bl_tok > 0 else 0
        acc_diff = dr_correct - bl_correct

        speedups.append(speedup)
        acc_diffs.append(acc_diff)

        lines.append(f"| {ds} | {len(bl_valid)} | {bl_acc} | {dr_acc} | "
                     f"{bl_e2e:.1f} | {dr_e2e:.1f} | {speedup:.1f}% | "
                     f"{bl_tok:.0f} | {dr_tok:.0f} | {tok_red:.1f}% | {acc_diff:+d} |")

    avg_speedup = statistics.mean(speedups) if speedups else 0
    min_acc = min(acc_diffs) if acc_diffs else 0
    lines.append("")
    lines.append(f"- **Avg Time Speedup: {avg_speedup:.1f}%**")
    lines.append(f"- **Min Accuracy Diff: {min_acc}**")
    lines.append("")
    passed = avg_speedup >= 20 and min_acc >= 0
    lines.append(f"## Conclusion: {'PASSED' if passed else 'NEEDS OPTIMIZATION'}")
    lines.append(f"- Speedup >= 20%: {'MET' if avg_speedup >= 20 else 'NOT MET'} ({avg_speedup:.1f}%)")
    lines.append(f"- Accuracy no drop: {'MET' if min_acc >= 0 else 'NOT MET'} (min: {min_acc})")

    report_path = os.path.join(OUT_DIR, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport: {report_path}")
    print("\n".join(lines))


async def main():
    method = sys.argv[1] if len(sys.argv) > 1 else "all"
    skip_judge = "--skip-judge" in sys.argv
    skip_report = "--skip-report" in sys.argv

    methods = ["baseline", "deer"] if method == "all" else [method]

    for m in methods:
        with open(LOG_FILE, "w") as f:
            f.write("")
        await run_full_mixed(m)

    if not skip_judge:
        await run_judge_all()

    if not skip_report:
        generate_report()


if __name__ == "__main__":
    asyncio.run(main())
