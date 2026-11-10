"""Synthetic serving calibration for Exp A (no research data, no benchmark content).

Usage: python serve_probe.py <port> <concurrency> <public_source_file> <out_json>
Measures: /tokenize with tools, native tool-call emission, token-id return,
aggregate throughput for N concurrent distinct prompts, then the same prompts
again (prefix-cache reuse).
"""
import concurrent.futures
import json
import sys
import time
import urllib.request

PORT, N, SRC, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
BASE = "http://127.0.0.1:%d" % PORT
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
MODEL = "absent-edges-gemma4-31b-bf16"
TOOLS = [
    {"type": "function", "function": {"name": "source_reference_search",
        "description": "Search repository Python source for names and aliases. Matches are candidates.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
                       "additionalProperties": False}}},
    {"type": "function", "function": {"name": "submit_answer", "description": "Finish with call sites.",
        "parameters": {"type": "object", "properties": {"call_sites": {"type": "array", "items": {"type": "string"}}},
                       "required": ["call_sites"], "additionalProperties": False}}},
]


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read())


source = open(SRC, encoding="utf-8").read()


def messages(tag):
    return [{"role": "system", "content": "You are a careful software engineer. Always use a tool."},
            {"role": "user", "content": "[variant %s] Here is a Python module:\n```python\n%s\n```\n"
             "Which functions in this module call `_dedent`-like helpers? Use source_reference_search first." % (tag, source)}]


def one(i, tag):
    t = time.time()
    r = post("/v1/chat/completions", {"model": MODEL, "messages": messages(tag), "tools": TOOLS,
             "tool_choice": "auto", "max_tokens": 256, "temperature": 0.2, "seed": 11 + i,
             "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": False}})
    u = r["usage"]
    m = r["choices"][0]["message"]
    return {"i": i, "seconds": round(time.time() - t, 3), "prompt_tokens": u["prompt_tokens"],
            "completion_tokens": u["completion_tokens"], "tool_calls": [c["function"]["name"] for c in (m.get("tool_calls") or [])],
            "token_ids_returned": isinstance(r.get("prompt_token_ids"), list) and isinstance(r["choices"][0].get("token_ids"), list)}


def round_(tags):
    t = time.time()
    with concurrent.futures.ThreadPoolExecutor(len(tags)) as ex:
        rows = list(ex.map(lambda a: one(*a), list(enumerate(tags))))
    wall = time.time() - t
    return {"wall_seconds": round(wall, 3), "requests": len(rows),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "completion_tokens_per_second": round(sum(r["completion_tokens"] for r in rows) / wall, 1),
            "prompt_tokens_per_second": round(sum(r["prompt_tokens"] for r in rows) / wall, 1),
            "tool_call_rate": sum(1 for r in rows if r["tool_calls"]) / len(rows),
            "token_ids_all_returned": all(r["token_ids_returned"] for r in rows), "rows": rows}


result = {"models": post("/v1/models", None) if False else None}
req = urllib.request.Request(BASE + "/v1/models")
with OPENER.open(req, timeout=30) as r:
    result["models"] = json.loads(r.read())
tok = post("/tokenize", {"model": MODEL, "messages": messages("tok"), "tools": TOOLS, "add_generation_prompt": True,
                         "add_special_tokens": False, "chat_template_kwargs": {"enable_thinking": False}})
result["tokenize"] = {"count": tok.get("count"), "max_model_len": tok.get("max_model_len")}
result["single"] = one(0, "single")
result["round_cold"] = round_(["cold-%d" % i for i in range(N)])
result["round_warm"] = round_(["cold-%d" % i for i in range(N)])
json.dump(result, open(OUT, "w"), indent=1)
print(json.dumps({k: (v if k in ("tokenize", "single") else {kk: vv for kk, vv in v.items() if kk != "rows"} if isinstance(v, dict) and "rows" in v else "ok") for k, v in result.items()}, indent=1))
