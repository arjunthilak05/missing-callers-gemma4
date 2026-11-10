"""Live /tokenize audit of the Exp A v2 wording pairs against the served Gemma tokenizer (no research data)."""
import json
import sys
import urllib.request

PORT, OUT = int(sys.argv[1]), sys.argv[2]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
MODEL = "absent-edges-gemma4-31b-bf16"
PAIRS = {
    "A": {"action": "Inspect the relevant source and search for additional references before you submit.",
          "C2": "Be careful with this answer. Review your assumptions and weigh the evidence before relying on it in your final list.",
          "C3": "This index answer is not an exhaustive list of callers. A missing relationship does not show that no other caller exists."},
    "B": {"action": "Check the relevant source and look for further references before you finish.",
          "C2": "Treat this evidence with caution. Question your assumptions and double-check your reasoning before you finalize the list.",
          "C3": "Treat these indexed relationships as partial evidence. An absent relationship does not mean that no further caller exists."},
}


def tok(body):
    req = urllib.request.Request("http://127.0.0.1:%d/tokenize" % PORT, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with OPENER.open(req, timeout=60) as r:
        return json.loads(r.read())["count"]


rows = {}
for pair, p in PAIRS.items():
    row = {}
    for cond in ("C2", "C3"):
        text = p[cond] + " " + p["action"]
        payload = json.dumps({"guidance": text, "matches": []}, sort_keys=True, separators=(",", ":"))
        row[cond] = {
            "text_tokens": tok({"model": MODEL, "prompt": text, "add_special_tokens": False}),
            "chat_tokens": tok({"model": MODEL, "messages": [{"role": "user", "content": payload}],
                                "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}),
        }
    d_text = row["C3"]["text_tokens"] - row["C2"]["text_tokens"]
    d_chat = row["C3"]["chat_tokens"] - row["C2"]["chat_tokens"]
    row["diff_text"], row["diff_chat"] = d_text, d_chat
    row["relative_text"] = round(abs(d_text) / max(row["C2"]["text_tokens"], row["C3"]["text_tokens"]), 4)
    row["within_tolerance"] = abs(d_text) <= 6 and row["relative_text"] <= 0.10 and abs(d_chat) <= 6
    rows[pair] = row
json.dump(rows, open(OUT, "w"), indent=1, sort_keys=True)
print(json.dumps(rows, indent=1, sort_keys=True))
