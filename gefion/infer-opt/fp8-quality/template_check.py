"""Do our trace contexts match what NVIDIA's chat template renders?

  template_check.py <traces_dir> <model_dir> [n_records]

Re-renders each sampled turn's context from the trace messages with the model's HF chat template
(truncate_history_thinking False and True) and compares the token ids with the trace's prompt ids
(nodes[:i] token_ids + the turn's header tokens). Counts exact matches and differing turns per harness.
"""
import collections, json, re, sys, glob, subprocess
from transformers import AutoTokenizer
tdir, mdir = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
tok = AutoTokenizer.from_pretrained(mdir)
def conv(m):
    m = dict(m)
    if isinstance(m.get("content"), list):
        m["content"] = "".join(p.get("text", "") for p in m["content"])
    if m.get("tool_calls"):
        tc = []
        for c in m["tool_calls"]:
            f = c.get("function", c); a = f.get("arguments")
            try: a = json.loads(a) if isinstance(a, str) else a
            except Exception: pass
            tc.append({"type": "function", "function": {"name": f["name"], "arguments": a}})
        m["tool_calls"] = tc
    return m
stat = collections.Counter(); ex = {}
files = sorted(glob.glob(f"{tdir}/stream/*.jsonl"))  # the uncompressed chunk is enough
n = 0
for line in open(files[-1], "rb"):
    try: d = json.loads(line)
    except Exception: continue
    if d["run"]["work"]["type"] != "train": continue
    t = d["traces"][0]; h = t["agent"]["config"].get("harness", {}).get("id") or d["env"]["name"]
    tools = [{"type": "function", "function": x} for x in (t.get("tools") or [])]
    nodes = t["nodes"]; calls = {c["node"]: c for c in t["calls"] if "node" in c}
    sampled = [i for i, x in enumerate(nodes) if x.get("sampled") and x.get("logprobs")]
    if not sampled: continue
    n += 1
    if n > N: break
    for i in (sampled[0], sampled[len(sampled) // 2], sampled[-1]):
        ids = nodes[i]["token_ids"]; hdr = len(ids) - len(nodes[i]["logprobs"])
        trace = [x for y in nodes[:i] for x in y["token_ids"]] + ids[:hdr]
        if (calls.get(i, {}).get("usage") or {}).get("prompt_tokens") != len(trace):
            stat[(h, "compacted/branched (skipped)")] += 1; continue
        msgs = [conv(x["message"]) for x in nodes[:i]]
        out = {}
        for trunc in (False, True):
            try:
                txt = tok.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=True, enable_thinking=True,
                                              truncate_history_thinking=trunc, tokenize=False)
                out[trunc] = tok.encode(txt, add_special_tokens=False)
            except Exception as e:
                out[trunc] = None; ex.setdefault((h, "error"), repr(e)[:200])
        k = "match" if out[False] == trace else "differs"
        stat[(h, f"F:{k}")] += 1
        stat[(h, "trunc==notrunc" if out[True] == out[False] else "trunc!=notrunc")] += 1
        if k == "differs" and (h, "diff") not in ex and out[False]:
            j = next((p for p, (a, b) in enumerate(zip(out[False], trace)) if a != b), min(len(out[False]), len(trace)))
            ex[(h, "diff")] = (len(out[False]), len(trace), j, repr(tok.decode(trace[max(0, j-30):j+30])), repr(tok.decode(out[False][max(0, j-30):j+30])))
for k in sorted(stat): print(k, stat[k])
for k, v in ex.items(): print(k, v)
