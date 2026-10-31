"""Count empty problem_statement rows in the pinned SWE-smith snapshots, overall and within the task_ids_file."""
import collections, glob, json, sys
import pyarrow.parquet as pq
HUB = "/dcai/projects/iu_0145/rasmus/.cache/huggingface/hub"
REV = {"py": "77cab9055d42ab4a5c25c89a8f937096db13558e", "go": "d2f6ecc2cf8ae1af00fc790801bbacfa2913e015",
       "java": "1172df9919a8232e066562dfdb6e0193156de0c4", "js": "d6aeb41ba9fb045c39e1c6d2322a8f08ee31cef4",
       "ts": "c243d86f112d01b8507d311af912668aa1d74449", "rs": "1ae20d63edd611309d86133466247e3a9723a27d",
       "cpp": "b1554a5f12603c825f9b48b361b9b97032347e48", "php": "6b5da17f1e3543fb2ed5d151a6f4972a457a8c36"}
ids = set(json.load(open(sys.argv[1])))
out = {}
print(f"{'lang':5s} {'rows':>7s} {'empty':>7s} {'%':>6s} | {'in ids':>6s} {'empty':>6s} {'%':>6s}")
for lang, rev in REV.items():
    files = sorted(glob.glob(f"{HUB}/datasets--SWE-bench--SWE-smith-{lang}/snapshots/{rev}/data/train-*.parquet"))
    n = e = ni = ei = 0
    for f in files:
        t = pq.read_table(f, columns=["instance_id", "problem_statement"]).to_pylist()
        for r in t:
            empty = not (r["problem_statement"] or "").strip()
            n += 1; e += empty
            if f"{lang}:{r['instance_id']}" in ids:
                ni += 1; ei += empty
                if empty: out.setdefault(lang, []).append(f"{lang}:{r['instance_id']}")
    print(f"{lang:5s} {n:7d} {e:7d} {e/max(n,1):6.1%} | {ni:6d} {ei:6d} {ei/max(ni,1):6.1%}  ({len(files)} files)")
tot = sum(len(v) for v in out.values())
print(f"empty in task_ids_file: {tot} / {len(ids)} = {tot/len(ids):.1%}")
json.dump(out, open(sys.argv[2], "w"), indent=1)
