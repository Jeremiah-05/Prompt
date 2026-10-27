import json

with open('eval/results.json', encoding='utf-8') as f:
    rows = json.load(f)

v1_v2 = [r for r in rows if r['version'] in ['V1 — Naive', 'V2 — Role + Rules']]
for r in v1_v2:
    if int(r.get('task_success') or 0) == 1:
        print(f"{r['version']} | {r['query_id']} | task_success={r.get('task_success')}")
        print(f"QUERY: {r.get('query')}")
        print(f"ANSWER:\n{r.get('answer')}")
        print('-'*60)
