import json

with open('eval/results.json', encoding='utf-8') as f:
    rows = json.load(f)

by_query = {}
for r in rows:
    qid = r['query_id']
    by_query.setdefault(qid, {})[r['version']] = r

for qid in [f"Q0{i}" for i in range(1, 10)]:
    print("=" * 80)
    print(f"QUERY {qid}")
    v_dict = by_query.get(qid, {})
    for vname in ['V1 — Naive', 'V2 — Role + Rules', 'V3 — Few-Shot', 'V4 — CoT', 'V5 — ReAct', 'V6 — RAG + ReAct']:
        if vname in v_dict:
            r = v_dict[vname]
            ans = (r.get('answer') or '').replace('\n', ' ')[:140]
            succ = r.get('task_success')
            tool_acc = r.get('tool_accuracy')
            hall = r.get('hallucination')
            tools = r.get('tools_called') or 'none'
            print(f"  {vname[:12]:12} | succ={succ} tool={tool_acc} hall={hall} | tools=[{tools}] | ans: {ans}")
