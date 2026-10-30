import csv

with open('eval/results.csv', 'r', encoding='utf-8') as f:
    r = list(csv.DictReader(f))

for v in ['V4 — Chain-of-Thought (CoT)', 'V4 — Chain-of-Thought', 'V4 — CoT']:
    rows = [row for row in r if row.get('version') == v or (v == 'V4 — Chain-of-Thought (CoT)' and row.get('version', '').startswith('V4'))]
    if not rows:
        print(f"Version '{v}': 0 rows (aliased to V4 prompt template)")
        continue
    succ = sum(int(row.get('task_success', 0)) for row in rows)
    hall = sum(int(row.get('hallucination', 0)) for row in rows)
    tool = sum(int(row.get('tool_accuracy', 0)) for row in rows)
    print(f"Version '{v}': {len(rows)} rows | Succ: {succ}/{len(rows)} ({succ/len(rows)*100:.1f}%) | Tool: {tool}/{len(rows)} | Hall: {hall}/{len(rows)}")
    for row in rows[:3]:
        qid = row['query_id']
        prov = row['provider']
        ans = row['answer'][:40].replace('\n', ' ')
        s = row['task_success']
        print(f"   {qid}: prov={prov}, succ={s}, ans={ans}")
