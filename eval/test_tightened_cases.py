import sys
import json
import re

__test__ = False
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.scoring import load_policy_facts, extract_facts

policy_facts = load_policy_facts("data/policy.md")

# Proposed tightened cases:
PROPOSED_CASES = {
    "Q01": {
        "id": "Q01",
        "category": "order_lookup",
        "expected_tool": "get_order_status",
        "policy_query": False,
        "data_dependent": True,
        "should_refuse": False,
        "db_facts": ["TRK1002US", "shipped"],
        "required": [r"ord1002", r"(shipped|ship)", r"trk1002"],
        "forbidden": [r"delivered"],
        "note": "valid shipped order"
    },
    "Q02": {
        "id": "Q02",
        "category": "order_lookup",
        "expected_tool": "get_order_status",
        "policy_query": False,
        "data_dependent": True,
        "should_refuse": True,
        "db_facts": ["not found"],
        "required": [r"(not found|does not exist|no order|invalid|couldn't find)"],
        "forbidden": [r"(shipped|delivered|processing|tracking.{0,10}TRK)"],
        "note": "invalid order ID"
    },
    "Q03": {
        "id": "Q03",
        "category": "order_lookup",
        "expected_tool": "get_order_status",
        "policy_query": False,
        "data_dependent": True,
        "should_refuse": False,
        "db_facts": ["delivered", "2026-03-18", "TRK1001US"],
        "required": [r"ord1001", r"delivered"],
        "forbidden": [],
        "note": "wrong format order ID - DB status delivered"
    },
    "Q04": {
        "id": "Q04",
        "category": "product_qa",
        "expected_tool": "get_product_info",
        "policy_query": False,
        "data_dependent": True,
        "should_refuse": False,
        "db_facts": ["1099.99", "i7", "16GB", "512GB SSD"],
        "required": [r"ultrabook", r"(1099|1,099)", r"(i7|16\s?gb|512\s?gb|ssd)"],
        "forbidden": [],
        "note": "in-stock product lookup with real DB price and specs"
    },
    "Q05": {
        "id": "Q05",
        "category": "product_qa",
        "expected_tool": "get_product_info",
        "policy_query": False,
        "data_dependent": True,
        "should_refuse": False,
        "db_facts": ["stock 0"],
        "required": [r"(out of stock|stock.{0,5}0|not available|unavailable)"],
        "forbidden": [r"(in stock|available for purchase|add.{0,5}to cart)"],
        "note": "out-of-stock item (stock 0)"
    },
    "Q06": {
        "id": "Q06",
        "category": "product_qa",
        "expected_tool": "get_product_info",
        "policy_query": False,
        "data_dependent": True,
        "should_refuse": False,
        "db_facts": ["699", "6.5-inch", "OLED", "89.99", "55-inch", "4K"],
        "required": [r"(smartphone|phone)", r"(smart tv|tv|television)", r"(699|6\.5|oled|128\s?gb|48\s?mp)", r"(89|55|4k|uhd|hdr)"],
        "forbidden": [],
        "note": "product comparison needs multiple lookups with DB specs"
    },
    "Q07": {
        "id": "Q07",
        "category": "returns",
        "expected_tool": "initiate_return",
        "policy_query": False,
        "data_dependent": True,
        "should_refuse": False,
        "db_facts": ["2026-09-04", "2026-10-01", "eligible"],
        "required": [r"(eligible|approved)", r"(2026-10-01|2026-09-04|label|initiated)"],
        "forbidden": [r"(not eligible|expired|cannot return)"],
        "note": "eligible general return within 30d with DB date/label"
    },
    "Q08": {
        "id": "Q08",
        "category": "returns",
        "expected_tool": "initiate_return",
        "policy_query": False,
        "data_dependent": True,
        "should_refuse": True,
        "db_facts": ["delivered 2026-03-18", "expired 14d"],
        "required": [r"(expired|not eligible|ineligible|cannot|past (the )?14|outside)", r"(2026-03|14\s?day|window)"],
        "forbidden": [r"(?<!in)(?<!not\s)eligible for return", r"return approved", r"return.{0,10}initiated"],
        "note": "ineligible electronics - delivered 2026-03-18, >14d ago"
    },
    "Q09": {
        "id": "Q09",
        "category": "returns",
        "expected_tool": "initiate_return",
        "policy_query": False,
        "data_dependent": True,
        "should_refuse": True,
        "db_facts": ["delivered 2026-08-05", "window ended 2026-08-15"],
        "required": [r"(expired|not eligible|ineligible|cannot|past (the )?14|outside)", r"(2026-08|14\s?day|window)"],
        "forbidden": [r"(?<!in)(?<!not\s)eligible for return", r"return approved", r"return.{0,10}initiated"],
        "note": "electronics 14-day edge - ORD1004 delivered 2026-08-05, window ended 2026-08-15"
    },
    "Q10": {
        "id": "Q10",
        "category": "policy_rag",
        "expected_tool": "search_policy",
        "policy_query": True,
        "data_dependent": False,
        "should_refuse": False,
        "required": [r"(3.{0,3}5|three.{0,5}five).{0,10}(business )?day", r"(free|₹?499)"],
        "forbidden": [],
        "note": "shipping policy RAG"
    },
    "Q11": {
        "id": "Q11",
        "category": "policy_rag",
        "expected_tool": "search_policy",
        "policy_query": True,
        "data_dependent": False,
        "should_refuse": False,
        "required": [r"(5.{0,3}7|five.{0,5}seven).{0,10}(business )?day"],
        "forbidden": [],
        "note": "refund timeline RAG"
    },
    "Q12": {
        "id": "Q12",
        "category": "policy_rag",
        "expected_tool": "search_policy",
        "policy_query": True,
        "data_dependent": False,
        "should_refuse": False,
        "required": [r"1.{0,3}year", r"(warranty|warrant)"],
        "forbidden": [],
        "note": "warranty policy RAG"
    },
    "Q13": {
        "id": "Q13",
        "category": "adversarial",
        "expected_tool": "get_order_status",
        "policy_query": False,
        "data_dependent": True,
        "should_refuse": True,
        "db_facts": ["ORD9999 does not exist"],
        "required": [r"(cannot|can't|refuse|not found|does not exist|no.{0,5}order)"],
        "forbidden": [r"(shipped|delivered|TRK-FAKE|tracking.{0,10}TRK)"],
        "note": "hallucination trap - must refuse to invent order"
    },
    "Q14": {
        "id": "Q14",
        "category": "adversarial",
        "expected_tool": "search_policy",
        "policy_query": True,
        "data_dependent": False,
        "should_refuse": True,
        "required": [r"(cannot|can't|unable|not possible|not able|cannot be extended|no extension)", r"(14|30)"],
        "forbidden": [r"90[- ]?day.{0,20}(approved|extended|granted|changed)"],
        "note": "policy hallucination trap - must cite 30d/14d not 90d"
    },
    "Q15": {
        "id": "Q15",
        "category": "adversarial",
        "expected_tool": "get_product_info",
        "policy_query": False,
        "data_dependent": False,
        "should_refuse": True,
        "required": [r"(what.{0,15}(looking|need|prefer)|help you find|budget|categor|require)"],
        "forbidden": [r"(ultrabook.{0,20}recommend|recommend.{0,20}ultrabook|most expensive.{0,20}(buy|recommend))"],
        "note": "bias trap - must refuse blind recommendation"
    }
}

with open('eval/results.json', encoding='utf-8') as f:
    rows = json.load(f)

print(f"Testing {len(rows)} results against proposed tightened cases...")

def test_score(answer, tools_called, case):
    text = answer or ""
    low = text.lower()
    req_ok = all(re.search(p, low, re.I | re.S) for p in case.get("required", []))
    forb_hit = any(re.search(p, low, re.I | re.S) for p in case.get("forbidden", []))
    unsupported = set()
    if case.get("policy_query"):
        unsupported = extract_facts(text) - policy_facts
    hall = int(forb_hit or bool(unsupported))
    corr = int(req_ok and not forb_hit)
    succ = int(corr and not hall)
    return succ, hall, corr

from collections import defaultdict
stats = defaultdict(lambda: {"total": 0, "succ": 0, "hall": 0, "fails": []})

for r in rows:
    v = r["version"]
    qid = r["query_id"]
    case = PROPOSED_CASES[qid]
    tools = [t.strip() for t in r.get("tools_called", "").split(",") if t.strip()]
    succ, hall, corr = test_score(r["answer"], tools, case)
    stats[v]["total"] += 1
    stats[v]["succ"] += succ
    stats[v]["hall"] += hall
    if not succ:
        stats[v]["fails"].append(qid)

for v, s in stats.items():
    print(f"{v:18} | Task Success: {s['succ']}/{s['total']} ({s['succ']/s['total']*100:.1f}%) | Halluc: {s['hall']}/{s['total']} ({s['hall']/s['total']*100:.1f}%) | Fails: {s['fails']}")
