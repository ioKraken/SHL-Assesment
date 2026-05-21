"""Evaluation script for the SHL Assessment Recommender Agent.

Uses the OFFICIAL 10 conversation traces provided by SHL.
Computes Recall@10, schema compliance, URL validation, and behavior probes.

Usage:
    # Start the server first:
    uvicorn app.main:app --port 8000

    # Then run evaluation:
    python eval/evaluate.py [--base-url http://localhost:8000]
"""

import json
import sys
import os
import re
import argparse
import httpx


# ─── Parse official traces from markdown files ──────────────────

def parse_trace_md(filepath: str) -> dict:
    """Parse a single official conversation trace .md file.

    Extracts:
    - All user messages (for sending to the agent)
    - Expected recommendation URLs (from the LAST table)
    - Expected recommendation names
    """
    with open(filepath) as f:
        content = f.read()

    # Extract user messages in order
    user_blocks = re.findall(
        r'\*\*User\*\*\s*\n([\s\S]*?)(?=\n\*\*Agent\*\*|\Z)',
        content
    )
    user_messages = []
    for block in user_blocks:
        # Extract quoted text (lines starting with >)
        lines = [
            line.lstrip("> ").strip()
            for line in block.strip().split("\n")
            if line.strip().startswith(">")
        ]
        if lines:
            user_messages.append(" ".join(lines))

    # Find all recommendation tables
    tables = list(re.finditer(
        r'\| # \| Name.*?\n(?:\|.*\n)+', content
    ))

    # Extract expected URLs from the LAST table (final recommendations)
    expected_urls = []
    expected_names = []
    if tables:
        last_table = tables[-1].group()
        url_pattern = r'<(https://www\.shl\.com/products/product-catalog/view/[^>]+)>'
        expected_urls = re.findall(url_pattern, last_table)

        name_pattern = r'\| \d+ \| ([^|]+) \|'
        expected_names = [n.strip() for n in re.findall(name_pattern, last_table)]

    return {
        "user_messages": user_messages,
        "expected_urls": list(set(expected_urls)),
        "expected_names": expected_names,
    }


def load_official_traces(traces_dir: str) -> list[dict]:
    """Load all 10 official conversation traces."""
    traces = []
    for i in range(1, 11):
        filepath = os.path.join(traces_dir, f"C{i}.md")
        if not os.path.exists(filepath):
            print(f"  ⚠️  Missing trace: {filepath}")
            continue

        parsed = parse_trace_md(filepath)
        traces.append({
            "id": f"C{i}",
            "filepath": filepath,
            "first_message": parsed["user_messages"][0] if parsed["user_messages"] else "",
            "all_user_messages": parsed["user_messages"],
            "expected_urls": parsed["expected_urls"],
            "expected_names": parsed["expected_names"],
        })

    return traces


# ─── Evaluation functions ───────────────────────────────────────

def test_health(base_url: str) -> bool:
    """Test the /health endpoint."""
    try:
        resp = httpx.get(f"{base_url}/health", timeout=10)
        data = resp.json()
        ok = resp.status_code == 200 and data.get("status") == "ok"
        print(f"  /health: {'✅ PASS' if ok else '❌ FAIL'} — {data}")
        return ok
    except Exception as e:
        print(f"  /health: ❌ FAIL — {e}")
        return False


def test_schema(response: dict) -> tuple[bool, list[str]]:
    """Validate response matches the required schema."""
    issues = []

    if "reply" not in response:
        issues.append("Missing 'reply' field")
    elif not isinstance(response["reply"], str):
        issues.append(f"'reply' should be str, got {type(response['reply'])}")

    if "recommendations" not in response:
        issues.append("Missing 'recommendations' field")
    elif not isinstance(response["recommendations"], list):
        issues.append(f"'recommendations' should be list")
    else:
        for i, rec in enumerate(response["recommendations"]):
            if not isinstance(rec, dict):
                issues.append(f"rec[{i}] should be dict")
                continue
            for field in ["name", "url", "test_type"]:
                if field not in rec:
                    issues.append(f"rec[{i}] missing '{field}'")

    if "end_of_conversation" not in response:
        issues.append("Missing 'end_of_conversation' field")

    return len(issues) == 0, issues


def compute_recall_at_k(recommended_urls: list[str], expected_urls: list[str], k: int = 10) -> float:
    """Compute Recall@K: fraction of expected URLs found in top-K recommendations."""
    if not expected_urls:
        return 1.0  # No expected = vacuously correct

    # Normalize URLs for comparison (remove trailing slashes)
    norm = lambda url: url.rstrip("/").lower()
    rec_set = {norm(u) for u in recommended_urls[:k]}
    exp_set = {norm(u) for u in expected_urls}

    hits = rec_set & exp_set
    recall = len(hits) / len(exp_set)
    return recall


def run_trace(base_url: str, trace: dict, catalog_urls: set) -> dict:
    """Run a trace against the agent and compute metrics.

    Strategy: Send the first user message to get recommendations.
    This simulates the evaluator's approach of starting a conversation.
    """
    result = {
        "id": trace["id"],
        "first_message": trace["first_message"][:80],
        "expected_count": len(trace["expected_urls"]),
        "schema_pass": False,
        "urls_in_catalog": False,
        "recall_at_10": 0.0,
        "num_recs": 0,
        "issues": [],
    }

    # Build conversation messages for the request
    # Use the first user message to start (evaluator sends one message at a time)
    messages = [{"role": "user", "content": trace["first_message"]}]

    try:
        resp = httpx.post(
            f"{base_url}/chat",
            json={"messages": messages},
            timeout=30,
        )
        data = resp.json()
    except Exception as e:
        result["issues"].append(f"Request failed: {e}")
        return result

    # Schema compliance
    schema_ok, schema_issues = test_schema(data)
    result["schema_pass"] = schema_ok
    result["issues"].extend(schema_issues)

    # Get recommended URLs
    recs = data.get("recommendations", [])
    result["num_recs"] = len(recs)
    rec_urls = [r.get("url", "") for r in recs]

    # Check all URLs are in catalog
    url_issues = []
    for url in rec_urls:
        norm_url = url.rstrip("/").lower()
        if norm_url not in catalog_urls:
            url_issues.append(f"URL not in catalog: {url}")
    result["urls_in_catalog"] = len(url_issues) == 0
    result["issues"].extend(url_issues)

    # Compute Recall@10
    result["recall_at_10"] = compute_recall_at_k(rec_urls, trace["expected_urls"])

    # Show which expected items were hit/missed
    norm = lambda u: u.rstrip("/").lower()
    rec_set = {norm(u) for u in rec_urls[:10]}
    for exp_url, exp_name in zip(trace["expected_urls"], trace["expected_names"]):
        hit = "✓" if norm(exp_url) in rec_set else "✗"
        result.setdefault("hits", []).append(f"{hit} {exp_name}")

    result["reply_preview"] = data.get("reply", "")[:100]
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate SHL Assessment Recommender Agent")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    print(f"\n{'='*65}")
    print(f"  SHL Assessment Recommender — Official Evaluation")
    print(f"  Target: {base_url}")
    print(f"{'='*65}\n")

    # Load catalog URLs for validation
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    catalog_path = os.path.join(project_root, "data", "catalog.json")
    with open(catalog_path) as f:
        catalog = json.load(f)
    catalog_urls = {c["url"].rstrip("/").lower() for c in catalog}
    print(f"  Catalog: {len(catalog)} assessments loaded")
    print()

    # Load official traces
    traces_dir = os.path.join(project_root, "eval", "traces", "GenAI_SampleConversations")
    traces = load_official_traces(traces_dir)
    print(f"  Traces: {len(traces)} official conversation traces loaded")
    print()

    # Test health
    print("1. Health Check")
    health_ok = test_health(base_url)
    print()

    if not health_ok:
        print("❌ Server is not healthy. Is it running?")
        sys.exit(1)

    # Run traces
    print(f"2. Running {len(traces)} official conversation traces\n")
    results = []
    total_recall = 0.0

    for trace in traces:
        print(f"  [{trace['id']}] {trace['first_message'][:65]}...")
        result = run_trace(base_url, trace, catalog_urls)
        results.append(result)

        status = "✅" if result["recall_at_10"] > 0 else "⚠️ "
        print(f"    {status} Recall@10: {result['recall_at_10']:.2f} | Recs: {result['num_recs']} | Schema: {result['schema_pass']} | Catalog URLs: {result['urls_in_catalog']}")

        # Show hit/miss detail
        for hit in result.get("hits", []):
            print(f"      {hit}")

        if result["issues"]:
            for issue in result["issues"]:
                print(f"      ⚠️  {issue}")
        print()

        total_recall += result["recall_at_10"]

    # Summary
    mean_recall = total_recall / len(results) if results else 0
    schema_pass = sum(1 for r in results if r["schema_pass"])
    url_pass = sum(1 for r in results if r["urls_in_catalog"])
    total = len(results)

    print(f"{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Mean Recall@10:     {mean_recall:.3f}")
    print(f"  Schema compliance:  {schema_pass}/{total}")
    print(f"  URLs from catalog:  {url_pass}/{total}")
    print()
    print(f"  Per-trace Recall@10:")
    for r in results:
        bar = "█" * int(r["recall_at_10"] * 20) + "░" * (20 - int(r["recall_at_10"] * 20))
        print(f"    {r['id']}: {bar} {r['recall_at_10']:.2f}  ({r['num_recs']} recs, expected {r['expected_count']})")
    print()

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "eval_results.json")
    summary = {
        "mean_recall_at_10": mean_recall,
        "schema_pass": schema_pass,
        "url_pass": url_pass,
        "total_traces": total,
        "traces": results,
    }
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved to: {results_file}")
    print()

    return 0 if mean_recall >= 0.3 else 1


if __name__ == "__main__":
    sys.exit(main())
