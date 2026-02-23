"""
Vector Native Density Test

Direct empirical test of VN's core hypothesis:
  "Structured symbols activate denser pre-trained representations"

Experimental Design:
1. Symbol comparison: ● vs "attention" vs "important"
2. Format comparison: VN syntax vs natural language (same meaning)
3. Structure comparison: config-style vs prose

Metrics:
- n_active: Total active features
- density: Features per character
- concentration: How focused the activation is

Usage:
    # Run on Modal (requires GPU)
    modal run scripts/modal_vn_density.py

    # Analyze results after Modal run
    python experiments/vn_density_test.py --results vn_density_results.json
"""

import json
from dataclasses import dataclass
from typing import List, Dict
import argparse


@dataclass
class TestCase:
    """A single comparison test case."""
    name: str
    category: str  # symbol, format, structure
    variants: Dict[str, str]  # variant_name -> text
    hypothesis: str  # What we expect


# =============================================================================
# TEST CASES
# =============================================================================

SYMBOL_TESTS = [
    TestCase(
        name="attention_symbol",
        category="symbol",
        variants={
            "vn_symbol": "●",
            "word_attention": "attention",
            "word_important": "important",
            "word_focus": "focus",
            "phrase": "pay attention to this",
        },
        hypothesis="● has highest density (most associations per char)"
    ),
    TestCase(
        name="merge_symbol",
        category="symbol",
        variants={
            "vn_symbol": "⊕",
            "word_add": "add",
            "word_merge": "merge",
            "word_combine": "combine",
            "phrase": "add these together",
        },
        hypothesis="⊕ has highest density (math training data)"
    ),
    TestCase(
        name="flow_symbol",
        category="symbol",
        variants={
            "vn_symbol": "→",
            "word_then": "then",
            "word_next": "next",
            "phrase": "and then do",
        },
        hypothesis="→ has high density (diagrams, code)"
    ),
    TestCase(
        name="block_symbol",
        category="symbol",
        variants={
            "vn_symbol": "≠",
            "word_not": "not",
            "word_block": "block",
            "word_reject": "reject",
            "phrase": "do not allow",
        },
        hypothesis="≠ has highest density (math/programming)"
    ),
]

FORMAT_TESTS = [
    TestCase(
        name="simple_instruction",
        category="format",
        variants={
            "vn": "●analyze|data:sales",
            "natural": "Please analyze the sales data",
            "terse": "analyze sales data",
        },
        hypothesis="VN format has highest density"
    ),
    TestCase(
        name="complex_instruction",
        category="format",
        variants={
            "vn": "●analyze|dataset:Q4_sales|focus:revenue|output:summary",
            "natural": "Please analyze the Q4 sales dataset, focusing on revenue, and output a summary",
            "terse": "analyze Q4 sales revenue summary",
        },
        hypothesis="VN maintains density advantage at scale"
    ),
    TestCase(
        name="workflow",
        category="format",
        variants={
            "vn": "●workflow|id:review→●analyze|data:Q4→●generate|type:report",
            "natural": "Start a workflow called review. First analyze the Q4 data. Then generate a report.",
            "terse": "workflow review: analyze Q4, generate report",
        },
        hypothesis="VN workflow syntax is denser"
    ),
    TestCase(
        name="state_declaration",
        category="format",
        variants={
            "vn": "●STATE|Ψ:editor|Ω:documentation|mode:execute",
            "natural": "You are an editor. Your goal is documentation. Execute mode.",
            "terse": "editor, goal: documentation, mode: execute",
        },
        hypothesis="VN state symbols are densest"
    ),
]

STRUCTURE_TESTS = [
    TestCase(
        name="key_value_pairs",
        category="structure",
        variants={
            "config": "name:John|age:30|role:engineer",
            "json_like": '{"name": "John", "age": 30, "role": "engineer"}',
            "prose": "John is 30 years old and works as an engineer",
        },
        hypothesis="Pipe-delimited config is densest"
    ),
    TestCase(
        name="parameter_list",
        category="structure",
        variants={
            "vn_pipes": "dataset:sales|metrics:revenue,profit|period:Q4",
            "comma_list": "dataset=sales, metrics=revenue and profit, period=Q4",
            "prose": "using the sales dataset, looking at revenue and profit metrics for Q4",
        },
        hypothesis="VN pipe syntax is densest"
    ),
]

ALL_TESTS = SYMBOL_TESTS + FORMAT_TESTS + STRUCTURE_TESTS


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze experimental results from real Modal run."""
    analysis = {
        "summary": {},
        "by_category": {},
        "hypotheses": [],
    }

    # Group by test case
    by_test = {}
    for r in results:
        test_name = r["test_name"]
        if test_name not in by_test:
            by_test[test_name] = {}
        by_test[test_name][r["variant"]] = r

    # Analyze each test
    for test in ALL_TESTS:
        if test.name not in by_test:
            continue

        variants = by_test[test.name]

        # Find highest density variant
        densities = {v: variants[v].get("density", 0) for v in variants}
        if not densities:
            continue

        max_variant = max(densities, key=densities.get)

        # Check if VN variant won
        vn_variants = ["vn_symbol", "vn", "config", "vn_pipes"]
        vn_won = any(v in max_variant for v in vn_variants)

        test_result = {
            "test": test.name,
            "category": test.category,
            "hypothesis": test.hypothesis,
            "densities": densities,
            "winner": max_variant,
            "vn_won": vn_won,
        }

        analysis["hypotheses"].append(test_result)

        # Aggregate by category
        cat = test.category
        if cat not in analysis["by_category"]:
            analysis["by_category"][cat] = {"vn_wins": 0, "total": 0}
        analysis["by_category"][cat]["total"] += 1
        if vn_won:
            analysis["by_category"][cat]["vn_wins"] += 1

    # Summary
    total_tests = len(analysis["hypotheses"])
    vn_wins = sum(1 for h in analysis["hypotheses"] if h["vn_won"])
    analysis["summary"] = {
        "total_tests": total_tests,
        "vn_wins": vn_wins,
        "vn_win_rate": vn_wins / total_tests if total_tests > 0 else 0,
    }

    return analysis


def print_results(analysis: Dict):
    """Pretty-print analysis results."""
    print("=" * 70)
    print("VECTOR NATIVE DENSITY TEST RESULTS")
    print("=" * 70)
    print()

    # Summary
    s = analysis["summary"]
    print(f"OVERALL: VN won {s['vn_wins']}/{s['total_tests']} tests ({100*s['vn_win_rate']:.0f}%)")
    print()

    # By category
    print("BY CATEGORY:")
    print("-" * 50)
    for cat, stats in analysis["by_category"].items():
        rate = stats["vn_wins"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {cat}: {stats['vn_wins']}/{stats['total']} ({100*rate:.0f}%)")
    print()

    # Individual tests
    print("INDIVIDUAL TESTS:")
    print("-" * 70)
    for h in analysis["hypotheses"]:
        status = "✓" if h["vn_won"] else "✗"
        print(f"\n{status} {h['test']} ({h['category']})")
        print(f"  Hypothesis: {h['hypothesis']}")
        print(f"  Winner: {h['winner']}")
        print(f"  Densities: ", end="")
        for v, d in sorted(h["densities"].items(), key=lambda x: -x[1]):
            print(f"{v}={d:.0f} ", end="")
        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Vector Native Density Test")
    parser.add_argument("--results", type=str, help="Path to results JSON from Modal run")
    args = parser.parse_args()

    print("Vector Native Density Test")
    print("=" * 50)
    print(f"Test cases: {len(ALL_TESTS)}")
    print(f"Total variants: {sum(len(t.variants) for t in ALL_TESTS)}")
    print()

    if not args.results:
        print("No results file provided.")
        print()
        print("To run the experiment:")
        print("  modal run scripts/modal_vn_density.py")
        print()
        print("To analyze results:")
        print("  python experiments/vn_density_test.py --results vn_density_results.json")
        return

    # Load real results
    with open(args.results) as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print(f"No results found in {args.results}")
        return

    print(f"Loaded {len(results)} results from {args.results}")
    print()

    # Analyze
    analysis = analyze_results(results)
    print_results(analysis)


if __name__ == "__main__":
    main()
