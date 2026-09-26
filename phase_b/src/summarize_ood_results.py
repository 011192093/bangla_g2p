"""Summarize verified OOD results with per-domain and overall Wilson CIs.

Usage:
    python summarize_ood_results.py \
        --result news=../data/ood_news.tsv \
        --result literary=../data/ood_literary.tsv

Only rows marked ``correct`` or ``wrong`` in the final column are scored.
"""

import argparse
import math


def wilson_interval(correct, total, z=1.96):
    if not total:
        return None
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return centre - margin, centre + margin


def read_result(path):
    correct = wrong = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            verdict = line.rstrip("\n").split("\t")[-1].strip().lower()
            if verdict.startswith("correct"):
                correct += 1
            elif verdict.startswith("wrong"):
                wrong += 1
    return correct, wrong


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", action="append", required=True, metavar="DOMAIN=PATH",
        help="Verified result TSV and its domain label; repeat for each domain",
    )
    args = parser.parse_args()

    totals = []
    print("Domain                 Correct/Scored       Accuracy       95% CI")
    print("-" * 70)
    for specification in args.result:
        if "=" not in specification:
            parser.error("--result must be DOMAIN=PATH")
        domain, path = specification.split("=", 1)
        correct, wrong = read_result(path)
        total = correct + wrong
        interval = wilson_interval(correct, total)
        totals.append((correct, total))
        if interval is None:
            print(f"{domain:<22} {correct}/{total:<17} unverified       n/a")
        else:
            print(
                f"{domain:<22} {correct}/{total:<17} "
                f"{correct / total * 100:6.1f}%       "
                f"{interval[0] * 100:5.1f}-{interval[1] * 100:5.1f}%"
            )

    correct = sum(item[0] for item in totals)
    total = sum(item[1] for item in totals)
    interval = wilson_interval(correct, total)
    print("-" * 70)
    if interval is None:
        print("Overall: no verified rows")
    else:
        print(
            f"Overall                 {correct}/{total:<17} "
            f"{correct / total * 100:6.1f}%       "
            f"{interval[0] * 100:5.1f}-{interval[1] * 100:5.1f}%"
        )


if __name__ == "__main__":
    main()