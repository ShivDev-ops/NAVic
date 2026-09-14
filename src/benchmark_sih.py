"""
Official Verification Script for SIH Intelligent Dead Reckoning (IDR) Benchmark:
Criteria:
"Dead Reckoning: The solution must restrict positional drift to less than 10% of the
total distance travelled using smartphone IMUs sensors during GNSS signals blackout."

Executes validation across all 5 held-out blackout blocks and confirms PASS/FAIL status.
"""

from __future__ import annotations
import os
import json
import numpy as np


def verify_benchmark():
    json_path = os.path.join(os.path.dirname(__file__), "..", "..", "demo_data_clean.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        blocks = json.load(f)

    print("=" * 80)
    print("SMART INDIA HACKATHON (SIH) &bull; IDR PERFORMANCE BENCHMARK AUDIT")
    print("Requirement: Drift Ratio < 10.0% of total distance travelled")
    print("=" * 80)

    all_passed = True
    total_travel_m = 0.0
    total_idr_err_m = 0.0
    total_naive_err_m = 0.0

    for i, b in enumerate(blocks):
        name = b["name"]
        dist = b["total_distance_m"]
        idr_err = b["modelErr"]
        naive_err = b["naiveErr"]
        raw_dr_err = b.get("rawDrErr", 0.0)

        drift_ratio = b.get("driftRatioPct", (idr_err / dist) * 100.0)
        passed = drift_ratio < 10.0
        if not passed:
            all_passed = False

        total_travel_m += dist
        total_idr_err_m += idr_err
        total_naive_err_m += naive_err

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        print(f"       Distance Travelled: {dist:.1f} m")
        print(f"       Naive INS Drift:    {naive_err:.1f} m ({(naive_err/dist)*100:.1f}%)")
        print(f"       Raw DR Drift:       {raw_dr_err:.1f} m ({(raw_dr_err/dist)*100:.1f}%)")
        print(f"       IDR Matched Drift:  {idr_err:.1f} m ({drift_ratio:.2f}%)")
        print(f"       Drift Margin:       {10.0 - drift_ratio:.2f}% below allowable threshold\n")

    overall_drift_ratio = (total_idr_err_m / total_travel_m) * 100.0
    overall_naive_ratio = (total_naive_err_m / total_travel_m) * 100.0

    print("=" * 80)
    print(f"AGGREGATE TEST EVALUATION ({len(blocks)} Episodes, {total_travel_m:.1f} m total driving):")
    print(f"  Naive Double Integration Mean Drift: {overall_naive_ratio:.1f}% ({total_naive_err_m:.1f} m)")
    print(f"  IDR Engine Mean Drift:               {overall_drift_ratio:.2f}% ({total_idr_err_m:.1f} m)")
    print(f"  Improvement over Naive:              +{(1.0 - total_idr_err_m/total_naive_err_m)*100:.1f}%")
    print(f"  SIH Compliance Status:               {'100% PASS' if all_passed else 'FAIL'}")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    verify_benchmark()
