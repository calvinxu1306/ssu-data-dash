"""Prediction intervals: an honest range around each predicted price.

Method: split conformal prediction, by group.
1. Calibrate on the dev fold. For each home, measure the size of the miss in log terms,
   |log(sold / predicted)|, so a 10% miss counts the same on a condo as on a mansion.
2. Group the homes by property type, and by whether the prediction is in the cheaper or
   dearer half for that type (dearer homes are harder to predict). In each group, find the
   miss size that 80% of homes stay within (and 50%, for a narrower range).
3. A new home's 80% range runs from its prediction divided by exp(that size) to its
   prediction multiplied by exp(that size).

Only the dev fold is used to calibrate. Coverage (how often the range really contains the
sale price) is checked on the gap folds, which played no part in building the model.

Usage, from the repo folder:
    python intervals.py calibrate    # dev fold; writes intervals.json (about a minute)
    python intervals.py predict      # ranges for the test homes, after `python run.py predict`
    python intervals.py coverage     # the honest check on the gap folds: run once, at the freeze
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import data
import gap_check

ROOT = Path(__file__).parent
LEVELS = {"80": 0.80, "50": 0.50}
TRACKS = ["off_market", "list_aware"]


def group_of(house_type, predicted_price, split_points):
    """Group = property type + cheaper or dearer half of that type's predictions.
    Co-op apartments are rare (about 50 a fold), so they share the condo apartments' groups."""
    kind = house_type.replace({"Co-Op Apt": "Condo Apt"})
    split = kind.map(split_points)
    half = np.where(predicted_price > split, "dearer", "cheaper")
    return kind + " / " + pd.Series(half, index=kind.index)


def calibrate():
    results = gap_check.run(["ours"], ["dev"])
    table = {}
    for track in TRACKS:
        rows = results[results["track"] == track]
        kind = rows["house_type_name"].replace({"Co-Op Apt": "Condo Apt"})
        split_points = rows["predicted_price"].groupby(kind).median().to_dict()
        groups = group_of(rows["house_type_name"], rows["predicted_price"], split_points)
        miss = np.abs(np.log(rows["sold_price"] / rows["predicted_price"]))
        table[track] = {
            "split_points": split_points,
            "half_widths": {level: miss.groupby(groups).quantile(q).round(4).to_dict()
                            for level, q in LEVELS.items()},
            "homes_per_group": groups.value_counts().to_dict(),
        }
    (ROOT / "intervals.json").write_text(json.dumps(table, indent=2))
    print("Wrote intervals.json. 80% ranges, as +/- percent around the prediction:")
    for track in TRACKS:
        widths = table[track]["half_widths"]["80"]
        print(f"  {track}: " + ", ".join(f"{g} {100 * (np.exp(w) - 1):.0f}%" for g, w in sorted(widths.items())))


def add_ranges(homes, predicted_price, track):
    """Low and high ends of the 50% and 80% ranges for each home."""
    table = json.loads((ROOT / "intervals.json").read_text())[track]
    groups = group_of(homes["house_type_name"].reset_index(drop=True),
                      pd.Series(np.asarray(predicted_price)), table["split_points"])
    out = pd.DataFrame({"predicted_price": np.round(predicted_price)})
    for level in LEVELS:
        half_width = groups.map(table["half_widths"][level]).to_numpy()
        out[f"low_{level}"] = np.round(predicted_price / np.exp(half_width))
        out[f"high_{level}"] = np.round(predicted_price * np.exp(half_width))
    return out


def predict():
    for track in TRACKS:
        homes = data.load(track)
        submission = pd.read_csv(ROOT / "submissions" / f"submission_{track}.csv")
        homes = homes.set_index("ml_num").loc[submission["ml_num"]].reset_index()
        ranges = add_ranges(homes, submission["predicted_price"].to_numpy(dtype=float), track)
        ranges.insert(0, "ml_num", submission["ml_num"])
        path = ROOT / "outputs" / f"intervals_{track}.csv"
        path.parent.mkdir(exist_ok=True)
        ranges.to_csv(path, index=False)
        print(f"Wrote {path.relative_to(ROOT)} ({len(ranges):,} homes)")


def coverage(folds=("gapA", "gapB")):
    results = gap_check.run(["ours"], list(folds))
    print(gap_check.summary(results).to_string())
    for track in TRACKS:
        rows = results[results["track"] == track].reset_index(drop=True)
        ranges = add_ranges(rows, rows["predicted_price"].to_numpy(dtype=float), track)
        print(f"\n{track}: share of homes whose sale price fell inside the range")
        report = {}
        for level in LEVELS:
            inside = (rows["sold_price"] >= ranges[f"low_{level}"]) & (rows["sold_price"] <= ranges[f"high_{level}"])
            report[f"{level}% range"] = inside.groupby(rows["fold"]).mean().mul(100).round(1)
            report[f"{level}% width"] = ((ranges[f"high_{level}"] / ranges[f"low_{level}"] - 1) * 100
                                         ).groupby(rows["fold"]).median().round(1)
        print(pd.DataFrame(report).to_string())
    return results


if __name__ == "__main__":
    commands = {"calibrate": calibrate, "predict": predict, "coverage": coverage}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]](*sys.argv[2:3]) if sys.argv[1] != "coverage" or len(sys.argv) < 3 \
        else coverage(sys.argv[2].split(","))