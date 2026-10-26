"""Gap-matched validation: score models the way the hidden test works.

The real test trains on sales up to May 31, 2026 and predicts homes sold June to
September 2026: one to four months after the data ends. The starter's `run.py validate`
only ever predicts one month ahead, so it flatters every model. Here each fold trains
on sales before a cutoff and predicts the four months after it.

Usage, from the repo folder:
    python gap_check.py                          every model, every fold
    python gap_check.py max_error,starter_gbm    only these models (quicker)
    python gap_check.py starter_gbm dev          one model, one fold

Use the dev fold to choose features and settings. Run the gapA and gapB folds once,
at the end, for the report.
"""
import sys
import time

import pandas as pd

import data
import features
import models

# fold name: (cutoff, end). Train on sales before the cutoff; predict sales from the cutoff to the end.
FOLDS = {
    "dev":  ("2025-06-01", "2025-10-01"),   # Jun to Sep 2025: the same months of the year as the test
    "gapA": ("2025-12-01", "2026-04-01"),   # Dec 2025 to Mar 2026
    "gapB": ("2026-02-01", "2026-06-01"),   # Feb to May 2026
}

# model name: a function that returns a fresh, untrained model for a track
CANDIDATES = {
    "max_error": lambda track: models.NeighbourhoodMedianBaseline(),
    "starter": lambda track: (models.ListPriceBaseline() if track == "list_aware"
                              else models.NearestSalesBaseline(k=10)),
    "starter_gbm": lambda track: models.FeatureModel(
        features.list_aware_features if track == "list_aware" else features.off_market_features),
}
TRACKS = ["off_market", "list_aware"]


def run(model_names, fold_names):
    """Fit and predict every model on every fold and track. One row per home per run."""
    listings = data.load("listings")
    results = []
    for fold in fold_names:
        cutoff, end = map(pd.Timestamp, FOLDS[fold])
        train = listings[listings["date_sold"] < cutoff]
        homes = listings[(listings["date_sold"] >= cutoff) & (listings["date_sold"] < end)]
        # 1 = sold in the first month after the cutoff, ..., 4 = the fourth month
        months_ahead = (homes["date_sold"].dt.year - cutoff.year) * 12 + homes["date_sold"].dt.month - cutoff.month + 1
        for name in model_names:
            for track in TRACKS:
                start = time.time()
                model = CANDIDATES[name](track).fit(train)
                predicted = model.predict(data.hide_answers(homes, track))
                results.append(pd.DataFrame({
                    "fold": fold, "model": name, "track": track, "ml_num": homes["ml_num"],
                    "house_type_name": homes["house_type_name"], "community": homes["community"],
                    "months_ahead": months_ahead, "sold_price": homes["sold_price"],
                    "predicted_price": predicted}))
                print(f"  {fold:5s} {name:12s} {track:10s} {time.time() - start:5.1f}s", file=sys.stderr)
    return pd.concat(results, ignore_index=True)


def summary(results):
    """MdAPE, MAPE and median signed error per track, model and fold, plus MdAPE by months ahead."""
    signed = 100 * (results["predicted_price"] - results["sold_price"]) / results["sold_price"]
    results = results.assign(abs_error=signed.abs(), signed_error=signed)
    groups = results.groupby(["track", "model", "fold"], sort=False)
    table = pd.DataFrame({
        "homes": groups.size(),
        "MdAPE": groups["abs_error"].median().round(2),
        "MAPE": groups["abs_error"].mean().round(2),
        "median_signed": groups["signed_error"].median().round(2),
    })
    by_month = results.pivot_table(index=["track", "model", "fold"], columns="months_ahead",
                                   values="abs_error", aggfunc="median", sort=False).round(1)
    by_month.columns = [f"MdAPE_{m}mo" for m in by_month.columns]
    return table.join(by_month)


if __name__ == "__main__":
    model_names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(CANDIDATES)
    fold_names = sys.argv[2].split(",") if len(sys.argv) > 2 else list(FOLDS)
    results = run(model_names, fold_names)
    pd.set_option("display.width", 200)
    print(summary(results).to_string())
    results.to_csv("outputs/gap_predictions.csv", index=False)
    print("Wrote outputs/gap_predictions.csv")