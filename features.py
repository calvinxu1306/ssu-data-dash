"""EDIT ME: turn a table of homes into numbers a model can learn from.

Each function gets a table of homes with the columns of its test file (so never
sold_price or date_sold) and returns a table of numbers, one row per home.
FeatureModel in models.py uses them; switch it on in make_model() to try your features.

Ideas to try:
- lot size: lot_front * lot_depth (empty for condos, which is fine for tree models)
- garage, house_style, municipality
- how pricey a community is: the median price per square foot of its past sales
  (work it out from the training sales only, never from the homes you predict)
- how prices have moved recently in the area or for that property type
- words in the remarks, such as "renovated" or "needs work"
- off-market and list-aware can use different features: list-aware has list_price
"""
import pandas as pd

PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]


def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def off_market_features(homes):
    """Features for homes with no list price."""
    X = pd.DataFrame(index=homes.index)
    X["square_feet"] = square_feet(homes["house_area"])
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking"] = homes["parking_total"]
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
    # Months since January 2021, so the model can follow prices over time.
    X["months"] = (homes["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44
    # One 0/1 column per property type.
    for kind in PROPERTY_TYPES:
        X["is_" + kind] = (homes["house_type_name"] == kind).astype(int)
    return X


def list_aware_features(homes):
    """Features for homes that have a list price: everything above, plus the list price."""
    X = off_market_features(homes)
    X["list_price"] = homes["list_price"]
    return X


# ---------------------------------------------------------------------------
# Our features. Everything below is fitted on past sales only: each class learns
# from the training table in fit() and is then applied to the homes we predict.
# ---------------------------------------------------------------------------
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KDTree

CATEGORIES = ["house_type_name", "house_style", "municipality", "community"]


def learn_categories(train):
    """The values each category column can take, fixed from the training sales so the
    model sees the same codes when it predicts. A value it never saw becomes missing."""
    return {column: pd.CategoricalDtype(sorted(train[column].dropna().unique())) for column in CATEGORIES}


def home_features(homes, categories):
    """What the home is and where it is: size, rooms, lot, coordinates and four categories.
    The gradient-boosted trees handle the categories and the missing lot sizes of condos."""
    X = pd.DataFrame(index=homes.index)
    X["square_feet"] = square_feet(homes["house_area"])
    X["area_is_range"] = homes["house_area"].astype(str).str.contains("-").astype(int)
    for column in ["bedrooms", "washrooms", "parking_total", "garage",
                   "lot_front", "lot_depth", "latitude", "longitude"]:
        X[column] = homes[column]
    X["lot_area"] = homes["lot_front"] * homes["lot_depth"]
    for column, dtype in categories.items():
        X[column] = pd.Categorical(homes[column], dtype=dtype)
    return X


def expected_sale_month(homes, last_month):
    """The month a home will probably sell in. Half of homes sell within 12 days of listing,
    so we take two weeks after the listing date. Every home we predict sold after our data
    ends (in the test and in validation alike), so the month is never earlier than that."""
    first_unseen_day = (last_month + 1).start_time
    return (homes["date_listed"] + pd.Timedelta(days=14)).clip(lower=first_unseen_day).dt.to_period("M")


class MonthlyIndex:
    """A market index: the median of some value per month, and a forecast for later months.

    The forecast starts from the last month we have and adds a share of the change seen
    between the same months in earlier years (prices rise every spring and fall every
    summer). Only a share, because a seasonal pattern measured from a few years is noisy:
    in a backtest on cutoffs from 2022 to early 2025, adding all of it did worse than
    adding none, and part of it did best.
    """

    def __init__(self, seasonal_share):
        self.seasonal_share = seasonal_share

    def fit(self, values, months):
        self.level = values.groupby(months).median()
        self.last_month = months.max()
        return self

    def at(self, months):
        """The index in months we have data for (used for the training sales)."""
        return self.level.reindex(months).to_numpy()

    def forecast(self, months):
        """The index in months after our data ends (used for the homes we predict)."""
        unique = pd.unique(months)
        forecasts = {}
        for month in unique:
            past_changes = [self.level[month - 12 * years] - self.level[self.last_month - 12 * years]
                            for years in range(1, 6)
                            if (month - 12 * years) in self.level.index
                            and (self.last_month - 12 * years) in self.level.index]
            seasonal = np.mean(past_changes) if past_changes else 0.0
            forecasts[month] = self.level[self.last_month] + self.seasonal_share * seasonal
        return np.array([forecasts[month] for month in months])


class ComparableSales:
    """The k nearest sales of the same property type, valued as log price per square foot
    after removing the market index, so a 2021 sale and a 2025 sale are comparable.
    Condo units in one building share coordinates, so their neighbours are the other units.

    For a training sale, the sale itself is left out: otherwise its own price would be
    one of its features and the model would learn to trust this feature too much.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, train, adjusted_log_ppsf):
        self.trees, self.values, self.positions = {}, {}, {}
        for kind, sales in train.groupby("house_type_name"):
            self.trees[kind] = KDTree(np.c_[sales["latitude"], sales["longitude"] * 0.72])
            self.values[kind] = adjusted_log_ppsf.loc[sales.index].to_numpy()
            self.positions[kind] = pd.Series(np.arange(len(sales)), index=sales.index)
        return self

    def transform(self, homes, leave_self_out=False):
        out = pd.DataFrame(np.nan, index=homes.index, columns=["comp_value", "comp_distance", "comp_same_spot"])
        for kind, group in homes.groupby("house_type_name"):
            if kind not in self.trees:
                continue
            k = min(self.k + 1, len(self.values[kind]))
            distance, index = self.trees[kind].query(np.c_[group["latitude"], group["longitude"] * 0.72], k=k)
            if leave_self_out:
                own = self.positions[kind].loc[group.index].to_numpy()
                distance = np.where(index == own[:, None], np.inf, distance)   # push the home itself last
            order = np.argsort(distance, axis=1, kind="stable")[:, : self.k]
            index, distance = np.take_along_axis(index, order, 1), np.take_along_axis(distance, order, 1)
            out.loc[group.index, "comp_value"] = np.median(self.values[kind][index], axis=1)
            out.loc[group.index, "comp_distance"] = distance.mean(axis=1)
            out.loc[group.index, "comp_same_spot"] = (distance < 1e-9).sum(axis=1)
        return out


class ConditionText:
    """A condition score read from the listing remarks ("fully renovated", "turnkey",
    "original but clean", "offered as-is"). Words and word pairs are weighted (TF-IDF) and a
    ridge regression learns how they relate to a home's price per square foot compared with
    its type and community.

    For the training sales the score is cross-fitted: each sale is scored by a model that
    did not see it, so the score is as noisy in training as it will be on new homes.
    """

    def __init__(self, alpha=30.0):
        self.alpha = alpha

    def fit(self, remarks, target):
        self.words = TfidfVectorizer(ngram_range=(1, 2), min_df=20, sublinear_tf=True)
        W = self.words.fit_transform(remarks)
        self.train_scores = np.zeros(len(remarks))
        for fit_rows, score_rows in KFold(5, shuffle=True, random_state=0).split(W):
            ridge = Ridge(alpha=self.alpha).fit(W[fit_rows], target.iloc[fit_rows])
            self.train_scores[score_rows] = ridge.predict(W[score_rows])
        self.ridge = Ridge(alpha=self.alpha).fit(W, target)
        return self

    def transform(self, remarks):
        return self.ridge.predict(self.words.transform(remarks))