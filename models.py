"""EDIT ME: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track. The two starting models are baselines:
deliberately simple, so you can measure how much your own ideas add.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import features


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if track == "list_aware":
        return ListAwareModel()
    return OffMarketModel()
    # Baselines to compare against: ListPriceBaseline(), NearestSalesBaseline(k=10),
    # NeighbourhoodMedianBaseline(), FeatureModel(features.off_market_features)


class ListPriceBaseline:
    """List-aware baseline: every home sells for exactly its list price."""

    def fit(self, train):
        return self   # nothing to learn

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float)


class NearestSalesBaseline:
    """Off-market baseline: the median sold price of the k nearest earlier sales of
    the same property type.

    "Nearest" means closest by latitude and longitude. "Earlier" means sold before the
    home was listed, so the model never looks into the future, even when the
    training table holds later sales. It makes no adjustment for size, condition or
    for prices moving since those sales: improving on that is your job.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = ["ml_num", "house_type_name", "date_sold", "latitude", "longitude", "sold_price"]
        # Newest sales first, so when several sales are equally near (a condo
        # building shares one location), the most recent ones are picked.
        self.sales = train[columns].sort_values(["date_sold", "ml_num"], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    def predict(self, homes):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        sales = self.sales_by_type.get(home["house_type_name"], self.sales)
        earlier = sales[sales["date_sold"] < home["date_listed"]]
        if len(earlier) == 0:   # no earlier sale of this type: use every type
            earlier = self.sales[self.sales["date_sold"] < home["date_listed"]]

        # Distance in degrees. In Toronto a degree of longitude is only 0.72 times as
        # long as a degree of latitude (0.72 = cosine of 43.7 degrees north).
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        nearest = np.argsort(distance, kind="stable")[: self.k]
        return float(np.median(earlier["sold_price"].to_numpy()[nearest]))


class FeatureModel:
    """A machine-learning model (gradient-boosted trees) on the features in features.py.

    It learns log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. Not used until you switch it on
    in make_model().
    """

    def __init__(self, make_features):
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(random_state=0)

    def fit(self, train):
        self.model.fit(self.make_features(train), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.make_features(homes)))


class NeighbourhoodMedianBaseline:
    """The handout's maximum-error baseline: the median sale of the same type in the
    same community over the previous 12 months. The bar we must beat."""

    def fit(self, train):
        last_sale = train["date_sold"].max()
        recent = train[train["date_sold"] >= last_sale - pd.DateOffset(months=12)]
        self.by_type_and_community = (recent.groupby(["house_type_name", "community"])["sold_price"]
                                      .median().rename("median_price"))
        self.by_type = recent.groupby("house_type_name")["sold_price"].median()
        return self

    def predict(self, homes):
        price = homes.join(self.by_type_and_community, on=["house_type_name", "community"])["median_price"]
        fallback = homes["house_type_name"].map(self.by_type)
        return price.fillna(fallback).to_numpy(dtype=float)


def boosted_trees():
    """The gradient-boosted trees both of our models use. Settings are sensible defaults,
    not tuned on the validation folds."""
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.08, max_leaf_nodes=63,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=0)


class PricingModel:
    """The parts both tracks share, all learned from the training sales.

    1. A market index. Prices moved a lot from 2021 to 2026 and rise every spring and fall
       every summer. Trees cannot follow a trend past their last training month, so instead
       the trees learn price relative to the index, and the index is forecast separately.
    2. Comparable sales: the index-adjusted price per square foot of the 10 nearest sales of
       the same type.
    3. A condition score from the listing remarks.
    """

    def fit_shared(self, train):
        self.categories = features.learn_categories(train)
        months = train["date_sold"].dt.to_period("M")
        log_ppsf = np.log(train["sold_price"] / features.square_feet(train["house_area"]))
        neighbourhood = [train["house_type_name"], train["community"]]

        # 1. Compare each sale with the average for its type and community, so a month that
        #    happens to have many expensive neighbourhoods does not look like a price rise.
        relative = log_ppsf - log_ppsf.groupby(neighbourhood).transform("mean")
        self.price_index = features.MonthlyIndex(seasonal_share=0.75).fit(relative, months)
        self.train_market = self.price_index.at(months)
        adjusted = log_ppsf - self.train_market

        # 2. and 3.
        self.comps = features.ComparableSales(k=10).fit(train, adjusted)
        text_target = (adjusted - adjusted.groupby(neighbourhood).transform("mean")).fillna(0)
        self.text = features.ConditionText(alpha=30.0).fit(train["remarks"], text_target)

    def shared_features(self, homes, training=False):
        X = features.home_features(homes, self.categories)
        X = X.join(self.comps.transform(homes, leave_self_out=training))
        X["text_score"] = self.text.train_scores if training else self.text.transform(homes["remarks"])
        return X

    def market_forecast(self, homes):
        months = features.expected_sale_month(homes, self.price_index.last_month)
        return self.price_index.forecast(months), months


class OffMarketModel(PricingModel):
    """Off-market: log(price) = market index + what the trees learn from the home's features."""

    def fit(self, train):
        self.fit_shared(train)
        X = self.shared_features(train, training=True)
        self.trees = boosted_trees().fit(X, np.log(train["sold_price"]) - self.train_market)
        return self

    def predict(self, homes):
        market, _ = self.market_forecast(homes)
        return np.exp(self.trees.predict(self.shared_features(homes)) + market)


class ListAwareModel(PricingModel):
    """List-aware: predict the sold-to-list ratio, not the price. The list price already
    carries most of what the market knows; the question is how far above or below it the
    home will close. The ratio has its own index (homes sold over asking in 2021-22 and
    about 5% under it in 2025-26), forecast the same way."""

    def fit(self, train):
        self.fit_shared(train)
        months = train["date_sold"].dt.to_period("M")
        log_ratio = np.log(train["sold_price"] / train["list_price"])
        self.ratio_index = features.MonthlyIndex(seasonal_share=0.5).fit(log_ratio, months)
        X = self.list_features(train, self.train_market, training=True)
        self.trees = boosted_trees().fit(X, log_ratio - self.ratio_index.at(months))
        return self

    def list_features(self, homes, market, training=False):
        X = self.shared_features(homes, training)
        X["log_list_price"] = np.log(homes["list_price"])
        # How the asking price compares with what nearby sales suggest. Below zero means the
        # home is listed low, which often sets off a bidding war.
        X["list_vs_comps"] = (np.log(homes["list_price"] / X["square_feet"]) - market - X["comp_value"])
        return X

    def predict(self, homes):
        market, months = self.market_forecast(homes)
        log_ratio = self.trees.predict(self.list_features(homes, market)) + self.ratio_index.forecast(months)
        return homes["list_price"].to_numpy(dtype=float) * np.exp(log_ratio)