"""
intent_classifier.py
---------------------
STEP 2 + STEP 3 of the assignment.

STEP 2 - Intent discovery:
    Embed customer messages, cluster them (KMeans), inspect top TF-IDF
    terms per cluster, and let a human-readable label be assigned per
    cluster (auto-suggested from keywords, confirmable/overridable).

STEP 3 - Intent classification, three tiers:
    1. Random baseline           -> establishes the floor.
    2. TF-IDF + Logistic Reg.    -> strong, cheap, interpretable baseline.
    3. Sentence-embeddings + LR  -> "final" model; swap LogisticRegression
                                     for LightGBM via `use_lightgbm=True`
                                     if the lightgbm package is available.

All three share the same train/test split and the same evaluation
function (see evaluation.py) so comparisons in STEP 9 are apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

from src.utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# STEP 2: Intent discovery via clustering
# --------------------------------------------------------------------------- #

# Curated keyword -> label lookup used only to turn a cluster's top TF-IDF
# terms into a human-readable name automatically. This keeps intent
# discovery "unsupervised first, labeled second" rather than hard-coding
# categories up front, matching the assignment's "discover intents from
# the data" requirement.
_KEYWORD_LABEL_MAP: dict[str, str] = {
    "refund": "Refund", "money": "Refund", "back": "Refund",
    "delay": "Order Delay", "late": "Order Delay", "shipping": "Order Delay",
    "locked": "Account Locked", "lockout": "Account Locked", "access": "Account Locked",
    "password": "Password Reset", "reset": "Password Reset", "login": "Password Reset",
    "delivery": "Delivery Issue", "damaged": "Delivery Issue", "wrong": "Delivery Issue",
    "charge": "Billing Problem", "charged": "Billing Problem", "billing": "Billing Problem",
    "crash": "Technical Issue", "bug": "Technical Issue", "update": "Technical Issue",
    "cancel": "Subscription Cancellation", "subscription": "Subscription Cancellation",
    "plan": "Subscription Cancellation",
}


def discover_intents(texts: pd.Series, embeddings: np.ndarray, n_clusters: int,
                      random_state: int = 42) -> tuple[np.ndarray, dict[int, str], pd.DataFrame]:
    """Cluster message embeddings and auto-name each cluster from its top
    TF-IDF terms. Returns (cluster_ids, {cluster_id: label}, top_terms_df).
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)

    vectorizer = TfidfVectorizer(max_features=2000, stop_words="english", min_df=1)
    tfidf = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    cluster_labels: dict[int, str] = {}
    top_terms_rows = []
    for cid in range(n_clusters):
        mask = cluster_ids == cid
        if mask.sum() == 0:
            cluster_labels[cid] = f"Cluster_{cid}"
            continue
        cluster_tfidf_mean = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        top_idx = cluster_tfidf_mean.argsort()[::-1][:8]
        top_terms = terms[top_idx].tolist()

        label = None
        for term in top_terms:
            if term in _KEYWORD_LABEL_MAP:
                label = _KEYWORD_LABEL_MAP[term]
                break
        label = label or f"Cluster_{cid}_({top_terms[0]})"
        cluster_labels[cid] = label
        top_terms_rows.append({"cluster_id": cid, "label": label,
                                "size": int(mask.sum()), "top_terms": ", ".join(top_terms)})

    top_terms_df = pd.DataFrame(top_terms_rows).sort_values("size", ascending=False)
    logger.info("Discovered %d intent clusters:\n%s", n_clusters, top_terms_df.to_string(index=False))
    return cluster_ids, cluster_labels, top_terms_df


# --------------------------------------------------------------------------- #
# STEP 3: Classifiers
# --------------------------------------------------------------------------- #

@dataclass
class TrainedModel:
    name: str
    model: Any
    vectorizer: Any = None  # only set for TF-IDF baseline
    label_classes: list[str] = None


def split_data(texts: pd.Series, labels: pd.Series, test_size: float, seed: int):
    return train_test_split(texts, labels, test_size=test_size, random_state=seed, stratify=labels)


def train_random_baseline(y_train: pd.Series, seed: int) -> TrainedModel:
    """Baseline 1: predicts the label distribution's prior, ignoring text
    entirely. This is the floor every other model must clear."""
    clf = DummyClassifier(strategy="stratified", random_state=seed)
    clf.fit(np.zeros((len(y_train), 1)), y_train)
    return TrainedModel(name="random_baseline", model=clf)


def train_tfidf_logreg(X_train_text: pd.Series, y_train: pd.Series,
                        max_features: int, seed: int) -> TrainedModel:
    """Baseline 2: TF-IDF bag-of-words + Logistic Regression. Strong,
    fast, and fully interpretable via feature weights -- a realistic
    "should we even need embeddings?" sanity check."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words="english")
    X_train = vectorizer.fit_transform(X_train_text)
    clf = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
    clf.fit(X_train, y_train)
    return TrainedModel(name="tfidf_logreg", model=clf, vectorizer=vectorizer)


def train_embedding_classifier(X_train_emb: np.ndarray, y_train: pd.Series, seed: int,
                                use_lightgbm: bool = False) -> TrainedModel:
    """Final model: sentence-embedding features + a classifier head.

    Defaults to Logistic Regression for robustness on small datasets
    (LightGBM needs more rows per class to avoid overfitting); pass
    use_lightgbm=True when enough data is available (see decision log).
    """
    if use_lightgbm:
        try:
            from lightgbm import LGBMClassifier
            clf = LGBMClassifier(random_state=seed, n_estimators=200, verbosity=-1)
        except ImportError:
            logger.warning("lightgbm not installed; falling back to LogisticRegression.")
            clf = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
    else:
        clf = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
    clf.fit(X_train_emb, y_train)
    return TrainedModel(name="embedding_classifier", model=clf)


def predict(trained: TrainedModel, X_text: pd.Series = None, X_emb: np.ndarray = None) -> np.ndarray:
    """Unified prediction interface across the three model types."""
    if trained.name == "random_baseline":
        return trained.model.predict(np.zeros((len(X_text), 1)))
    if trained.name == "tfidf_logreg":
        return trained.model.predict(trained.vectorizer.transform(X_text))
    if trained.name == "embedding_classifier":
        return trained.model.predict(X_emb)
    raise ValueError(f"Unknown model name: {trained.name}")
