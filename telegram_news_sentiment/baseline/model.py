from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline(cfg):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=cfg.model.max_features,
                    ngram_range=(1, cfg.model.ngrams),
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=cfg.model.C,
                    max_iter=cfg.model.max_iter,
                    class_weight="balanced",
                ),
            ),
        ]
    )
