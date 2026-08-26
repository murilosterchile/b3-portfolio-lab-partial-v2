import numpy as np

from portfolio_core.ml.evaluation import top_k_metrics


def test_perfect_topk_ranking_has_perfect_precision_recall_and_ndcg() -> None:
    truth = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    prediction = truth.copy()
    metrics = top_k_metrics(truth, prediction, k=2)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.ndcg == 1.0
    assert metrics.top_minus_bottom > 0
