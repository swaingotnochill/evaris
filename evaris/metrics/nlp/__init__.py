"""Traditional NLP evaluation metrics.

This module provides classic NLP metrics:
- BLEUMetric: Bilingual Evaluation Understudy score
- ROUGEMetric: Recall-Oriented Understudy for Gisting Evaluation
- METEORMetric: Metric for Evaluation of Translation with Explicit ORdering
"""

from evaris.metrics.nlp.bleu import BLEUConfig, BLEUMetric
from evaris.metrics.nlp.meteor import METEORConfig, METEORMetric
from evaris.metrics.nlp.rouge import ROUGEConfig, ROUGEMetric

__all__ = [
    "BLEUMetric",
    "BLEUConfig",
    "ROUGEMetric",
    "ROUGEConfig",
    "METEORMetric",
    "METEORConfig",
]
