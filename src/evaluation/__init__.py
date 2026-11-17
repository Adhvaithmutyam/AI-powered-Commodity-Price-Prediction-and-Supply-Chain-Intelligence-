"""
NLP Evaluation Framework and Data-Quality Validation.

Provides data-quality auditing, bias-free human annotation workflows,
and classification performance evaluation for commodity news intelligence.
"""

from src.evaluation.config import EvaluationConfig
from src.evaluation.data_quality import DataQualityValidator, validate_article_features
from src.evaluation.annotation import (
    AnnotationSampler,
    generate_annotation_sample,
    validate_annotation_dataset,
    ANNOTATION_GUIDELINES,
)
from src.evaluation.metrics import (
    compute_confusion_matrix,
    compute_classification_metrics,
)
from src.evaluation.evaluator import (
    RelevanceEvaluator,
    EventExtractionEvaluator,
    SentimentEvaluator,
    NLPEvaluationPipeline,
    evaluate_nlp_pipeline,
)

__all__ = [
    "EvaluationConfig",
    "DataQualityValidator",
    "validate_article_features",
    "AnnotationSampler",
    "generate_annotation_sample",
    "validate_annotation_dataset",
    "ANNOTATION_GUIDELINES",
    "compute_confusion_matrix",
    "compute_classification_metrics",
    "RelevanceEvaluator",
    "EventExtractionEvaluator",
    "SentimentEvaluator",
    "NLPEvaluationPipeline",
    "evaluate_nlp_pipeline",
]
