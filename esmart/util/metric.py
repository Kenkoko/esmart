from typing import Iterable, List, Optional, Union

from esmart import Config
from esmart.job import Job
from tensorflow_addons.metrics.f_scores import FBetaScore
from tensorflow_addons.utils.types import AcceptableDTypes, FloatTensorLike
from typeguard import typechecked

import tensorflow as tf

class Metric:
    "Utility class for comparing metrics."

    def __init__(self, metric_max: Union[Job, Config, bool]):
        "Params: metric_max=True means higher is better."
        if isinstance(metric_max, Job):
            metric_max = metric_max.config
        if isinstance(metric_max, Config):
            metric_max = metric_max.get("valid.metric_max")
        self._metric_max = metric_max

    def better(self, metric1: float, metric2: float) -> bool:
        if self._metric_max:
            return metric1 > metric2
        else:
            return metric1 < metric2

    def best(self, metrics: Iterable[float]) -> float:
        if self._metric_max:
            return max(metrics)
        else:
            return min(metrics)

    def best_index(self, metrics: List[float]) -> int:
        return metrics.index(self.best(metrics))

    def worst(self) -> float:
        if self._metric_max:
            return float("-Inf")
        else:
            return float("Inf")


class RecallMultiClass(FBetaScore):
    r"""Computes multi-class recall score.

    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro` and
            `weighted`. Default value is None.
        beta: Determines the weight of recall. Default value is 1.
        threshold: Elements of `y_pred` greater than threshold are
            converted to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
        name: (Optional) String name of the metric instance.
        dtype: (Optional) Data type of the metric result.

    Returns:
        Multi-class Recall Score: float.

    Raises:
        ValueError: If the `average` has values other than
        `[None, 'micro', 'macro', 'weighted']`.

        ValueError: If the `beta` value is less than or equal
        to 0.

    `average` parameter behavior:

        None: Scores for each class are returned.

        micro: True positivies, false positives and
            false negatives are computed globally.

        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.

        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.

    """
    
    @typechecked
    def __init__(
        self,
        num_classes: FloatTensorLike,
        average: Optional[str] = None,
        beta: FloatTensorLike = 1.0,
        threshold: Optional[FloatTensorLike] = None,
        name: str = "multiclass_recall",
        dtype: AcceptableDTypes = None,
        **kwargs,
    ):
        super().__init__(num_classes, average, beta, threshold, name=name, dtype=dtype)

    def result(self):
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )

        if self.average == "weighted":
            weights = tf.math.divide_no_nan(
                self.weights_intermediate, tf.reduce_sum(self.weights_intermediate)
            )
            recall = tf.reduce_sum(recall * weights)

        elif self.average is not None:  # [micro, macro]
            recall = tf.reduce_mean(recall)

        return recall


class PrecisionMultiClass(FBetaScore):
    r"""Computes multi-class precision score.

    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro` and
            `weighted`. Default value is None.
        beta: Determines the weight of precision Default value is 1.
        threshold: Elements of `y_pred` greater than threshold are
            converted to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
        name: (Optional) String name of the metric instance.
        dtype: (Optional) Data type of the metric result.

    Returns:
        Multi-class precision Score: float.

    Raises:
        ValueError: If the `average` has values other than
        `[None, 'micro', 'macro', 'weighted']`.

        ValueError: If the `beta` value is less than or equal
        to 0.

    `average` parameter behavior:

        None: Scores for each class are returned.

        micro: True positivies, false positives and
            false negatives are computed globally.

        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.

        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.

    """
    
    @typechecked
    def __init__(
        self,
        num_classes: FloatTensorLike,
        average: Optional[str] = None,
        beta: FloatTensorLike = 1.0,
        threshold: Optional[FloatTensorLike] = None,
        name: str = "multiclass_precision",
        dtype: AcceptableDTypes = None,
        **kwargs,
    ):
        super().__init__(num_classes, average, beta, threshold, name=name, dtype=dtype)

    def result(self):
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives
        )

        if self.average == "weighted":
            weights = tf.math.divide_no_nan(
                self.weights_intermediate, tf.reduce_sum(self.weights_intermediate)
            )
            precision = tf.reduce_sum(precision * weights)

        elif self.average is not None:  # [micro, macro]
            precision = tf.reduce_mean(precision)

        return precision