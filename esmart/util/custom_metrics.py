from typing import Iterable, List, Optional, Union

from tensorflow_addons.metrics.f_scores import FBetaScore
from tensorflow_addons.utils.types import AcceptableDTypes, FloatTensorLike
from typeguard import typechecked

import tensorflow as tf

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
        if average:
            name = f'{average}_{name}'
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
        if average:
            name = f'{average}_{name}'
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


class MetricWrapper(tf.keras.metrics.Metric):
    def __init__(self, num_classes: int, config: dict = {}, name='wrapper', metric_name: str = 'accuracy', **kwargs):
        super().__init__(name=f'{metric_name}_{name}', **kwargs)
        self.metric_name = metric_name
        self.num_classes = num_classes
        if metric_name == 'accuracy':
            self.metric = tf.keras.metrics.Accuracy(**config)
        elif metric_name == 'f1_score':
            import tensorflow_addons as tfa
            self.metric = tfa.metrics.F1Score(num_classes=self.num_classes, **config)
        else:
            raise ValueError(f'{metric_name} not supported')
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, (-1, self.num_classes))
        y_pred = tf.reshape(y_pred, (-1, self.num_classes))
        if self.metric_name in ['accuracy',]:
            y_true = tf.argmax(y_true, axis=-1)
            y_pred = tf.argmax(y_pred, axis=-1)
        self.metric.update_state(y_true, y_pred, sample_weight)
    def result(self):
        return self.metric.result()
    def reset_state(self):
        self.metric.reset_state()
    def reset_states(self):
        self.metric.reset_states()
    def get_config(self):
        return {
            'metric_name': self.metric_name,
            'num_classes': self.num_classes,
            'config': self.metric.get_config()
        }

