from typing import Iterable, List, Optional, Union

from tensorflow_addons.metrics.f_scores import FBetaScore
from tensorflow_addons.utils.types import AcceptableDTypes, FloatTensorLike
from typeguard import typechecked

import tensorflow as tf


class LossWrapper(tf.keras.losses.Loss):
    def __init__(
        self, 
        number_classes:int, 
        name='wrapper',
        loss_name: str = 'kl_divergence',
        reduce: str = 'mean',
        config: dict = {}):
        super().__init__(name=f'{loss_name}_{name}')
        self.loss_name = loss_name
        if loss_name == 'kl_divergence':
            self.loss = tf.keras.losses.KLDivergence(**config)
        elif loss_name == 'categorical_crossentropy':
            self.loss = tf.keras.losses.CategoricalCrossentropy(**config)
        else:
            raise ValueError(f'{loss_name} not supported')
        self.number_classes = number_classes
        if reduce not in ['mean', 'sum']:
            raise ValueError(f'{reduce} not supported')
        self.reduce = reduce
    def call(self, y_true, y_pred):
        reshape_y_true = tf.reshape(y_true, (-1, self.number_classes))
        reshape_y_pred = tf.reshape(y_pred, (-1, self.number_classes))
        loss = self.loss(reshape_y_true, reshape_y_pred)
        if self.reduce == 'mean':
            batch_size = tf.shape(reshape_y_true)[0]
            tf.debugging.Assert(tf.greater(batch_size, 0), [batch_size, reshape_y_true, reshape_y_pred])
            loss = loss / tf.cast(batch_size, tf.float32)
        tf.debugging.assert_all_finite(loss, f'loss should be finite')
        return loss
    def get_config(self):
        config = {
            'number_classes': self.number_classes,
            'name': self.name,
            'loss_name': self.loss_name,
            'reduce': self.reduce,
            'config': self.loss.get_config()
        }
        return config
