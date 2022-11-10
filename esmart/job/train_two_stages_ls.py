
import time
from typing import Dict

import tensorflow as tf
import yaml
from tensorflow.keras import layers

from esmart.job import Job
from esmart.job.train import TrainingJob
from esmart.job.train_two_stages import TrainingTwoStages
from esmart.processor.tokenizer import Tokenizer
from esmart.util.custom_loss import LossWrapper
from esmart.util.custom_metrics import MetricWrapper
import os
import tempfile

class TrainingTwoStagesLS(TrainingJob):

    def __init__(
        self, config, dataset, parent_job=None, builder=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, builder=builder, forward_only=forward_only
        )
        config.log("Initializing two-stage training job...")
        self.type_str = "two-stage"
        self.current_stage = None
        self.unfreeze = self.config.get("two_stages_training.unfreeze").lower()
        if self.__class__ == TrainingTwoStages:
            for f in Job.job_created_hooks:
                f(self)
        
        self.optimizer_1 = self.get_optimizer(
            name=self.config.get('two_stages_training.optimizer_1.name'),
            lr=self.config.get('two_stages_training.optimizer_1.lr'))

        self.optimizer_2 = self.get_optimizer(
            name=self.config.get('two_stages_training.optimizer_2.name'),
            lr=self.config.get('two_stages_training.optimizer_2.lr'))

        self.history_first = None
        self.history_second = None
        self.model_weight = None


    def create_metrics(self, eval_type):
        if eval_type == 'classification':
            ## TODO: make this configurable
            metrics = [
                MetricWrapper(
                    num_classes=self.config.get('dataset.data_arg.num_classes'), metric_name='accuracy'),
                MetricWrapper(
                    num_classes=self.config.get('dataset.data_arg.num_classes'), metric_name='f1_score', 
                    config={'average': 'macro'})
            ]
        else:
            raise ValueError(f'Unknow eval type: {eval_type}')

        # logging metrics
        for metric in metrics:
            if type(metric) == str:
                self.config.log(metric)
                continue
            metric_config = metric.get_config().copy()
            self.config.log(metric_config['name'])
            metric_config.pop('name', None)
            self.config.log(yaml.dump(metric_config, default_flow_style=False), prefix="  ")
        return metrics

    def create_loss(self, loss_name):
        return LossWrapper(
            number_classes=2,
            loss_name=loss_name, 
            config={'reduction': tf.keras.losses.Reduction.SUM}
        )

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()
        self.config.log('Loading LS data...')
        self.config.log('Preparing the training dataset')
        self.ds_train = tf.data.Dataset.from_tensor_slices(self.dataset.split('train'))
        self.ds_train = self.ds_train.map(
            lambda imgpath, map_file: tf.py_function(self.parse_func['training'], [imgpath, map_file], [tf.float32, tf.float32]), 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        self.ds_train = self.ds_train.shuffle(buffer_size=self.shuffle_buffer_size)
        self.ds_train = self.ds_train.batch(self.batch_size, drop_remainder=True)
        self.ds_train = self.ds_train.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.ds_train = self.ds_train.cache(os.path.join('/mnt/disks/sdb/tmp_data', 'train.tmp'))
        self.ds_train = self.ds_train.repeat()

        self.config.log('Preparing the validation dataset')
        self.ds_val = tf.data.Dataset.from_tensor_slices(self.dataset.split('valid'))
        self.ds_val = self.ds_val.map(
            lambda imgpath, map_file: tf.py_function(self.parse_func['inference'], [imgpath, map_file], [tf.float32, tf.float32]), 
            num_parallel_calls=tf.data.AUTOTUNE)
        self.ds_val = self.ds_val.batch(self.batch_size, drop_remainder=True)
        self.ds_val = self.ds_val.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.ds_val = self.ds_val.cache(os.path.join('/mnt/disks/sdb/tmp_data', 'val.tmp'))

    
    def _init_trace_entry(self, epoch):
        super()._init_trace_entry(epoch)
        if not self.is_forward_only:
            self.current_trace["epoch"].update(
                {"current stage": self.current_stage}
            )

    def save_to(self, checkpoint: Dict) -> Dict:
        checkpoint = super().save_to(checkpoint)
        checkpoint.update({
            'current_stage': self.current_stage,
            'weight': self.model.get_weights(),
            'history_first':  self.history_first,
            'history_second': self.history_second,
        })
        return checkpoint

    def _load(self, checkpoint: Dict) -> str:
        super()._load(checkpoint)
        self.current_stage = checkpoint['current_stage']
        self.model_weight = checkpoint['weight']
        self.history_first = checkpoint['history_first']
        self.history_second = checkpoint['history_second']

    def unfreeze_layers(self, model):
        if self.unfreeze == 'all': 
            self.config.log('unfreezing all layers')
            for layer in model.layers:
                if not isinstance(layer, layers.BatchNormalization):
                    self.config.log(f'unfreezing layer - {layer.name}')
                    layer.trainable = True
        elif self.unfreeze == 'none':
            for layer in model.layers[-None:]:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True

    def _run(self):
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        self.metrics = self.create_metrics(self.config.get('eval.type'))
        self.model = self.builder.build_model(self.model_weight)
        self.save(self.config.checkpoint_file(0))
        if self.current_stage == None:
            self.optimizer = self.optimizer_1
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                metrics=self.metrics,
            )
            self.config.log("Running first stage")
            self.current_stage = 'first stage'
            result_first = self.model.fit(
                self.ds_train, 
                steps_per_epoch=self.steps_per_epoch, 
                epochs=self.max_epochs, 
                validation_data=self.ds_val,
                callbacks=self.callbacks, 
                verbose=1)
            self.config.log("Finished first stage")
            self.history_first = result_first.history
        ## TODO: IMPORTANT trace back, don't need to run if finished first stage
            self.save(self.config.checkpoint_file(0))

        if self.current_stage == 'first stage':
            self.unfreeze_layers(self.model)
            self.optimizer = self.optimizer_2
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                #TODO accuracy to self.metrics
                metrics=self.metrics,
            )

            self.config.log("Running second stage")
            self.current_stage = 'second stage'

            result_second = self.model.fit(
                self.ds_train, 
                steps_per_epoch=self.steps_per_epoch, 
                epochs=self.max_epochs, 
                validation_data=self.ds_val, 
                callbacks=self.callbacks, 
                verbose=1)

            self.trace(event="train_completed")
            self.history_second = result_second.history

            self.current_stage = 'Training completed'
            self.save(self.config.checkpoint_file(0))
        
        return (self.history_first, self.history_second)
