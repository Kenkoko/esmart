import time
from esmart.config import Configurable

import tensorflow as tf

from esmart.job import Job
from esmart.job.train import TrainingJob
from tensorflow.keras import layers
# import tensorflow_addons as tfa

class TrainingTwoStages(TrainingJob):

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

    def _init_trace_entry(self, epoch):
        super()._init_trace_entry(epoch)
        if not self.is_forward_only:
            self.current_trace["epoch"].update(
                {"current stage": self.current_stage}
            )


    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()
        
        filepaths, labels = self.dataset.split('train')
        self.config.log('Preparing the training dataset')
        self.ds_train = tf.data.Dataset.from_tensor_slices((filepaths, labels))
        self.ds_train = self.ds_train.map(self.parse_func['training'], num_parallel_calls=tf.data.AUTOTUNE)
        self.ds_train = self.ds_train.shuffle(buffer_size=self.shuffle_buffer_size)
        self.ds_train = self.ds_train.batch(self.batch_size, drop_remainder=True)
        self.ds_train = self.ds_train.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.ds_train = self.ds_train.repeat()

        filepaths, labels = self.dataset.split('valid')
        self.config.log('Preparing the validation dataset')
        self.ds_val = tf.data.Dataset.from_tensor_slices((filepaths, labels))
        self.ds_val = self.ds_val.map(self.parse_func['inference'], num_parallel_calls=tf.data.AUTOTUNE)
        self.ds_val = self.ds_val.batch(self.batch_size, drop_remainder=True)
        self.ds_val = self.ds_val.prefetch(buffer_size=tf.data.AUTOTUNE)


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
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.metrics = self.create_metrics(self.config.get('eval.type'))
            model = self.builder.build_model()
            
            self.config.log("Running first stage")
            self.current_stage = 'first stage'
            self.optimizer = self.optimizer_1
            model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                #TODO accuracy to self.metrics
                metrics=self.metrics,
            )
            
            result_first = model.fit(
                self.ds_train, 
                steps_per_epoch=self.steps_per_epoch, 
                epochs=self.max_epochs, 
                validation_data=self.ds_val, 
                callbacks=self.callbacks, 
                verbose=1)
            self.config.log("Finished first stage")
            ## TODO: IMPORTANT trace back, don't need to run if finished first stage
            self.config.log("Running second stage")
            self.current_stage = 'second stage'
            self.unfreeze_layers(model)
            self.optimizer = self.optimizer_2
            model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                #TODO accuracy to self.metrics
                metrics=self.metrics,
            )
            
            result_second = model.fit(
                self.ds_train, 
                steps_per_epoch=self.steps_per_epoch, 
                epochs=self.max_epochs, 
                validation_data=self.ds_val, 
                callbacks=self.callbacks, 
                verbose=1)

            self.trace(event="train_completed")
        return (result_first, result_second)


