import time

import tensorflow as tf

from esmart.job import Job
from esmart.job.train import TrainingJob


class TrainingJobNormal(TrainingJob):
    """Samples SPO pairs and queries sp_ and _po, treating all other entities as negative."""


    def __init__(
        self, config, dataset, parent_job=None, builder=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, builder=builder, forward_only=forward_only
        )
        config.log("Initializing normal training job...")
        self.type_str = "normal"

        self.optimizer = self.get_optimizer(
            name=self.config.get('normal_training.optimizer.name'),
            lr=self.config.get('normal_training.optimizer.lr'))

        if self.__class__ == TrainingJobNormal:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()
        filepaths, labels = self.dataset.split('train')
        self.ds_train = tf.data.Dataset.from_tensor_slices((filepaths, labels))
        self.ds_train = self.ds_train.map(self.parse_func['training'], num_parallel_calls=tf.data.AUTOTUNE)
        self.ds_train = self.ds_train.shuffle(buffer_size=self.shuffle_buffer_size)
        self.ds_train = self.ds_train.batch(self.batch_size, drop_remainder=True)
        self.ds_train = self.ds_train.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.ds_train = self.ds_train.repeat()

        filepaths, labels = self.dataset.split('valid')
        self.ds_val = tf.data.Dataset.from_tensor_slices((filepaths, labels))
        self.ds_val = self.ds_val.map(self.parse_func['inference'], num_parallel_calls=tf.data.AUTOTUNE)
        self.ds_val = self.ds_val.batch(self.batch_size, drop_remainder=True)
        self.ds_val = self.ds_val.prefetch(buffer_size=tf.data.AUTOTUNE)

    
    def _run(self):
        model = self.get_model()
        self.metrics = self.create_metrics(self.config.get('eval.type'))
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            #TODO accuracy to self.metric
            metrics=self.metrics,
        )
        self.config.log("Running")
        result = model.fit(
            self.ds_train, 
            steps_per_epoch=self.steps_per_epoch, 
            epochs=self.max_epochs, 
            validation_data=self.ds_val, 
            callbacks=self.callbacks, 
            verbose=1)

        self.trace(event="train_completed")
        return result