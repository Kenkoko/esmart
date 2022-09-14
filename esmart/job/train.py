
import gc
from logging import StreamHandler
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from esmart.util.custom_metrics import RecallMultiClass, PrecisionMultiClass
from esmart import Config, Dataset
from esmart.builder import BaseBuilder
from esmart.job import Job, TrainingOrEvaluationJob
from esmart.job.trace import format_trace_entry
from esmart.misc import init_from
from tensorflow.keras.optimizers import Optimizer
from esmart.util.metric import Metric
from esmart.job.trace import Trace
import matplotlib.pyplot as plt
import torch
import yaml

def plot_hist(self, result):
    def merge_hist(dict_result):
        hist = {}
        
        for history in dict_result:
            for key, value in history.items():
                hist.setdefault(key, []).append(value)

        def flatten(l):
            return [item for sublist in l for item in sublist]

        for key, value in hist.items():
            hist[key] = flatten(hist[key])
        return hist
    hist = None
    if type(result) == tuple:
        hist = merge_hist(result)
    elif isinstance(result, tf.keras.callbacks.History):
        hist = result.history
    else:
        raise ValueError(f'unknown type - {type(result)} of result to plot the metrics')
    metric = 'loss'
    plt.plot(hist[metric])
    plt.plot(hist[f"val_{metric}"])
    plt.title(f"model {metric}")
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(f'{self.config.folder}/model_{metric}.png')

def trace_best_result(self, result):
    self.config.log("Best result in this training job:")
    best = None
    best_metric = None
    metric_name = self.config.get("valid.metric")
    for trace_entry in self.valid_trace:
        metric_value = Trace.get_metric(trace_entry, metric_name)
        if not best or Metric(self).better(metric_value, best_metric):
            trace_entry["metric_name"] = metric_name
            trace_entry["metric_value"] = metric_value
            best = trace_entry
            best_metric = metric_value
    for k in ["job", "job_id", "parent_job_id", "event"]:
        if k in best:
            del best[k]
    self.trace(
        event="train_completed",
        echo=True,
        echo_prefix="  ",
        log=True,
        **best,
    )




        

class TrainingJob(TrainingOrEvaluationJob):
    def __init__(
        self, 
        config: Config,
        dataset: Dataset, 
        parent_job: Job = None,
        builder=None,
        forward_only: bool =False
    ) -> None:
        super().__init__(config, dataset, parent_job)
        if builder is None:
            self.builder: BaseBuilder = BaseBuilder.create(config, dataset)
        else:
            self.builder: BaseBuilder = builder
        self.loss = config.get("train.loss")
        self.batch_size: int = config.get("train.batch_size")
        self.is_forward_only = forward_only
        self.shuffle_buffer_size = self.batch_size*config.get("train.shuffle_buffer_size_factor")
        self.max_epochs = config.get("train.max_epochs")
        self.train_split = config.get("train.split")

        self.metrics = None
        self.parse_func = {}

        if not self.is_forward_only:
            self.optimizer: Optimizer = None

            self.backup_path = ""
            self.bestmodel_path = ""
            self.trace_path = f'./checkpoints/test/trace'
            self.callbacks = self.get_callbacks(config)
            self.steps_per_epoch = config.get("train.steps_per_epoch")

            self.valid_trace: List[Dict[str, Any]] = []
            self.training_img_size = self.config.get('train.parsing_img.training.size')
            self.training_img_size = self.builder.image_size if self.training_img_size == -1 else self.training_img_size
            self.parse_func['training'] = self.create_parse_func('training')

        # Hooks run after validation. The corresponding valid trace entry can be found
        # in self.valid_trace[-1] Signature: job
        self.post_valid_hooks: List[Callable[[Job], Any]] = []

        self.inference_img_size = self.builder.image_size
        self.parse_func['inference'] = self.create_parse_func('inference')

        self.ds_train = None
        self.ds_val = None
        self.num_examples = None
        self.type_str: Optional[str] = None

        if self.__class__ == TrainingJob:
            # hooks = Job.job_created_hooks.copy()
            # hooks.append(_custom_handler)
            for f in Job.job_created_hooks:
                f(self)

    # TODO: Move to eval class
    def create_metrics(self, eval_type):
        if eval_type == 'classification':
            ## TODO: make this configurable
            metrics = [
                'accuracy',
                tfa.metrics.F1Score(
                    num_classes=self.dataset.get_option('data_arg.num_classes'), 
                    threshold=None, 
                    average='macro'),
                PrecisionMultiClass(
                    num_classes=self.dataset.get_option('data_arg.num_classes'), 
                    threshold=None, 
                    average='macro'),
                RecallMultiClass(
                    num_classes=self.dataset.get_option('data_arg.num_classes'), 
                    threshold=None, 
                    average='macro'
                ),
                # PrecisionMultiClass(
                #     num_classes=self.dataset.get_option('data_arg.num_classes')),
                # RecallMultiClass(
                #     num_classes=self.dataset.get_option('data_arg.num_classes')),

            ]
        elif eval_type == 'object_detection':
            raise NotImplemented
            return []
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

    def create_parse_func(self, type_func):
        r"""
        create parsing function based on configuaration
        """
        def _parse_func(file_data, label=None):
            
            # loading image
            try:
                image_decoded = tf.image.decode_jpeg(
                    tf.io.read_file(file_data), channels=self.builder.get_option('img_channels'))
            except BaseException as e:
                self.config.log(
                    f"Aborting loading due to failure of loading file {file_data}"
                )
                raise e

            # resizing image
            img_size = getattr(self, f'{type_func}_img_size')
            resize_method = self.config.get(f'train.parsing_img.{type_func}.method')
            self.config.log(f'resize images by {resize_method} to {img_size} x {img_size} for {type_func} dataset')

            ## get the resizing function
            resize_func = getattr(tf.image, resize_method)
            if resize_method == 'resize':
                image_decoded = resize_func(image_decoded, (img_size, img_size))
            elif resize_method == 'resize_with_pad':
                image_decoded = resize_func(image_decoded, img_size, img_size)
            else:
                raise ValueError(f'Unknown resize method {resize_method}')

            # encoding labels
            if label is not None:
                label = tf.one_hot(label, self.dataset.get_option('data_arg.num_classes'))
                return image_decoded, label
            else:
                return image_decoded

        # returing the parsing function
        return _parse_func

    def create_current_trace(self, epoch, logs):
        self.current_trace["epoch"].update(
            logs
        )

        # run hooks (may modify trace)
        for f in self.post_epoch_hooks:
            f(self)

        # output the trace, then clear it
        trace_entry = self.trace(**self.current_trace["epoch"], echo=False, log=True)
        self.config.log("\n")
        self.config.log(
            format_trace_entry("train_epoch", trace_entry, self.config), prefix="  "
        )
        self.current_trace["epoch"] = None
        # self.valid_job.epoch = epoch
        # trace_entry = self.valid_job.run()
        self.valid_trace.append(trace_entry)
        for f in self.post_valid_hooks:
            f(self)
        # self.model.meta["valid_trace_entry"] = trace_entry

    def _init_trace_entry(self, epoch):
        self.current_trace["epoch"] = dict(
            type=self.type_str,
            scope="epoch",
            epoch=epoch,
            split=self.train_split,
            batches=self.ds_train.cardinality().numpy(),
            size=self.num_examples,
        )
        if not self.is_forward_only:
            self.current_trace["epoch"].update(
                {"lr": float(self.optimizer.learning_rate.numpy())}
            )

        # run pre-epoch hooks (may modify trace)
        for f in self.pre_epoch_hooks:
            f(self)

    def _prepare(self):
        super()._prepare()
        self.config.log(f'Preparing for training job')
        if not self.is_forward_only:
            ## logging
            tracer = tf.keras.callbacks.LambdaCallback(
                on_epoch_begin=lambda epoch, logs: self._init_trace_entry(epoch),
                on_epoch_end=lambda epoch, logs: self.create_current_trace(epoch=epoch, logs=logs)
            )
            self.callbacks.append(tracer)

            ## backup and restore
            folder = self.config.log_folder if self.config.log_folder else self.config.folder
            self.backup_path = f"{folder}/backup"
            backup = tf.keras.callbacks.BackupAndRestore(backup_dir=self.backup_path)
            self.callbacks.append(backup)

            ## best model saving
            folder = self.config.log_folder if self.config.log_folder else self.config.folder
            self.bestmodel_path = f"{folder}"
            best_model = tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{self.bestmodel_path}' + '/best_model.h5',
                save_weights_only=False,
                save_best_only=True,
                monitor=f"{self.config.get('valid.metric')}",
                mode="max" if self.config.get('valid.metric_max') else 'auto',
                verbose=0)
            self.callbacks.append(best_model)

            ## early stopping
            es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
            self.callbacks.append(es_callback)

            ## add hook
            ## TODO: make this configuarable

            self.post_run_hooks.append(plot_hist)
            self.post_run_hooks.append(trace_best_result)


    def save(self, filename) -> None:
        """Save current state to specified file"""
        self.config.log("Saving checkpoint to {}...".format(filename))
        checkpoint = self.save_to({})
        torch.save(
            checkpoint,
            filename,
        )
        self.config.log("Saving completed")

    def save_to(self, checkpoint: Dict) -> Dict:
        """Adds trainjob specific information to the checkpoint"""
        train_checkpoint = {
            "type": "train",
            "valid_trace": self.valid_trace,
            # "lr_scheduler_state_dict": self.kge_lr_scheduler.state_dict(),
            "job_id": self.job_id,
        }
        train_checkpoint = self.config.save_to(train_checkpoint)
        checkpoint.update(train_checkpoint)
        return checkpoint

    def _load(self, checkpoint: Dict) -> str:
        if checkpoint["type"] != "train":
            raise ValueError("Training can only be continued on trained checkpoints")
        self.valid_trace = checkpoint["valid_trace"]
        self.resumed_from_job_id = checkpoint.get("job_id")
        self.trace(
            event="job_resumed",
            checkpoint_file=checkpoint["file"],
        )
        self.config.log(
            "Resuming training from {} of job {}".format(
                checkpoint["file"], self.resumed_from_job_id
            )
        )

    @staticmethod
    def create(
        config: Config,
        dataset: Dataset,
        parent_job: Job = None,
        builder=None,
        forward_only=False,
    ) -> "TrainingJob":
        """Factory method to create a training job."""
        train_type = config.get("train.type")
        class_name = config.get_default(f"{train_type}.class_name")
        return init_from(
            class_name,
            config.modules(),
            config,
            dataset,
            parent_job,
            builder=builder,
            forward_only=forward_only,
        )

    def get_callbacks(self, config: Config):
        callbacks = []
        callback_configs = config.get('train.callbacks')
        for callback_name in callback_configs.keys():
            callback = None
            if callback_name == 'lr_scheduler':
                callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                    patience=10, 
                                    min_lr=0.0001, 
                                    verbose=1,
                                    cooldown=2) ## https://stackoverflow.com/questions/51889378/how-to-use-keras-reducelronplateau
            if callback != None: 
                callbacks.append(callback)

        return callbacks

    def get_optimizer(self, name, lr):
        if name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif name == 'adagrad':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
        elif name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        else:
            raise ValueError(
                "invalid value train.loss={}".format(name)
            )
        return optimizer







