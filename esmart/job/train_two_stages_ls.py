
from esmart.job.train_two_stages import TrainingTwoStages
from esmart.util.custom_loss import LossWrapper
from esmart.util.custom_metrics import MetricWrapper
import yaml
import tensorflow as tf

class TrainingTwoStagesLS(TrainingTwoStages):

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

