#!/usr/bin/env python
# esmart dev-code local/experiments/20221014-140027-toy-example-pull-request --cloud.model_path moap-crossarmtype-cls/m20220907 --cloud.model_id MOAP_CROSSARM_TYPE_CLASSIFIER --cloud.valid_dataset moap_crossarm_material_infer --cloud.train_dataset moap_test_request_train,moap_test_request_val
import copy
import os
import sys
from esmart.config import Config
from esmart.dataset import Dataset
from esmart.job.job import Job
from esmart.job.search import SearchJob
from esmart.job.train import TrainingJob
from jinja2 import Template, meta, FileSystemLoader, Environment
from pathlib import Path
import inspect
from esmart.util.custom_metrics import RecallMultiClass, PrecisionMultiClass
from esmart.misc import esmart_base_dir
import yaml

def add_dev_code_parser(subparsers):
    # subparsers.add_argument('--cloud.valid_dataset', nargs="*", type=str, help='valid dataset', default=[])
    # subparsers.add_argument('--cloud.train_dataset', nargs="*", type=str, help='train dataset', default=[])
    pass


def dev_code(config: Config, dataset: Dataset, job):
    
    
    if isinstance(job, SearchJob):
        raise Exception("Search job is not supported now")
    elif isinstance(job, TrainingJob):
        print("Training job")
        training_job = job

    dev_folder = f'{config.folder}/development_code'
    if not os.path.exists(dev_folder):
        os.makedirs(dev_folder)

    parent_folder = Path(__file__).parent
    template_folder = f'{parent_folder}/templates'


    cloud_model_path = config.get('cloud.model_path')
    save_config: dict = copy.deepcopy(config.options)
    
    for item in ['console', 'modules', 'import', 'search', 'ax_search']:
        save_config.pop(item, None)
    premise_model_path = os.path.join(esmart_base_dir(), config.folder)
    with open(os.path.join(dev_folder, "config_dev.yaml"), "w+") as file:
        file.write(yaml.dump(save_config))

    jinja_env = Environment(
        loader=FileSystemLoader(searchpath=template_folder),
    )

    parameters = {
        # general parameters
        'id_model': config.get('cloud.model_id'),
        'LOCAL_MODELPATH': f'{premise_model_path}/best_model.h5',
        'MODELPATH': f"gs://esmart-model-repo/{cloud_model_path}/model.h5",
        'valid_datasets': config.get('cloud.valid_dataset'),
        'train_datasets': config.get('cloud.train_dataset'),
        # general parameters
        'label_list': training_job.dataset.class_names,
        'train_image_size': training_job.processor.train_img_size,
        'valid_image_size': training_job.processor.valid_img_size,
        'channels': training_job.processor.img_channels,
        'train_moap_preprocessor': training_job.processor.get_moap_preprocessor()['train_preprocessor'],
        'valid_moap_preprocessor': training_job.processor.get_moap_preprocessor()['valid_preprocessor'],
        'batch_size': config.get('train.batch_size'),
        'shuffle_buffer_size_factor': config.get('train.shuffle_buffer_size_factor'),
        'moap_lib': ', '.join(list(set(training_job.processor.get_moap_preprocessor().values()))),
        'RecallMultiClass': inspect.getsource(RecallMultiClass),
        'PrecisionMultiClass': inspect.getsource(PrecisionMultiClass),
        'steps_per_epoch': config.get("train.steps_per_epoch"),
        'max_epochs': config.get("train.max_epochs"),
        'loss': config.get('train.loss'),
    }

    training_type = config.get('train.type')
    if training_type == 'normal_training':
        template_train = jinja_env.get_template('normal_training.j2')
        optimizer = config.get('normal_training.optimizer.name')
        learning_rate = config.get('normal_training.optimizer.lr')
        parameters.update({
            "optimizer": f"'{optimizer}'",
            "learning_rate": str(learning_rate), 
        })
    elif training_type == 'two_stages_training':
        template_train = jinja_env.get_template('two_stages_training.j2')
        parameters.update({
            "optimizer_1": config.get('two_stages_training.optimizer_1.name'),
            "learning_rate_1": config.get('two_stages_training.optimizer_1.lr'),
            "optimizer_2": config.get('two_stages_training.optimizer_2.name'),
            "learning_rate_2": config.get('two_stages_training.optimizer_2.lr'),
            "unfreeze": config.get('two_stages_training.unfreeze').lower(),
        })
    else:
        raise Exception("Training type is not supported")

    # convert to string
    for key, value in parameters.items():
        if type(value) != str:
            parameters[key] = f'{value}'

    

    template_module = jinja_env.get_template('modules.v2.j2')
    template_test = jinja_env.get_template('test.j2')
    template_import = jinja_env.get_template('config.j2')

    template_module.stream(**parameters).dump(f'{dev_folder}/modules.py')
    template_train.stream(**parameters).dump(f'{dev_folder}/train.py')
    template_test.stream(**parameters).dump(f'{dev_folder}/test.py')
    template_import.stream(**parameters).dump(f'{dev_folder}/config.py')




