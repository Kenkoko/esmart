#!/usr/bin/env python
# esmart dev-code local/experiments/20221014-140027-toy-example-pull-request --cloud.model_path moap-crossarmtype-cls/m20220907 --cloud.model_id MOAP_CROSSARM_TYPE_CLASSIFIER --cloud.valid_dataset moap_crossarm_material_validation,cloud --cloud.train_dataset moap_crossarm_material_validation,cloud
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

    print(config.options)
    save_config: dict = copy.deepcopy(config.options)
    
    for item in ['console', 'modules', 'import', 'search', 'ax_search']:
        save_config.pop(item, None)
    premise_model_path = os.path.join(esmart_base_dir(), config.folder)
    with open(os.path.join(dev_folder, "config_dev.yaml"), "w+") as file:
        file.write(yaml.dump(save_config))


    parameters = {
        'id_model': config.get('cloud.model_id'),
        'LOCAL_MODELPATH': f'{premise_model_path}/best_model.h5',
        'MODELPATH': f"gs://esmart-model-repo/{cloud_model_path}/best_model.h5",
        'valid_datasets': config.get('cloud.valid_dataset'),
        'train_datasets': config.get('cloud.train_dataset'),
        'label_list': training_job.dataset.class_names,
        'train_image_size': training_job.processor.train_img_size,
        'valid_image_size': training_job.processor.valid_img_size,
        'train_moap_preprocessor': training_job.processor.get_moap_preprocessor()['train_preprocessor'],
        'valid_moap_preprocessor': training_job.processor.get_moap_preprocessor()['valid_preprocessor'],
        'batch_size': config.get('train.batch_size'),
        'shuffle_buffer_size_factor': config.get('train.shuffle_buffer_size_factor'),
        'moap_lib': ', '.join(list(set(training_job.processor.get_moap_preprocessor().values()))),
        'RecallMultiClass': inspect.getsource(RecallMultiClass),
        'PrecisionMultiClass': inspect.getsource(PrecisionMultiClass),

    }

    # convert to string
    for key, value in parameters.items():
        if type(value) != str:
            parameters[key] = f'{value}'

    
    jinja_env = Environment(
        loader=FileSystemLoader(searchpath=template_folder),
    )
    
    template = jinja_env.get_template('modules.py.j2')
    template.stream(**parameters).dump(f'{dev_folder}/modules.py')


