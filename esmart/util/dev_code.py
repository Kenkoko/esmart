#!/usr/bin/env python

import os
import sys
from esmart.config import Config
from esmart.dataset import Dataset
from esmart.job.job import Job
from esmart.job.search import SearchJob
from esmart.job.train import TrainingJob
from jinja2 import Template
from pathlib import Path
import inspect
from esmart.util.custom_metrics import RecallMultiClass, PrecisionMultiClass

def add_dev_code_parser(subparsers):
    parser = subparsers.add_parser('dev-code', help='Add development code')
    # parser.add_argument('file', help='File to add development code to')
    # parser.add_argument('code', help='Code to add')
    # parser.set_defaults(func=dev_code)

def dev_code(config: Config, dataset: Dataset, job: Job):
    
    
    if isinstance(job, SearchJob):
        raise Exception("Search jobs are not supported")
    elif isinstance(job, TrainingJob):
        print("Training job")
        training_job = job

    dev_folder = f'{config.folder}/development_code'
    if not os.path.exists(dev_folder):
        os.makedirs(dev_folder)


    cloud_model_path = 'moap-crossarmtype-cls/m20220907'
    premise_model_path = '/home/dhuynh/workspaces/joint-splice-detector/esmart/local/experiments/0.A.finished/20220824-223134-cross-arm-efficient-net-two-stages-padding'

    config.set("deployment.mode_path", f"gs://esmart-model-repo/{cloud_model_path}/best_model.h5", create=True)
    config.save(os.path.join(dev_folder, "config_dev.yaml"))

    parameters = {
        'ID_MODEL': 'crossarmtype-cls',
        'LOCAL_MODELPATH': f'{premise_model_path}/best_model.h5',
        'MODELPATH': f"gs://esmart-model-repo/{cloud_model_path}/best_model.h5",
        'RecallMultiClass': inspect.getsource(RecallMultiClass),
        'PrecisionMultiClass': inspect.getsource(PrecisionMultiClass),
        'validation_data': 'moap_crossarm_material_validation',
        'val_image_size': job.builder.image_size,
        'get_preprocessor': get_code(training_job.processor.get_processor, remove_space=True),
    }

    parent_folder = Path(__file__).parent.parent
    with open(f"{parent_folder}/util/templates/modules.py.jinja") as file_: # open template
        template = Template(file_.read())
    msg = template.render(parameters)

    with open(f"{dev_folder}/modules.py", "w") as file_:
        file_.write(msg)

def get_code(object, replace_fun_name=None, remove_space=False):
    import re

    ## object level
    regex = r"self.*\([\w, =\"]*\)|self[.\w]*"
    self_obj = object.__self__
    fun_name = object.__name__
    object_code = inspect.getsource(object)


    object_code = object_code.replace('(self, ', '(')
    object_code = object_code.replace('self.config.log', 'print')


    if replace_fun_name:
        object_code = object_code.replace(fun_name, replace_fun_name)
    matches = re.finditer(regex, object_code, re.MULTILINE)

    for matchNum, match in enumerate(matches, start=1):
        exec_code = match.group()
        print(exec_code)
        results = {}
        exec(f"result = {exec_code.replace('self', 'self_obj')}", {"self_obj":self_obj}, results)
        print(results)
        if type(results['result']) == str:
            object_code = object_code.replace(exec_code, f'"{results["result"]}"')
        else:
            object_code = object_code.replace(exec_code, f'{results["result"]}')
    
    if remove_space:
        matches = re.finditer(r"def ", object_code, re.MULTILINE)
        num_space = list(matches)[0].start()
        object_code = object_code.replace(f'\n{" " * num_space}', "\n")
        object_code = object_code.strip()

    # ## class level
    # class_name = self_obj.__class__.__name__
    # regex = fr"{class_name}[.\w]*"
    
    # matches = re.finditer(regex, object_code, re.MULTILINE)
    # all_matches = set([match.group() for matchNum, match in enumerate(matches, start=1)])
    # for match in all_matches:
    #     exec(f"result = inspect.isfunction({match})", {class_name:class_name}, results)

    return object_code

