#!/usr/bin/env python

import sys
from jinja2 import Template
from pathlib import Path
import inspect
from esmart.util.custom_metrics import RecallMultiClass, PrecisionMultiClass

def add_dev_code_parser(subparsers):
    parser = subparsers.add_parser('dev-code', help='Add development code')
    # parser.add_argument('file', help='File to add development code to')
    # parser.add_argument('code', help='Code to add')
    # parser.set_defaults(func=dev_code)

def dev_code(args):
    
    class Foo():
        def __init__(self):
            self.bar = 1
            self.baz = 2
        def generate_parser(self):
            def parser():
                return self.bar + self.baz
            return parser

    foo = Foo()

    cloud_model_path = 'moap-crossarmtype-cls/m20220907'
    premise_model_path = '/home/dhuynh/workspaces/joint-splice-detector/esmart/local/experiments/0.A.finished/20220824-223134-cross-arm-efficient-net-two-stages-padding'
    parameters = {
        'LOCAL_MODELPATH': f'{premise_model_path}/best_model.h5',
        'MODELPATH': f"gs://esmart-model-repo/{cloud_model_path}/best_model.h5",
        'RecallMultiClass': inspect.getsource(RecallMultiClass),
        'PrecisionMultiClass': inspect.getsource(PrecisionMultiClass),
        'validation_data': 'moap_crossarm_material_validation',
        'generate_parser': inspect.getsource(foo.generate_parser()),
    }

    parent_folder = Path(__file__).parent.parent
    with open(f"{parent_folder}/util/templates/modules.py.jinja") as file_: # open template
        template = Template(file_.read())
    msg = template.render(parameters)

    print(msg)

    print(foo.generate_parser()())


