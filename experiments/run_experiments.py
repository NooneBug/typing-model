from typing_model.runner.experimenters import ExperimentRoutine, BertBaselineExperiment
from typing_model.data_models.base_dataclass import BertBaselineDataclass

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

config_file_path = 'experiments/exp_config.ini'

# exp_list is a list of experiment configurations: each element has to be a dict with:
# exp_name: a tag present in the config file at `config_file_path`
# Dataclass: a dataclass which can take in input the parameters in the above configfile at the `exp_name` tag
# ExperimentClass: a class which follows the typing_model.runner.experimenters.BaseExperimentClass interface
exp_list = [
            {
                'exp_name': 'BertBaseline',
                'Dataclass': BertBaselineDataclass,
                'ExperimentClass': BertBaselineExperiment
            }
        ]

exp_routine = ExperimentRoutine(exp_list = exp_list, config_file=config_file_path)
exp_routine.perform_experiments()