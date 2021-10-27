"""
How to use this benchmark:
--------------------------

We recommend using the containerized version of this benchmark.
If you want to use this benchmark locally (without running it via the corresponding container),
you need to perform the following steps.

Prerequisites:
==============
Conda environment in which the HPOBench is installed (pip install .). Activate your environment.
```
conda activate <Name_of_Conda_HPOBench_environment>
```

1. Download data:
=================
The data will be downloaded automatically.

If you want to download the data on your own, you can download the data with the following command and then link the
hpobench-config's data-path to it.
You can download the requried data [here](https://syncandshare.lrz.de/getlink/fiCMkzqj1bv1LfCUyvZKmLvd/).

```python
from yahpo_gym import local_config
local_config.init_config()
local_config.set_data_path("path-to-data")
```

The data consist of surrogates for different data sets. Each surrogate is a compressed ONNX neural network.


1. Clone from github:
=====================
```
git clone HPOBench
```

2. Clone and install
====================
```
cd /path/to/HPOBench
pip install .[yahpo_gym]

```

Changelog:
==========
0.0.1:
* First implementation
"""

import warnings
import logging
from typing import Union, Dict

import ConfigSpace as CS
import numpy as np

from yahpo_gym.configuration import cfg, Configuration
import yahpo_gym.benchmarks

from hpobench.abstract_benchmark import AbstractBenchmark
__version__ = '0.0.1'

logger = logging.getLogger('YAHPOGym')


class YAHPOGymBenchmark(AbstractBenchmark):

    def __init__(self, scenario: str, instance: str,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        Parameters
        ----------
        dataset : str
            Name for the surrogate data. Must be one of ["lcbench", "fcnet", "nb301", "rbv2_svm",
            "rbv2_ranger", "rbv2_rpart", "rbv2_glmnet", "rbv2_aknn", "rbv2_xgboost", "rbv2_super"]
        rng : np.random.RandomState, int, None
        """
        self.scenario = scenario
        self.instance = instance
        self.benchset = BenchmarkSet(scenario, active_session = True, download = False)
        self.benchset.set_instance(instance)
        self.instance.config.download_files()
        logger.info(f'Start Benchmark for scenario {scenario} and instance {instance}')


    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        self.benchset.get_configuration_space(drop_fidelity_params = True)

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        self.benchset.get_fidelity_space()

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        if isinstance(configuration, CS.Configuration):
            configuration = configuration.get_dictionary()
        if isinstance(fidelity, CS.Configuration):
            fidelity = fidelity.get_dictionary()

        out = self.benchset.objective_function({**configuration, **fidelity})
        cost = 0
        return {'function_value': obj_value,
                "cost": cost,
                'info': {'fidelity': fidelity}}

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        return self.objective_function(configuration, fidelity=fidelity, rng=rng)


    @staticmethod
    def get_meta_information():
        """ Returns the meta information for the benchmark """
        return {'name': 'YAHPO Gym',
                'references': ['@misc{pfisterer2021yahpo,',
                               'title={YAHPO Gym -- Design Criteria and a new Multifidelity Benchmark for Hyperparameter Optimization},',
                               'author    = {Florian Pfisterer and Lennart Schneider and Julia Moosbauer and Martin Binder and Bernd Bischl},',
                               'eprint={2109.03670},',
                               'archivePrefix={arXiv},',
                               'year      = {2021}}'],
                'code': 'https://github.com/pfistfl/yahpo_gym/yahpo_gym'
                }
