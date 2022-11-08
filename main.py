import os
import random

#from absl import app, flags

#import gin
#import gin.torch
#import gin.torch.external_configurables
import neptune
import numpy as np
import sys
import torch
import yaml
from quinine import Quinfig

import data
import models
import trainers, contrastive_trainers
from utils import gin_config_to_dict
from hydra import compose, initialize, initialize_config_module


class ExperimentManager():
    def __init__(self, config, prefix):
        self.config = config
        self.exp_config = config["EXPERIMENT"]
        self.prefix = prefix

        self.setup_environment()
        self.setup_torch()
        self.setup_trainer()

    def setup_environment(self):
        os.makedirs(self.exp_config.DATADIR, exist_ok=True)

        if self.exp_config.LOGDIR is not None:
            self.logdir = os.path.join(self.exp_config.LOGDIR, self.prefix)
            os.makedirs(self.exp_config.LOGDIR, exist_ok=True)

    def setup_torch(self):
        use_cuda = not self.exp_config.USE_CUDA and torch.cuda.is_available()

        torch.manual_seed(self.exp_config.seed)
        if use_cuda:
            torch.cuda.manual_seed(self.exp_config.seed)
            torch.cuda.manual_seed_all(self.exp_config.seed)
        random.seed(self.exp_config.seed)
        np.random.seed(self.exp_config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:
            self.dataloader_kwargs = {'num_workers': 3, 'pin_memory': True}
        else:
            self.dataloader_kwargs = {'num_workers': self.exp_config.NUM_WORKERS, 'pin_memory': False}
        print("Device: {}".format(self.device))

    def setup_trainer(self):
        self.data = data.Data(self.exp_config.DATADIR,
                              self.dataloader_kwargs)

        self.model = models.Model(self.device,
                                  self.data.input_shape,
                                  self.data.num_classes)

        self.trainer = trainers.trainer_maker(self.data.target_type,
                                              self.device,
                                              self.model.build(),
                                              self.data,
                                              self.exp_config.LOGDIR)

    def run_experiment(self):
        self.trainer.train()


def main(argv):
    #with open('my_config.yaml') as file:
    #    config = yaml.safe_load(file)
    #config = Quinfig(config_path=argv[0].split('=')[-1])   

    # Compose the overrides with "vissl/config/defaults.yaml"
    #with initialize_config_module(config_module="configs"):
    with initialize(config_path="configs"):
        config = compose("defaults", overrides=argv)
    
    print(config)

    #gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)
    #print("Gin parameter bindings:\n{}".format(gin.config_str()))

    use_neptune = "NEPTUNE_API_TOKEN" in os.environ
    exp_id = ''

    if use_neptune:
        neptune.init(project_qualified_name='bbeatrix/scl')
        exp = neptune.create_experiment(params=gin_config_to_dict(gin.config_str()),
                                        name=FLAGS.gin_file[0].split('/')[-1][:-4],
                                        upload_source_files=['./*.py'])
        exp_id = exp.id
    else:
        neptune.init('shared/onboarding', 'ANONYMOUS', backend=neptune.OfflineBackend())

    neptune.log_text('gin_config', gin.config_str())
    neptune.log_artifact(*FLAGS.gin_file, 'gin_config_{}.gin'.format(exp_id))

    exp_manager = ExperimentManager(prefix=exp_id)
    exp_manager.run_experiment()

    neptune.stop()
    print("Fin")


if __name__ == '__main__':
    #flags.DEFINE_multi_string('gin_file', None, "List of paths to the config files.")
    #flags.DEFINE_multi_string('gin_param', None, "Newline separated list of Gin param bindings.")
    #FLAGS = flags.FLAGS
    
    #app.run(main)

    main(sys.argv[1:])
