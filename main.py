import os
import random

from absl import app, flags
import gin
import gin.torch
import gin.torch.external_configurables
import neptune
import numpy as np
import torch

import data
import models
import trainers
from utils import gin_config_to_dict


@gin.configurable
class ExperimentManager():
    def __init__(self, seed=0, no_cuda=False, num_workers=2, outdir=None, prefix='',
                 datadir=os.path.expanduser('~/datasets')):
        self.seed = seed
        self.no_cuda = no_cuda
        self.num_workers = num_workers
        self.outdir = outdir
        self.prefix = prefix
        self.datadir = datadir

        self.setup_environment()
        self.setup_torch()
        self.setup_trainer()

    def setup_environment(self):
        os.makedirs(self.datadir, exist_ok=True)

        if self.outdir is not None:
            self.imagesdir = os.path.join(self.outdir, self.prefix, 'images')
            self.chkptdir = os.path.join(self.outdir, self.prefix, 'models')
            os.makedirs(self.imagesdir, exist_ok=True)
            os.makedirs(self.chkptdir, exist_ok=True)

    def setup_torch(self):
        use_cuda = not self.no_cuda and torch.cuda.is_available()

        torch.manual_seed(self.seed)
        if use_cuda:
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:
            self.dataloader_kwargs = {'num_workers': 3, 'pin_memory': True}
        else:
            self.dataloader_kwargs = {'num_workers': self.num_workers, 'pin_memory': False}
        print("Device: {}".format(self.device))

    def setup_trainer(self):
        self.data_factory = data.DataFactory(self.datadir, self.dataloader_kwargs)
        self.model_builder = models.ModelBuilder(self.device,
                                                 self.data_factory.input_shape,
                                                 self.data_factory.num_classes)

        self.trainer = trainers.SupervisedTrainer(device=self.device,
                                                  model=self.model_builder.model,
                                                  batch_size=self.data_factory.batch_size,
                                                  num_tasks=self.data_factory.num_tasks,
                                                  data_loaders=self.data_factory.loaders)

    def run_experiment(self):
        self.trainer.train()


def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)
    print("Gin parameter bindings:\n{}".format(gin.config_str()))

    use_neptune = "NEPTUNE_API_TOKEN" in os.environ
    exp_id = ''

    if use_neptune:
        neptune.init(project_qualified_name='bbeatrix/curl')
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
    flags.DEFINE_multi_string('gin_file', None, "List of paths to the config files.")
    flags.DEFINE_multi_string('gin_param', None, "Newline separated list of Gin param bindings.")
    FLAGS = flags.FLAGS

    app.run(main)
