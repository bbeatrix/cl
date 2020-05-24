import neptune
import gin, gin.torch
import gin.torch.external_configurables
from absl import flags, app

import os
import random
import numpy as np
import torch
from torchsummary import summary

import data, models, trainers


@gin.configurable
class ExperimentRunner():

    def __init__(self, seed=0, no_cuda=False, num_workers=2, outdir=None, prefix='',
                 datadir='~/datasets', model_class=models.resnet50):

        self.seed = seed
        self.no_cuda = no_cuda
        self.num_workers = num_workers
        self.outdir = outdir
        self.prefix = prefix
        self.datadir = datadir
        self.model_class = model_class

        self.setup_environment()
        self.setup_torch()
        self.setup_data_loaders()
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
        print('Device: ', self.device)


    def setup_data_loaders(self):
        self.data = data.Data(self.datadir,
                              self.dataloader_kwargs)

        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader


    def setup_trainer(self):
        self.model = self.model_class(in_channels=self.data.input_shape[0],
                                      n_classes=self.data.num_classes)

        self.model.to(self.device)
        summary(self.model, self.data.input_shape)

        self.trainer = trainers.SupervisedTrainer(model=self.model,
                                                  device=self.device,
                                                  batch_size=self.data.batch_size,
                                                  train_loader=self.train_loader,
                                                  test_loader=self.test_loader)


    def run_exp(self):
        self.trainer.train()


def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)

    use_neptune = "NEPTUNE_API_TOKEN" in os.environ
    exp_id = ''

    if use_neptune:
        neptune.init(project_qualified_name="bbeatrix/curl")
        exp = neptune.create_experiment(params={}, name="exp")
        exp_id = exp.id
    else:
        neptune.init('shared/onboarding', 'ANONYMOUS', backend=neptune.OfflineBackend())

    neptune.log_text('gin_config', gin.config_str())
    neptune.log_artifact(*FLAGS.gin_file, 'gin_config_{}.gin'.format(exp_id))

    exp_runner = ExperimentRunner(prefix=exp_id)

    exp_runner.run_exp()
    neptune.stop()
    print('fin')


if __name__ == '__main__':
    flags.DEFINE_multi_string('gin_file', None, 'List of paths to the config files.')
    flags.DEFINE_multi_string('gin_param', None, 'Newline separated list of Gin parameter bindings.')
    FLAGS = flags.FLAGS

    app.run(main)
