import os
import random
import numpy as np
import torch
from torchsummary import summary

import neptune
import gin, gin.torch
from absl import flags, app

import data, models, trainers


@gin.configurable
class ExperimentRunner():

    def __init__(self, seed=0, no_cuda=False, num_workers=2, epochs=10, batch_size=64,
                 log_interval=100, outdir=None, prefix='', target_type='supervised',
                 datadir='~/datasets', dataset_name='cifar100', model_class=gin.REQUIRED):

        self.seed = seed
        self.no_cuda = no_cuda
        self.num_workers = num_workers
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.outdir = outdir
        self.prefix = prefix
        self.datadir = datadir
        self.target_type = target_type
        self.dataset_name = dataset_name
        self.model_class = model_class

        self.setup_environment()
        self.setup_torch()


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


    def setup_trainers(self):
        self.model = self.model_class(in_channels=self.data.input_shape[0],
                                      n_classes=self.data.num_classes)

        self.model.to(self.device)
        summary(self.model, self.data.input_shape)

        self.trainer = trainers.SupervisedTrainer(self.model,
                                                  self.device,
                                                  train_loader=self.train_loader)


    def setup_data_loaders(self):
        self.data = data.Data(self.datadir,
                              self.dataset_name,
                              self.batch_size,
                              self.target_type,
                              self.dataloader_kwargs)

        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader


    def train(self):
        self.setup_data_loaders()
        self.setup_trainers()

        self.global_iters = 0
        for self.current_epoch in range(1, self.epochs + 1):
            for batch_idx, (x, y) in enumerate(self.train_loader, start=0):
                self.global_iters += 1
                batch_results = self.trainer.train_on_batch(x, y)
                if self.global_iters % self.log_interval == 0:
                    template = "Train\tglobal iter: {}, epoch: {}, batch: {}/{}, metrics:  "
                    template += ": {:.3f}  ".join(list(batch_results.keys()) + [''])
                    print(template.format(self.global_iters,
                                          self.current_epoch,
                                          batch_idx + 1,
                                          len(self.train_loader),
                                          *[item.data for item in batch_results.values()]))
                    for metric, result in batch_results.items():
                        neptune.send_metric('batch_' + metric, x=self.global_iters, y=result)

                if self.global_iters % (10 * self.log_interval) == 0:
                    self.test()
        self.test()


    def test(self):
        test_results = None

        with torch.no_grad():
            for test_batch_idx, (x_test, y_test) in enumerate(self.test_loader, start=0):
                test_batch_results = self.trainer.test_on_batch(x_test, y_test)

                if test_results is None:
                    test_results = test_batch_results.copy()
                else:
                    for metric, result in test_batch_results.items():
                        test_results[metric] += result.data

        total = len(self.test_loader)
        mean_test_results = {key: value/total for key, value in test_results.items()}

        template = "Test\tEpoch: {} ({:.2f}%), Metrics:  "
        template += ": {:.3f}  ".join(list(mean_test_results.keys()) + [''])
        print(template.format(self.current_epoch,
                              float(self.current_epoch) / (self.epochs) * 100.,
                              *[item.data for item in mean_test_results.values()]))

        for metric, result in mean_test_results.items():
            neptune.send_metric('test_' + metric, x=self.global_iters, y=result)


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
    exp_runner.train()
    neptune.stop()
    print('fin')


if __name__ == '__main__':
    flags.DEFINE_multi_string('gin_file', None, 'List of paths to the config files.')
    flags.DEFINE_multi_string('gin_param', None, 'Newline separated list of Gin parameter bindings.')
    FLAGS = flags.FLAGS

    app.run(main)
