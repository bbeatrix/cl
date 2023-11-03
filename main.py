import logging
import os
import random
import shutil

import contrastive_trainers
import data
import gin
import gin.torch
import gin.torch.external_configurables
import models
import numpy as np
import torch
import trainers
import wandb
from absl import app, flags
from utils import gin_config_to_dict, set_memory_limit


@gin.configurable
class ExperimentManager:
    def __init__(
        self, seed=0, no_cuda=False, num_workers=2, logdir=None, prefix="", datadir=os.path.expanduser("~/datasets")
    ):
        self.seed = seed
        self.no_cuda = no_cuda
        self.num_workers = num_workers
        self.logdir = logdir
        self.prefix = prefix
        self.datadir = datadir

        self.setup_environment()
        self.setup_torch()
        self.setup_trainer()

    def setup_environment(self):
        os.makedirs(self.datadir, exist_ok=True)

        if self.logdir is not None:
            self.logdir = os.path.join(self.logdir, self.prefix)
            os.makedirs(self.logdir, exist_ok=True)

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
        logging.info(f"Device: {self.device}")

        # Additional info when using cuda
        if self.device.type == "cuda":
            logging.info(f"Device name: {torch.cuda.get_device_name()}")
            logging.info(f"Allocated memory: {round(torch.cuda.memory_allocated()/1024**3,1)} GB")
            logging.info(f"Cached memory: {round(torch.cuda.memory_reserved()/1024**3,1)} GB")

        if use_cuda:
            self.dataloader_kwargs = {"num_workers": 10, "pin_memory": True}
        else:
            self.dataloader_kwargs = {"num_workers": self.num_workers, "pin_memory": False}

    def setup_trainer(self):
        self.data = data.Data(self.datadir, self.dataloader_kwargs)

        self.model = models.Model(
            self.device, self.data.input_shape, self.data.num_classes, self.data.num_classes_per_task
        )

        self.trainer = trainers.trainer_maker(
            self.data.target_type, self.device, self.model.build(), self.data, self.logdir
        )

    def run_experiment(self):
        self.trainer.train()


def main(argv):
    logging.info("DÉBUT")
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)
    logging.info(f"Gin config parameter bindings:\n{gin.config_str()}")

    os.environ["WANDB_DIR"] = gin_config_to_dict(gin.config_str())["ExperimentManager.logdir"]
    if "WANDB_API_TOKEN" in os.environ:
        wandb.init(project="clmem_revision", entity="bbea")
        exp_logdir = wandb.run.dir
        exp_prefix = ""
    else:
        wandb.init(anonymous="allow")
        exp_logdir = gin_config_to_dict(gin.config_str())["ExperimentManager.logdir"]
        exp_prefix = wandb.run.id

    run_counter = wandb.run.name.split("-")[-1]
    wandb.run.name = run_counter + "-" + FLAGS.gin_file[0].split("/")[-1][:-4]
    wandb.run.save()

    wandb.config.update(gin_config_to_dict(gin.config_str()))
    wandb.save(*FLAGS.gin_file)
    wandb.run.log_code("./", include_fn=lambda path: path.endswith(".py"))

    exp_manager = ExperimentManager(logdir=exp_logdir, prefix=exp_prefix)
    exp_manager.run_experiment()

    wandb.finish()
    if "WANDB_API_TOKEN" in os.environ:
        shutil.rmtree(exp_logdir)
    logging.info("FIN")


if __name__ == "__main__":
    set_memory_limit(50)
    flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
    flags.DEFINE_multi_string("gin_param", None, "Newline separated list of Gin param bindings.")
    FLAGS = flags.FLAGS

    app.run(main)
