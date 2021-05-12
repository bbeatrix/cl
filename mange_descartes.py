import os
import itertools
import gin
import gin.torch
from absl import flags, app

import main

@gin.configurable
def descartes_builder(name='out', params=[]):
    print("Building grid search with parameters: ", params)

    directory = os.path.join('configs', 'grids', name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    all_values = []
    all_params = []
    for param in params:
        values = gin.query_parameter(param)
        all_params.append(param)
        all_values.append(values)
    descartes = itertools.product(*all_values)

    i = 0
    for one in descartes:

        exp_directory = os.path.join(directory, str(i))
        if not os.path.exists(exp_directory):
            os.makedirs(exp_directory)

        with gin.unlock_config():
            for param_idx in range(len(all_params)):
                gin.bind_parameter(all_params[param_idx], one[param_idx])

        config_str = gin.config_str()
        with open(os.path.join(exp_directory, 'config.gin'), 'w+') as f:
            f.write(config_str)
        i += 1
    pass


def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)
    print("Gin parameter bindings:\n{}".format(gin.config_str()))

    descartes_builder()

if __name__ == '__main__':
    flags.DEFINE_multi_string('gin_file', None, 'List of paths to the config files.')
    flags.DEFINE_multi_string('gin_param', None, 'Newline separated list of Gin parameter bindings.')
    FLAGS = flags.FLAGS

    app.run(main)