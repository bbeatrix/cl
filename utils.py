import matplotlib
matplotlib.use("Agg")
import seaborn as sns; sns.set(style="ticks", font_scale=1.4, rc={"lines.linewidth": 2.5})
import matplotlib.pyplot as plt; plt.style.use('seaborn-deep')

import torchvision.utils as tvutils
import neptune


def save_image(x, name, it, filename, normalize=True):
    tvutils.save_image(x, filename, normalize=normalize)
    neptune.log_image(name, x=it, y=filename)
