import numpy as np
from torchvision import datasets, transforms

fscores = np.load("./outputs/all_scores/forget_scores/forgetscores_task=0_globaliter=202200.npy")
print(fscores.shape)

train_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
train_dataset = datasets.ImageFolder(
                "/home/bbea/datasets" + "/imagenet100/imagenet100_train", transform=train_transforms
            )

print(len(train_dataset))
print(train_dataset.classes)
#print("targets \n", train_dataset.targets)#  = np.array(train_dataset.targets)

targets = np.array(train_dataset.targets)
print(targets.shape)

fs_with_labels = np.vstack((fscores, targets))
print(fs_with_labels.shape)

#np.save("./outputs/data/imagenet100_train_precomputed_fscores_task=0_epoch=500_dytoxsetup_with_labels.npy", fs_with_labels)

# plot histogram
import matplotlib.pyplot as plt
plt.hist(fscores, bins=100)
plt.savefig("./outputs/data/imagenet100_train_precomputed_fscores_task=0_epoch=500_dytoxsetup.png")