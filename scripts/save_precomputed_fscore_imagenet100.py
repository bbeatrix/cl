import numpy as np
from torchvision import datasets, transforms

iters = [404400, 505500]
epochs= [400, 500]
for i, it in enumerate(iters):
    print(it)
    print(epochs[i])
    fscores = np.load(f"./outputs/all_scores/forget_scores/forgetscores_task=0_globaliter={it}.npy")
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

    np.save(f"./outputs/data/imagenet100_train_precomputed_fscores_task=0_epoch={epochs[i]}_dytoxsetup_with_labels.npy", fs_with_labels)

    # plot histogram
    import matplotlib.pyplot as plt
    plt.hist(fscores, bins=100)
    plt.savefig(f"./outputs/data/imagenet100_train_precomputed_fscores_task=0_epoch={epochs[i]}_dytoxsetup.png")