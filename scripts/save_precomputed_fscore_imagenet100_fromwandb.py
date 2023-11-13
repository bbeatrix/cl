import numpy as np
from torchvision import datasets, transforms
import os
import wandb
api = wandb.Api()

# runs = api.runs("bbea/clmem_revision", {"display_name": {"$regex": "66-imagenet100_supwfs_studysetup_base"}})
runs = api.runs("bbea/clmem_revision", {"display_name": {"$regex": "43-imagenet100_supwfs_dytoxsetup_base_simpleaugment"}})
print(len(runs))
print([run.id for run in runs])
print([run.name for run in runs])

download_dir = "./outputs/data/"

train_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
train_dataset = datasets.ImageFolder(
                "/home/bbea/datasets" + "/imagenet100/imagenet100_train", transform=train_transforms
            )

print(len(train_dataset))
print(train_dataset.classes)
#print("targets \n", train_dataset.targets)#  = np.array(train_dataset.targets)

labels = np.array(train_dataset.targets)
print(labels.shape)

for run in runs:
    print(run.name)
    run_id = run.id
    seed = run.config["ExperimentManager.seed"]
    run = api.run(f"bbea/clmem_revision/{run_id}")

    filename = f"all_scores/forget_scores/forgetscores_task=0_globaliter=505500.npy"
    print(filename)

    fscores = run.file(filename).download(root=download_dir, replace=True)
    fs = np.load(download_dir + filename)

    fs_with_labels = np.vstack((fs, labels))
    print(fs_with_labels.shape)

    np.save("./outputs/data/imagenet100_train_precomputed_fscores_task=0_epoch=500_dytoxsetupsimpleaugment_with_labels.npy", fs_with_labels)
print("Done")