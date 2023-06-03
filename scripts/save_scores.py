import numpy as np
import os
import wandb
api = wandb.Api()


runs = api.runs("bbea/cl", {"display_name": {"$regex": "^(1308)-.*$"}})
print(len(runs))
print([run.id for run in runs])
print([run.name for run in runs])

filename = f"all_scores/forget_scores/forgetscores_task=0_globaliter=78200.npy"
download_dir = "./data/"

labels = np.load("./data/cifar100_train_labels.npy")
print(labels.shape)

for run in runs:
    print(run.name)
    run_id = run.id
    seed = run.config["ExperimentManager.seed"]
    run = api.run(f"bbea/cl/{run_id}")
    fscores = run.file(filename).download(root=download_dir)
    fs = np.load(download_dir + filename)

    fs_with_labels = np.vstack((fs, labels))
    print(fs_with_labels.shape)
    np.save(f"{download_dir}cifar100_train_precomputed_fscores_task=0_epoch=200_studysetup_heavyaugment_with_labels.npy", fs_with_labels)
    os.remove(download_dir + filename)
    os.rmdir(download_dir + filename.split("/")[0])
print("Done")