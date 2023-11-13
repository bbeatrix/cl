import numpy as np
import os
import wandb
api = wandb.Api()

iters = [117300, 156400] # [19550, 39100, 78200, 195500]
epochs= [300, 400] # [50, 100, 200, 500]

runs = api.runs("bbea/clmem_revision", {"display_name": {"$regex": "41-cifar100_supwfs_dytoxsetup_base"}})
print(len(runs))
print([run.id for run in runs])
print([run.name for run in runs])

download_dir = "./outputs/data/"

labels = np.load("/home/bbea/archive/forget_mem_stuff/data/cifar100_train_labels.npy")

for run in runs:
    print(run.name)
    run_id = run.id
    seed = run.config["ExperimentManager.seed"]
    run = api.run(f"bbea/clmem_revision/{run_id}")

    for i, it in enumerate(iters):
        filename = f"all_scores/forget_scores/forgetscores_task=0_globaliter={it}.npy"
        print(filename)
        fscores = run.file(filename).download(root=download_dir, replace=True)
        fs = np.load(download_dir + filename)

        fs_with_labels = np.vstack((fs, labels))
        print(fs_with_labels.shape)
        np.save(f"{download_dir}cifar100_train_precomputed_fscores_task=0_epoch={epochs[i]}_dytoxsetup_with_labels.npy", fs_with_labels)
        os.remove(download_dir + filename)
        # os.rmdir(download_dir + filename.split("/")[0])
print("Done")