import numpy as np
import os
import wandb
api = wandb.Api()


runs = api.runs("bbea/cl_mem_grids", {"display_name": {"$regex": ""}})
print(len(runs))
print([run.id for run in runs])
print([run.name for run in runs])

filename = f"forget_scores/fs_task=0_globaliter=78200.npy"
download_dir = "./data/precomputed_fscores/"

labels = np.load("./data/cifar10_train_labels.npy")
    
for run in runs:
    print(run.name)
    run_id = run.id
    seed = run.config["ExperimentManager.seed"]
    run = api.run(f"bbea/cl_mem_grids/{run_id}")

    fscores = run.file(filename).download(root=download_dir)
    fs = np.load(download_dir + filename)

    fs_with_labels = np.vstack((fs, labels))
    print(fs_with_labels.shape)
    np.save(f"{download_dir}cifar10_train_precomputed_fscores_task=0_epoch=200_studysetup_woaugment_with_labels_seed{seed}.npy", fs_with_labels)
    os.remove(download_dir + filename)
    os.rmdir(download_dir + filename.split("/")[0])
print("Done")