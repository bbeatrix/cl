import numpy as np
import os
import wandb
api = wandb.Api()

ds = "miniimagenet"
stypes = ["base", "base_woaugment", "base_heavyaugment"]

for stype in stypes:

    runs = api.runs("bbea/cl_mem_grids", {"display_name": {"$regex": f"{ds}_supwforgetstats_studysetup_{stype}_seedgrid.*$"}})

    print(len(runs))
    print([run.id for run in runs])
    print([run.name for run in runs])

    #filename = f"forget_scores/fs_task=0_globaliter=78200.npy"
    filename = f"all_scores/forget_scores/forgetscores_task=0_globaliter=78200.npy"
    if not os.path.exists(f"./data/base_scores_for_rankcorr/{ds}_{stype}"):
        os.mkdir(f"./data/base_scores_for_rankcorr/{ds}_{stype}")
    download_dir = f"./data/base_scores_for_rankcorr/{ds}_{stype}/"

    for run in runs:
        print(run.name)
        run_id = run.id
        seed = run.config["ExperimentManager.seed"]
        #run = api.run(f"bbea/cl/{run_id}")
        fscores = run.file(filename).download(root=download_dir)
        fs = np.load(download_dir + filename)

        #fs_with_labels = np.vstack((fs, labels))
        #print(fs_with_labels.shape)
        np.save(f"{download_dir}{ds}_precomputed_fs_task=0_epoch=200_studysetup_{stype}_{seed}.npy", fs)
        os.remove(download_dir + filename)
        #os.rmdir(download_dir + filename.split("/")[0])
print("Done")