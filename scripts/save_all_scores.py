import numpy as np
import os
import wandb
api = wandb.Api()

runnames = {'1256': 'cifar10_base_ept=200',
            '1271': 'cifar10_cl_ept=200',
            '1272': 'cifar10_cl_ept=20',
            '1257': 'cifar10_baseheavyaugment_ept=200',
            '1270': 'cifar10_clheavyaugment_ept=200',
            '1248': 'cifar10_clwreplayfscoreunforgettables_ept=10',
            '1267': 'cifar10_clreservoirmem_ept=10',
            '1268': 'cifar10_clreservoirmem_ept=200',
            '1262': 'cifar10_clreservoirmemheavyaugment_ept=10',
            '1269': 'cifar10_clreservoirmemheavyaugment_ept=200'}

# runs = api.runs("bbea/cl", {"display_name": {"$regex": "^(1256|1255|1254|1257|1259|1248|1267|1268|1262|1269)-.*$"}})
# runs = api.runs("bbea/cl", {"display_name": {"$regex": "^(1260|1261|1262|1263)-.*$"}})

runs = api.runs("bbea/cl", {"display_name": {"$regex": "^(1256|1271|1272|1257|1270|1248|1267|1268|1262|1269)-.*$"}})
print(len(runs))
print([run.id for run in runs])
print([run.name for run in runs])

all_scores_descriptions = {"forget": "Number of forgetting events occurred", 
                           "softforget": "Number of soft forgetting events occurred",
                           "softforget2": "Number of second type of soft forgetting events occurred",
                           "el2n": "Sum of l2 norm of error vectors",
                           "negentropy": "Sum of negative entropies of softmax outputs",
                           "accuracy": "Sum of prediction accuracies",
                           "pcorrect": "Sum of correct prediction probabilities",
                           "pmax": "Sum of maximum prediction probabilities",
                           "firstlearniter": "Iteration of first learning event",
                           "finallearniter": "Iteration of final learning event",}
                           #"firstencounteriter": "Iteration of first encounter",
                           #"adjustedfirstencounteriter": "Adjusted iteration of first encounter"}
score_types = [item for item in all_scores_descriptions.keys()]
print(score_types)
download_dir = "./data/all_scores_for_rankcorr/"

for run in runs:
    lasttask = run.config['Data.num_tasks']
    lastiter = run.config['Trainer.iters']
    epoch = run.config['Trainer.epochs_per_task']
    # name = run.name.split('-')[-1]
    counter = run.name.split('-')[0]
    print(counter)

    if counter not in runnames.keys():
        raise ValueError
        continue
    else:
        name = runnames[counter]
        print(name)

    # if directory does not exist, creae one
    if not os.path.isdir(os.path.join("./data/all_scores_for_rankcorr", name)):
        os.makedirs(os.path.join("./data/all_scores_for_rankcorr", name))

    for score_type in score_types:
        print(run.name, score_type)

        try:
            scores = run.file(f"all_scores/{score_type}_scores/{score_type}scores_task={lasttask-1}_globaliter={lastiter}.npy").download(root=download_dir)
        except wandb.errors.CommError as err:
            print(err.message)
            continue
        scores = np.load(f"./data/all_scores_for_rankcorr/all_scores/{score_type}_scores/{score_type}scores_task={lasttask-1}_globaliter={lastiter}.npy")

        np.save(f"./data/all_scores_for_rankcorr/{name}/{name}_{score_type}_score_task={lasttask-1}_epoch={epoch}x{lasttask}.npy", scores)
        os.remove(f"./data/all_scores_for_rankcorr/all_scores/{score_type}_scores/{score_type}scores_task={lasttask -1}_globaliter={lastiter}.npy")
        os.rmdir(f"./data/all_scores_for_rankcorr/all_scores/{score_type}_scores/")
    os.rmdir(f"./data/all_scores_for_rankcorr/all_scores/")
print("Done")
