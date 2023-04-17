
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

sns.set_theme(style="white")

files = os.listdir("./data/all_scores_for_rankcorr/")
files = sorted([file for file in files if file.endswith(".npy")])
print(files)

all_scores = {}

#cscore = np.load("./data/cifar10_train_precomputed_cscores.npy")
#all_scores["cscore"] = cscore

for file in files:
    print(file)
    label = file.split(".")[0].split("_")[2] + "_" + file.split(".")[0].split("_")[3]
    print(label)
    if "heavyaugment" not in label and "epoch=200" in file and "base_forget" in label or "base_finallearniter" in label:
        score = np.load(f"./data/all_scores_for_rankcorr/{file}")
        print(score.shape)
        label = label #+ "_epoch200" # + "_transformed"
        if "cl" in label and "finallearniter" in label:
            score = np.square(score)
            label = label + "_transformed"
        all_scores[label] = score
    #else:
    #    all_scores[label] = score

base_forget_scores = all_scores["base_forget"]
base_finallearniter_scores = all_scores["base_finallearniter"]
unforgettables_indices = np.where(base_forget_scores == 0)[0]
print("unforgettables count: ", len(unforgettables_indices))
unforgettables_finallearniter_scores = base_finallearniter_scores[unforgettables_indices]
print("min unforgettables' finallearniter: ", min(unforgettables_finallearniter_scores))
print("max unforgettables' finallearniter: ", max(unforgettables_finallearniter_scores))

forgettables_indices = np.where(base_forget_scores > 0)[0]
print("forgettables count: ", len(forgettables_indices))
mean_forget_score = np.mean(base_forget_scores[forgettables_indices])
print("mean forget score: ", mean_forget_score)

forgettables_scores = base_forget_scores[forgettables_indices]
sorted_indices = np.argsort(forgettables_scores)
halfsize = int(len(sorted_indices)/2)
low_forgettables_indices = forgettables_indices[sorted_indices[0:1000]] # halfsize
high_forgettables_indices = forgettables_indices[sorted_indices[-1000:]]

print("high forgettables count: ", len(high_forgettables_indices))
print("low forgettables count: ", len(low_forgettables_indices))

print("min low forgettables' finallearniter: ", min(base_finallearniter_scores[low_forgettables_indices]))
print("max low forgettables' finallearniter: ", max(base_finallearniter_scores[low_forgettables_indices]))

print("min high forgettables' finallearniter: ", min(base_finallearniter_scores[high_forgettables_indices]))
print("max high forgettables' finallearniter: ", max(base_finallearniter_scores[high_forgettables_indices]))

print("low forget scores: \n", base_forget_scores[forgettables_indices][sorted_indices[0:1000]])
print("Done")

exit()
high_forgettables_indices = np.where(base_forget_scores >= mean_forget_score and base_forget_scores > 0)[0]
low_forgettables_indices = np.where(base_forget_scores < mean_forget_score and base_forget_scores > 0)[0]
print("high forgettables count: ", len(high_forgettables_indices))
print("low forgettables count: ", len(low_forgettables_indices))
exit()
all_scores_df = pd.DataFrame(all_scores)
print(all_scores_df.columns)
corr = all_scores_df.corr("spearman") # "kendall"

mask = np.transpose(np.triu(np.ones_like(corr, dtype=bool))) 
np.fill_diagonal(mask, False)

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(30, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(250, 30, as_cmap=True)
cmap = sns.color_palette("vlag", as_cmap=True)
cmap = sns.diverging_palette(10, 250, s=75, l=40,
                            n=5, center="light", as_cmap=True)
 
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
plt.xticks(rotation=90, ha='right')

# Highlight the specified row with a black border
row_idx = all_scores_df.columns.get_loc(list(all_scores_df.filter(like='base_forget').columns)[0])
for col_idx in range(row_idx, len(all_scores_df.columns)):
    ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='black', lw=3))
for col_idx in range(row_idx):
    ax.add_patch(plt.Rectangle((row_idx, col_idx), 1, 1, fill=False, edgecolor='black', lw=3))

plt.savefig("./outputs/figures/correlations_between_all_scores.jpg", dpi=300)
print("Done")
