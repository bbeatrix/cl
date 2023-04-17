
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
    if "heavyaugment" not in label and "epoch=200" in file:
        score = np.load(f"./data/all_scores_for_rankcorr/{file}")
        print(score.shape)
        # label = label #  + "_epoch200" # + "_transformed"
        # if "cl" in label
        all_scores[label] = score
    #else:
    #    all_scores[label] = score


all_scores_df = pd.DataFrame(all_scores)
print(all_scores_df.columns)

base_scores = all_scores_df[all_scores_df.filter(like='base_').columns]
base_forget_scores = all_scores_df[all_scores_df.filter(like='base_forget').columns]
base_forget_scores = np.array(base_forget_scores).squeeze()

cl_forget_scores = all_scores_df[all_scores_df.filter(like='cl_forget').columns]
cl_forget_scores = np.array(cl_forget_scores).squeeze()
print(cl_forget_scores.shape)

cl_firstencounteriter_scores = all_scores_df[all_scores_df.filter(like='cl_firstencounteriter').columns]
cl_firstencounteriter_scores = np.array(cl_firstencounteriter_scores).squeeze()
print(cl_firstencounteriter_scores.shape)

labels = np.load("./data/cifar10_train_labels.npy")
print(labels.shape)

print("labels: ", labels[:10])
label2task = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4}
#label2task = {i: 0 for i in range(10)}

tasks = np.array([label2task[label] for label in labels])

print("tasks: ", tasks[:10])

print("cl_firstencounteriters: ", cl_firstencounteriter_scores[:10])

y = base_forget_scores


for col in all_scores_df.columns:
    if "cl_el2n" in col:
        print(col)
        print("rank corr with base forget: ", all_scores_df[col].corr(all_scores_df["base_forget"], method="spearman"))

        x1 = np.array(all_scores_df[all_scores_df.filter(like=col).columns]).squeeze()
        if np.all(np.isfinite(x1)) == False:
            print("score contains nan, cannot fit linear regression")
            continue
            print(np.all(np.isfinite(y)))

        x2 = tasks
        x3 = cl_firstencounteriter_scores

        X = np.vstack([x1, x2, x3, np.ones(len(x1))]).T
        # X = np.vstack([x1, np.ones(len(x1))]).T
        # X = np.vstack([x1, x2, np.ones(len(x1))]).T
        print(f"\t y: base_forget \tx1: {col}  \t  x2: tasks  \t  x3:  cl_firstencounteriter \t x4: 1")
        #print(f"\t y: base_forget \tx1: {col}  \t x2: task \t  \t x3: 1")
        c1, c2, c3, c4 = np.linalg.lstsq(X, y, rcond=None)[0]
        print("\t Coefficients: ", c1, c2, c3, c4)
        #c1, c2, c3 = np.linalg.lstsq(X, y, rcond=None)[0]
        #print("\t Coefficients: ", c1, c2, c3)

        y_pred = c1 * x1 + c2 * x2 + c3 * x3 + c4

        print("\t L1 loss: ", np.linalg.norm(y - y_pred, ord = 1))
        print("\t L2 loss: ", np.linalg.norm(y - y_pred, ord = 2))


_ = plt.plot(range(len(base_forget_scores)), base_forget_scores[np.argsort(y_pred)], 'o', label='base forget scores', markersize=2)
_ = plt.plot(range(len(base_forget_scores)), np.sort(y_pred), 'x', label='fitted line', markersize=2, alpha=0.2)
_ = plt.legend()
plt.savefig("./outputs/figures/lsl.jpg", dpi=300)
print("Done")

exit()
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
