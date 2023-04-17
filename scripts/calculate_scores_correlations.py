
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

sns.set_theme(style="white")


def calc_and_plot_corr(all_scores_df, plot_name):
    corr = all_scores_df.corr("spearman") # "kendall"

    mask = np.transpose(np.triu(np.ones_like(corr, dtype=bool))) 
    np.fill_diagonal(mask, False)

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(40, 15))

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
    row_idx = all_scores_df.columns.get_loc(list(all_scores_df.filter(like='forget').columns)[0])
    for col_idx in range(row_idx, len(all_scores_df.columns)):
        ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='black', lw=3))
    for col_idx in range(row_idx):
        ax.add_patch(plt.Rectangle((row_idx, col_idx), 1, 1, fill=False, edgecolor='black', lw=3))

    plt.savefig(f"./outputs/figures/{plot_name}_score_correlations.jpg", dpi=300)


def main():
    dirs = os.listdir("./data/all_scores_for_rankcorr/")
    print(dirs)

    for dir1 in dirs:
        for dir2 in dirs:
            if dir1 != dir2:
                print(dir1, " vs ", dir2)
                files1 = os.listdir(f"./data/all_scores_for_rankcorr/{dir1}")
                files2 = os.listdir(f"./data/all_scores_for_rankcorr/{dir2}")
                files = [os.path.join(dir1, f) for f in files1] + [os.path.join(dir2, f) for f in files2]

                files = sorted([file for file in files if file.endswith(".npy")])
                print(files)
                all_scores = {}

                cscore = np.load("./data/cifar10_train_precomputed_cscores.npy")
                all_scores["cscore"] = cscore

                for file in files:
                    print(file)
                    if "/" in file:
                        onlyfile = file.split("/")[-1]
                    label = onlyfile.split(".")[0].split("_")[1] + "_" + onlyfile.split(".")[0].split("_")[3]
                    print(label)

                    #if "heavyaugment" not in label and "epoch=200" in file:
                    score = np.load(f"./data/all_scores_for_rankcorr/{file}")
                    print(score.shape)
                    all_scores[label] = score
                    #    label = label + "_epoch200" # + "_transformed"
                    #if "cl" in label and "finallearniter" in label:
                    #    score = np.square(score)
                    #    label = label + "_transformed"
                    
                    #else:
                    #    all_scores[label] = score

                all_scores_df = pd.DataFrame(all_scores)
                print(all_scores_df.columns)
                calc_and_plot_corr(all_scores_df, f"{dir1}_vs_{dir2}")

    print("Done")

if __name__ == '__main__':
    main()
