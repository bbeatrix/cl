
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

sns.set_theme(style="white")

def plot_corr(all_scores_df, corr, plot_name):
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
    if len(list(all_scores_df.filter(like='forget').columns)) > 0:
        row_idx = all_scores_df.columns.get_loc(list(all_scores_df.filter(like='forget').columns)[0])
        for col_idx in range(row_idx, len(all_scores_df.columns)):
            ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='black', lw=3))
        for col_idx in range(row_idx):
            ax.add_patch(plt.Rectangle((row_idx, col_idx), 1, 1, fill=False, edgecolor='black', lw=3))

    plt.savefig(f"./data/base_scores_for_rankcorr/figures/{plot_name}_score_correlations.jpg", dpi=300)
    plt.close()

def calc_and_plot_corr(all_scores_df, plot_name):
    corr = all_scores_df.corr("spearman") # "kendall"

    upper_tri = np.triu(corr, k=1)

    upper_tri_nonzero = upper_tri[np.nonzero(upper_tri)]

    # Calculate the mean and standard deviation of the non-zero elements
    mean_value = np.mean(upper_tri_nonzero)
    std_value = np.std(upper_tri_nonzero)
    print(f"Mean and std: {mean_value.round(2)} ({std_value.round(2)})")

    # plot_corr(all_scores_df, corr, plot_name)


def main():
    dirs = os.listdir("./data/base_scores_for_rankcorr/")
    print(dirs)

    for dir in sorted(dirs):
        print(dir)
        files = os.listdir(f"./data/base_scores_for_rankcorr/{dir}")

        files = sorted([file for file in files if file.endswith(".npy")])
        # print(files)
        if len(files) == 0:
            continue

        all_scores = {}

        for file in files:
            #print(file)
            onlyfile = file
            label = dir + "_seed" + file.split(".")[0].split("_")[-1]

            score = np.load(f"./data/base_scores_for_rankcorr/{dir}/{file}")
            #print(score.shape)
            all_scores[label] = score

        all_scores_df = pd.DataFrame(all_scores)
        # print(all_scores_df.columns)

        calc_and_plot_corr(all_scores_df, f"{dir}_seeds")

    print("Done")

if __name__ == '__main__':
    main()
