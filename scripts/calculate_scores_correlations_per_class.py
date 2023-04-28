
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

sns.set_theme(style="white")


def calc_and_plot_corr(all_scores_df, plot_name):
    corr = all_scores_df.corr("spearman") # "kendall"

    n = len(all_scores_df.columns) // 2
    corr = corr.iloc[:n, -n:]

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(40, 15))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(250, 30, as_cmap=True)
    cmap = sns.color_palette("vlag", as_cmap=True)
    cmap = sns.diverging_palette(10, 250, s=75, l=40,
                                n=5, center="light", as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, annot=True, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
    plt.xticks(rotation=90, ha='right')

    plt.savefig(f"./outputs/figures/corr_per_class/{plot_name}_score_correlations_per_class.jpg", dpi=300)


def main():
    dir1 = "cifar10_baseheavyaugment_ept=200"
    dir2 = "cifar10_clfixedunforgettablesmemsoftforgetsheavyaugment_ept=200"
    scoretype1 = "forget"
    scoretype2 = "softforget"

    print(dir1, " vs ", dir2)
    files1 = os.listdir(f"./data/all_scores_for_rankcorr/{dir1}")
    files2 = os.listdir(f"./data/all_scores_for_rankcorr/{dir2}")

    scorefile1 = [f for f in files1 if f"_{scoretype1}_" in f]
    scorefile2 = [f for f in files2 if f"_{scoretype2}_" in f]

    score1 = np.load(f"./data/all_scores_for_rankcorr/{dir1}/{scorefile1[0]}")
    score2 = np.load(f"./data/all_scores_for_rankcorr/{dir2}/{scorefile2[0]}")

    labels = np.load("./data/cifar10_train_labels.npy")

    num_classes = np.unique(labels).shape[0]
    print("num_classes: ", num_classes)

    scores1_per_class  = {}
    scores2_per_class  = {} 

    for i in range(num_classes):
        print(i)
        scores1_per_class[dir1 + "_" + scoretype1 + "_class" + str(i)] = score1[labels == i]
        scores2_per_class[dir2 + "_" + scoretype2 + "_class" + str(i)] = score2[labels == i]
        print(score1[labels == i].shape)
        print(score2[labels == i].shape)

    all_scores = {**scores1_per_class, **scores2_per_class}
    all_scores_df = pd.DataFrame(all_scores)
    print(all_scores_df.columns)
    calc_and_plot_corr(all_scores_df, f"{dir1 + '_' + scoretype1}_vs_{dir2 + '_' + scoretype2}")

    print("Done")

if __name__ == '__main__':
    main()
