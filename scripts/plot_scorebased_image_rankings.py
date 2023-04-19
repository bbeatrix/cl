
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torchvision

sns.set_theme(style="white")


def plot_image_grid(score_name, sorted_scores, sorted_labels, sorted_indices_in_ds, k, num_classes):
    train_data  = torchvision.datasets.CIFAR10(root='~/datasets', train=True, download=False)
    num_classes = np.unique(sorted_labels).shape[0]

    fig, axs = plt.subplots(num_classes, 2 * k + 1, figsize=(num_classes, 2 * k + 1))

    for c in range(num_classes):
        print("class", c)

        # add label in each row at the beginning rotated by 90 degrees
        plt.text(-2, 1, f"Class {c}", fontsize=15, fontweight='bold', ha='left', va='top', transform=axs[c, 0].transAxes, rotation=90)

        indices_per_class = np.where(sorted_labels == c)[0]

        high_score_indices_in_ds = sorted_indices_in_ds[indices_per_class[-k:]]
        low_score_indices_in_ds = sorted_indices_in_ds[indices_per_class[:k]]

        indices_in_ds_for_plot = np.concatenate((low_score_indices_in_ds, high_score_indices_in_ds))
        scores_for_plot = np.concatenate((sorted_scores[indices_per_class[:k]], sorted_scores[indices_per_class[-k:]]))

        for i, idx in enumerate(indices_in_ds_for_plot):
            image = train_data[idx][0]
            label = train_data[idx][1]
            if i >= k:
                j = i + 1
            else:
                j = i
            axs[c, j].imshow(image)
            axs[c, j].set_title(f"{scores_for_plot[i]:.3f}", fontsize=7)
            axs[c, j].axis('off')
            axs[c, j].set_aspect('equal')

        axs[c, k].axvline(x=k, color='black', linewidth=3)
        axs[c, k].axis('off')
        axs[c, k].set_aspect('equal')

    plt.tight_layout()
            
    plt.savefig(f"./outputs/figures/{score_name}_score_top{k}_highest_lowest_images_per_class.jpg", dpi=300)

def load_and_sort_data(score_path, label_path):
    scores = np.load(score_path)

    sorted_order = np.argsort(scores)
    sorted_scores = scores[sorted_order]

    labels = np.load(label_path)
    sorted_labels = labels[sorted_order]

    indices_in_ds = np.arange(len(labels))
    sorted_indices_in_ds = indices_in_ds[sorted_order]

    return sorted_scores, sorted_labels, sorted_indices_in_ds


def main():

    label_path = "./data/cifar10_train_labels.npy"
    scores_dict = {#"cscore": "./data/cifar10_train_precomputed_cscores.npy",
                   #"forgetscore_base": "./data/all_scores_for_rankcorr/cifar10_base_ept=200/cifar10_base_ept=200_forget_score_task=0_epoch=200x1.npy",
                   #"softforgetscore_base": "./data/all_scores_for_rankcorr/cifar10_base_ept=200/cifar10_base_ept=200_softforget_score_task=0_epoch=200x1.npy",
                   #"finallearniter_base": "./data/all_scores_for_rankcorr/cifar10_base_ept=200/cifar10_base_ept=200_finallearniter_score_task=0_epoch=200x1.npy",
                   "forgetscore_base_heavyaugment": "./data/all_scores_for_rankcorr/cifar10_baseheavyaugment_ept=200/cifar10_baseheavyaugment_ept=200_forget_score_task=0_epoch=200x1.npy",
                   "forgetscore_cl": "./data/all_scores_for_rankcorr/cifar10_cl_ept=200/cifar10_cl_ept=200_forget_score_task=4_epoch=200x5.npy",
                   "softforgetscore_cl": "./data/all_scores_for_rankcorr/cifar10_cl_ept=200/cifar10_cl_ept=200_softforget_score_task=4_epoch=200x5.npy",
                   "finallearniter_cl": "./data/all_scores_for_rankcorr/cifar10_cl_ept=200/cifar10_cl_ept=200_finallearniter_score_task=4_epoch=200x5.npy",
                   "forgetscore_cl_heavyaugment": "./data/all_scores_for_rankcorr/cifar10_baseheavyaugment_ept=200/cifar10_baseheavyaugment_ept=200_forget_score_task=0_epoch=200x1.npy",
                   }

    for score_name, score_path in scores_dict.items():
        print(score_name)
        sorted_scores, sorted_labels, sorted_indices_in_ds = load_and_sort_data(score_path, label_path)
        plot_image_grid(score_name, sorted_scores, sorted_labels, sorted_indices_in_ds, k=10, num_classes=10)

    print("Done")


if __name__ == '__main__':
    main()
