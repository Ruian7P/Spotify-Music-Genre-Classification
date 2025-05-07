import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
import autograd.numpy as np
from autograd import grad
import pandas as pd
import seaborn as sns
from sklearn import datasets
import scipy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
import random 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D


random.seed(13147286)

def plot_eigenvalues(pca):
    eigenvalues = pca.explained_variance_
    eigenvalues_above_1 = []
    for i in range(len(eigenvalues)):
        if eigenvalues[i] > 1:
            eigenvalues_above_1.append(eigenvalues[i])
    plt.figure(figsize=(16, 5))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--', color='b')
    plt.axhline(y=1, color='r', linestyle='-')
    plt.title('Eigenvalues of PCA Components')
    plt.xlabel('Component Number')
    plt.ylabel('Eigenvalue')
    plt.xticks(range(1, len(eigenvalues) + 1))
    plt.grid()
    plt.tight_layout()
    plt.show()
    print("Eigenvalues above 1:", eigenvalues_above_1)


def plot_explained_variance_ratio(pca):
    explained_var = pca.explained_variance_ratio_

    plt.figure()
    plt.plot(np.cumsum(explained_var), marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance by PCA Components")
    plt.grid(True)
    plt.show()
    print("explained variance ratio:", pca.explained_variance_ratio_)
    print("Total explained variance:", np.sum(pca.explained_variance_ratio_))
    print("explained variance ratio by first 2 components:", np.sum(pca.explained_variance_ratio_[:2]))



def plot_pca_2d(X_pca, y):

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='Set1', s=5)
    plt.title('PCA of Music Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    plt.legend(title='Music Genre', loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.tight_layout()
    plt.show()



def plot_pca_3d(X_pca, y):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='Set1', s=5)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D PCA of Music Data')
    plt.show()



def plot_silhouette_scores(ks, silhouette_scores):
    plt.figure(figsize=(8, 5))
    plt.plot(ks, silhouette_scores, marker='o', linestyle='--', color='b')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.xticks(ks)
    plt.grid()
    plt.tight_layout()
    plt.show()
    print("Silhouette scores:", silhouette_scores)
    best_k = ks[silhouette_scores.index(max(silhouette_scores))]
    print("Best k:", best_k)


def plot_roc_curve(y_true, y_probs):

    fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 1], pos_label=1)
    roc_auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
    plt.figure(figsize=(8, 6))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='MLP').plot()
    plt.title('ROC Curve for MLP after PCA')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.show()