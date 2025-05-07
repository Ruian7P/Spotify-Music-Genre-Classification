import argparse
import os
from data import preprocess_data, data_loader
from train_model import train_nn, train_adaboost, train_random_forest
from nn_models import MLP, Resnet
from visualization import plot_roc_curve, plot_eigenvalues, plot_explained_variance_ratio, plot_pca_2d, plot_pca_3d, plot_silhouette_scores
import random
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D

random.seed(13147286)

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate models.")
    parser.add_argument("--model", type=str, choices=["mlp", "resnet", "adaboost", "randomforest"], required=True,help="Model to train")
    parser.add_argument("--data_path", type=str, default="./musicData.csv", help="Path to the data file")
    parser.add_argument("--use_pca", action = "store_true", help="Train with PCA processed data")
    parser.add_argument("--plot_before", action = "store_true", help="Plot before PCA")
    parser.add_argument("--plot_after", action = "store_true", help="Plot after PCA")

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    model = args.model
    use_pca = args.use_pca
    plot_before = args.plot_before
    plot_after = args.plot_after

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data_path=data_path)
    pca = PCA().fit(X_train)
    X_train_pca = pca.transform(X_train)

    if plot_before:
        plot_eigenvalues(pca)
        plot_explained_variance_ratio(pca)
        plot_pca_2d(X_train_pca[:, :2], y_train)
        plot_pca_3d(X_train_pca[:, :3], y_train)

        silhouette_scores = []
        ks = range(2, 20)

        for k in ks:
            kmeans = KMeans(n_clusters=k, n_init= 'auto', random_state=13147286)
            labels = kmeans.fit_predict(X_train_pca)
            score = silhouette_score(X_train_pca, labels)
            silhouette_scores.append(score)
        plot_silhouette_scores(ks, silhouette_scores)
        best_k = ks[silhouette_scores.index(max(silhouette_scores))]
        kmeans = KMeans(n_clusters=best_k, n_init= 'auto', random_state=13147286)
        labels = kmeans.fit_predict(X_train_pca)
        plot_pca_2d(X_train_pca, labels)

    if use_pca:
        X_train = X_train_pca
        X_test = pca.transform(X_test)
        print("PCA applied to the data")

    if model == "randomforest":
        print("Training Random Forest model")
        model, auc, params = train_random_forest(X_train, y_train, X_test, y_test)
        print("Random Forest model trained")
        print("Best AUC:", auc)
        print("Best parameters:", params)

    elif model == "adaboost":
        print("Training AdaBoost model")
        model, auc, params = train_adaboost(X_train, y_train, X_test, y_test)
        print("AdaBoost model trained")
        print("Best AUC:", auc)
        print("Best parameters:", params)

    else:
        out_dim = len(np.unique(y_train))
        if use_pca:
            best_auc = 0
            best_n = 0
            best_model = None
            for n in range(2, X_train.shape[1]):
                X_train_n = X_train[:, :n]
                X_test_n = X_test[:, :n]
                train_loader, test_loader = data_loader(X_train_n, y_train, X_test_n, y_test)

                if model == "mlp":
                    print("Training MLP model")
                    nn_model = MLP(in_dim=X_train_n.shape[1], hidden_dim=256, out_dim = out_dim, dropout=0.3)
                    nn_model, auc, best_model = train_nn(nn_model, train_loader, test_loader)

                    if auc > best_auc:
                        best_auc = auc
                        best_n = n
                        best_model = nn_model

                elif model == "resnet":
                    print("Training ResNet model")
                    nn_model = Resnet(in_dim=X_train_n.shape[1], hidden_dim=256, out_dim = out_dim, dropout=0.3)
                    nn_model, auc, best_model = train_nn(nn_model, train_loader, test_loader)
                    if auc > best_auc:
                        best_auc = auc
                        best_n = n
                        best_model = nn_model


            print("Best AUC:", best_auc)
            print("Best n:", best_n)
            print("Model trained")


        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_loader, test_loader = data_loader(X_train, y_train, X_test, y_test)
            if model == "mlp":
                print("Training MLP model")
                nn_model = MLP(in_dim=X_train.shape[1], hidden_dim=256, out_dim = out_dim, dropout=0.3)
                nn_model, auc, best_model = train_nn(nn_model, train_loader, test_loader)

            elif model == "resnet":
                print("Training ResNet model")
                nn_model = Resnet(in_dim=X_train.shape[1], hidden_dim=256, out_dim = out_dim, dropout=0.3)
                nn_model, auc, best_model = train_nn(nn_model, train_loader, test_loader)

            print("Best AUC:", auc)
            print("Model trained")

    if plot_after:
        best_model.eval()

        if use_pca:
            X_train_n = X_train[:, :best_n]
            X_test_n = X_test[:, :best_n]
            train_loader, test_loader = data_loader(X_train_n, y_train, X_test_n, y_test)

        y_probs = []
        y_true = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = best_model(inputs)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                y_probs.append(probs)
                y_true.append(labels.cpu().numpy())
        y_probs = np.concatenate(y_probs, axis=0)
        y_true = np.concatenate(y_true)

        plot_roc_curve(y_true, y_probs)

        X_train_latent, y_train_latent = best_model.get_latent(train_loader, device)
        pca_latent = PCA().fit(X_train_latent)
        X_train_latent_pca = pca_latent.transform(X_train_latent)

        plot_eigenvalues(pca_latent)
        plot_explained_variance_ratio(pca_latent)
        plot_pca_2d(X_train_latent[:, :2], y_train_latent)
        plot_pca_3d(X_train_latent[:, :3], y_train_latent)

        silhouette_scores = []
        ks = range(2, 20)
        for k in ks:
            kmeans = KMeans(n_clusters=k, n_init= 'auto', random_state=13147286)
            labels = kmeans.fit_predict(X_train_latent_pca)
            score = silhouette_score(X_train_latent_pca, labels)
            silhouette_scores.append(score)
        plot_silhouette_scores(ks, silhouette_scores)
        best_k = ks[silhouette_scores.index(max(silhouette_scores))]
        kmeans = KMeans(n_clusters=best_k, n_init= 'auto', random_state=13147286)
        labels = kmeans.fit_predict(X_train_latent_pca)
        plot_pca_2d(X_train_latent_pca, labels)
        plot_pca_3d(X_train_latent_pca, labels)
        