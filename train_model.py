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
import math
from tqdm import tqdm

random.seed(13147286)



def train_random_forest(X_train, y_train, 
                        X_test, y_test, 
                        n_estimators = 200, 
                        criterion = 'gini', 
                        max_features = 'sqrt', 
                        max_depth = 15, 
                        min_samples_split = 10, 
                        min_samples_leaf = 4, 
                        random_forest_config = None,
                        pca_components = -1):
    """
    Train a Random Forest classifier and evaluate its performance.
    """

    if pca_components > 0:
        best_auc = 0
        best_model = None
        for n in range(2, X_train.shape[1]):
            X_train_n = X_train[:, :n]
            X_test_n = X_test[:, :n]
            model = RandomForestClassifier(n_estimators=n_estimators,
                                             criterion=criterion,
                                             max_features=max_features,
                                             max_depth=max_depth,
                                             min_samples_split=min_samples_split,
                                             min_samples_leaf=min_samples_leaf,
                                             random_state=random.seed(13147286),
                                             n_jobs=-1
                                            )
            model.fit(X_train_n, y_train)
            y_pred = model.predict_proba(X_test_n)
            auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
            if auc > best_auc:
                best_auc = auc
                best_model = model
        print("Best AUC:", best_auc)
        return best_model, best_auc, (n_estimators, criterion, max_features, max_depth, min_samples_split, min_samples_leaf)

    elif random_forest_config is not None:
        n_estimators_list = random_forest_config['n_estimators']
        criterion_list = random_forest_config['criterion']
        max_features_list = random_forest_config['max_features']
        max_depth_list = random_forest_config['max_depth']
        min_samples_split_list = random_forest_config['min_samples_split']
        min_samples_leaf_list = random_forest_config['min_samples_leaf']
        best_auc = 0
        best_model = None
        best_params = None
        for n_estimators in n_estimators_list:
            for criterion in criterion_list:
                for max_features in max_features_list:
                    for max_depth in max_depth_list:
                        for min_samples_split in min_samples_split_list:
                            for min_samples_leaf in min_samples_leaf_list:
                                model = RandomForestClassifier(
                                    n_estimators=n_estimators,
                                    criterion=criterion,
                                    max_features=max_features,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    random_state=random.seed(13147286),
                                    n_jobs=-1
                                )
                                model.fit(X_train, y_train)
                                y_pred = model.predict_proba(X_test)
                                auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
                                if auc > best_auc:
                                    best_auc = auc
                                    best_model = model
                                    best_params = (n_estimators, criterion, max_features, max_depth, min_samples_split, min_samples_leaf)
        print("Best AUC:", best_auc)
        print("Best parameters:", best_params)
        return best_model, best_auc, best_params
    


    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random.seed(13147286),
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
        print("AUC:", auc)
        return model, auc, (n_estimators, criterion, max_features, max_depth, min_samples_split, min_samples_leaf)



def train_adaboost(X_train, y_train,
                   X_test, y_test,
                   n_estimators = 200,
                   learning_rate = 1.0,
                   algorithm = 'SAMME',
                   adaboost_config = None,
                   pca_components = -1):
    """
    Train an AdaBoost classifier and evaluate its performance.
    """

    if pca_components > 0:
        best_auc = 0
        best_model = None
        for n in range(2, X_train.shape[1]):
            X_train_n = X_train[:, :n]
            X_test_n = X_test[:, :n]
            model = AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                algorithm=algorithm,
                random_state=random.seed(13147286)
            )
            model.fit(X_train_n, y_train)
            y_pred = model.predict_proba(X_test_n)
            auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
            if auc > best_auc:
                best_auc = auc
                best_model = model
        print("Best AUC:", best_auc)
        return best_model, best_auc, (n_estimators, learning_rate, algorithm)

    elif adaboost_config is not None:
        n_estimators_list = adaboost_config['n_estimators']
        learning_rate_list = adaboost_config['learning_rate']
        algorithm_list = adaboost_config['algorithm']
        best_auc = 0
        best_model = None
        best_params = None
        for n_estimators in n_estimators_list:
            for learning_rate in learning_rate_list:
                for algorithm in algorithm_list:
                    model = AdaBoostClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        algorithm=algorithm,
                        random_state=random.seed(13147286)
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_test)
                    auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
                    if auc > best_auc:
                        best_auc = auc
                        best_model = model
                        best_params = (n_estimators, learning_rate, algorithm)
        print("Best AUC:", best_auc)
        print("Best parameters:", best_params)
        return best_model, best_auc, best_params
    
    else:
        model = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random.seed(13147286)
        )
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
        print("AUC:", auc)
        return model, auc, (n_estimators, learning_rate, algorithm)
    

def warmup_cosine_lr(epoch, total_epochs, warmup_epochs=10, eta_min = 0):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs # Linear warmup
    else:
        return eta_min + 0.5 * (math.cos((epoch - warmup_epochs) / (total_epochs - warmup_epochs) * math.pi) + 1)
    
def train_nn(model, train_loader, test_loader, epochs = 200, criterion = nn.CrossEntropyLoss(), lr = 1e-3, weight_decay = 5e-4):
    best_auc = 0
    best_epoch = -1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: warmup_cosine_lr(epoch, epochs))

    for epoch in range(epochs):
        model.train()
        training_loss = 0.0
        for idx, batch in enumerate(tqdm(train_loader, desc= f"Epoch {epoch + 1}/{epochs}")):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        training_loss /= len(train_loader)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {training_loss:.4f}")

        # validation for auc
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            y_probs = []
            y_true = []
            
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                y_probs.append(probs)
                y_true.append(labels.cpu().numpy())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        y_probs = np.concatenate(y_probs, axis=0)
        y_true = np.concatenate(y_true)
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
        print(f"Epoch {epoch + 1}/{epochs}, Validation AUC: {auc:.4f}, Validation Accuracy: {accuracy:.2f}%")
        print("-------------------------------------------")


        # save model
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch + 1
            best_model = model
            torch.save(model.state_dict(), 'best_mlp_model.pth')


    print(f"Best AUC: {best_auc:.4f} at epoch {best_epoch}")
    return model, best_auc, best_model
