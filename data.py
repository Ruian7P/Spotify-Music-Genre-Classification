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


random.seed(13147286)


def preprocess_data(data_path = './musicData.csv', random_seed = 13147286):
    """
    Preprocess the data by loading it, process missing data, normalizing it, and splitting it into training and testing sets.
    """
    
    preview = pd.read_csv(data_path)
    

    print("---------------- check missing value of each column ----------------")
    for column in preview.columns:
        print(column, preview[column].isnull().sum())

    # print NaN rows
    print("NaN rows", preview[preview.isnull().any(axis=1)])
    # total row number
    print("total row number:", preview.shape[0])
    df = preview.dropna()
    print("total row number after drop nan:", df.shape[0])


    print("---------------- check non-numeric value of each numeric column ----------------")
    for col in ['duration_ms', 'tempo', 'loudness', 'energy', 'valence', 'speechiness', 'instrumentalness']:
        non_numeric_mask = pd.to_numeric(df[col], errors='coerce').isna() & df[col].notna()
        if non_numeric_mask.any():
            print(f"Non-numeric values in {col}:")
            print(df.loc[non_numeric_mask, col])

    print("---------------- check unreasonable numeric value of each numeric column ----------------")
    print("polularity", ((df['popularity'] > 99) | (df['popularity'] < 0)).sum())
    for column in ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        if column == 'duration_ms' or column == 'tempo':
            print(column, (df[column] < 0).sum())
        elif column == 'loudness':
            print(column, (df[column] > 0).sum())
        else:
            print(column, ((df[column] > 1) | (df[column] < 0)).sum())


    print("---------------- impute numeric values genre-wisely, drop useless columns, normalize numeric columns ----------------")
    for column in ['duration_ms', 'loudness', 'tempo']:
        for genre in df['music_genre'].unique():
            if column == 'duration_ms':
                genre_median = df.loc[(df['music_genre'] == genre) & (df[column] >= 0), column].median()
                df.loc[(df['music_genre'] == genre) & (df[column] < 0), column] = genre_median
            elif column == 'loudness':
                genre_median = df.loc[(df['music_genre'] == genre) & (df[column] <= 0), column].median()
                df.loc[(df['music_genre'] == genre) & (df[column] > 0), column] = genre_median
            else:
                genre_median = df.loc[(df['music_genre'] == genre) & (df[column].notna()), column].median()
                df.loc[(df['music_genre'] == genre) & (df[column].isna()), column] = genre_median


    drop_cols = ['instance_id', 'artist_name', 'track_name', 'obtained_date']

    df = df.drop(columns = drop_cols)

    norm_cols = ['popularity', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
    for col in norm_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()




    print("---------------- process categorical data ----------------")
    print("mode types:", df["mode"].unique())

    df.loc[df['mode'] == 'Major', 'mode'] = 1
    df.loc[df['mode'] == 'Minor', 'mode'] = 0
    df['mode'] = df['mode'].astype(int)

    print("key types:", df["key"].unique())

    # dummy encoding
    df['key'] = df['key'].astype('category')
    df = pd.get_dummies(df, columns=['key'], prefix='key', drop_first=True)

    # print dummy encoding
    print(df.filter(like='key_').head())

    print("music_genre types:", df["music_genre"].unique())

    # label
    genres = df['music_genre'].unique()
    for i, genre in enumerate(genres):
        print("genre", genre, "----> label", i)
        df.loc[df['music_genre'] == genre, 'music_genre'] = i
    df['music_genre'] = df['music_genre'].astype(int)

    print("total_num_rows:", df.shape[0])
    for genre in df['music_genre'].unique():
        print("genre", genre, "num_rows:", df[df['music_genre'] == genre].shape[0])


    ### save the processed data
    df.to_csv('./processed_musicData.csv', index=False)


    print("---------------- split data into train and test set ----------------")
    # train test split for each genre
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for genre in df['music_genre'].unique():
        genre_df = df[df['music_genre'] == genre]
        X_0 = genre_df.drop(columns=['music_genre'])
        y_0 = genre_df['music_genre']
        X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X_0, y_0, test_size=500, random_state = 13147286)
        X_train.append(X_train_0)
        X_test.append(X_test_0)
        y_train.append(y_train_0)
        y_test.append(y_test_0)

    X_train = pd.concat(X_train).reset_index(drop=True)
    X_test = pd.concat(X_test).reset_index(drop=True)
    y_train = pd.concat(y_train).reset_index(drop=True)
    y_test = pd.concat(y_test).reset_index(drop=True)

    # check whether the split is correct
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)


    bool_cols = X_train.select_dtypes(include=[bool]).columns
    for col in bool_cols:
        X_train[col] = X_train[col].astype(float)
        X_test[col] = X_test[col].astype(float)

    return X_train, X_test, y_train, y_test



def data_loader(X_train, y_train, X_test, y_test, batch_size = 1024, num_workers = 12):
    """
    Load the data into PyTorch tensors.
    """
    X_train_tensor = torch.tensor(X_train.values, dtype= torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype= torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype= torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype= torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader