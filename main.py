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


genre_map = {
    0: "Electornic",
    1: "Anime",
    2: "Jazz",
    3: "Alternative",
    4: "Country",
    5: "Rap",
    6: "Blues",
    7: "Rock",
    8: "Classical",
    9: "Hip-Hop"
}


def load_model(input_dim, hidden_dim, out_dim, dropout, checkpoint_path = "./genre_machine.pth"):
    model = MLP(in_dim = input_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model



def parse_args():  
    parser = argparse.ArgumentParser(description="Run the model.")
    parser.add_argument("--popularity", type = float, required=True, help="Popularity score to predict")
    parser.add_argument("--acousticness", type = float, required=True, help="Acousticness score to predict")
    parser.add_argument("--danceability", type = float, required=True, help="Danceability score to predict")
    parser.add_argument("--duration_ms", type = float, required=True, help="Duration in milliseconds to predict")
    parser.add_argument("--energy", type = float, required=True, help="Energy score to predict")
    parser.add_argument("--instrumentalness", type = float, required=True, help="Instrumentalness score to predict")
    parser.add_argument("--liveness", type = float, required=True, help="Liveness score to predict")
    parser.add_argument("--loudness", type = float, required=True, help="Loudness score to predict")
    parser.add_argument("--speechiness", type = float, required=True, help="Speechiness score to predict")
    parser.add_argument("--tempo", type = float, required=True, help="Tempo score to predict")
    parser.add_argument("--valence", type = float, required=True, help="Valence score to predict")
    parser.add_argument("--key", type = str, required=True, choices= ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"], help="Key to predict")
    parser.add_argument("--mode", type = str, required=True, choices=["Major", "Minor"], help="Mode to predict")


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    popularity = args.popularity
    acousticness = args.acousticness
    danceability = args.danceability
    duration_ms = args.duration_ms
    energy = args.energy
    instrumentalness = args.instrumentalness
    liveness = args.liveness
    loudness = args.loudness
    speechiness = args.speechiness
    tempo = args.tempo
    valence = args.valence
    key = args.key
    mode = args.mode

    X_train, X_test, y_train, y_test = preprocess_data(data_path="./musicData.csv")
    
    # convert input data to numpy array
    input_dict = {
        "popularity": [popularity],
        "acousticness": [acousticness],
        "danceability": [danceability],
        "duration_ms": [duration_ms],
        "energy": [energy],
        "instrumentalness": [instrumentalness],
        "liveness": [liveness],
        "loudness": [loudness],
        "mode": [1 if mode == "Major" else 0],
        "speechiness": [speechiness],
        "tempo": [tempo],
        "valence": [valence]
    }


    all_keys = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    key_dummies = {f"key_{k}": [1 if k == key else 0] for k in all_keys[1:]}  # drop_first=True --> drop A


    input_dict.update(key_dummies)
    input_df = pd.DataFrame(input_dict)
    input_df = input_df[X_train.columns]  # Ensure the order of columns matches the training data

    for col in input_df.columns:
        mean = X_train[col].mean()
        std = X_train[col].std()
        input_df[col] = (input_df[col] - mean) / std


    # convert input data to tensor
    input_tensor = torch.tensor(input_df.values, dtype=torch.float32)

    # load model
    input_dim = X_train.shape[1]
    hidden_dim = 256
    out_dim = len(np.unique(y_train))
    dropout = 0.3
    model = load_model(input_dim, hidden_dim, out_dim, dropout)

    # make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).numpy()
        predicted_class = np.argmax(probs, axis=1).item()
        predicted_probs = probs.squeeze()

    # print predicted class and probabilities
    print(f"Prediected Genre:{genre_map[predicted_class]}")
    print(f"Predicted Probabilities: {predicted_probs}")