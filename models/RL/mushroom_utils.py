import os

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

import utils


def load_mushroom(random_state=42, train_frac=0.8):
    """ Load mushroom dataset and preprocess it """
    mushroom_df = pd.read_csv(os.path.join(utils.get_project_root(), 'models/RL/data/mushroom.csv'))

    # shuffle lines before splitting between train and test sets
    mushroom_df = mushroom_df.sample(frac=1, random_state=random_state)

    # change categorical data to numerical
    for c in mushroom_df.columns:
        mushroom_df[c] = mushroom_df[c].astype('category').cat.codes  # 1: poisonous, 0: edible

    # split between train / test
    train_size = int(train_frac * len(mushroom_df))
    train_df, test_df = mushroom_df[:train_size], mushroom_df[train_size:]

    train_X, train_y = train_df.iloc[:, :-1].to_numpy(), train_df['class'].to_numpy()
    test_X, test_y = test_df.iloc[:, :-1].to_numpy(), test_df['class'].to_numpy()

    return mushroom_df, train_X, train_y, test_X, test_y


def plot_points(mushroom_df, train_X, test_X, fig):
    """ Plot 3D projection of mushroom dataset (PCA) """
    ax = Axes3D(fig, rect=[0, 0, .95, 1])

    pca = decomposition.PCA(n_components=3)
    pca.fit(mushroom_df.iloc[:, :-1].to_numpy())
    X = pca.transform(train_X)

    edible = mushroom_df['class'] == 0
    ax.scatter(X[edible[:len(X)], 0], X[edible[:len(X)], 1], X[edible[:len(X)], 2], color='b', label='edible - train')
    ax.scatter(X[~edible[:len(X)], 0], X[~edible[:len(X)], 1], X[~edible[:len(X)], 2], color='y',
               label='poisonous - train')
    offset = len(X)
    X = pca.transform(test_X)
    ax.scatter(X[edible[offset:], 0], X[edible[offset:], 1], X[edible[offset:], 2], edgecolor='k', color='r',
               label='edible - test')
    ax.scatter(X[~edible[offset:], 0], X[~edible[offset:], 1], X[~edible[offset:], 2], color='k',
               label='poisonous - test')

    ax.legend()