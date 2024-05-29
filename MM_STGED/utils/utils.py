import pickle
import json
import os
import numpy as np
import pandas as pd


def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_pkl_data(data, dir, file_name):
    create_dir(dir)
    pickle.dump(data, open(dir + file_name, 'wb'))


def load_pkl_data(dir, file_name):
    '''
    Args:
    -----
        path: path
        filename: file name
    Returns:
    --------
        data: loaded data
    '''
    file = open(dir+file_name, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def save_json_data(data, dir, file_name):
    create_dir(dir)
    with open(dir+file_name, 'w') as fp:
        json.dump(data, fp)


def load_json_data(dir, file_name):
    with open(dir+file_name, 'r') as fp:
        data = json.load(fp)
    return data

def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=',')
    A[A!=0.] = 1.
    A[A==0.] = 1e-10
    # A = calculate_laplacian_matrix(A, 'hat_rw_normd_lap_mat')
    A1 = A
    print(A1.shape)
    # A2 = np.matmul(A,A)
    # A3 = np.matmul(A2,A)
    # A2[A2!=0.] = 1.
    # A3[A3!=0.] = 1.
    # print(np.sum(A1)/A1.shape[0], np.sum(A2)/A1.shape[0], np.sum(A3)/A1.shape[0])
    
    return A1

def load_graph_node_features(path, feature1='start_lat', feature2='start_lng', feature3='end_lat', feature4='end_lng', feature5='length', feature6='level'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4, feature5, feature6]]
    X = rlt_df.to_numpy()
    length_max = X[:,4].max()
    X[:,4] = X[:,4]/length_max

    from sklearn.preprocessing import OneHotEncoder
    one_hot_encoder = OneHotEncoder()
    cat_list = list(X[:, 5])
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))
    one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
    num_cats = one_hot_rlt.shape[-1]
    final_x = np.zeros((X.shape[0], X.shape[-1] - 1 + num_cats), dtype=np.float32)
    final_x[:,:5] = X[:,:5]
    final_x[:,5:] = one_hot_rlt
    
    return final_x