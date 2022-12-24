from keras.preprocessing.image import load_img 
from keras_preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
from tqdm import tqdm
import shutil
import time
from numpy import savetxt
from numpy import loadtxt

input_path = r'E:\DataScience\LN_Keyed\wabr-1622-firstpage-tif'
feature_path = r'E:\DataScience\LN_Keyed\wabr-1622-firstpage-tif'
cluster_path = r'E:\DataScience\LN_Keyed\WI_Segmented_Files_form_wise'

pca_components = [100]
clusters_size_list = [100]

os.chdir(input_path)

def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features

def get_features(feature_path):
    files_list = []

    # scandir() function -- a better and faster directory iterator
    with os.scandir(input_path) as files:
        for file in files:
            if (file.name.endswith('.tif') or file.name.endswith('.tiff')):
                files_list.append(file.name)    

    if (not os.path.exists(feature_path + '.csv')):
        model = VGG16()
        model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

        data = {}

        # loop through each image in the dataset
        for file in tqdm(files_list):
            # try to extract the features and update the dictionary
            feat = extract_features(file, model)
            data[file] = feat

        # get a list of the filenames
        filenames = np.array(list(data.keys()))

        # get a list of just the features
        feat = np.array(list(data.values()))

        # reshape so that there are 210 samples of 4096 vectors
        feat = feat.reshape(-1, 4096)

        savetxt(feature_path + '.csv', feat, delimiter=',')
    else:
        feat = loadtxt(feature_path + '.csv', delimiter=',')
        filenames = np.array(files_list)

    return feat, filenames


def get_clusters(filenames, pca, clusters, cluster_path):
    # reduce the amount of dimensions in the feature vector
    pca = PCA(n_components=pca, random_state=22)
    pca.fit(feat)
    x = pca.transform(feat)

    kmeans = KMeans(n_clusters=clusters, random_state=22)
    kmeans.fit(x)

    # holds the cluster id and the images { id: [images] }
    groups = {}
    for file, cluster in zip(filenames, kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)

    for key, value in groups.items():
        print('Cluster:', str(key), ' -- Counts: ', str(len(value)))

    os.makedirs(cluster_path)

    path = os.path.join(cluster_path, 'cluster_' + str(clusters) + '.json')
    with open(path, 'wb') as fp:
        pickle.dump(groups, fp)
        fp.close()
    
    return groups


def segment_files(data, cluster_path):
    # path = os.path.join(cluster_path, 'cluster_' + clusters + '.json') 
    # with open(path, 'rb') as fp:
    #     data = pickle.load(fp)
    #     fp.close()
    # data = dict(data.items())
    for key, file_names in data.items():
        dir_path = cluster_path + '\\Cluster_' + str(key)
        os.mkdir(dir_path)
        if (file_names != []):
            for filename in tqdm(file_names):
                file_path = input_path + '\\' + filename
                shutil.copy(file_path, dir_path)
                

feat, filenames = get_features(feature_path)

time.sleep(3)

for size in clusters_size_list:
    for pca in pca_components:
        path = cluster_path + '_' + str(size) + '_' + str(pca)
        clusters = get_clusters(filenames, pca, size, path)
        segment_files(clusters, path)