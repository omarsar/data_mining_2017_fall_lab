import pandas as pd
import pickle
import numpy as np
from sklearn import preprocessing

""" Function to preprocess, transform, and analyze the data """

def hot_encoding(train_data, data, category_list = ['emo1', 'emo2', 'emo3']):
    """Convert the test set multiples labels into dictionaries"""
    labels_dict = [set(emos) & set(train_data.emotions) for emos in data[category_list].values]
    mlb = preprocessing.MultiLabelBinarizer()
    y_bin_emotions = mlb.fit_transform(labels_dict)
    data["bin_emotions"] = y_bin_emotions.tolist()
    return data

def split_test_by_emotions(data):
    """split the dataset by emotion category and store them in pickles"""
    emotions = ["joy", "sadness", "fear","anger", "trust", "surprise", "anticipation", "disgust"]
    base_directory = "data/emo_2017/pretrained/"
    for e in emotions:
        t_ = list(data.loc[(data["emotions"] == e)][["text", "bin_emotions"]].values)
        pickle.dump(t_, open(base_directory + e + ".p", "wb"))

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    Adopted from: "cnn_data_helpers"
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(files):
    """
    Loads emotion data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    x_text = []
    y = []
    # Load data from files
    for f in files:
        print (" in " , f, " records")
        raw_data = pickle.load(open(f,"rb"))
        x_samples = [s[0].strip() for s in raw_data]
        y.extend ([s[1] for s in list(raw_data)])
        x_text.extend(x_samples)
    y = np.array(y)

    return [x_text, y]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def explore_per_emotion(emotion):
    """explore the data by an emotion"""
    raw_data = pickle.load(open(emotion,"rb"))
    for s in list(raw_data):
        print(s[1])

def prepare_for_projector_embedding(data):
    """
    Prepares the data for projector embedding
    """
