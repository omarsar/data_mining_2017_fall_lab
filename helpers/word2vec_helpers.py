import nltk
import numpy as np
import gensim
import logging
logging.root.handlers = []  # Jupyter messes up logging so needs a reset
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

""" Helpers for Word2vec """

def remove_missing_lists(train_data, y_train):
    """
    Remove the missing empty for gensim word2vec to work
    TODO: merge with the other function below
    """
    remove_indices = []
    index_count = 0
    for x in train_data:
        if not x:
            remove_indices.append(index_count)
        index_count+=1
    keep_indices = list(sorted(set(range(len(train_data))) - set(remove_indices)))
    return train_data[keep_indices], y_train[keep_indices].reset_index(drop=True)

def remove_missing_tokens(wv, train_data, y):
    """
    Remove the text that has a missing token not found in Word2vec vocab list
    Support numpy lists
    TODO: can remove token instead of deleting the entire text
    """
    remove_indices = []
    index_count = 0
    for text in train_data:
        for word in text:
            if word not in wv.vocab:
                remove_indices.append(index_count)
        index_count+=1
    remove_indices = list(sorted(set(remove_indices)))
    keep_indices = list(set(range(len(train_data))) - set(remove_indices))

    return train_data[keep_indices], y[keep_indices].reset_index(drop=True)

def word_averaging(wv, words):
    """
    Word Averaging for a specific set of words
    """
    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.layer1_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_list(wv, text_list):
    """
    Return word averaging list
    """
    return np.vstack([word_averaging(wv, review) for review in text_list ])

def w2v_tokenize_text(text, remove_stopwords=False):
    """
    Tokenize text using word2vec
    """
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            if remove_stopwords == True:
                if word in stopwords.words('english'):
                    continue
            tokens.append(word)
    return tokens

def store_for_embddings_projector(X_train, X_train_metadata):
    """
    Store for word embeddings visualization porjector.tensorflow.org
    """
    np.savetxt("data/emo_2017/for_embeddings/10000_emo_text_word2vec.tsv", X_train_word_average , delimiter="\t")
    X_train_metadata = np.array([(" ".join(x)) for x in final_train_tokenized])
    X_train_metadata_combined = list(zip(X_train_metadata, final_train_y))

    # storing the metadata
    np.savetxt("data/emo_2017/for_embeddings/10000_emo_text_word2vec_metadata.tsv",\
           X_train_metadata_combined ,header="tweet \t emotion" ,delimiter="\t", fmt="%s")


"""
Visualization functions
"""

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def unpack_words_from_doc_vector(model):
    words_found = []
    for x in zip(model.wv.vocab):
        words_found.append(x[0])
    return words_found

def visualize_vectors(model, words):
    """
    Obtain and visualize T-SNE
    """
    X = model[model.wv.vocab]

    tsne = TSNE(n_components = 2)
    X_tsne = tsne.fit_transform(X[:1000,:])

    plt.scatter(X_tsne[:300, 0], X_tsne[:300, 1])

    # visualize
    for label, x, y in zip(words[:300], X_tsne[:300, 0], X_tsne[:300, 1]):
        plt.annotate(label, xy=(x,y), xytext=(0,0),  textcoords='offset points')
    plt.show
