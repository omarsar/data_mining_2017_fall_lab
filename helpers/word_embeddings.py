import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk

"""
Functions for Word embedding Sandbox code
"""

def plot_confusion_matrix(cm, my_tags, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots a confusion matrix given some input
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(my_tags))
    target_names = my_tags
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def evaluate_prediction(predictions, target, tags, title="Confusion matrix"):
    """
    Evaluate the results
    """
    print('accuracy %s' % accuracy_score(target, predictions))
    cm = confusion_matrix(target, predictions)
    print('confusion matrix\n %s' % cm)
    print('(row=expected, col=predicted)')

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, tags, title + ' Normalized')

def predict(vectorizer, classifier, data, tags):
    """
    Make a prediction
    """
    data_features = vectorizer.transform(data['plot'])
    predictions = classifier.predict(data_features)
    target = data['tag']
    evaluate_prediction(predictions, target, tags)

def tokenize_text(text):
    """
    Tokenization using NLTK tokenizer
    """
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

def get_tag_index(tags, tag_to_search):
    """
    Return the index for the particular class in the tags array
    """
    counter = 0
    for t in tags:
        if tag_to_search == t:
            break
        else:
            counter+=1
    return counter

def most_influential_words(model, vectorizer, genre_index=0, num_words=10):
    """
    Obtain the most important words for a class
    """
    features = vectorizer.get_feature_names()
    max_coef = sorted(enumerate(model.coef_[genre_index]), key=lambda x:x[1], reverse=True)
    return [[features[x[0]], x[1] ] for x in max_coef[:num_words]]

def check_word_overlap(tag_1, tag_2):
    """
    Return the words/position that overlap and the number of words that overlap
    """
    position_tracker = 0
    words_found = []
    for t1 in tag_1:
        for t2 in tag_2:
            if t1 == t2:
                words_found.append([t1,position_tracker])
        position_tracker+=1
    return [words_found, len(words_found)]

def most_influential_words_doc(doc, tfidf_words):
    """
    Obtain the most important words for a specific document
    """
    words_found = []
    for d in doc.split():
        for t in tfidf_words:
            if d == t[0]:
                if d not in words_found:
                    words_found.append(d)
    return words_found

"""
Similarity measures
"""
import math

def dot_product(v1, v2):
    return sum(map(lambda x: x[0] * x[1], zip(v1, v2)))

def cosine_measure(v1, v2):
    prod = dot_product(v1, v2)
    len1 = math.sqrt(dot_product(v1, v1))
    len2 = math.sqrt(dot_product(v2, v2))
    return prod / (len1 * len2)
