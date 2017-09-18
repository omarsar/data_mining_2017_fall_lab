import csv
import re
import pickle

'''
Provides helper functions for obtaining statistics from LIWC_2007

Notations:
d - document
D - documents
V - vowels
w - word
W - words
l - letter
'''

class LIWC(object):
    '''
    Usage:
    analysis_1 = foo.LIWC("testing this thing with the change of getting things thirsty")
    print(analysis_1.get_liwc_stats("1"))
    '''
    def __init__(self, D):
        self.D = D
        self.liwc_dict = pickle.load(open("data/liwc_pickle/liwc.p","rb"))
        self.liwc_cats = pickle.load(open("data/liwc_pickle/liwc_cat.p", "rb"))

    def get_liwc_categories(self):
        '''
        Usage: List all the liwc categories
        '''
        liwc_cats = self.liwc_cats
        return liwc_cats

    def get_category_terms(self, category):
        '''
        Usage: Returnt the actual list of terms in a liwc category
        '''
        return self.liwc_dict[category]

    def get_liwc_terms(self, category_type):
        '''
        Usage: return the liwc terms for a set of documents D, given category_type/s
        analysis_1 = liwc_helpers.LIWC(D)
        print(analysis_1.get_liwc_stats(["1","2"]))
        '''
        counter = 0
        D = self.D
        W = []

        for d in D:
            for w in d.split():
                for cat in category_type:
                    if len(re.compile(r"^"+'$|'.join(self.liwc_dict[cat])).findall(w)) >= 1:
                        W.append(w)
                        counter+=1

        return [W, counter]

    def word_count(self, word_list):
        '''
        Usage: get a word count of the LIWC words for some categories
        TODO: could be a static method
        '''
        W = {}

        for w in word_list:
            if w not in W:
                W[w] = 0
            W[w]+=1
        return W

#############################################################################
'''
Other Utility Functions
TODO: refactor to standard notation (w, W, D, d)
'''
#############################################################################
def create_liwc_categories():
    '''
    Usage: create individual dictionaries for LIWC categories
    '''
    psych_words= csv.reader(open("../data/liwc_2007/LIWC2007cats.dic"),delimiter ="\t" )
    category_list = {}
    count = 0
    my_dict = {}

    for word in psych_words:
        for w in word[1:]:
            if w != "":
                if w not in category_list:
                    category_list[w] = []
                count+=1
                category_list[w].append(word[0])
        my_dict[word[0]] = count
        count = 0
    return category_list

def store_liwc_categories():
    '''
    Usage: process the categories for the LIWC dictionary (all categories)
    '''
    cat_list = {}
    liwc_cats= csv.reader(open("data/liwc_pickle/liwc_cats.csv"),delimiter ="," )
    for cat in liwc_cats:
        if cat[0] not in cat_list:
            cat_list[cat[0]] = []
        cat_list[cat[0]].append(cat[1])
    return cat_list
