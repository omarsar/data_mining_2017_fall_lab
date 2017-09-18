'''
Notations:
d - document
D - documents
v - vowel
V - vowels
w - word
W - words
l - letter
'''

def get_vowels_basic_info(d):
    '''
    Usage: Return basic info of vowel information in a document
    '''
    vowel_list = ['a', 'e', 'i', 'o', 'u']
    V = {}
    counter = 0
    window = 0
    positions = []

    for w in d.split():
        for l in list(w.lower()):
            window+=1
            if l in vowel_list:
                positions.append(window)
                counter+=1
                if l not in V:
                    V[l] = 0
                V[l]+=1
    return {"vowel_frequency":V, "total_vowels": counter, "positions": positions}

def get_vowels_average_distance(positions):
    '''
    Usage: Return the distances of vowels in a document
    '''
    distances = []
    size = len(positions)
    for index, item in enumerate(positions):
        if index < size-1:
            distances.append(positions[index+1] - positions[index])
    return distances
