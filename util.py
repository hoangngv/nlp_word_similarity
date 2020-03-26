from sklearn.metrics.pairwise import cosine_similarity as cs
import numpy as np
import os

# Load embeddings into dict
def load_word_dict():
    word_dict = {}
    with open('./word2vec/W2V_150.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines()[2:]:
            word, vector = line.split(' ', maxsplit=1)
            word_dict[word] = np.fromstring(vector, dtype=np.float, sep=' ').reshape(1, -1)
    return word_dict

# Calculate cosine similarity
def cosine_similarity(w1, w2, word_dict):
    if w1 not in word_dict or w2 not in word_dict:
        return 0
    return cs(word_dict[w1], word_dict[w2])[0][0]

# Load ViSim400
def load_visim400(word_dict):
    visim_results = []
    sim1 = []
    sim2 = []
    with open('./datasets/ViSim-400/Visim-400.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            visim_word = line.split()
            visim_results.append(cosine_similarity(visim_word[0], visim_word[1], word_dict))
            sim1.append(float(visim_word[3]))
            sim2.append(float(visim_word[4]))
    return visim_results, sim1, sim2

# K most-similar words
def most_similar_words(word, k:int, word_dict):
    temporary_dict = {}
    temporary_list = []
    if word not in word_dict.keys():
        print("Your word does not exist.")
    else:
        for key in word_dict.keys():
            if key != word:
                temporary_dict[key] = cosine_similarity(word, key, word_dict)
        for key, value in sorted(temporary_dict.items(), key=lambda x : x[1], reverse=True):
            temporary_list.append(key)
        print('The ', str(k), 'most-similar words to "', word, '" are: ')
        for i in range(0, k):
            print(temporary_list[i])

# Read training Antonym_Synonym dataset
def read_training_data(word_dict):
    X_train,y_train = [], []
    with open('./antonym-synonym set/Antonym_vietnamese.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            w1, w2 = line.split(' ', 1)
            w1, w2 = w1.strip(), w2.strip()
            try:
                X_train.append((word_dict[w1] - word_dict[w2])[0,:])
                y_train.append(0)
            except KeyError:
                pass

    with open('./antonym-synonym set/Synonym_vietnamese.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            w1, w2 = line.split(' ', 1)
            w1, w2 = w1.strip(), w2.strip()
            try:
                X_train.append((word_dict[w1] - word_dict[w2])[0,:])
                y_train.append(1)
            except KeyError:
                pass
    return X_train, y_train

# Read testing Antonym_Synonym dataset
def read_testing_data(word_dict):
    X_test, y_test = [], []
    file_paths = ['./datasets/Vicon-400/400_noun_pairs.txt', './datasets/Vicon-400/400_verb_pairs.txt', './datasets/Vicon-400/600_adj_pairs.txt']
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                w1, w2, rls = line.split('\t', 2)
                w1, w2, rls = w1.strip(), w2.strip(), rls.strip()
                try:
                    X_test.append((word_dict[w1] - word_dict[w2])[0,:])
                    y_test.append(0 if rls=='ANT' else 1)
                except KeyError:
                    pass
    return X_test, y_test
