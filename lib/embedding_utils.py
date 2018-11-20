from numpy import array
from numpy import asarray
from numpy import zeros

def get_vector_by_word_from(file, words):
    vector_by_word = dict()
    for line in file:
        values = line.split()
        word = values[0]
        if(word in words):
            vector_by_word[word] = asarray(values[1:], dtype='float32')
    return vector_by_word

def build_embedding_matrix(vector_by_word):
    embedding_vectors = zeros((len(vector_by_word) + 1, value_size(vector_by_word)))
    for index, embedding_vector in enumerate(list(vector_by_word.values())):
        if embedding_vector is not None:
            embedding_vectors[index] = embedding_vector
    return embedding_vectors

def value_size(dictionary):
    values = list(dictionary.values())
    return len(values[0])

def build_word_embedding(words, embedding_file):
    file = open(embedding_file)
    vector_by_word = get_vector_by_word_from(file, words)
    file.close()
    print(vector_by_word.keys())
    return build_embedding_matrix(vector_by_word)