from numpy import asarray
import numpy as np


def build_embedding_vectors_indexed_by_word(embedding_file, words):
    file = open(embedding_file)
    vector_by_word = dict()
    for line in file:
        values = line.strip().split()
        word = values[0]
        if(word in words): vector_by_word[word] = asarray(values[1:], dtype='float32')
    file.close()
    return vector_by_word


def calculate_mean_and_std(embedding_vectors):
    all_embs = np.stack(embedding_vectors)
    return all_embs.mean(), all_embs.std()    


def create_embedding_matrix(
    embedding_vectors,
    rows_count,
    columns_count,
    words_count
):
    emb_mean,emb_std = calculate_mean_and_std(embedding_vectors)
    shape = (min(rows_count, words_count), columns_count)
    return np.random.normal(emb_mean, emb_std, shape)


def build_embedding_matrix(
    embedding_file,
    rows_count,
    columns_count,
    word_index
):
    embedding_by_word = build_embedding_vectors_indexed_by_word(embedding_file, word_index.keys())

    embedding_matrix = create_embedding_matrix(
        embedding_by_word.values(),
        rows_count,
        columns_count,
        len(word_index)
    )

    for word, index in word_index.items():
        if index >= rows_count: continue

        embedding_vector = embedding_by_word.get(word)

        if embedding_vector is not None: embedding_matrix[index] = embedding_vector

    return embedding_matrix