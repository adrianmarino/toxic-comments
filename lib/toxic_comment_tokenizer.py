from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from lib.model.utils import random_index, to_input

class ToxicCommentTokenizer:
    def __init__(self, config, all_samples):
        tokenizer = Tokenizer(num_words=config['word_embeding.rows_count'])
        tokenizer.fit_on_texts(all_samples)
        self.__tokenizer = tokenizer
        self.__config = config

    def texts_to_paded_sequences(self, samples):
        tokenized_samples = self.__tokenizer.texts_to_sequences(samples)

        return pad_sequences(
            tokenized_samples, 
            maxlen=self.__config['word_embeding.columns_count']
        )

    def word_index(self): return self.__tokenizer.word_index
    
    def sequences_to_texts(self, samples): return self.__tokenizer.sequences_to_texts(samples)