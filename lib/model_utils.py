import numpy as np

def to_input(sample): return np.array([sample])

def random_index(samples): return np.random.randint(0, high=len(samples))

class ToxicCommentPredictionPrinter:
    def __init__(self, model, tokenizer, config, samples):
        self.__model = model
        self.__tokenizer = tokenizer
        self.__config = config
        self.__samples = samples

    def __to_categories(self, prediction, config, min_percent):    
        all_categories = config['dataset']['labels']

        categories = []
        for index in range(len(prediction)):
            if prediction[index] > min_percent: 
                percent = prediction[index] * 100
                categories.append(f'{all_categories[index]}({percent:0.3}%)')

        return categories
    
    def __to_text(self, sample): return self.__tokenizer.sequences_to_texts(to_input(sample))

    def print_any(self, min_percent=0.1):
        index = random_index(self.__samples)
        self.print_by_index(index, min_percent)

    def print_by_index(self, index, min_percent=0.1):
        sample = self.__samples[index]

        text = self.__to_text(sample)[0]
        if len(text) > 0:
            text = f'{self.__to_text(sample)[0].capitalize()}.'
        else:
            text = 'Empty'

        print(f'Comment (idx:{index}) => {text}')

        prediction = self.__model.predict_one(sample)
        print("\nCategories => ", self.__to_categories(prediction[0], self.__config, min_percent))
