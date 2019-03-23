from keras.layers    import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers    import Bidirectional, GlobalMaxPool1D
from keras.models    import Model
from keras.callbacks import ModelCheckpoint
from keras           import initializers, regularizers, constraints, optimizers, layers
from lib.model.utils import to_input

from lib.model.metrics.metrics_plotter import MetricsPlotter

class ToxicCommentPredictor:
    def __init__(self, config, embedding_matrix, validation_data, epochs=2, batch_size=32, lr=0.0001, dropout=0.2, recurrent_dropout=0.05, filepath='best_weights.hdf5'):
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__validation_data = validation_data
        self.__filepath = filepath
        
        _input = Input(shape=(config['word_embeding.columns_count'],))

        net = Embedding(
            config['word_embeding.rows_count'],
            config['word_embeding.columns_count'],
            weights=[embedding_matrix],
            trainable=True
        )(_input)

        net = Bidirectional(
            LSTM(
                config['word_embeding.columns_count'], 
                return_sequences=True, 
                dropout=dropout,
                recurrent_dropout=recurrent_dropout
            )
        )(net)

        net = GlobalMaxPool1D()(net)
        net = Dense(512, activation="relu")(net)
        net = Dropout(dropout)(net)
        output = Dense(6, activation="sigmoid")(net)

        model = Model(inputs=_input, outputs=output)

        optimizer = optimizers.adam(lr=lr)

        model.compile(
            loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy']
        )

        self.__model = model

        self.__callbacks = [
            ModelCheckpoint(
                filepath=self.__filepath, 
                verbose=1,
                save_best_only=True
            ),
            MetricsPlotter(
                validation_data=self.__validation_data,
                plot_interval=600,
                evaluate_interval=600,
                batch_size=self.__batch_size
            )
        ]

    def fit(self, samples, labels):
        self.__model.fit(
            samples,
            labels,
            batch_size=self.__batch_size,
            epochs=self.__epochs,
            validation_data=self.__validation_data,
            verbose=1,
            callbacks=self.__callbacks
    )
        
    def predict_one(self, sample): return self.predict_many(to_input(sample))
    
    def predict_many(self, samples): return self.__model.predict(samples, batch_size=32)
    
    def summary(self): self.__model.summary()
        
    def load_weights(self, path='best_weights.hdf5'): self.__model.load_weights(path)
