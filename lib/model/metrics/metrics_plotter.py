import matplotlib.pyplot as plt
from IPython.display import clear_output
from keras.callbacks import Callback

from lib.model.metrics.utils import MetricMeterBuilder


class MetricsPlotter(Callback):
    def __init__(self, validation_generator=None, validation_data=None, plot_interval=10, evaluate_interval=50, batch_size=32):
        super().__init__()
        self.plot_interval = plot_interval
        self.evaluate_interval = evaluate_interval
        self.validation_generator = validation_generator
        self.validation_data = validation_data
        self.batch_size = batch_size
        self.i = 0
        self.val_bach_index = 0
        self.x = []
        self.logs = []
        self.metrics_values = {}
        self.val_metrics_values = {}

    def on_train_begin(self, logs={}):
        for metric in self.model.metrics_names:
            self.metrics_values[metric] = []
            self.val_metrics_values[metric] = []

    def on_batch_end(self, batch, logs={}):
        if batch % self.plot_interval == 0 and len(self.logs) > 1:
            clear_output(wait=True)
            f, axes = plt.subplots(1, len(self.model.metrics_names), figsize=(30, 8))

            index = 0
            for metric in self.model.metrics_names:
                axes[index].plot(self.x, self.metrics_values[metric], label=metric)
                axes[index].plot(self.x, self.val_metrics_values[metric], label=f'val_{metric}')
                axes[index].legend()
                index += 1

            plt.show()

        if batch % self.evaluate_interval == 0:
            self.i += 1
            self.logs.append(logs)
            self.x.append(self.i)

            val_features, val_labels = self.get_validation_data()
            score = self.model.evaluate(val_features, val_labels, batch_size=self.batch_size, verbose=1)

            index = 0
            output = []
            meter_builder = MetricMeterBuilder(self.val_metrics_values)
            for metric in self.model.metrics_names:
                self.metrics_values[metric].append(logs.get(metric))
                self.val_metrics_values[metric].append(score[index])
                output.append(meter_builder.build(metric))
                index += 1

            print('\nValidation:')
            for line in output:
                print(f'  - {line}')

        if self.val_bach_index < len(self.get_validation_data()):
            self.val_bach_index += 1
        else:
            self.val_bach_index = 0

    def validation_data_len(self):
        return len(self.validation_generator) if self.validation_data is None else len(self.validation_data[0])

    def get_validation_data(self):
        if self.validation_data is None:
            return self.validation_generator[self.val_bach_index]

        return self.validation_data[0], self.validation_data[1]
