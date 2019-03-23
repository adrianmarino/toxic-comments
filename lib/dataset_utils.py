import pandas as pd

def load_dataset(config):
    train = pd.read_csv(config['dataset.path.train'])
    train_comments = train[config['dataset.features'][0]].fillna("_na_").values
    train_labels = train[config['dataset.labels']].values
    
    test = pd.read_csv(config['dataset.path.test'])
    test_samples = test[config['dataset.features'][0]].fillna("_na_").values

    return train_comments, train_labels, test_samples