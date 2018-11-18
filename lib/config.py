import yaml


def find_keys(node, kv):
    if isinstance(node, list):
        for i in node:
            for x in find_keys(i, kv):
                yield x
    elif isinstance(node, dict):
        if kv in node:
            yield node[kv]
        for j in node.values():
            for x in find_keys(j, kv):
                yield x


class Config:
    def __init__(self, path):
        with open(path, 'r') as file:
            self.config = yaml.load(file)

    def get(self): return self.config

    def property(self, name):
        results = list(find_keys(self.config, name))

        if len(results) == 0:
            Exception(f'Not found {name} property!')
        elif len(results) > 1:
            Exception(f'More than one property with {name} name!')

        return results[0]

