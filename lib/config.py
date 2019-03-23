import re
from functools import reduce

import yaml

from lib.pretty_json_formatter import PrettyJsonFormatter


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


def replace_values(yaml_file):
    def _get(dict, list):
        return reduce(lambda d, k: d[k], list, dict)

    def _replace(obj):
        for k, v in obj.items():
            if isinstance(v, dict):
                _replace(v)
            if isinstance(v, str):
                match = re.match(r'\${(.*)}', v)
                if match:
                    reference = match.group(1).split('.')
                    replace = _get(yaml_file, reference)
                    obj[k] = replace

    _replace(yaml_file)
    return yaml_file


class Config:
    def __init__(self, path):
        with open(path, 'r') as file:
            self.dict = yaml.load(file)
            self.dict = replace_values(self.dict)

    def __getitem__(self, property_name):
        if "." in property_name:
            value = self.dict
            for key in property_name.split('.'):
                value = value[key]
                if value is None:
                    raise Exception(f'Config: Not found value for {key} property of {property_name} path!')
            return value

        return self.dict[property_name]

    def property(self, name):
        results = list(find_keys(self.dict, name))

        if len(results) == 0:
            Exception(f'Not found {name} property!')
        elif len(results) > 1:
            Exception(f'More than one property with {name} name!')

        return results[0]
