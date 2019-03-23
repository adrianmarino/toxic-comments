import json


class PrettyJsonFormatter:
    def format(self, dict):
        return json.dumps(
            dict,
            indent=2,
            sort_keys=False,
            default=lambda x: x.__dict__
        )
