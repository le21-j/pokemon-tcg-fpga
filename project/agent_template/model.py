"""A runnable weighted scoring demo. These weights are hand-set, not trained."""

import json


class DemoModel:
    def __init__(self, weights):
        self.weights = weights

    @classmethod
    def load(cls, path):
        return cls(json.loads(path.read_text())["option_type_scores"])

    def score_options(self, observation):
        return [self.weights.get(str(option["type"]), 0.0)
                for option in observation["select"]["option"]]
