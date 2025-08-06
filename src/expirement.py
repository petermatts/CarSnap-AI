# the main file for running ML expirements using argparse to specify model, hyperparams, datasets, etc.

from pathlib import Path
import yaml
from args import make_args


class Config:
    def __init__(self, name: str):
        self.config_name = name.replace(".yaml", "")
        self.path = (Path(__file__).parent /
                     f"../config/{self.config_name}.yaml").resolve()
        if not self.path.exists():
            raise FileNotFoundError(
                f"Could not find config file {self.config_name}.yaml in {self.path.parent}")

        self.dataset = None
        self.name = None
        self.lr = None
        self.optimizer = None
        self.split = None

        self._load()

        # todo throw error/warning if some config params arent set

    def _load(self):
        config: dict = None
        with open(self.path, "r") as f:
            config = yaml.safe_load(f)

        for k, v in config.items():
            setattr(self, k, v)


def experiment():
    pass  # todo


if __name__ == "__main__":
    exp_args = make_args()

    config = Config(exp_args.config)
