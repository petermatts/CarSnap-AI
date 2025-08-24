# the main file for running ML expirements using argparse to specify model, hyperparams, datasets, etc.

import torchvision.transforms.v2 as transforms
from expirement import getDataSplit, Trainer
from args import make_args
import yaml
from pathlib import Path
from data import DMV_Cars  # nopep8
from models import VIT  # nopep8
from resources import getOptimizer  # nopep8


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

    def __str__(self) -> str:
        msg = f"" \
            f"Name: {self.name}\n" \
            f"Dataset: {self.dataset}\n" \
            f"Split:\n" \
            f"\ttrain: {self.split['train']}%\n" \
            f"\tval:   {self.split['val']}%\n" \
            f"\ttest:  {self.split['test']}%\n" \
            f"Optimizer: {self.optimizer}\n" \
            f"Learning Rate: {self.lr}\n"

        return msg.expandtabs(4)


def expirement(config: Config):
    transform = transforms.Compose(
        transforms.Resize((224, 224)),
        transforms.Normalize(*DMV_Cars.getStats())
    )
    dataset = DMV_Cars(transform=transform)
    train, val, test = getDataSplit(dataset, batch_size=1)

    model = VIT(num_classes=dataset.num_classes())

    machine = Trainer(
        model=model,
        train_loader=train,
        val_loader=val,
        optimizer=getOptimizer("Adam", model),
        num_epochs=1,
    )
    machine.learn()  # ! continue working from here


if __name__ == "__main__":
    exp_args = make_args()

    config = Config(exp_args.config)
    print(config)

    expirement(config)
