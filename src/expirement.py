# the main file for running ML expirements using argparse to specify model, hyperparams, datasets, etc.
from torch import device, cuda
from torch.nn import Module
from torch.optim import Optimizer
import torchvision.transforms.v2 as transforms
from expirement import getDataSplit, Trainer
from args import make_args
import yaml
from pathlib import Path
import logging
from data.dmv import DMV_Cars  # nopep8
from data.vmmr import VMMRdb  # nopep8
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
        self.params = None
        self.optimizer = None
        self.split = None

        self._load()
        self._none_check()

        # todo throw error/warning if some config params arent set

    def _load(self) -> None:
        config: dict = None
        with open(self.path, "r") as f:
            config = yaml.safe_load(f)

        for k, v in config.items():
            setattr(self, k, v)

    def _none_check(self) -> None:
        for k, v in vars(self).items():
            if v is None:
                logging.warning(
                    f" Config variable {k} is None. This may cause issues.")

    def getDataset(self) -> DMV_Cars | VMMRdb:
        match(self.dataset):
            case "DMV":
                return DMV_Cars
            case "VMMR":
                return VMMRdb
            case _:
                raise ValueError(
                    f"Could not find implmented dataset for {self.dataset}, valid options are [DMV,VMMR]")

    def getOptim(self, model: Module) -> Optimizer:
        return getOptimizer(self.optimizer, model, **self.params)

    def __str__(self) -> str:
        msg = f"" \
            f"Name: {self.name}\n" \
            f"Dataset: {self.dataset}\n" \
            f"Split:\n" \
            f"\ttrain: {self.split['train']}%\n" \
            f"\tval:   {self.split['val']}%\n" \
            f"\ttest:  {self.split['test']}%\n" \
            f"Optimizer: {self.optimizer}\n" \
            "Params: {\n"

        for m in [f"\t{k}: {v}\n" for k, v in self.params.items()]:
            msg += m

        msg += "}"

        return msg.expandtabs(4)


def expirement(config: Config):
    dev = device("cuda" if cuda.is_available() else "cpu")
    dataset_class = config.getDataset()

    mean, stdv = dataset_class.getStats()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean, stdv)
    ])
    # dataset = DMV_Cars(transform=transform)
    dataset = dataset_class(transform=transform)
    train, val, test = getDataSplit(dataset, batch_size=16)

    model = VIT(num_classes=dataset.num_classes())
    model.to(dev)

    machine = Trainer(
        model=model,
        train_loader=train,
        val_loader=val,
        optimizer=getOptimizer("Adam", model),
        num_epochs=1,
        plot_loss=True
    )
    machine.learn()


if __name__ == "__main__":
    exp_args = make_args()

    config = Config(exp_args.config)
    print(config)

    expirement(config)
