from argparse import ArgumentParser, Namespace
from enum import StrEnum
from rich_argparse import ArgumentDefaultsRichHelpFormatter


class DatasetArgs(StrEnum):
    VMMR = "vmmr"
    DMV = "dmv"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

    @staticmethod
    def argparse(s: str) -> str:
        def parse():
            try:
                return DatasetArgs[DatasetArgs(s).name].value
            except:
                raise ValueError(
                    f"Invalid dataset: {s}. Choose from: {[e.value for e in DatasetArgs]}")
        return parse


def make_args() -> Namespace:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsRichHelpFormatter)

    # parser.add_argument("--dataset", type=DatasetArgs.argparse(),
    #                     choices=list(DatasetArgs), help="The dataset to run the expirement on")
    parser.add_argument("--config", type=str, default="default",
                        help="name of config yaml file (located in the config folder)")

    return parser.parse_args()


if __name__ == "__main__":
    print(make_args())
