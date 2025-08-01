from argparse import ArgumentParser, Namespace
import yaml
from enum import StrEnum
from rich_argparse import ArgumentDefaultsRichHelpFormatter


class DatasetArgs(StrEnum):
    # todo revise construction
    VMMR = "vmmr"
    DMV = "dmv"

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(s: str):
        try:
            # return DatasetArgs(s.lower()).value
            return DatasetArgs[s.lower()].value
        except:
            raise ValueError(
                f"Invalid dataset: {s}. Choose from: {[e.value for e in DatasetArgs]}")


def make_args() -> Namespace:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsRichHelpFormatter)

    # dataset args
    datasets = parser.add_argument_group()
    datasets.add_argument("--dataset", type=DatasetArgs.from_string,
                          choices=list(DatasetArgs), help="The dataset to run the expirement on")
    split = datasets.add_mutually_exclusive_group()
    split.add_argument("--split", nargs=3, default=[
                       80, 10, 10], help="train/val/test split in percent. Example: --split 80 10 10. Must sum to 100.")
    split.add_argument("--nsplit", type=int, nargs=3,
                       help="train/val/test split in # of elements. Must sum to size of dataset.")

    # training args
    # training = parser.add_argument_group()

    # saving args

    return parser.parse_args()


if __name__ == "__main__":
    print(make_args())
