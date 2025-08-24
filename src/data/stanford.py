from torch.utils.data import Dataset  # , DataLoader
from datasets import load_dataset  # , Dataset as DS
from pathlib import Path

# https://huggingface.co/docs/datasets/en/use_with_pytorch


# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#:~:text=PyTorch%20provides%20two%20data%20primitives,easy%20access%20to%20the%20samples.

# todo figure this out

STANFORD = load_dataset(Path("../STANFORD"))

STANFORD.set_format("torch")
