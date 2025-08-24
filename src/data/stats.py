import argparse
from tqdm import tqdm
from pathlib import Path
from enum import StrEnum
from pandas import read_csv
from torchvision.io import decode_image
import torch


class DatasetNames(StrEnum):
    STANFORD = "STANFORD"
    DMV = "DMV"
    VMMR = "VMMR"
    COMPCARS = "COMPCARS"
    VEHICLEID = "VEHICLEID"


def getImageStats(dataset: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset, writes the key dataset statistics
    """

    csv_path = Path(__file__).parent / f'{dataset}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path.absolute()}")

    df = read_csv(csv_path, dtype=str)
    image_paths = list(map(lambda e: Path(e), df["Image"]))

    M = 0
    V = 0

    for p in tqdm(image_paths):
        tensor = decode_image(p).float()
        v, m = torch.std_mean(tensor, dim=[1, 2])

        M += m
        V += v

    M /= len(image_paths)
    V /= len(image_paths)

    torch.save({"mean": M, "stdv": V}, Path(f"./classes/{dataset}.pt"))

    return M, V


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ds = parser.add_mutually_exclusive_group(required=True)

    ds.add_argument("--stanford", action="store_const", dest="ds",
                    const=DatasetNames.STANFORD.value, help="Downloads the Stanford Cars dataset")
    ds.add_argument("--dmv", action="store_const", dest="ds",
                    const=DatasetNames.DMV.value, help="Downloads the DMV Cars dataset")
    ds.add_argument("--vmmr", action="store_const", dest="ds",
                    const=DatasetNames.VMMR.value, help="Downloads the VMMR dataset")
    ds.add_argument("--compcars", action="store_const", dest="ds",
                    const=DatasetNames.COMPCARS.value, help="Downloads the CompCars dataset")

    args = parser.parse_args()

    getImageStats(args.ds)
