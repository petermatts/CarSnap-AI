from torch import tensor, Tensor, load, float32
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import Compose
from pathlib import Path
from dotenv import load_dotenv
import os
from glob import glob
import yaml
from pandas import read_csv, DataFrame
if __package__ or "." in __name__:
    from .stats import getImageStats
else:
    from stats import getImageStats


class VMMRdb(Dataset):
    def __init__(self, transform: Compose = None, clean: bool = False):
        load_dotenv(Path(__file__).parent / "../../.env")

        self.CSV_PATH = Path(__file__).parent / "./VMMR.csv"
        self.YML_PATH = Path(__file__).parent / "./VMMR.yml"
        self.root_dir = Path(os.getenv("DATA_PATH")
                             or str(Path(__file__).parent.parent)).absolute() / "VMMR"

        self.transforms = transform
        self.clean = clean

        self.image_files, self.classes = self._index_file()

    def num_classes(self) -> int:
        return (
            len(self.classes["Brands"]),
            len(self.classes["Models"]),
            len(self.classes["Years"]),
            len(self.classes["Trims"])
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[Tensor, tuple[Tensor, ...]]:
        data = self.image_files.iloc[index]
        image = decode_image(data["Image"]).to(float32)

        label = (
            tensor(self.classes["Brands"][data["Brand"]]),
            tensor(self.classes["Models"][data["Model"]]),
            tensor(self.classes["Years"][data["Year"]]),
            tensor(self.classes["Trims"][data["Trim"]])
        )

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def _index_file(self) -> tuple[DataFrame, dict]:
        if self.CSV_PATH.exists() and not self.clean:
            df = read_csv(self.CSV_PATH, dtype=str).fillna("")
            if not self.YML_PATH.exists():
                data = self._write_yml(df)
            else:
                with open(self.YML_PATH, "r") as f:
                    data = yaml.safe_load(f)
        else:
            paths = glob(f"{self.root_dir}/**/*.jpg", recursive=True)
            brand, model, trim, year = list(
                zip(*(map(lambda x: self._parse(x), paths))))
            paths = list(map(lambda x: Path(x).absolute().resolve(), paths))

            df = DataFrame({
                "Image": paths,
                "Brand": brand,
                "Model": model,
                "Year": year,
                "Trim": trim
            }, dtype=str)

            data = self._write_yml(df)

            df.to_csv(self.CSV_PATH, index=False)

        return df, data

    def _write_yml(self, df: DataFrame) -> dict:
        data = {
            "Brands": {label: idx for idx, label in enumerate(df["Brand"].unique().tolist())},
            "Models": {label: idx for idx, label in enumerate(df["Model"].unique().tolist())},
            "Years": {label: idx for idx, label in enumerate(df["Year"].unique().tolist())},
            "Trims": {label: idx for idx, label in enumerate(df["Trim"].unique().tolist())}
        }

        with open(self.YML_PATH, "w") as f:
            yaml.dump(data, f)

        return data

    def _parse(self, path: str) -> tuple[str]:
        p = Path(path)
        parts = p.parts[-2].lower().split('_')

        if len(parts) == 4:
            return parts
        elif len(parts) in [3, 5]:
            return parts[0], parts[1], "", parts[-1]
        else:
            print(f"Unexpected model: {parts}")

    @staticmethod
    def getStats() -> tuple[Tensor, Tensor]:
        """Returns the mean and standard deviation of the dataset"""
        pt_path = Path(__file__).parent / "VMMR.pt"
        if pt_path.exists():
            stats = load(pt_path)
        else:
            stats = getImageStats("DMV")

        return stats['mean'], stats['stdv']


if __name__ == "__main__":
    # ds = VMMRdb(clean=True)
    ds = VMMRdb()
    print(ds.image_files.iloc[0])
    print(ds.image_files.iloc[0]["Image"])
    print(ds.num_classes())
    item, targets = ds.__getitem__(0)
    # print(item)
    print(targets)

    print(VMMRdb.getStats())
