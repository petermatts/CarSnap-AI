from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import Compose
from pathlib import Path
from dotenv import load_dotenv
import os
from glob import glob
import yaml
from pandas import read_csv, DataFrame


class VMMRdb(Dataset):
    def __init__(self, transform: Compose = None, label_mode: str = 'brand', clean: bool = False):
        assert label_mode in ['all', 'brand', 'model', 'year']
        load_dotenv(Path(__file__).parent / "../../.env")

        self.CSV_PATH = Path(__file__).parent / "./VMMR.csv"
        self.YML_PATH = Path(__file__).parent / "./VMMR.yml"
        self.root_dir = Path(os.getenv("DATA_PATH")
                             or str(Path(__file__).parent.parent)).absolute() / "VMMR"

        self.transforms = transform
        self.label_mode = label_mode
        self.clean = clean

        self.image_files = self._index_file()

    def num_classes(self) -> int:
        with open(self.YML_PATH, "r") as f:
            specs = yaml.safe_load(f)

        if self.label_mode == "all":
            return len(specs["Years"])*len(specs["Brands"])*len(specs["Models"])
        else:
            return len(specs[f"{self.label_mode.capitalize()}s"])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[Tensor, str | tuple[str]]:
        data = self.image_files.iloc[index]
        image = decode_image(data["Image"])
        if self.label_mode == "all":
            label = (data["Year"], data["Brand"], data["Model"])
        else:
            label = data[self.label_mode.capitalize()]

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def _index_file(self) -> DataFrame:
        if self.CSV_PATH.exists() and not self.clean:
            df = read_csv(self.CSV_PATH, dtype=str).fillna("")
            if not self.YML_PATH.exists():
                self._write_yaml(df)
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

            self._write_yaml(df)

            df.to_csv(self.CSV_PATH, index=False)

        return df

    def _write_yaml(self, df: DataFrame) -> None:
        with open(self.YML_PATH, "w") as f:
            yaml.safe_dump({
                "Brands": df["Brand"].unique().tolist(),
                "Models": df["Model"].unique().tolist(),
                "Years": df["Year"].unique().tolist(),
                "Trims": df["Trim"].unique().tolist()
            }, f)

    def _parse(self, path: str) -> tuple[str]:
        p = Path(path)
        parts = p.parts[-2].lower().split('_')

        if len(parts) == 4:
            return parts
        elif len(parts) in [3, 5]:
            return parts[0], parts[1], "", parts[-1]
        else:
            print(f"Unexpected model: {parts}")


if __name__ == "__main__":
    ds = VMMRdb(label_mode="all")
    print(ds.image_files.iloc[0])
    print(ds.image_files.iloc[0]["Image"])
    print(ds.__getitem__(0)[1])
    print(ds.num_classes())
