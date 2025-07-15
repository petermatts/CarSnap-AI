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


class DMV_Cars(Dataset):
    def __init__(self, transform: Compose = None, label_mode: str = 'brand'):
        assert label_mode in ['all', 'brand', 'model', 'year']
        load_dotenv(Path(__file__).parent / "../../.env")

        self.INDEX_PATH = Path("./DMV.csv")
        self.root_dir = Path(os.getenv("DATA_PATH") or "..").absolute() / "DMV"

        self.transforms = transform
        self.label_mode = label_mode

        self.image_files = self._index_file()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        data = self.image_files.iloc[index]
        image = decode_image(data["Image"])
        if self.label_mode == "all":
            label = " ".join([data["Color"], data["Year"],
                             data["Brand"], data["Model"]])
        else:
            label = data[self.label_mode.capitalize()]

        # todo use transforms
        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def _index_file(self) -> DataFrame:
        if self.INDEX_PATH.exists():
            return read_csv(self.INDEX_PATH, dtype=str)
        else:
            paths = glob(f"{self.root_dir}/**/*.jpg", recursive=True)
            brand, model, year, color = list(
                zip(*(map(lambda x: self._parse(x), paths))))
            paths = list(map(lambda x: Path(x).absolute().resolve(), paths))

            df = DataFrame({
                "Image": paths,
                "Brand": brand,
                "Model": model,
                "Year": year,
                "Color": color
            }, dtype=str)
            df.to_csv(self.INDEX_PATH, index=False)
            return df

    def _parse(self, path: str) -> tuple[str]:
        p = Path(path)
        parts = p.name.split('$$')
        brand = parts[0]
        model = parts[1]
        year = parts[2]
        color = parts[3]

        return brand, model, year, color


if __name__ == "__main__":
    ds = DMV_Cars(label_mode="all")
    print(ds.image_files.iloc[0])
    print(ds.image_files.iloc[0]["Image"])
    print(ds.__getitem__(0)[1])
