from torch.utils.data import Dataset
# from torchvision.io import read_image
from torchvision.transforms import Compose
from pathlib import Path
from dotenv import load_dotenv
import os
from glob import glob
import yaml
from pandas import read_csv, DataFrame


class VMMRdb(Dataset):
    def __init__(self, transform: Compose = None):
        # self.INDEX_PATH = Path("./VMMR.yml")
        self.INDEX_PATH = Path("./VMMR.csv")
        load_dotenv(Path(__file__).parent / "../../.env")
        self.root_dir = Path(os.getenv("DATA_PATH") or ".").absolute() / "VMMR"
        self.transforms = transform
        self.image_files = self._index_file()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int):
        pass  # todo

    def _index_file(self) -> DataFrame:
        if self.INDEX_PATH.exists():
            # with open(self.INDEX_PATH, "r") as f:
            #     return yaml.safe_load(self.INDEX_PATH)
            return read_csv(self.INDEX_PATH)
        else:
            paths = glob("../VMMR/**/*.jpg", recursive=True)
            brand, model, year = list(
                zip(*(map(lambda x: self._parse(x), paths))))

            paths = list(map(lambda x: Path(x).absolute().resolve(), paths))

            df = DataFrame({
                "Image": paths,
                "Brand": brand,
                "Model": model,
                "Year": year
            })
            df.to_csv(self.INDEX_PATH, index=False)
            return df
            # with open(self.INDEX_PATH, "w") as f:
            #     yaml.safe_dump(paths)
            # return paths

    def _parse(self, path: str) -> tuple[str]:
        # todo this still needs some polishing
        p = Path(path)
        parts = p.parts[-2].split('_')
        brand = parts[0].title() if len(
            parts[0].replace('-', '')) > 3 else parts[0].upper()
        model = parts[1].title() if len(
            parts[1].replace('-', '')) > 3 else parts[1].upper()
        year = parts[2]
        return brand, model, year


if __name__ == "__main__":
    ds = VMMRdb()
