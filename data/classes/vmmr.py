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
    def __init__(self, transform: Compose = None, label_mode: str = 'brand'):
        assert label_mode in ['all', 'brand', 'model', 'year']
        load_dotenv(Path(__file__).parent / "../../.env")

        self.INDEX_PATH = Path("./VMMR.csv")
        self.root_dir = Path(os.getenv("DATA_PATH") or ".").absolute() / "VMMR"

        self.transforms = transform
        self.label_mode = label_mode

        self.image_files = self._index_file()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        data = self.image_files.iloc[index]
        image = decode_image(data["Image"])
        if self.label_mode == "all":
            label = " ".join([data["Year"], data["Brand"], data["Model"]])
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
            paths = glob("../VMMR/**/*.jpg", recursive=True)
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
            df.to_csv(self.INDEX_PATH, index=False)
            return df

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
