from torch.utils.data import Dataset
# from torchvision.io import read_image
from torchvision.transforms import Compose
from pathlib import Path
from dotenv import load_dotenv
import os
from glob import glob
import yaml


class VMMRdb(Dataset):
    def __init__(self, transform: Compose = None):
        self.INDEX_PATH = Path("./VMMR.yml")
        load_dotenv(Path(__file__).parent / "../../.env")
        self.root_dir = Path(os.getenv("DATA_PATH") or ".").absolute() / "VMMR"
        self.transforms = transform
        self.image_files = self._index_file()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int):
        pass  # todo

    def _index_file(self) -> list[str]:
        if self.INDEX_PATH.exists():
            with open(self.INDEX_PATH, "r") as f:
                return yaml.safe_load(self.INDEX_PATH)
        else:
            paths = glob("./VMMR/**/*.jpg", recursive=True)
            with open(self.INDEX_PATH, "w") as f:
                yaml.safe_dump(paths)
            return paths
