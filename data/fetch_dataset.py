import argparse
import os
import requests
from tqdm import tqdm
import tarfile
import zipfile
from enum import StrEnum

from pathlib import Path
from yarl import URL

from dotenv import load_dotenv
from datasets import load_dataset


load_dotenv(Path(__file__).parent / "../.env")
DATA_DIR = Path()
path = os.getenv("DATA_PATH")
if path is None:
    print(
        f"No DATA_PATH env variable detected. Datasets will be saved to {DATA_DIR.absolute()}")
else:
    print(f"Found DATA_PATH env. Datasets will be saved to {path}")
    DATA_DIR = Path(path)
del path


class DatasetNames(StrEnum):
    STANFORD = "STANFORD"
    DMV = "DMV"
    VMMR = "VMMR"


def download(url, dest_path, chunk_size=1024):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as file, tqdm(
        desc=os.path.basename(dest_path),
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))


def extract(filepath: Path, output_dir: Path):
    """
    Decompress a tar or zip archive to the specified directory.

    Args:
        filepath (Path): Path to the archive file.
        output_dir (Path): Path to the output directory.

    Raises:
        ValueError: If the file type is unsupported.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    # else:
    #     print(f"{output_dir} already exists. Skipping extraction.")

    # ZIP file
    if zipfile.is_zipfile(filepath):
        print("Unzipping... this may take a while")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted ZIP file to '{output_dir}'")

    # TAR files (including .tar.gz, .tgz, .tar.bz2, etc.)
    elif tarfile.is_tarfile(filepath):
        print("Untarring... this may take a while")
        with tarfile.open(filepath, r'r:gz') as tar_ref:
            tar_ref.extractall(output_dir)
        print(f"Extracted TAR file to '{output_dir}'")
    # else:
    #     raise ValueError(f"Unsupported archive format: '{filepath.suffix}'")


def download_and_extract_stanford_cars():
    ds = load_dataset("tanganke/stanford_cars")
    ds.save_to_disk(DATA_DIR / "STANFORD")


def download_and_extract_dmv_cars():
    dataset_dir = DATA_DIR / DatasetNames.DMV.value
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True)

    data_file = dataset_dir / "data.zip"
    if not os.path.exists(data_file):
        url = URL("https://figshare.com/ndownloader/files/34792453")
        download(url, data_file)
    else:
        print("File exists. Skipping download.")

    extract(data_file, dataset_dir)
    data_file.unlink()


def download_and_extract_vmmrdb():
    dataset_dir = DATA_DIR / DatasetNames.VMMR.value
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True)

    data_file = dataset_dir / "data.zip"
    if not os.path.exists(data_file):
        url = URL(
            "https://www.dropbox.com/scl/fi/3r3a0oz71nlvhqhu3tocf/VMMRdb.zip?rlkey=e5bb62ae88x2pzxnsggobxpe0&e=2&dl=1")
        download(url, data_file)
    else:
        print("File exists. Skipping download.")

    extract(data_file, dataset_dir)
    data_file.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ds = parser.add_mutually_exclusive_group(required=True)

    ds.add_argument("--stanford", action="store_const", dest="ds",
                    const=DatasetNames.STANFORD.value, help="Downloads the Stanford Cars dataset")
    ds.add_argument("--dmv", action="store_const", dest="ds",
                    const=DatasetNames.DMV.value, help="Downloads the DMV Cars dataset")
    ds.add_argument("--vmmr", action="store_const", dest="ds",
                    const=DatasetNames.VMMR.value, help="Downloads the VMMR dataset")

    args = parser.parse_args()

    download_map = {
        DatasetNames.STANFORD.value: download_and_extract_stanford_cars,
        DatasetNames.DMV.value: download_and_extract_dmv_cars,
        DatasetNames.VMMR.value: download_and_extract_vmmrdb
    }

    download_map[args.ds]()
