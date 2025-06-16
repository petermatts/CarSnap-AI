import argparse
from enum import StrEnum


class DatasetNames(StrEnum):
    STANFORD = "STANFORD"
    DMV = "DMV"
    VMMR = "VMMR"


def update_git_ignore() -> None:
    pass # todo

# todo, write functionality to fetch and download the datasets from the dataset papers in ../docs/getRefs.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
