"""
This script fetches and downloads (as a PDF) any of the reference papers
listed in the Papers dictionary below. Papers are saved to the references
directory
"""

from pathlib import Path
import requests
import os
import re

# references are a key value pair: key=name of paper, value=url to paper pdf
Papers = {
    # model papers
    "AlexNet": "https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf",
    "Deep Residual Learning for Image Recognition": "https://arxiv.org/pdf/1512.03385",
    "Scaling Vision with Sparse Mixture of Experts": "https://arxiv.org/pdf/2106.05974",
    "Sparsely-gated Mixture-of-Expert Layers for CNN Interpretability": "https://www.semanticscholar.org/reader/7947124631cf9a02aeb2980475455598e92478cc",
    "Distilling the Knowledge in a Neural Network": "https://arxiv.org/pdf/1503.02531",
    "Deep Learning Based Vehicle Make-Model Classification": "https://arxiv.org/pdf/1809.00953",
    "Enhancing Vehicle Make and Model Recognitio with 3D Attention Modules": "https://arxiv.org/pdf/2502.15398",
    "Federated learning Applications, challenges and future directions": "https://arxiv.org/pdf/2205.09513",
    "Communication-Efficient Learning of Deep Networks from Decentralized Data": "https://arxiv.org/pdf/1602.05629", # intro paper for FL
    "You Only Look Once Unified, Real-Time Object Detection": "https://arxiv.org/pdf/1506.02640",
    "A Comprhensive Review of YOLO Architectures In Computer Vision From YOLOv1 to YOLOv8 and YOLO-NAS": "https://arxiv.org/pdf/2304.00501",

    # dataset papers
    "Car-1000 A New Large Scale Fine-Grained Visual Categorization Dataset": "https://arxiv.org/pdf/2503.12385",
    "Collecting a Large-Scale Dataset of Fine-Grained Cars": "https://ai.stanford.edu/~jkrause/papers/fgvc13.pdf",
    "A Large and Diverse Dataset for Improved Vehicle Make and Model Recognition": "https://openaccess.thecvf.com/content_cvpr_2017_workshops/w9/papers/Tafazzoli_A_Large_and_CVPR_2017_paper.pdf",
    "A Large-Scale Car Dataset for Fine-Grained Categorization and Verification": "https://arxiv.org/pdf/1506.08959",
}

BASE_PATH = Path(__file__).parent / "references"
os.makedirs(BASE_PATH, exist_ok=True)

downloaded = False
for k, v in Papers.items():
    response = requests.get(v)
    temp = re.sub(r",|\s", "_", k)
    file_path = BASE_PATH / f'{temp}.pdf'

    if os.path.isfile(file_path):
        continue
    else:
        downloaded = True
        name = (k + '.pdf ' if len(k) < 64 else k[:64] + '... .pdf').ljust(72, '.')
        print(f"Getting: {name} ... ", end='')
        with open(file_path, "wb") as f:
            f.write(response.content)
            print(" Done")

if not downloaded:
    print("Refs upto date :)")
    