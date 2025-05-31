"""
Script to make a stable requirements.txt for devices that may not support CUDA and a separate
requirements_cuda.txt file for devices that support CUDA.
"""

import re
import sys
import subprocess


reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode("utf-8").split("\r\n")
# print(reqs)

cuda_reqs = list(filter(lambda x: re.search(r"\+cu\d{3}", x), reqs))
no_cuda_reqs = list(map(lambda x: re.sub(r"\+cu\d{3}", "", x), cuda_reqs))
base_reqs = list(filter(lambda x: re.search(r"\+cu\d{3}", x) is None, reqs))

if len(cuda_reqs) != 0:
    cuda_ver = re.search(r"cu\d{3}", cuda_reqs[0]).group()
    # cuda_link = "--find-links https://download.pytorch.org/whl/torch_stable.html"
    cuda_link = f"--extra-index-url https://download.pytorch.org/whl/{cuda_ver}"
    cuda_reqs = [cuda_link] + cuda_reqs

    with open("requirements_cuda.txt", "w") as f:
        f.write("\n".join(base_reqs))
        f.write("".join(list(map(lambda x: f"{x}\n", cuda_reqs))))

with open("requirements.txt", "w") as f:
    f.write("".join(list(map(lambda x: f"{x}\n", list(map(lambda x: re.sub(r"\+cu\d{3}", "", x), reqs))))))
    