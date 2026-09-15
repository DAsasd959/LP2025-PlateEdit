import os
import shutil
import torch
import numpy as np
from torchvision.datasets.folder import default_loader
from pytorch_fid import fid_score

gt_path = "/media/avlab/Transcend/CCPD2019/split_ccpd/test_1000/plates"
img_path = "/media/avlab/76b02113-59b9-4d03-9078-45cf20e14592/0_paper/FluxText_plate/CCPD_test1000_clear2/plates"

tmp_gt = "./tmp_fid_gt"
tmp_img = "./tmp_fid_img"

# -----------------------
# clean folders
# -----------------------
for p in [tmp_gt, tmp_img]:
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(p, exist_ok=True)

# -----------------------
# match filenames
# -----------------------
img_dict = {os.path.splitext(f)[0]: f for f in os.listdir(img_path)}
gt_dict  = {os.path.splitext(f)[0]: f for f in os.listdir(gt_path)}

common_keys = sorted(list(img_dict.keys() & gt_dict.keys()))

print("Matched pairs:", len(common_keys))

# -----------------------
# copy paired images
# -----------------------
for k in common_keys:
    shutil.copy(
        os.path.join(gt_path, gt_dict[k]),
        os.path.join(tmp_gt, gt_dict[k])
    )
    shutil.copy(
        os.path.join(img_path, img_dict[k]),
        os.path.join(tmp_img, img_dict[k])
    )

# -----------------------
# FID computation
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fid_value = fid_score.calculate_fid_given_paths(
    [tmp_gt, tmp_img],
    batch_size=1,
    device=device,
    dims=2048
)

print("FID:", fid_value)