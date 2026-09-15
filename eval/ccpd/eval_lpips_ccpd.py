import os
import random
import numpy as np
from PIL import Image
import torch
import lpips
import torchvision.transforms as T
from tqdm import tqdm
import cv2

# ============================================================
# LPIPS model
# ============================================================
loss_fn = lpips.LPIPS(net='alex').cuda()

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def load_img(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).cuda()

# ============================================================
# CCPD parsing
# ============================================================
def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    x_sorted   = pts[np.argsort(pts[:, 0])]
    left_most  = x_sorted[:2][np.argsort(x_sorted[:2, 1])]
    right_most = x_sorted[2:][np.argsort(x_sorted[2:, 1])]
    tl, bl = left_most
    tr, br = right_most
    return np.array([tl, tr, br, bl], dtype=np.float32)

def parse_ccpd_quad(filename):
    parts = filename.split('-')
    quad_str = parts[3]
    pts = quad_str.split('_')
    quad = []
    for p in pts:
        x, y = map(int, p.split('&'))
        quad.append([x, y])
    quad = np.array(quad, dtype=np.float32)

    # 用 min(x), min(y) 轉換到 crop 圖座標（跟可視化腳本一致）
    x_min = np.min(quad[:, 0])
    y_min = np.min(quad[:, 1])
    quad[:, 0] -= x_min
    quad[:, 1] -= y_min
    return np.array(quad, dtype=np.float32)

def generate_ccpd_mask(
    filename,
    H,
    W,
    orig_H,
    orig_W,
    rng
):
    """
    Generate CCPD binary mask.

    與 inference 完全一致：

        1. Canonical plate (440x140)
        2. Character boundaries
        3. Random k,s
        4. Perspective projection
        5. Binary mask

    Returns:
        mask : [1,1,H,W]
    """

    quad = parse_ccpd_quad(filename)

    if quad is None:
        return None

    quad = order_points(quad)

    # ========================================================
    # Canonical plate
    # ========================================================

    src_pts = np.array(
        [
            [0, 0],
            [440, 0],
            [440, 140],
            [0, 140]
        ],
        dtype=np.float32
    )

    M = cv2.getPerspectiveTransform(
        src_pts,
        quad.astype(np.float32)
    )

    # ========================================================
    # Last-5 character boundaries
    # ========================================================

    char_boundaries = [
        140,
        197,
        254,
        311,
        368,
        425
    ]

    pad_y = 12

    # ========================================================
    # Random mask selection
    # ========================================================

    k = rng.randint(1, 5)

    s = rng.randint(
        0,
        5 - k
    )

    start_x = char_boundaries[s]
    end_x = char_boundaries[s + k]

    # ========================================================
    # Canonical mask polygon
    # ========================================================

    mask_canon = np.array(
        [
            [start_x, pad_y],
            [end_x,   pad_y],
            [end_x,   140 - pad_y],
            [start_x, 140 - pad_y]
        ],
        dtype=np.float32
    )

    # ========================================================
    # Project back to image plane
    # ========================================================

    mask_quad = cv2.perspectiveTransform(
        mask_canon.reshape(-1, 1, 2),
        M
    ).reshape(4, 2)

    # ========================================================
    # Scale to network size
    # ========================================================

    scale_x = W / orig_W
    scale_y = H / orig_H

    mask_quad[:, 0] *= scale_x
    mask_quad[:, 1] *= scale_y

    # ========================================================
    # Rasterize binary mask
    # ========================================================

    mask = np.zeros(
        (H, W),
        dtype=np.float32
    )

    cv2.fillPoly(
        mask,
        [mask_quad.astype(np.int32)],
        1.0
    )

    # ========================================================
    # Tensor
    # ========================================================

    mask = (
        torch.from_numpy(mask)
        .unsqueeze(0)
        .unsqueeze(0)
        .cuda()
    )

    return mask

# ============================================================
# main evaluation
# ============================================================
def compute_lpips(image_dir, output_dir):
    full_scores = []
    mask_scores = []

    names = sorted([
        n for n in os.listdir(output_dir)
        if n.endswith(".png") and "_debug" not in n
    ])

    # ★ 全域一個 rng，每張圖推進狀態（跟 inference 腳本一致）
    sample_rng = random.Random(42)

    for name in tqdm(names):
        gt_path   = os.path.join(image_dir, name.replace(".png", ".jpg"))
        pred_path = os.path.join(output_dir, name)

        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            # rng 仍要推進，保持 index 對齊
            sample_rng.randint(1, 5)
            sample_rng.randint(0, 4)
            continue

        # ★ 讀取原始 crop 圖尺寸，用於座標 scale
        orig_img = Image.open(gt_path)
        orig_W, orig_H = orig_img.size

        gt   = load_img(gt_path)
        pred = load_img(pred_path)

        mask = generate_ccpd_mask(name, 512, 512, orig_H, orig_W, sample_rng)
        mask_3c = mask.repeat(1, 3, 1, 1)

        # Full LPIPS
        full_lpips = loss_fn(pred, gt)
        full_scores.append(full_lpips.item())

        # Region LPIPS
        if mask.sum() > 0:
            pred_m = pred * mask_3c
            gt_m   = gt   * mask_3c
            mask_lpips = loss_fn(pred_m, gt_m)
            mask_scores.append(mask_lpips.item())

    print("\n===== LPIPS Results =====")
    print(f"N (full)   : {len(full_scores)}")
    print(f"N (region) : {len(mask_scores)}")
    print(f"Full LPIPS mean  : {np.mean(full_scores):.4f}")
    print(f"Region LPIPS mean: {np.mean(mask_scores):.4f}")


if __name__ == "__main__":
    image_dir  = "/media/avlab/Transcend/CCPD2019/split_ccpd/test_1000/plates"
    output_dir = "/media/avlab/76b02113-59b9-4d03-9078-45cf20e14592/0_paper/FluxText_plate/CCPD_test1000_finetune2/plates"

    compute_lpips(image_dir, output_dir)