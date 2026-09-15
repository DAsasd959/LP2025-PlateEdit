import os
import cv2
import numpy as np

# === 路徑設定 ===
base_path = "/media/avlab/Transcend/Synthtext/20251206_valid1k"
img_dir = os.path.join(base_path, "i_s")
bbox_dir = os.path.join(base_path, "i_s_bbox")
out_dir = os.path.join(base_path, "i_s_rect_mask")

os.makedirs(out_dir, exist_ok=True)

# === 取得所有 PNG ===
img_list = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

for img_name in img_list:
    img_path = os.path.join(img_dir, img_name)
    txt_path = os.path.join(bbox_dir, img_name.replace(".png", ".txt"))
    out_path = os.path.join(out_dir, img_name)

    # 讀取圖片尺寸
    img = cv2.imread(img_path)
    if img is None:
        print(f"[Skip] Cannot read: {img_path}")
        continue

    h, w = img.shape[:2]

    # 建立全黑 mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # 讀 bbox txt
    if not os.path.exists(txt_path):
        print(f"[Skip] Missing bbox: {txt_path}")
        continue

    with open(txt_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    parts = first_line.split()

    # parts[0] = 字串，例如 ARG-7030
    # 後面 8 個是 bbox 座標
    coords = list(map(int, parts[1:9]))  # 取 8 個數字

    if len(coords) != 8:
        print(f"[Error] bbox 格式不正確：{txt_path}")
        continue

    # 解析四點座標
    lt = (coords[0], coords[1])
    rt = (coords[2], coords[3])
    rb = (coords[4], coords[5])
    lb = (coords[6], coords[7])

    polygon = np.array([lt, rt, rb, lb], dtype=np.int32)

    # 在 mask 上把四邊形塗白
    cv2.fillPoly(mask, [polygon], 255)

    # 儲存 mask
    cv2.imwrite(out_path, mask, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    print(f"[OK] {out_path}")
