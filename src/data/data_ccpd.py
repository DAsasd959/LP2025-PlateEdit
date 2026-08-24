import os
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2

# ============================================================
# CCPD 字典對照表 (只保留英數字)
# ============================================================
ADS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "O"]

class PlateDataset(Dataset):
    """
    CCPD 車牌部分遮蔽數據集 (Finetune 預處理專用版本)
    
    特點：
    - 從 CCPD 檔名直接解析後 5 碼文字與區域
    - 固定 per-image seed，確保離線預處理 (Preprocessing) 可完全重現
    - 即時產生 Mask 與 Glyph 供 tokenizer 使用
    """
    
    def __init__(self, data_root, split='train', image_size=512, seed=42):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.seed = seed
        
        # 載入所有 CCPD 圖片檔名
        valid_exts = {'.jpg', '.jpeg', '.png'}
        self.image_files = [f.name for f in self.data_root.iterdir() if f.suffix.lower() in valid_exts]
        self.image_files = sorted(self.image_files)
        
        print(f"[{split}] 載入 {len(self.image_files)} 張 CCPD 圖片 (Fixed Seed = {seed})")
        
        # 載入字體
        # Out of scope for this LP2025 release (only reached when PP_DATASET=ccpd);
        # kept configurable so the import does not carry a machine-specific path.
        self.font_path = os.environ.get("PP_LATIN_FONT", "fonts/TWGen7_V1.ttf")
        self.base_font = ImageFont.truetype(self.font_path, size=60)
        
        # Transform
        self.to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    # ============================================================
    # 解析 CCPD 檔名 (只取 Quad 與後 5 碼)
    # ============================================================
    def parse_ccpd_filename(self, filename):
        parts = filename.split('-')

        # 解析 quad（原圖座標）
        quad_str = parts[3]
        pts = quad_str.split('_')
        quad = []
        for p in pts:
            x, y = map(int, p.split('&'))
            quad.append([x, y])
        quad = np.array(quad, dtype=np.float32)

        # 用 min(x), min(y) 轉換到 crop 圖座標
        x_min = np.min(quad[:, 0])
        y_min = np.min(quad[:, 1])
        quad[:, 0] -= x_min
        quad[:, 1] -= y_min

        # 解析文字
        text_str = parts[4]
        indices = list(map(int, text_str.split('_')))
        last_5_text = "".join([ADS[idx] for idx in indices[2:7]])

        return quad, last_5_text

    def order_points(self, pts):
        """抗傾斜頂點排序 (左上, 右上, 右下, 左下)"""
        pts = np.array(pts, dtype=np.float32)
        x_sorted = pts[np.argsort(pts[:, 0]), :]
        left_most = x_sorted[:2, :]
        right_most = x_sorted[2:, :]

        left_most = left_most[np.argsort(left_most[:, 1]), :]
        tl, bl = left_most[0], left_most[1]

        right_most = right_most[np.argsort(right_most[:, 1]), :]
        tr, br = right_most[0], right_most[1]

        return np.array([tl, tr, br, bl], dtype=np.float32)


    def generate_dynamic_mask_and_text(self, quad, last_5_text):
        ordered_quad = self.order_points(quad)
        
        # 1. 建立一個虛擬的「標準車牌畫布」 (中國車牌標準尺寸: 440mm x 140mm)
        # 這樣我們就可以在完美的長方形上做切割，不用管原圖有多歪
        src_pts = np.array([
            [0, 0], 
            [440, 0], 
            [440, 140], 
            [0, 140]
        ], dtype=np.float32)
        
        # 2. 計算從「標準畫布」到「實際圖片傾斜座標」的透視轉換矩陣
        M = cv2.getPerspectiveTransform(src_pts, ordered_quad)
        
        # 3. 根據物理規範，精確定義後 5 碼的邊界 (單位: mm)
        # 第 1 碼: 140 ~ 197
        # 第 2 碼: 197 ~ 254
        # 第 3 碼: 254 ~ 311
        # 第 4 碼: 311 ~ 368
        # 第 5 碼: 368 ~ 425
        char_boundaries = [140, 197, 254, 311, 368, 425]
        
        # 4. 隨機決定遮蔽長度 (k=1~5) 與起始位置 (s)
        k = random.randint(1, 5)
        s = random.randint(0, 5 - k)
        
        start_x = char_boundaries[s]
        end_x = char_boundaries[s + k]
        
        # Y 軸稍微內縮，避免切到車牌金屬邊框 (總高 140mm，上下各縮 12mm)
        pad_y = 12
        start_y = pad_y
        end_y = 140 - pad_y
        
        # 5. 在「標準畫布」上畫出完美的矩形 Mask
        canon_mask = np.array([
            [start_x, start_y],
            [end_x, start_y],
            [end_x, end_y],
            [start_x, end_y]
        ], dtype=np.float32)
        
        # 6.利用矩陣將標準 Mask 反向轉換回實際圖片的傾斜座標！
        # perspectiveTransform 需要的 shape 是 (N, 1, 2)
        mask_quad = cv2.perspectiveTransform(canon_mask.reshape(-1, 1, 2), M).reshape(4, 2)
        
        target_text = last_5_text[s:s+k]

        return mask_quad, target_text, f"{s}-{s+k}"

    # ============================================================
    # 渲染 Glyph
    # ============================================================
    def find_best_font_size(self, font, text, w, h):
        if not text: return 60
        canvas = Image.new('L', (w * 4, h * 4), 0)
        ImageDraw.Draw(canvas).text((w, h), text, font=font.font_variant(size=100), fill=255)
        arr = np.array(canvas)
        rows = np.any(arr > 0, axis=1)
        if not np.any(rows): return 60
        return max(10, min(int(100 * (h * 0.9) / (np.sum(rows) + 1e-6)), 600))

    def draw_glyph(self, render_text, polygon, width, height):
        if not render_text: 
            return np.zeros((height, width, 1), dtype=np.float32)
        
        p0, p1, p2, p3 = polygon
        w = int(np.linalg.norm(p0 - p1))
        h = int(np.linalg.norm(p1 - p2))
        
        if w <= 1 or h <= 1: 
            return np.zeros((height, width, 1), dtype=np.float32)

        new_font = self.base_font.font_variant(size=self.find_best_font_size(self.base_font, render_text, w, h))
        big = Image.new('L', (w * 4, h * 4), 0)
        ImageDraw.Draw(big).text((w, h), render_text, font=new_font, fill=255)
        arr = np.array(big)

        rows = np.any(arr > 0, axis=1)
        cols = np.any(arr > 0, axis=0)
        if not np.any(rows) or not np.any(cols): 
            return np.zeros((height, width, 1), dtype=np.float32)

        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]

        resized = cv2.resize(arr[r_min:r_max + 1, c_min:c_max + 1], (w, h), interpolation=cv2.INTER_LINEAR)
        M = cv2.getPerspectiveTransform(np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32), polygon.astype(np.float32))
        warped = cv2.warpPerspective(resized, M, (width, height), borderValue=0)
        
        return warped[..., None].astype(np.float32) / 255.0

    # ============================================================
    # DataLoader 取件邏輯
    # ============================================================
    def __getitem__(self, idx):
        # 恢復固定隨機種子 (配合 preprocess.py，保證存出的資料可重現)
        worker_seed = self.seed + idx
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        
        filename = self.image_files[idx]
        image_path = self.data_root / filename
        stem = Path(filename).stem
        
        # 1. 讀取圖像
        image = Image.open(image_path).convert('RGB')
        W, H = image.size
        
        # 2. 解析座標與後 5 碼
        quad, last_5_text = self.parse_ccpd_filename(filename)
        
        # 例外處理
        if quad is None or not last_5_text:
            mask_pil = Image.new('L', (W, H), 0)
            condition_pil = Image.new('L', (W, H), 0)
            masked_text = ""
            mask_range = "0-0"
        else:
            # 3. 生成局部遮罩與目標文字
            mask_quad, masked_text, mask_range = self.generate_dynamic_mask_and_text(quad, last_5_text)
            
            # 4. 繪製 Binary Mask
            mask_np = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask_np, [mask_quad.astype(np.int32)], 255)
            mask_pil = Image.fromarray(mask_np, mode="L")
            
            # 5. 渲染 Glyph
            glyph_arr = self.draw_glyph(masked_text, mask_quad, W, H)
            condition_pil = Image.fromarray((glyph_arr[..., 0] * 255).astype(np.uint8))

        # 6. Transform
        image_tensor = self.to_tensor(image)
        mask_tensor = self.mask_transform(mask_pil)
        condition_tensor = self.mask_transform(condition_pil)
        
        # 7. 返回給 preprocess.py 的字典格式
        return {
            'image': image_tensor,
            'hint': mask_tensor,
            'condition': condition_tensor,
            'description': masked_text,
            'filename': stem,  # 取 stem，這樣 preprocess_partial.py 存檔時才會是 xxx.pt
            'full_text': last_5_text, # 只存後5碼
            'mask_range': mask_range,
        }