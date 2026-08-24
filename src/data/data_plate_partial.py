from pathlib import Path
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import random

class PlateDataset(Dataset):
    """
    車牌部分遮蔽數據集 (BBox 快速版本)
    
    改進：
    - 直接用首字左上左下 + 尾字右上右下構成矩形 Mask
    - 更快、更簡潔
    """
    
    def __init__(self, data_root, split='train', image_size=512, seed=42):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.condition_type = "word_fill"
        self.seed = seed
        
        # 固定隨機種子
        random.seed(seed)
        np.random.seed(seed)
        
        # 資料夾路徑
        self.i_s_dir = self.data_root / 'i_s'
        self.i_s_bbox_dir = self.data_root / 'i_s_bbox'
        self.mask_s_dir = self.data_root / 'mask_s'
        
        # 獲取所有圖片檔名
        self.image_files = sorted([f.stem for f in self.i_s_dir.glob('*.png')])
        
        print(f"[{split}] 載入 {len(self.image_files)} 張圖片 (BBox-based Masking, seed={seed})")
        
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
    
    def parse_bbox_file(self, bbox_path):
        """
        解析 BBox 文件
        Returns:
            full_text: str
            chars: List[str]
            bboxes: List[np.array] - 每個字的 4 個角點 [4, 2]
        """
        with open(bbox_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        full_text = lines[0].strip().split()[0]
        
        chars = []
        bboxes = []
        
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            
            char = parts[0]
            # [x1, y1, x2, y2, x3, y3, x4, y4]
            coords = np.array(list(map(int, parts[1:9])), dtype=np.int32).reshape((4, 2))
            
            chars.append(char)
            bboxes.append(coords)
        
        return full_text, chars, bboxes
    
    def select_random_mask_range(self, chars):
        """
        規則：
        1. 用 "-" 分割左右
        2. 依照左右字數比例選擇 side
        3. 在該 side uniform 遮 1 ~ 全部字
        4. "-" 永遠不會被遮
        """

        if '-' not in chars:
            # 如果沒有 dash，fallback：遮任一字
            num_chars = len(chars)
            if num_chars == 0:
                return 0, 0
            # 以 PP_PROVINCE_PROB 機率強制遮「省份中文字(位置0)」, 讓中文獲得足夠訓練
            import os
            _pp = float(os.environ.get("PP_PROVINCE_PROB", "0"))
            if _pp > 0 and random.random() < _pp:
                return 0, 1
            start = random.randint(0, num_chars - 1)
            return start, start + 1

        dash_idx = chars.index('-')

        left_len = dash_idx
        right_len = len(chars) - dash_idx - 1

        # 如果某邊沒有字
        if left_len == 0 and right_len == 0:
            return 0, 0
        elif left_len == 0:
            side = 'right'
        elif right_len == 0:
            side = 'left'
        else:
            # 按比例選 side
            total = left_len + right_len
            if random.random() < left_len / total:
                side = 'left'
            else:
                side = 'right'

        # 在選中的 side 裡 uniform 遮
        if side == 'left':
            mask_len = random.randint(1, left_len)
            start = random.randint(0, left_len - mask_len)
            end = start + mask_len

        else:  # right
            mask_len = random.randint(1, right_len)
            start = random.randint(dash_idx + 1, len(chars) - mask_len)
            end = start + mask_len

        return start, end
    
    def create_bbox_mask(self, first_bbox, last_bbox, H, W):
        """
        ⭐ 用首尾 BBox 構成矩形 Mask
        
        Args:
            first_bbox: [4, 2] - 首字的 4 個角點
            last_bbox: [4, 2] - 尾字的 4 個角點
            H, W: 圖像尺寸
        
        Returns:
            mask: PIL.Image (L mode)
        
        邏輯:
        首字: [左上, 右上, 右下, 左下] = [p0, p1, p2, p3]
        尾字: [左上, 右上, 右下, 左下] = [q0, q1, q2, q3]
        
        矩形 4 個點:
        - 左上: 首字左上 (p0)
        - 右上: 尾字右上 (q1)
        - 右下: 尾字右下 (q2)
        - 左下: 首字左下 (p3)
        """
        # 首字的點
        p0 = first_bbox[0]  # 左上
        p3 = first_bbox[3]  # 左下
        
        # 尾字的點
        q1 = last_bbox[1]   # 右上
        q2 = last_bbox[2]   # 右下
        
        # 構成矩形的 4 個點
        rect_points = [
            tuple(p0),  # 左上
            tuple(q1),  # 右上
            tuple(q2),  # 右下
            tuple(p3),  # 左下
        ]
        
        # 繪製
        mask_img = Image.new('L', (W, H), 0)
        draw = ImageDraw.Draw(mask_img)
        draw.polygon(rect_points, fill=255)
        
        return mask_img
    
    def create_partial_glyph(self, glyph_full, mask):
        """
        從完整 Glyph 中提取被遮蔽部分
        """
        glyph_np = np.array(glyph_full)
        mask_np = np.array(mask)
        
        # 只保留 Mask 區域的 Glyph
        partial_glyph_np = np.where(mask_np > 0, glyph_np, 0)
        
        return Image.fromarray(partial_glyph_np.astype(np.uint8))
    
    def __getitem__(self, idx):
        # 固定每個樣本的隨機種子
        worker_seed = self.seed + idx
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        
        filename = self.image_files[idx]
        
        # 1. 讀取圖像
        image_path = self.i_s_dir / f"{filename}.png"
        image = Image.open(image_path).convert('RGB')
        W, H = image.size
        
        # 2. 讀取完整 Glyph
        glyph_path = self.mask_s_dir / f"{filename}.png"
        glyph_full = Image.open(glyph_path).convert('L')
        
        # 3. 解析 BBox
        bbox_path = self.i_s_bbox_dir / f"{filename}.txt"
        full_text, chars, bboxes = self.parse_bbox_file(bbox_path)
        
        # 4. 隨機選擇遮蔽範圍
        start_idx, end_idx = self.select_random_mask_range(chars)
        
        # 5. 生成 Mask 和 Prompt
        if start_idx < end_idx:
            masked_text = "".join(chars[start_idx:end_idx])
            
            # 用首尾 BBox 構成矩形
            first_bbox = bboxes[start_idx]
            last_bbox = bboxes[end_idx - 1]
            
            mask_pil = self.create_bbox_mask(first_bbox, last_bbox, H, W)
            condition_pil = self.create_partial_glyph(glyph_full, mask_pil)
        else:
            masked_text = ""
            mask_pil = Image.new('L', (W, H), 0)
            condition_pil = Image.new('L', (W, H), 0)
        
        # 6. Transform
        image_tensor = self.to_tensor(image)
        mask_tensor = self.mask_transform(mask_pil)
        condition_tensor = self.mask_transform(condition_pil)
        
        # 7. 返回
        return {
            'image': image_tensor,
            'ref_image': image_tensor.clone(),
            'hint': mask_tensor,
            'condition': condition_tensor,
            'description': masked_text,
            'filename': filename,
            'position_delta': torch.tensor([[0.0, 0.0]]),
            'full_text': full_text,
            'mask_range': f"{start_idx}-{end_idx}",
        }