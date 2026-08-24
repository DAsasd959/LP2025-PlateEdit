from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class PlateDataset(Dataset):
    """
    LP2025遮蔽數據集
    """
    
    def __init__(self, data_root, split='train', image_size=512, seed=42):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.condition_type = "word_fill"
        self.seed = seed
        
        
        # 資料夾路徑
        self.i_s_dir = self.data_root / 'filtered_plate'
        self.i_s_bbox_dir = self.data_root / 'partial_labels_txt'
        self.mask_s_dir = self.data_root / 'partial_glyphs'
        self.mask_dir = self.data_root / 'partial_masks'
        
        # 獲取所有圖片檔名
        self.image_files = sorted([f.stem for f in self.mask_s_dir.glob('*.png')])
        
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
            bboxes: List[np.array] - 每個詞的 4 個角點 [4, 2]
        """
        with open(bbox_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        full_text = lines[0].strip().split()[0]
        
        return full_text
    
    def __getitem__(self, idx):        
        filename = self.image_files[idx]
        
        # 1. 讀取圖像
        image_path = self.i_s_dir / f"{filename}.jpg"
        image = Image.open(image_path).convert('RGB')
        W, H = image.size
        
        # 2. 讀取完整 Glyph
        glyph_path = self.mask_s_dir / f"{filename}.png"
        glyph_full = Image.open(glyph_path).convert('L')

        # 3. 讀取完整 Mask
        mask_path = self.mask_dir / f"{filename}.png"
        mask_pil = Image.open(mask_path).convert('L')
        
        # 4. 解析 BBox
        bbox_path = self.i_s_bbox_dir / f"{filename}.txt"
        text = self.parse_bbox_file(bbox_path)
        
        
        # 5. 生成 Mask 和 Prompt
        masked_text = text
        condition_pil = glyph_full

        
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
            'full_text': text,
        }