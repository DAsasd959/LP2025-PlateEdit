import torch
from torch.utils.data import Dataset
from pathlib import Path

class PlateCacheDataset(Dataset):
    def __init__(self, data_root):
        """
        Args:
            data_root: 存放 .pt 檔案的目錄 (例如 .../train100k_latents)
        """
        self.data_root = Path(data_root)
        # 抓取所有 .pt 檔案
        self.files = sorted(list(self.data_root.glob("*.pt")))
        print(f"🔥 Loaded Cached Dataset: {len(self.files)} files from {data_root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        # 讀取 .pt
        data = torch.load(file_path, map_location="cpu") # 讀到 CPU，由 DataLoader 搬運
        
        
        return {
            # ==============================
            # Transformer Tokens
            # ==============================
            "img_tokens": data["img_tokens"],                     # [L, 64]
            "img_ids": data["img_ids"],
            "ref_tokens": data["ref_tokens"],
            "masked_ref_tokens": data["masked_ref_tokens"],       # [L, 64]
            "mask_tokens": data["mask_tokens"],                   # [L, 64]
            "cond_tokens": data["cond_tokens"],                   # [L, 64]
            "cond_ids": data["cond_ids"], 

            # ==============================
            # Text Conditioning
            # ==============================
            "prompt_embeds": data["prompt_embeds"],
            "pooled_prompt_embeds": data["pooled_prompt_embeds"],
            "text_ids": data["text_ids"],

            # ==============================
            # Pixel-space (for ODM / logging)
            # ==============================
            "pixel_gt": data["pixel_gt"],                         # [3, 512, 512]
            "mask_pixel": data["mask_pixel"],
            # ==============================
            # Meta
            # ==============================
            "description": data["description"],
        }