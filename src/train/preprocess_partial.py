import sys
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from diffusers import FluxFillPipeline
from transformers import BitsAndBytesConfig
import torch.nn.functional as F
import random

# 路徑設定
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.flux.pipeline_tools import prepare_text_input
# 用環境變數 PP_DATASET 切換: plate(合成,預設) / real(LP2025) / ccpd(CCPD)
_ppds = os.environ.get("PP_DATASET", "plate")
if _ppds == "ccpd":
    from src.data.data_ccpd import PlateDataset
elif _ppds == "real":
    from src.data.data_real import PlateDataset
else:
    from src.data.data_plate_partial import PlateDataset
print(f"[preprocess] PP_DATASET={_ppds} -> {PlateDataset.__module__}")

# --- Config ---
# 記得修改你的 DATA_ROOT 指向新的 Partial Mask 資料集路徑
DATA_ROOT = os.environ.get("PP_DATA_ROOT")
OUTPUT_ROOT = os.environ.get("PP_OUTPUT_ROOT")
FLUX_PATH = os.environ.get("PP_FLUX_PATH","black-forest-labs/FLUX.1-Fill-dev")
BATCH_SIZE = 4
IMAGE_SIZE = 512

prompt_templates = [
    "Fill the masked character '{text}' using the same color, font, and style as the surrounding text.",
    "The missing character '{text}' should match the style and color of neighboring glyphs.",
    "Generate '{text}' in the same font, size, and color as adjacent text.",
    "Replace the placeholder with '{text}', preserving the appearance of nearby letters.",
    "Complete the masked letter '{text}' with consistent color, font, and style of surrounding characters.",
    "Fill in '{text}' so it blends seamlessly with the nearby text in font and color.",
    "Render '{text}' matching the style, size, and color of the context text around it."
]

def encode_prompt(pipe, prompts):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
            pipe, prompts 
        )
    return prompt_embeds.cpu(), pooled_prompt_embeds.cpu(), text_ids.cpu()

def encode_image_to_packed_tokens(pipe, images):
    """
    Pixel -> VAE -> Pack
    支援 1 channel (glyph) 和 3 channel (RGB) 輸入
    值域：[0, 1] tensor
    """
    # 1. channel 保護：1ch → 3ch
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    # 2. [0,1] → [-1,1]
    images = 2.0 * images - 1.0
    images = images.to(pipe.device, dtype=torch.bfloat16)

    with torch.no_grad():
        # 3. VAE encode
        latents = pipe.vae.encode(images).latent_dist.sample()
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

        # 4. Pack latents
        packed = pipe._pack_latents(
            latents,
            latents.shape[0],  # B
            latents.shape[1],  # C
            latents.shape[2],  # H
            latents.shape[3],  # W
        )

        # 5. 計算 img_ids（和 encode_images 一致，含 shape 修正）
        img_ids = pipe._prepare_latent_image_ids(
            latents.shape[0],
            latents.shape[2],
            latents.shape[3],
            pipe.device,
            pipe.dtype,
        )
        if packed.shape[1] != img_ids.shape[0]:
            img_ids = pipe._prepare_latent_image_ids(
                latents.shape[0],
                latents.shape[2] // 2,
                latents.shape[3] // 2,
                pipe.device,
                pipe.dtype,
            )

    return packed, img_ids

def prepare_mask_tokens_official(pipe, mask, height, width):
    """ 
    ⭐ 256-dim High-Res Mask (Pixel Patchify) 
    這是配合 FLUX-Fill 原生權重的關鍵
    """
    B, _, _, _ = mask.shape
    vae_scale = 8
    
    # 1. Resize Mask
    mask = F.interpolate(mask, size=(height, width), mode='nearest')
    
    # 2. Patchify Logic (Space to Depth)
    latent_h = height // vae_scale
    latent_w = width // vae_scale
    
    # [B, 1, 512, 512] -> [B, 64, 64, 64]
    mask_reshaped = mask.view(B, 1, latent_h, 8, latent_w, 8)
    mask_reshaped = mask_reshaped.permute(0, 1, 3, 5, 2, 4) 
    mask_reshaped = mask_reshaped.reshape(B, 64, latent_h, latent_w)
    
    # 3. Pack -> [B, L, 256]
    mask_tokens = pipe._pack_latents(
        mask_reshaped, B, 64, latent_h, latent_w
    ) 
    return mask_tokens

def main():
    print(f"🚀 Starting Preprocess (Partial Masking Strategy)...")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    print("Loading FLUX models...")
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    pipe = FluxFillPipeline.from_pretrained(
        FLUX_PATH, torch_dtype=torch.bfloat16
    )
    # preprocess 只用 VAE + 文字編碼器, transformer 用不到 → 丟棄以省大量顯存(避免 OOM)
    import gc
    pipe.transformer = None
    gc.collect(); torch.cuda.empty_cache()
    pipe.to("cuda")
    
    dataset = PlateDataset(data_root=DATA_ROOT, split='train', image_size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    print("Dataset length:", len(dataset))
    
    print(f"Processing {len(dataset)} images...")
    
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        filenames = batch['filename']
        prompts = []
        for desc in batch['description']:
            text = desc  # 原本的 placeholder 或缺字
            template = random.choice(prompt_templates)
            prompt = template.format(text=text)
            prompts.append(prompt)
        
        # 1. Text
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(pipe, prompts)
        
        # 2. Images & Mask
        # 在 Partial Masking 任務中，GT 就是原圖
        gt_pixel = batch['image'].to("cuda")      
        hint_mask = batch['hint'].to("cuda")          
        cond_pixel = batch['condition'].to("cuda")    # Partial Glyph
        
        mask_weight = hint_mask[:, 0:1, :, :] # [B, 1, 512, 512]
        
        # 3. ⭐ 製作 Masked Image (挖空)
        # 這是最關鍵的一步：
        # 必須把被遮住的字 (Target) 挖掉，填成黑色 (或其他 noise)
        # 這樣模型才不會 "看到答案"
        masked_pixel = gt_pixel * (1 - mask_weight)
        
        # 4. Encode Tokens
        
        # A. Target (GT) -> 用來算 Loss
        img_tokens, img_ids = encode_image_to_packed_tokens(pipe, gt_pixel)
        
        # B. Masked Ref (Main Branch) -> 只能看到被挖空的圖
        masked_ref_tokens, _ = encode_image_to_packed_tokens(pipe, masked_pixel)
        
        # C. Ref Tokens (Condition Branch) -> ⭐ 這裡也要用 Masked Image!
        # 在 Self-Reconstruction 中，如果 Ref 給 Full Image，模型會直接 Copy 答案 (Leakage)
        # 給 Masked Image 強迫模型去 "看鄰居 (Context)" 來學顏色
        ref_tokens = masked_ref_tokens.clone()
        
        # D. Mask (256 dim) & Glyph
        mask_tokens = prepare_mask_tokens_official(pipe, mask_weight, IMAGE_SIZE, IMAGE_SIZE).to(pipe.device, dtype=torch.bfloat16)
        cond_tokens, cond_ids = encode_image_to_packed_tokens(pipe, cond_pixel)

        # 5. Pixel GT (for ODM Loss)
        pixel_gt = batch['image'].clone() # CPU

        # Save
        for i, fname in enumerate(filenames):
            save_path = os.path.join(OUTPUT_ROOT, f"{fname}.pt")
            data_dict = {
                "prompt_embeds": prompt_embeds[i].cpu().clone(),
                "pooled_prompt_embeds": pooled_prompt_embeds[i].cpu().clone(),
                "text_ids": text_ids.cpu().clone(),
                
                "img_tokens": img_tokens[i].cpu().clone(),
                "img_ids": img_ids.cpu().clone(),
                "masked_ref_tokens": masked_ref_tokens[i].cpu().clone(),
                "ref_tokens": ref_tokens[i].cpu().clone(), # Masked!
                "mask_tokens": mask_tokens[i].cpu().clone(),
                "cond_tokens": cond_tokens[i].cpu().clone(),
                "cond_ids": cond_ids.cpu().clone(),
                
                "mask_pixel": mask_weight[i].cpu().clone(),
                "pixel_gt": pixel_gt[i].cpu().clone(),
                "description": prompts[i],
            }
            torch.save(data_dict, save_path)

    print(f"✅ Preprocessing Complete! Saved to {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()