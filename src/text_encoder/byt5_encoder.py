import torch
import torch.nn.functional as F

from .glyph_byt5.glyph_sdxl.utils import (
    parse_config,
    huggingface_cache_dir,
    load_byt5_and_byt5_tokenizer
)
from .glyph_byt5.glyph_sdxl.modules import T5EncoderBlockByT5Mapper

byt5_mapper_dict = [T5EncoderBlockByT5Mapper]
byt5_mapper_dict = {mapper.__name__: mapper for mapper in byt5_mapper_dict}

class GlyphByt5Encoder:
    def __init__(self, config_dir, byt5_path, byt5mapper_path, color_ann_path, font_ann_path, byt5_max_length=77):
        config = parse_config(config_dir)

        config['byt5_config']['color_ann_path'] = color_ann_path
        config['byt5_config']['font_ann_path'] = font_ann_path
        self.byt5_model, self.byt5_tokenizer = load_byt5_and_byt5_tokenizer(
            **config.byt5_config,
            huggingface_cache_dir=huggingface_cache_dir,
        )
        self.byt5_mapper = byt5_mapper_dict[config.byt5_mapper_type](
            self.byt5_model.config,
            **config.byt5_mapper_config,
        )

        byt5_mapper_para = torch.load(byt5mapper_path, map_location='cpu')
        self.byt5_mapper.load_state_dict(byt5_mapper_para)
        
        byt5_model_para = torch.load(byt5_path, map_location='cpu')
        self.byt5_model.load_state_dict(byt5_model_para)
        self.byt5_max_length = byt5_max_length

    def encode(self, text_prompt, text_attn_mask=None):
        byt5_text_inputs = self.byt5_tokenizer(
            text_prompt,
            padding="max_length",
            max_length=self.byt5_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        byt5_text_input_ids = byt5_text_inputs.input_ids
        byt5_attention_mask = byt5_text_inputs.attention_mask.to(self.byt5_model.device) if text_attn_mask is None else text_attn_mask.to(self.byt5_model.device, dtype=byt5_text_inputs.attention_mask.dtype)
        with torch.cuda.amp.autocast(enabled=False):
            byt5_prompt_embeds = self.byt5_model(
                byt5_text_input_ids.to(self.byt5_model.device),
                attention_mask=byt5_attention_mask.float(),
            )
            byt5_prompt_embeds = byt5_prompt_embeds[0]
            byt5_prompt_embeds = self.byt5_mapper(byt5_prompt_embeds, byt5_attention_mask)
        target_channels = 3072
        batch, length, channels = byt5_prompt_embeds.shape
        byt5_prompt_embeds = byt5_prompt_embeds.reshape(-1, channels).unsqueeze(1)
        byt5_prompt_embeds = F.interpolate(byt5_prompt_embeds, size=target_channels, mode='linear', align_corners=False)
        byt5_prompt_embeds = byt5_prompt_embeds.squeeze(1).reshape(batch, length, target_channels)
        return byt5_prompt_embeds   # batch, length, 3072