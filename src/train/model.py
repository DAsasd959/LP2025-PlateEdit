import os
from diffusers.pipelines import FluxFillPipeline
import lightning as L
from peft import LoraConfig
import prodigyopt
from safetensors.torch import load_file
import torch

from ..flux.transformer import tranformer_forward
from ..flux.condition import Condition
from ..loss.ocr_loss.odm_loss import ODMLoss

from diffusers import FluxTransformer2DModel
from transformers import BitsAndBytesConfig 
from peft import LoraConfig, get_peft_model


class OminiModelFIll(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        reuse_lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        odm_loss_config: dict = None,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # --- [1. QLoRA 設定] ---
        # 即使本地已經是 nf4，這裡依然要宣告，目的是設定 compute_dtype=bfloat16
        print(f"Loading Local FLUX Model from: {flux_pipe_id}")
        
        # bitsandbytes' NF4 kernels are written for CUDA; its ROCm support is a
        # separate, experimental backend. Set FLUX_QUANTIZE=none to load the
        # transformer in bfloat16 instead — about 34 GB rather than 12 GB, which
        # a 192 GB MI300X absorbs easily, and it takes bitsandbytes out of the
        # picture entirely. Numerics then differ slightly from the released
        # checkpoint, which was tuned against the quantised base.
        quantize = os.environ.get("FLUX_QUANTIZE", "nf4").lower()
        nf4_config = None if quantize in ("none", "off", "0", "bf16") else \
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        print(f"[flux] quantisation: {'none (bfloat16)' if nf4_config is None else 'nf4'}")

        # --- [2. 載入模型] ---
        # A. 先載入 Transformer
        # 因為您的模型已經是 nf4，from_pretrained 會直接讀取那些 safetensors
        # quantization_config 確保它被正確識別為 QLoRA 準備狀態
        transformer = FluxTransformer2DModel.from_pretrained(
            flux_pipe_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=dtype,
            local_files_only=True # 強制只讀本地文件，防止它嘗試聯網
        )
        

        self.transformer = transformer
        # 強制凍結參數 (雖然 bitsandbytes 載入時通常已凍結，但保險起見)
        self.transformer.requires_grad_(False)

        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing() 

        # C. 載入 Pipeline 其餘部分
        # 把我們剛讀好的 transformer 塞進去
        self.flux_pipe = FluxFillPipeline.from_pretrained(
            flux_pipe_id,
            transformer=self.transformer,
            torch_dtype=dtype,
            local_files_only=True
        )

        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()

        # 手動將其他部分移至 GPU (因為 Transformer 已經由 bitsandbytes 管理)
        # self.flux_pipe.text_encoder.to(device)
        # self.flux_pipe.text_encoder_2.to(device)
        # self.flux_pipe.vae.to(device)
        self.vae_scale_factor = 8

        # Initialize LoRA layers
        self.init_lora(lora_config)
        # reuse the weight
        if reuse_lora_path is not None:
            print(f"reuse the lora path: {reuse_lora_path}")
            state_dict = load_file(reuse_lora_path)
            state_dict1 = {x.replace('lora_A', 'lora_A.default').replace('lora_B', 'lora_B.default').replace('transformer.', ''): v for x, v in state_dict.items()}
            self.transformer.load_state_dict(state_dict1, strict=False)

        # Initialize ODM layers
        if odm_loss_config is not None:
            self.odm_loss = ODMLoss(**odm_loss_config)
        else:
            self.odm_loss = None

        self.to(device).to(dtype)

   # [修改] 重寫 init_lora 方法
    def init_lora(self, lora_config: dict):
        peft_config = LoraConfig(**lora_config)
        self.transformer = get_peft_model(self.transformer, peft_config)
        # self.transformer.print_trainable_parameters()
        self.lora_layers = [p for p in self.transformer.parameters() if p.requires_grad]

    def save_lora(self, path: str):
        # 使用 PEFT 的 save_pretrained 比較穩
        self.transformer.save_pretrained(path)

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            # 使用 bitsandbytes 的 32bit AdamW 以節省顯存並保持精度
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW32bit(self.trainable_params, **opt_config["params"])
            except ImportError:
                optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def validation_step(self, batch, batch_idx):
        """
        驗證步驟：計算 Loss 但不更新權重
        """
        # 複用 step 函數計算 Loss
        loss = self.step(batch)
        
        # 記錄驗證 Loss
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def step(self, batch):
        # ============================================================
        # 1. 讀取 Packed Tokens
        # ============================================================
        x_0 = batch["img_tokens"].to(self.device, self.dtype)         # GT [B, L, 64]
        # source_tokens = batch["ref_tokens"].to(self.device, self.dtype) 
        masked_ref_tokens = batch["masked_ref_tokens"].to(self.device, self.dtype) # BG [B, L, 64]
        mask_tokens = batch["mask_tokens"].to(self.device, self.dtype) # Mask [B, L, 256]
        cond_tokens = batch["cond_tokens"].to(self.device, self.dtype) # Glyph [B, L, 64]
        txt_ids = batch["text_ids"][0].to(self.device, self.dtype)
        
        # Pixel GT for ODM
        mask_pixel = batch["mask_pixel"].to(self.device, self.dtype)  # [B,1,H,W]
        pixel_gt = batch["pixel_gt"].to(self.device, self.dtype)
        
        batch_size = x_0.shape[0]
        
        # Prompt Embeddings
        prompt_embeds = batch["prompt_embeds"].to(self.device, self.dtype)
        pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.device, self.dtype)

        img_ids = batch["img_ids"][0].to(self.device, self.dtype)
        condition_ids = batch["cond_ids"][0].to(self.device, self.dtype)

        # ============================================================
        # 2. Flow Matching
        # ============================================================
        t = torch.sigmoid(torch.randn(batch_size, device=self.device))
        x_1 = torch.randn_like(x_0).to(self.device)
        t_exp = t.view(batch_size, 1, 1) # [B, 1, 1]
        
        x_t = ((1 - t_exp) * x_0 + t_exp * x_1).to(self.dtype) # [B, L, 64]

        # ============================================================
        # 3. Context & Input Construction 
        # ============================================================
        
        # --- Context (320 dim) ---
        # 64 (Ref) + 256 (Mask) = 320
        context_main = torch.cat((masked_ref_tokens, mask_tokens), dim=2)
        context_cond = torch.cat((masked_ref_tokens, mask_tokens), dim=2)
        
        # --- Path 1: Main Branch (384 dim) ---
        # 64 (Noise) + 320 (Context) = 384
        hidden_states = torch.cat((x_t, context_main), dim=2)
        
        # --- Path 2: Condition Branch (384 dim) ---
        # 64 (Glyph) + 320 (Context) = 384
        condition_latents = torch.cat((cond_tokens, context_cond), dim=2)

        # Condition Type
        ct_id = Condition.get_type_id("word_fill")
        condition_type_ids = (
            torch.ones(batch_size, device=self.device, dtype=self.dtype) * ct_id
        ).unsqueeze(1)

        guidance = torch.ones_like(t).to(self.device) if self.transformer.config.guidance_embeds else None

        # ============================================================
        # 5. Forward Pass
        # ============================================================
        transformer_out = tranformer_forward(
            self.transformer,
            model_config=self.model_config,
            hidden_states=hidden_states,
            condition_latents=condition_latents,
            condition_ids=condition_ids,
            condition_type_ids=condition_type_ids,
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=txt_ids, 
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred_velocity = transformer_out[0]

        # ============================================================
        # 6. Loss: Velocity Matching
        # ============================================================
        target_velocity = x_1 -  x_0
        
        # ============================================================
        # Mask-Weighted Velocity Loss (最穩的主損失)
        # ============================================================
        latent_mask_weight = mask_tokens.mean(dim=2, keepdim=True) 
        
        # 核心精神：背景權重 1x，Mask 區域權重 3x (1 + 2*1)
        weight_sd = 1.0 + 2.0 * latent_mask_weight
        
        diff_sq = (pred_velocity - target_velocity) ** 2
        loss_sd =  (diff_sq * weight_sd).mean()

        ori_height = pixel_gt.shape[2]
        ori_width = pixel_gt.shape[3]
        odm_loss = torch.tensor(0.0, device=self.device)
        if self.odm_loss is not None:
            latents = self.flux_pipe._unpack_latents(x_1 - pred_velocity, ori_height, ori_width, self.vae_scale_factor)
            latents = (
                latents / self.flux_pipe.vae.config.scaling_factor
            ) + self.flux_pipe.vae.config.shift_factor
            self.flux_pipe.vae.to(self.device)
            self.flux_pipe.vae.eval()
            image_pred = self.flux_pipe.vae.decode(latents, return_dict=False)[0]

            pixel_gt_normalized = 2.0 * pixel_gt - 1.0
            odm_loss = self.odm_loss.loss(image_pred, pixel_gt_normalized, mask_pixel)
            
            lambda_odm = 1.0 
            loss = loss_sd + lambda_odm * odm_loss

        # Logging
        self.res = {
            "loss_sd": loss_sd,
            "loss_odm" : odm_loss,
        }
        self.last_t = t.mean().item()
        
        return loss