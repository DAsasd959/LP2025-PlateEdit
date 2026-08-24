import os
import lightning as L
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

try:
    import wandb
except ImportError:
    wandb = None

from ..flux.condition import Condition
from ..flux.generate_fill import generate_fill


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 3000)
        self.save_interval = training_config.get("save_interval", 3000)
        self.sample_interval = training_config.get("sample_interval", 3000)
        self.save_path = training_config.get("save_path", "./output")
        self.target_size = training_config.get("dataset", {}).get("image_size", 512)

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )
        if not self.use_wandb:
            self.writer = SummaryWriter(log_dir=f"{self.save_path}/{self.run_name}/logs")
        else:
            self.writer = None
        self.to_tensor = T.ToTensor()
        self.to_pil = T.ToPILImage()

        self.total_steps = 0
        # [新增] 用來暫存驗證集的 Loss
        self.val_loss_buffer = {} 

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # 2. Logging (WandB / TensorBoard)
        loss_value = outputs["loss"].item()
        # 注意：Lightning 自動處理 accumulate_grad_batches 的 loss 縮放
        # 如果 trainer.accumulate_grad_batches > 1，通常 outputs['loss'] 是平均後的

        # 2. 準備 Log Data (動態抓取 model.py 裡的 self.res)
        log_dict = {
            "epoch": trainer.current_epoch,
            "steps": self.total_steps,
            "train/loss": loss_value,
        }
        
        # 自動讀取 model.res 裡的所有 Tensor
        if hasattr(pl_module, 'res'):
            for k, v in pl_module.res.items():
                if isinstance(v, torch.Tensor):
                    log_dict[f"train/{k}"] = v.item()
        
        # Print training progress every n steps
        if self.use_wandb:
            # 加上 Gradient Size 監控 
            grad_norm = self._get_gradient_norm(pl_module)
            log_dict["train/grad_norm"] = grad_norm
            wandb.log(log_dict)
        else:
            # Tensorboard Logging
            for k, v in log_dict.items():
                if isinstance(v, (int, float)):
                    # Tensorboard 不喜歡斜線，可以改用底線或群組
                    self.writer.add_scalar(k, v, self.total_steps)


        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # 5. Generate Sample
        if self.total_steps % self.sample_interval == 0:
            print(f"Generating sample at step {self.total_steps}")
            # 直接傳入當前的 batch 進行測試
            self.generate_a_sample(
                pl_module,
                batch,
                f"{self.save_path}/{self.run_name}/output",
                f"step_{self.total_steps}"
            )
    
    def on_validation_epoch_start(self, trainer, pl_module):
        """每個驗證 Epoch 開始前，清空暫存區"""
        self.val_loss_buffer = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.val_loss_buffer:
            return

        log_dict = {}

        # 平均每個 loss
        for k, v_list in self.val_loss_buffer.items():
            mean_val = float(np.mean(v_list))
            log_dict[f"val/{k}"] = mean_val

        # 設定 epoch 和 step
        log_dict["epoch"] = trainer.current_epoch
        log_dict["steps"] = self.total_steps

        if self.use_wandb:
            wandb.log(log_dict)
        else:
            for k, v in log_dict.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, self.total_steps)

        print(f"Validation: {log_dict}")
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """每個驗證 Batch 結束時，收集 Loss"""
        
        # 1. 第一個 Batch 生成預覽圖
        if batch_idx == 0:
            print(f"Generating Validation Sample at Epoch {trainer.current_epoch}")
            self.generate_a_sample(
                pl_module,
                batch,
                f"{self.save_path}/{self.run_name}/val",
                f"val_epoch_{trainer.current_epoch}",
            )
        
        # 2. [核心] 收集 Loss (從 pl_module.res 抓取)
        if hasattr(pl_module, 'res'):
            for k, v in pl_module.res.items():
                if isinstance(v, torch.Tensor):
                    val = v.item()
                    if k not in self.val_loss_buffer:
                        self.val_loss_buffer[k] = []
                    self.val_loss_buffer[k].append(val)
    
    def _get_gradient_norm(self, pl_module):
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm**2  # GPU tensor 不用 .item()
        total_norm = total_norm**0.5
        return total_norm.item()  # 最後才回 CPU

    def _save_checkpoint(self, pl_module):
        save_dir = f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving LoRA weights to {save_dir}")
        pl_module.save_lora(save_dir)
    
    @torch.no_grad()
    def generate_a_sample(
        self,
        pl_module,
        batch,
        save_path,
        file_prefix,
    ):
        os.makedirs(save_path, exist_ok=True)
        idx = 0
        print("Generating Sample: Moving models to GPU...")

        # 1. 搬運模型
        pl_module.flux_pipe.vae.to(pl_module.device)
        pl_module.flux_pipe.text_encoder.to(pl_module.device)
        pl_module.flux_pipe.text_encoder_2.to(pl_module.device)

        try:
            # 定義一個解包並解碼的輔助函數
            def unpack_and_decode(packed_tokens):
                # packed_tokens: [1, 1024, 64]
                
                # 1. Unpack: 還原成 [1, 16, 64, 64]
                # 注意：這裡傳入的 height/width 是原始圖片尺寸 (512)，函數內部會自動除以 scale_factor
                latents_unpacked = pl_module.flux_pipe._unpack_latents(
                    packed_tokens, 
                    self.target_size, self.target_size, 
                    pl_module.flux_pipe.vae_scale_factor
                )
                
                # 2. Scale & Shift (反標準化)
                latents_scaled = (latents_unpacked / pl_module.flux_pipe.vae.config.scaling_factor) + pl_module.flux_pipe.vae.config.shift_factor
                
                # 3. Decode
                image = pl_module.flux_pipe.vae.decode(latents_scaled, return_dict=False)[0]
                
                # 4. Post-process
                image = (image / 2 + 0.5).clamp(0, 1)
                return self.to_pil(image.squeeze(0).float().cpu())

            # --- Ref Image (Masked Background / Color Hint) ---
            ref_tokens = batch['masked_ref_tokens'][idx].unsqueeze(0).to(pl_module.device, pl_module.dtype)
            # 使用輔助函數解包
            ref_image_pil = unpack_and_decode(ref_tokens)

            # --- Target (GT) ---
            # GT 本來就是 pixel，直接轉
            target_tensor = batch['pixel_gt'][idx].to(pl_module.device, pl_module.dtype).clamp(0,1)
            target_image_pil = self.to_pil(target_tensor.float().cpu())

            # --- Glyph / Condition ---
            cond_tokens = batch['cond_tokens'][idx].unsqueeze(0).to(pl_module.device, pl_module.dtype)
            # 使用輔助函數解包
            cond_pil = unpack_and_decode(cond_tokens)
            cond_np = np.array(cond_pil)

            # --- Mask Pixel ---
            mask_pixel = batch['mask_pixel'][idx].unsqueeze(0).to(pl_module.device, pl_module.dtype)
            # 插值確保大小對齊
            mask_up = F.interpolate(mask_pixel, size=(self.target_size, self.target_size), mode='nearest')
            mask_np = mask_up[0].permute(1, 2, 0).float().cpu().numpy() 

            # =========================================================
            # E. 推理生成 (Inference)
            # =========================================================
            prompt = batch['description'][idx]
            
            # Position Delta 處理 (Inpainting 通常是 0)
            if 'position_delta' in batch:
                pos_tensor = batch['position_delta'][idx]
                if isinstance(pos_tensor, torch.Tensor):
                    position_delta = pos_tensor.tolist()
                else:
                    position_delta = pos_tensor
                if isinstance(position_delta[0], list):
                    position_delta = position_delta[0]
            else:
                position_delta = [0.0, 0.0]

            condition = Condition(
                condition_type="word_fill",
                condition=[cond_np, mask_np, ref_image_pil], 
                position_delta=position_delta
            )

            generator = torch.Generator(device=pl_module.device).manual_seed(42)

            # 呼叫 pipeline 生成
            res = generate_fill(
                pl_module.flux_pipe,
                prompt=prompt,
                conditions=[condition],
                height=self.target_size,
                width=self.target_size,
                generator=generator,
                model_config=pl_module.model_config,
                default_lora=True,
            )

            generated_image = res.images[0]

            # =========================================================
            # F. 拼接 4 格畫布
            # =========================================================
            w, h = generated_image.size
            canvas = Image.new("RGB", (w * 4, h))
            
            canvas.paste(ref_image_pil, (0, 0))          # 1. 舊圖/Input
            canvas.paste(target_image_pil, (w, 0))       # 2. GT
            canvas.paste(cond_pil.convert("RGB"), (w * 2, 0)) # 3. Glyph
            canvas.paste(generated_image, (w * 3, 0))    # 4. 生成結果

            final_filename = f"{file_prefix}_step_{self.total_steps}.jpg"
            final_path = os.path.join(save_path, final_filename)
            canvas.save(final_path)
            print(f"✅ Sample saved to {final_path}")

            if self.use_wandb:
                mode = "train" if pl_module.training else "val"
                wandb.log({f"{mode}_sample": wandb.Image(canvas, caption=f"Prompt: {prompt}")})
        
        except Exception as e:
            print(f"⚠️ Sample generation failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            print("Cleaning up: Moving Text Encoders & VAE back to CPU...")
            pl_module.flux_pipe.text_encoder.to("cpu")
            pl_module.flux_pipe.text_encoder_2.to("cpu")
            pl_module.flux_pipe.vae.to("cpu")
            torch.cuda.empty_cache()