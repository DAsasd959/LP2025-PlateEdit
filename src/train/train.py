import os
import time

import lightning as L
from lightning.pytorch.strategies import DDPStrategy
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import yaml

from .callbacks import TrainingCallback
from src.train.data_cache import PlateCacheDataset
from .model import OminiModelFIll


def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# def setup_device():
#     """
#     Setup the device and the distributed coordinator.

#     Returns:
#         tuple[torch.device, DistCoordinator]: The device and the distributed coordinator.
#     """
#     assert torch.cuda.is_available(), "Training currently requires at least one GPU."
#     # NOTE: A very large timeout is set to avoid some processes exit early
#     dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
#     torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


def main():
    # Initialize
    config = get_config()
    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")

    L.seed_everything(training_config.get("seed", 1024))

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        init_wandb(wandb_config, run_name)


    # ==========================================
    # 1. 準備 Training Loader
    # ==========================================
    print("Initializing Training PlateDataset...")
    train_data_root = training_config["dataset"].get("train_cache_root")
    assert train_data_root is not None, "Config 中缺少 train.dataset.train_cache_root"
    

    train_dataset = PlateCacheDataset(
        data_root=train_data_root
    )
    print(f"Train Dataset length: {len(train_dataset)}")


    num_workers = training_config.get("dataloader_workers", 8)
    batch_size = training_config.get("batch_size", 1)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    # ==========================================
    # 2. [新增] 準備 Validation Loader
    # ==========================================
    print("Initializing Validation PlateDataset...")
    val_data_root = training_config["dataset"].get("valid_cache_root")
    val_dataset = PlateCacheDataset(
        data_root=val_data_root
    )
    print(f"Valid Dataset length: {len(val_dataset)}")
        
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, 
        shuffle=False,         # 驗證集不需要 shuffle
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Initialize model
    print("Initializing Model...")
    if 'model_type' in config:
        if config['model_type'] == 'flux_fill':
            trainable_model = OminiModelFIll(
                flux_pipe_id=config["flux_path"],
                reuse_lora_path=training_config.get("reuse_lora_path", None),
                lora_config=training_config["lora_config"],
                device="cuda",
                dtype=getattr(torch, config.get("dtype", "bfloat16")),
                optimizer_config=training_config["optimizer"],
                model_config=config.get("model", {}),
                gradient_checkpointing=training_config.get("gradient_checkpointing", False),
                odm_loss_config=training_config.get("odm_loss", None),
            )
        else:
            raise NotImplementedError

    # Callbacks for logging and saving checkpoints
    callbacks = []
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        callbacks.append(TrainingCallback(run_name, training_config=training_config))
        
        # 儲存 Config 備份
        save_path = training_config.get("save_path", "./output")
        os.makedirs(f"{save_path}/{run_name}", exist_ok=True)
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)


    # 7. 初始化 Trainer
    # 自動偵測 GPU 數量
    devices = torch.cuda.device_count()
    num_nodes = int(os.environ.get("NODE_NUM", 1))
    
    print(f"Trainer Setup: {devices} GPUs, {num_nodes} Nodes")
    
    # 決定策略：如果是多卡，使用 DDP；單卡則自動
    strategy = "auto"
    if devices > 1 or num_nodes > 1:
        strategy = DDPStrategy(find_unused_parameters=False) # 設為 False 稍微快一點，除非你有參數沒用到

    # Initialize trainer
    trainer = L.Trainer(
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        accelerator="gpu",
        precision="bf16-mixed", # 混合精度，這很重要！
        accumulate_grad_batches=training_config.get("accumulate_grad_batches", 1),
        callbacks=callbacks,
        enable_checkpointing=False, # callback 裡自己寫了 save，所以這裡關掉
        enable_progress_bar=True, 
        logger=False, # 用 wandb
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 1),
        use_distributed_sampler=True, 
        # --- 新增驗證設定 ---
        # 這裡設定每 1000 個 step 驗證一次（視您的數據量而定，100k data 若 batch=4，1 epoch = 25000 steps）
        # 如果設 1.0，就是每個 epoch 跑一次
        val_check_interval=training_config["dataset"].get("val_check_interval", 1000), 
        # 如果驗證集很大，可以限制只跑前 N 個 batch，這裡設 0 或 None 代表跑完
        limit_val_batches=training_config.get("limit_val_batches", None), 
        # ------------------
    )

    setattr(trainer, "training_config", training_config)


    # Start training
    print("Start Training...")
    trainer.fit(
        trainable_model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )


if __name__ == "__main__":
    # setup_device()
    main()
