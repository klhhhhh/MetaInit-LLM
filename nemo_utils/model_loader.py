import torch
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from pathlib import Path
from omegaconf import OmegaConf

def load_nemo_model(model_path: str):
    """
    适用于 3D 并行训练的 NeMo Checkpoint 加载
    :param model_path: 训练好的模型路径 (.nemo 或 .ckpt)
    :return: 加载的模型
    """
    model_path = Path(model_path)

    # ✅ 直接加载 `.nemo`
    if model_path.suffix == ".nemo":
        print(f"Loading NeMo model from {model_path}...")
        model = MegatronGPTModel.restore_from(restore_path=str(model_path))
        return model

    # ✅ 加载 `.ckpt`
    elif model_path.suffix == ".ckpt":
        print(f"Loading checkpoint from {model_path}...")
        cfg_path = model_path.parent / "megatron_gpt_config.yaml"

        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config file: {cfg_path}")

        cfg = OmegaConf.load(cfg_path)
        model = MegatronGPTModel(cfg.model)

        # ✅ 加载 checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")

        # 处理 DataParallel / DistributedDataParallel
        state_dict = checkpoint["state_dict"]
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(new_state_dict, strict=False)
        return model

    else:
        raise ValueError("Unsupported model format! Only .nemo and .ckpt are supported.")

def merge_3d_parallel_checkpoints(ckpt_paths, output_path):
    """
    适用于 3D 并行训练的 checkpoint 合并
    """
    merged_state_dict = {}

    for path in ckpt_paths:
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint["state_dict"]

        for k, v in state_dict.items():
            if k not in merged_state_dict:
                merged_state_dict[k] = v
            else:
                merged_state_dict[k] += v  # 累加多个 GPU 训练的部分

    torch.save({"state_dict": merged_state_dict}, output_path)
    print(f"Merged checkpoint saved to {output_path}")


def extract_weights(model, save_path="models/gpt2_checkpoint/gpt2_weights.pth"):
    """
    提取 GPT2 关键权重，并存储为 PyTorch 格式
    :param model: NeMo 训练好的 GPT2 模型
    :param save_path: 保存路径
    """
    state_dict = model.state_dict()
    
    # 提取 Transformer 层的 key 权重
    extracted_weights = {k: v for k, v in state_dict.items() if "transformer" in k}
    
    torch.save(extracted_weights, save_path)
    print(f"Extracted weights saved to {save_path}")
