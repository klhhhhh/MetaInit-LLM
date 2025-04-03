import logging
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import to_absolute_path

from nemo.utils import exp_manager
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder

def load_model_to_cpu(nemo_path: str):
    # 初始化 Hydra 配置目录（相对于当前脚本）
    with initialize_config_dir(config_dir="../conf"):
        cfg = compose(config_name="megatron_gpt_124m_weight_mapping_config")

    logging.info("\n\n************** Model Loader Configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    # 初始化 trainer
    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    # 加载 .nemo 文件
    abs_path = to_absolute_path(nemo_path)
    logging.info(f"Loading model from {abs_path} onto CPU.")

    model = MegatronGPTModel.restore_from(
        restore_path=abs_path,
        trainer=trainer,
        map_location="cpu"
    )

    logging.info("Model successfully loaded onto CPU.")
    return model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 你可以通过命令行或者程序传参这里设置模型路径
    model = load_model_to_cpu(nemo_path="/pscratch/sd/k/klhhhhh/checkpoints/nemo/gpt/megatron_gpt.nemo")

    # 可选：打印模型结构或者保存为 state_dict
    # print(model)
