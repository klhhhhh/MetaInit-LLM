import logging
from pathlib import Path
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import to_absolute_path

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder

def load_model_to_cpu(nemo_path: str):
    # 获取 ../conf 的绝对路径
    current_file = Path(__file__).resolve()
    config_dir = (current_file.parent.parent / "conf").as_posix()
    config_path = Path(config_dir) / "megatron_gpt_124m_weight_mapping_config.yaml"

    # 加载配置
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="megatron_gpt_124m_weight_mapping_config")

    logging.info("\n\n************** Model Loader Configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()

    abs_path = to_absolute_path(nemo_path)
    logging.info(f"Loading model from {abs_path} onto CPU.")

    model = MegatronGPTModel.restore_from(
        restore_path=abs_path,
        override_config_path=config_path,
        trainer=trainer,
        map_location="cpu"
    )

    logging.info("Model successfully loaded onto CPU.")
    return model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = load_model_to_cpu(nemo_path="/pscratch/sd/k/klhhhhh/checkpoints/nemo/gpt/megatron_gpt.nemo")
