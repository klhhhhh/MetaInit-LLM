import logging
import torch

from pathlib import Path
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

def build_model(cfg_name, map_location="cpu"):
    """
    Initialize a MegatronGPTModel from config and place it on the specified device.

    Args:
        cfg_path (DictConfig): full config or model config from Hydra
        map_location (str): device name ("cpu" or "cuda")

    Returns:
        model (MegatronGPTModel): initialized model on given device
    """
    # Get the absolute path of ../conf
    current_file = Path(__file__).resolve()
    config_dir = (current_file.parent.parent / "conf").as_posix()

    # Load configuration
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name=cfg_name)

    logging.info("************** Initializing Model **************")

    # Setup trainer
    trainer = MegatronTrainerBuilder(cfg).create_trainer()

    # Instantiate the model
    model = MegatronGPTModel(cfg.model, trainer)

    return model, trainer, cfg.exp_manager

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    print("Building model...")
    model, trainer, exp_manager = build_model("megatron_gpt_350m_config")
    print("Model built successfully.")

