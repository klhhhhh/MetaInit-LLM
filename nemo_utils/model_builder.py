import logging
import torch
from omegaconf import OmegaConf
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.utils import AppState
import pytorch_lightning as pl
from nemo.utils.model_utils import MegatronTrainerBuilder

@hydra_runner(config_path="../conf", config_name="megatron_gpt_config")
def build_model(cfg, map_location="cpu"):
    """
    Initialize a MegatronGPTModel from config and place it on the specified device.

    Args:
        cfg (DictConfig): full config or model config from Hydra
        map_location (str): device name ("cpu" or "cuda")

    Returns:
        model (MegatronGPTModel): initialized model on given device
    """

    logging.info("************** Initializing Model **************")

    # Step 1: Setup dummy trainer
    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    # Optional: log config
    logging.info("Model config:\n" + OmegaConf.to_yaml(model_cfg))

    # Step 4: Instantiate the model
    model = MegatronGPTModel(cfg.model, trainer)

    return model
