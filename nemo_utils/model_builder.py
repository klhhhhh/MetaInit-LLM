import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import torch

from pathlib import Path
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from nemo_utils.init_hook import patch_layer

def build_model(cfg_name, callbacks= None, map_location="cpu"):
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

    dtype = cfg.trainer.precision
    num_heads = cfg.model.num_attention_heads

    # Setup trainer
    trainer = MegatronTrainerBuilder(cfg).create_trainer(callbacks=callbacks)

    #Patch the ColumnParallelLinear and RowParallelLinear classes
    patch_layer(ColumnParallelLinear, "ColumnParallelLinear")
    patch_layer(RowParallelLinear, "RowParallelLinear")

    # Instantiate the model
    model = MegatronGPTModel(cfg.model, trainer)

    return model, trainer, cfg.exp_manager, dtype, num_heads

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    print("Building model...")
    model, trainer, exp_manager = build_model("megatron_gpt_350m_config")
    print("Model built successfully.")

