import logging
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from hydra.utils import to_absolute_path
from hydra import initialize_config_dir, compose
from nemo.core.config import hydra_runner
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.utils.model_utils import MegatronTrainerBuilder

@hydra_runner(config_path="../conf", config_name="megatron_gpt_config")
def load_model_to_cpu(cfg, nemo_path: str):
    logging.info("\n\n************** Model Loader Configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    # Step 1: Setup a dummy trainer for model initialization
    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    # Step 2: Load the small model from .nemo file onto CPU
    logging.info(f"Loading model from {nemo_path} onto CPU.")

    model = MegatronGPTModel.restore_from(
        restore_path=to_absolute_path(nemo_path),
        trainer=trainer,
        map_location="cpu")

    logging.info("Model successfully loaded onto CPU.")
    # You can now return or manipulate the model as needed
    # e.g., return small_model or prepare for weight mapping
    return model

if __name__ == "__main__":
    # Initialize the Hydra config directory
    model = load_model_to_cpu(nemo_path="path/to/your/nemo_file.nemo")
