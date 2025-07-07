import torch._dynamo
import torch.multiprocessing as mp
import argparse

from omegaconf.omegaconf import OmegaConf
from pathlib import Path

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

from nemo_utils.model_projection import ModelProjectionUtils

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

torch._dynamo.config.suppress_errors = True
mp.set_start_method("spawn", force=True)

def main() -> None:
    logging.info("Loading and projecting small model to large model (CPU stage)...")
    
    small_model_path = "/work/hdd/bdrw/klin4/checkpoints/nemo/gpt/megatron_gpt.nemo"
    large_model_cfg_name = "megatron_gpt_350m_config"
    device = "cpu"
    projection_utils = ModelProjectionUtils(small_model_path, large_model_cfg_name, device)
    projection_utils.project_parameters(rank=64, learnable=False)
    model = projection_utils.large_model
    trainer = projection_utils.get_large_model_trainer()
    large_model_exp_manager = projection_utils.get_large_model_exp_manager()

    # Step 3: Move model to appropriate device (GPU)
    model = model.to(torch.cuda.current_device())

    exp_manager(trainer, large_model_exp_manager)

    # Step 4: Begin training
    logging.info("Starting training with projected model...")
    trainer.fit(model)

if __name__ == '__main__':

    main()
