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

from projection.model_projection import ModelProjectionUtils
from projection.projection_freeze import AlphaScheduleAndFreeze

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

torch._dynamo.config.suppress_errors = True
mp.set_start_method("spawn", force=True)


def main(
    small_model_path: str,
    large_model_cfg_name: str,
    project_device: str,
    train_device: str,
    rank: int,
    learnable: bool,
) -> None:
    logging.info("Loading and projecting small model to large model (CPU stage)...")

    if learnable:
        logging.info("Projected parameters will be learnable.")
        projection_training_callback = [AlphaScheduleAndFreeze(schedule="cosine", mode="decay", start_value=0.0, end_value=1.0, start_step=0, end_step=2000, freeze_at=2000, cache_on_freeze=True, drop_small_on_freeze=True, verbose=True)]
    else:
        logging.info("Projected parameters will not be learnable.")
        projection_training_callback = None

    projection_utils = ModelProjectionUtils(small_model_path, large_model_cfg_name, projection_training_callback, project_device)
    projection_utils.project_parameters(rank=rank, learnable=learnable)

    projection_utils.free_small_model()
    torch.cuda.empty_cache()

    model = projection_utils.large_model
    trainer = projection_utils.get_large_model_trainer()
    large_model_exp_manager = projection_utils.get_large_model_exp_manager()

    exp_manager(trainer, large_model_exp_manager)

    # Begin training
    logging.info("Starting training with projected model...")
    trainer.fit(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Project small model to large model and train.")
    parser.add_argument(
        "--small_model_path",
        type=str,
        default="/work/hdd/bdrw/klin4/checkpoints/nemo/gpt/megatron_gpt.nemo",
        help="Path to the small model .nemo file.",
    )
    parser.add_argument(
        "--large_model_cfg_name",
        type=str,
        default="megatron_gpt_350m_config",
        help="Configuration name for the large model.",
    )
    parser.add_argument(
        "--project_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to load the model onto (default: cpu).",
    )
    parser.add_argument(
        "--train_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to load the model onto (default: cpu).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="Rank for the model projection (default: 64).",
    )
    parser.add_argument(
        "--learnable",
        action='store_true',
        help="Whether to make the projected parameters learnable (default: False).",
    )
    args = parser.parse_args()

    main(
        small_model_path=args.small_model_path,
        large_model_cfg_name=args.large_model_cfg_name,
        project_device=args.project_device,
        train_device=args.train_device,
        rank=args.rank,
        learnable=args.learnable,
    )
