import logging
import argparse

from pathlib import Path
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import to_absolute_path

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder

def load_model_to_cpu(nemo_path: str):
    # Get the absolute path of ../conf
    current_file = Path(__file__).resolve()
    config_dir = (current_file.parent.parent / "conf").as_posix()
    config_path = Path(config_dir) / "megatron_gpt_124m_weight_mapping_config.yaml"

    # Load configuration
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

    parser = argparse.ArgumentParser(description="A simple checkpoint loading processing script.")
    parser.add_argument("--nemo_path", type=str, help="The path to .nemo file for loading.")
    args = parser.parse_args()

    nemo_path = args.nemo_path
    logging.info(f"Loading model from {nemo_path} onto CPU.")
    # Load the model

    model = load_model_to_cpu(nemo_path=nemo_path)
    
