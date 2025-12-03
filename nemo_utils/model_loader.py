import logging
import argparse
import torch

from pathlib import Path
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import to_absolute_path

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder

def load_model_to_cpu(nemo_checkpoint_path: str, config_path=None):
    # Get the absolute path of ../conf
    current_file = Path(__file__).resolve()
    config_dir = (current_file.parent.parent / "conf").as_posix()
    if config_path == None:
        config_path = Path(config_dir) / "megatron_gpt_124m_weight_mapping_config.yaml"
    else:
        config_path = config_path

    # Load configuration
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="megatron_gpt_124m_weight_mapping_config")

    logging.info("\n\n************** Model Loader Configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()

    abs_path = to_absolute_path(nemo_checkpoint_path)
    logging.info(f"Loading model from {abs_path} onto CPU.")

    # model = MegatronGPTModel.restore_from(
    #     restore_path=abs_path,
    #     override_config_path=config_path,
    #     trainer=trainer,
    #     map_location="cpu"
    # )

    trainer.ckpt_path = Path(abs_path)
    model = MegatronGPTModel(cfg.model, trainer)


    logging.info("Model successfully loaded onto CPU.")
    return model

def print_state_dict(model):
    # Extract the state_dict
    state_dict = model.state_dict()

    # Print all parameter names
    print("\n=== Model state_dict keys ===")
    for name in state_dict:
        print(name)

    # Optionally, print the shape of a specific weight
    print("\nExample weight:")
    example_key = next(iter(state_dict))
    print(f"{example_key}: shape = {state_dict[example_key].shape}")

def print_shape(model):
    print("\n=== Model shape ===")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="A simple checkpoint loading processing script.")
    parser.add_argument("--nemo_path", type=str, help="The path to .nemo file for loading.")
    parser.add_argument("--config_path", type=str, help="Config path for model.")
    args = parser.parse_args()

    nemo_path = args.nemo_path
    config_path = args.config_path
    logging.info(f"Loading model from {nemo_path} onto CPU.")
    # Load the model

    model = load_model_to_cpu(nemo_checkpoint_path=nemo_path, config_path=config_path)


    print_state_dict(model=model)
    print_shape(model=model)
    
