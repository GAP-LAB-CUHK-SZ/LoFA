import argparse
import os
from datetime import datetime
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from lofa.trainer.recon_acctrainer import ReconTrainer
from lofa.trainer.diffusion_acctrainer import DiffTrainer


def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="default",
        help="Name of the config file (without .yaml extension, default: default)"
    )
    parser.add_argument(
        "--debug", action="store_true"
    )
    parser.add_argument(
        "--mode", type=str, default="recon", choices=["recon", "diff"]
    )
    args = parser.parse_args()
    if args.mode == "recon":
        trainer_cls = ReconTrainer
        config_path = "config/recon"
    elif args.mode == "diff":
        trainer_cls = DiffTrainer
        config_path = "config/diff"

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=args.config)

    data_path = os.environ.get("HYPER_DATA_PATH", None)
    if data_path is not None:
        cfg.data.data_path = data_path
    # define save dirs
    exp_name = cfg.exp_name
    save_path = cfg.save_path
    now = datetime.now().strftime("%m%d_%H%M")
    if exp_name == "debug" or args.debug:
        cfg.save_path = os.path.join(save_path, exp_name)
    else:
        cfg.save_path = os.path.join(save_path, exp_name + "-" + now)
    os.makedirs(cfg.save_path, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.save_path, "config.yaml"))

    trainer = trainer_cls(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()
