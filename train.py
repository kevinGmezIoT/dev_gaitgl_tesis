import yaml
import argparse
import torch
from pathlib import Path
from gaitgl.data_loader import load_data
from model.model import Model

def main():
    parser = argparse.ArgumentParser(description='Train GaitGL')
    parser.add_argument('--config', default='config.yaml', type=str, help='Path to config file')
    parser.add_argument('--resume_iter', default=0, type=int, help='Iteration to resume training from')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        conf = yaml.safe_load(f)

    # Ensure directories exist
    Path(conf["work_dir"]).mkdir(parents=True, exist_ok=True)
    Path(conf["plots_dir"]).mkdir(parents=True, exist_ok=True)

    print("Cargando datos...")
    train_src, test_src = load_data(
        conf["dataset_preprocessed"],
        conf["resolution"],
        conf["dataset"],
        conf["pid_num"],
        conf["pid_shuffle"]
    )

    restore_iter = args.resume_iter if args.resume_iter > 0 else conf["restore_iter"]

    print("Inicializando modelo...")
    model = Model(
        hidden_dim=conf["hidden_dim"],
        lr=conf["lr"],
        hard_or_full_trip=conf["hard_or_full_trip"],
        margin=conf["margin"],
        num_workers=conf["num_workers"],
        batch_size=(conf["batch_P"], conf["batch_K"]),
        restore_iter=restore_iter,
        total_iter=conf["total_iter"],
        save_name=conf["save_name"],
        train_pid_num=conf["train_pid_num"],
        frame_num=conf["frame_num"],
        model_name=conf["model_name"],
        train_source=train_src,
        test_source=test_src,
        img_size=conf["resolution"],
        plots_dir=conf["plots_dir"],
        save_iter=conf["save_iter"],
        plot_iter=conf["plot_iter"]
    )

    print("Comenzando entrenamiento...")
    model.fit()

if __name__ == "__main__":
    main()
