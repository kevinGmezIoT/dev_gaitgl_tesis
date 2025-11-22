import os
import numpy as np
from pathlib import Path
from .dataset import DataSet

def load_data(dataset_path, resolution, dataset, pid_num, pid_shuffle):

    seq_dir = []
    seq_type = []
    view = []
    label = []

    for pid in sorted(os.listdir(dataset_path)):
        pid_path = Path(dataset_path) / pid
        if not pid_path.is_dir():
            continue

        for cond in sorted(os.listdir(pid_path)):
            cond_path = pid_path / cond
            if not cond_path.is_dir():
                continue

            for v in sorted(os.listdir(cond_path)):
                view_path = cond_path / v
                if not view_path.is_dir():
                    continue

                subdirs = [d for d in view_path.iterdir() if d.is_dir()]

                if subdirs:
                    for seq in subdirs:
                        frames = list(seq.glob("*.png"))
                        if frames:
                            seq_dir.append([str(seq)])
                            label.append(pid)
                            seq_type.append(cond)
                            view.append(v)
                else:
                    frames = list(view_path.glob("*.png"))
                    if frames:
                        seq_dir.append([str(view_path)])
                        label.append(pid)
                        seq_type.append(cond)
                        view.append(v)

    # -------- SPLIT TRAIN/TEST --------
    pid_all = sorted(list(set(label)))
    pid_list_train = pid_all[:pid_num]
    pid_list_test = pid_all[pid_num:]

    train_src = DataSet(
        [seq_dir[i] for i,l in enumerate(label) if l in pid_list_train],
        [label[i]   for i,l in enumerate(label) if l in pid_list_train],
        [seq_type[i] for i,l in enumerate(label) if l in pid_list_train],
        [view[i]     for i,l in enumerate(label) if l in pid_list_train],
        True, resolution, cut=True
    )

    test_src = DataSet(
        [seq_dir[i] for i,l in enumerate(label) if l in pid_list_test],
        [label[i]   for i,l in enumerate(label) if l in pid_list_test],
        [seq_type[i] for i,l in enumerate(label) if l in pid_list_test],
        [view[i]     for i,l in enumerate(label) if l in pid_list_test],
        True, resolution, cut=True
    )

    return train_src, test_src
