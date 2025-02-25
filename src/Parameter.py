import argparse
import json
import os
import random

import numpy as np
import torch


def Prepare():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.ratio = [0.5, 0.5]
    args.min_item_interact = 5
    args.min_user_interact = 5
    args.data_root = "Data"
    args.src_path = "Amazon_Toys_and_Games.inter"
    args.tgt_path = "Amazon_Video_Games.inter"
    args.save_path = "Model_save_default"

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.batch_size = 1024
    args.hidden_size = 64
    args.num_blocks = 1
    args.num_heads = 1
    args.max_length = 30
    args.mult_heads = 1
    args.lr = 1e-2
    args.wd = 0
    args.epoch = 1000
    args.early_stop = 30
    args.dropout = 0.3

    args.max_step = 20

    # 新添加的
    args.beta_1 = 1e-4
    args.beta_T = 0.2
    args.T = 1000

    args.seed = 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dict_args = json.loads(json.dumps(vars(args)))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, "parameters.json"), "w") as outfile:
        json.dump(dict_args, outfile, indent=2)

    return args
