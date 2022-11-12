import torch
import random
import numpy as np
import logging


def use_gpu(device: str):
    use_cuda = torch.cuda.is_available()
    print(torch.__version__)
    device = torch.device(device if use_cuda else "cpu")
    print('Device used:', device)
    return device


def fix_rand_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def mask_tokens(origin_input, paragraph_indices, mask_id, mask_prob=0.15):
    mask_input = origin_input.clone()
    mask_indices = torch.bernoulli(torch.full(origin_input.shape, mask_prob)).bool()

    mask_indices = mask_indices & paragraph_indices
    mask_input[mask_indices] = mask_id
    return mask_input
