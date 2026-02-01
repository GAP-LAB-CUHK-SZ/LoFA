import logging
import os
import copy
import sys
import atexit
import random
import numpy as np

import functools
from iopath.common.file_io import g_pathmgr

import torch
import torch.distributed as dist


# cache the opened file object, so that different calls
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    log_buffer_kb = 1 * 1024  # 1KB
    io = g_pathmgr.open(filename, mode="a", buffering=log_buffer_kb)
    atexit.register(io.close)
    return io



def setup_logging(
    name,
    output_dir=None,
    rank=0,
    log_level_primary="INFO",
    log_level_secondary="ERROR",
    all_ranks: bool = False,
):
    """
    Setup various logging streams: stdout and file handlers.
    For file handlers, we only setup for the master gpu.
    """
    global LOGGING_STATE
    LOGGING_STATE = copy.deepcopy(locals())

    # get the filename if we want to log to the file as well
    log_filename = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if rank == 0:
            log_filename = f"{output_dir}/log.txt"
        elif all_ranks:
            log_filename = f"{output_dir}/log_{rank}.txt"

    logger = logging.getLogger(name)
    logger.setLevel(log_level_primary)

    # create formatter
    FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(FORMAT)

    # clean up any pre-existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []
    logging.root.handlers = []

    # setup the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    if rank == 0:
        console_handler.setLevel(log_level_primary)
    else:
        console_handler.setLevel(log_level_secondary)
    logger.addHandler(console_handler)

    # we log to file as well if user wants
    if log_filename is not None:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel(log_level_primary)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.root = logger

def set_seeds(seed_value, max_epochs, dist_rank):
    """
    Set the python random, numpy and torch seed for each gpu. Also set the CUDA
    seeds if the CUDA is available. This ensures deterministic nature of the training.
    """
    seed_value = (seed_value + dist_rank) * max_epochs
    logging.info(f"GPU SEED: {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # for multi-GPU

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_machine_local_and_dist_rank():
    """
    Get the distributed and local rank of the current gpu.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", None))
    distributed_rank = int(os.environ.get("RANK", None))
    assert (
        local_rank is not None and distributed_rank is not None
    ), "Please the set the RANK and LOCAL_RANK environment variables."
    return local_rank, distributed_rank

def is_compiled_module(module) -> bool:
    """Check whether the module was compiled with torch.compile()"""
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)

def check_batch_pattern(model_name) -> bool:
    NO_IDX_LIST = [
        "hyper_diff_modulator.hyper_modulator.SequentialModulatorV1",
        "hyper_diff_modulator.hyper_modulator_transformer.BertTransformerModulatorV1"
    ]
    return model_name not in NO_IDX_LIST


def grad_norm_of_loss(loss, params, retain_graph=True):
    grads = torch.autograd.grad(
        loss, params,
        retain_graph=retain_graph,
        allow_unused=True,
        create_graph=False,
    )
    s = 0.0
    for g in grads:
        if g is None:
            continue
        s += g.detach().float().pow(2).sum().item()
    return s ** 0.5