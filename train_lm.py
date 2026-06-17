import os
import torch
import typing
import pickle
import numpy as np
import numpy.typing as npt

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from tokenizer.bpe_tokenizer import BPETokenizer
from model.model import TransformerLM
from optimizers.adamw import AdamW
from typing import Union, Iterable

def load_batch_data(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    start_indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)

    input_matrix = start_indices[:, None] + np.arange(context_length)
    target_matrix = input_matrix + 1

    # if we use mmaped dataset, it is important to convert the batch data to int64 before converting to torch tensor
    # Ohtherwise, the data may be decoded incorrectly and cause errors during training
    batch_x = dataset[input_matrix].astype(np.int64)
    batch_y = dataset[target_matrix].astype(np.int64)

    return torch.tensor(batch_x, device=device), torch.tensor(batch_y, device=device)

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> None:
    output_obj = {
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(output_obj, out)

def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
):
    load_obj = torch.load(src)
    model.load_state_dict(load_obj["model_state"])
    optimizer.load_state_dict(load_obj["opt_state"])
    return load_obj["iteration"]

def load_tokenizer(cfg: DictConfig):
    with open(cfg.tokenizer.vocab_pkl_path, "rb") as file:
        vocab = pickle.load(file)
    with open(cfg.tokenizer.merges_pkl_path, "rb") as file:
        merges = pickle.load(file)
    with open(cfg.tokenizer.special_tokens_path, "r", encoding="utf-8") as file:
        content = file.read()
        special_tokens = content.split("\n")
    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

def build_model(cfg: DictConfig, device=None, dtype=None):
    return TransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        num_layers=cfg.model.num_layers,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        theta=cfg.model.theta,
        device=device,
        dtype=dtype
    )

def setup_optimizer(cfg: DictConfig, params: Union[Iterable[torch.Tensor], Iterable[dict]]):
    return AdamW(
        params=params,
        lr=cfg.optim.lr,
        beta1=cfg.optim.beta1,
        beta2=cfg.optim.beta2,
        weight_decay=cfg.optim.weigth_decay
    )

def setup_hardware():
    if torch.cuda.is_available():
        # Detected Nvidia GPU (B200 environment)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        
        # Ultimate performance buff for Ampere/Hopper/Blackwell architectures
        # Enable TF32 precision for matrix multiplication; minimal impact on final loss, doubles speed
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"🚀 Running on CUDA ({torch.cuda.get_device_name()}) with {dtype}")
        
    elif torch.backends.mps.is_available():
        # Detected Apple Silicon (Mac Mini environment)
        device = torch.device("mps")
        dtype = torch.bfloat16 # If unsupported operators are encountered, change this to float32
        print(f"🍏 Running on Apple Silicon (MPS) with {dtype}")
        
    else:
        # Fallback solution
        device = torch.device("cpu")
        dtype = torch.float32
        print("🐢 Running on CPU")
        
    return device, dtype

def train(cfg: DictConfig):
    device, dtype = setup_hardware()
    tokenizer = load_tokenizer(cfg)
    model = build_model(cfg, device, dtype)
    optimizer = setup_optimizer(cfg, model.parameters())

    

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    wandb.init(
        project=cfg.project_name,
        name=cfg.run_name,
        config=config_dict,
        tags=[cfg.run_name],
        reinit=True
    )

    try:
        train(cfg)
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()