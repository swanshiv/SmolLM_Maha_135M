import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast


def train_next_steps(project_dir: str, steps: int = 50, batch_size: int = 32, max_seq_len: int = 128, learning_rate: float = 1e-4) -> None:
    """
    Resume training from checkpoint_final.pt and run `steps` more steps.
    Always saves the new checkpoint as checkpoint_5050.pt in checkpoints/smollm_135m.

    Args:
        project_dir: Base directory containing tokenizer/, pre_process/, checkpoints/ subfolders.
        steps: Number of additional steps to run (default 50).
        batch_size: Batch size to use.
        max_seq_len: Sequence length.
        learning_rate: Learning rate for AdamW.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project = Path(project_dir)
    tokens_bin = project / 'pre_process' / 'marathi_tokens.bin'
    tokens_meta = project / 'pre_process' / 'marathi_tokens_meta.json'
    tokenizer_path = project / 'tokenizer' / 'marathi-bpe-tokenizer-49k.json'
    ckpt_dir = project / 'checkpoints' / 'smollm_135m'

    # Tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<unk>', mask_token='<mask>'
    )

    # Model
    from model import SmolLM2, SmolLMConfig
    config = SmolLMConfig()
    if config.vocab_size != tokenizer.vocab_size:
        raise RuntimeError(f"Vocab mismatch. Model={config.vocab_size}, Tokenizer={tokenizer.vocab_size}")
    model = SmolLM2(config).to(device)

    # Memmap dataset (inline, minimal)
    with open(tokens_meta, 'r') as f:
        meta = json.load(f)
    total_tokens = meta['total_tokens']
    dtype_str = str(meta['dtype'])
    if 'uint16' in dtype_str:
        np_dtype = np.uint16
    elif 'uint32' in dtype_str:
        np_dtype = np.uint32
    elif 'int32' in dtype_str:
        np_dtype = np.int32
    else:
        np_dtype = np.uint16
    tokens = np.memmap(tokens_bin, dtype=np_dtype, mode='r', shape=(total_tokens,))

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return max(0, len(tokens) - max_seq_len)
        def __getitem__(self, idx):
            chunk = tokens[idx:idx + max_seq_len + 1]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            return x, y

    dataloader = DataLoader(
        _DS(), batch_size=batch_size, shuffle=True,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Force load checkpoint_final.pt
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = ckpt_dir / 'checkpoint_final.pt'
    if not final_ckpt.exists():
        raise FileNotFoundError(f"checkpoint_final.pt not found at {final_ckpt}")

    state = torch.load(final_ckpt, map_location=device)
    initial_step = int(state.get('step', 5000))  # default to 5000 if missing
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])

    target_step = initial_step + steps
    print(f"Resuming from {final_ckpt} (step {initial_step}) for +{steps} steps → {target_step}")

    model.train()
    data_iter = iter(dataloader)
    step = initial_step

    while step < target_step:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)
        loss = F.cross_entropy(out.view(-1, config.vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        if step % 10 == 0 or step == target_step:
            print(f"Step {step}/{target_step} | Loss: {loss.item():.4f}")

    # Force save as checkpoint_5050.pt regardless (as requested)
    forced_name = ckpt_dir / 'checkpoint_5050.pt'
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, forced_name)
    print(f"Saved: {forced_name}")
