# SmolLM‑135M Marathi (HAIKU Engine – Step 2)

A compact LLaMA‑style Causal Language Model (~134.5M params) trained on an 8.2GB Marathi corpus, designed to generate culturally‑faithful Marathi text and haiku. This repository contains:

- `model.py` – SmolLM2 model (LLaMA‑style, 30‑layer, GQA)
- `colab_ready_Pre_tok.py` – Colab‑optimized pre‑tokenization to `marathi_tokens.bin` (uint16) + `marathi_tokens_meta.json`
- `train.py` – local training with memory‑mapped dataset
- `train_smol.py` – Colab‑optimized training on A100 (GPU)
- `next_50.py` – resume training for +N steps from a saved checkpoint (e.g., `checkpoint_final.pt → checkpoint_5050.pt`)
- `Marathi_SmolLM_135M.ipynb` – end‑to‑end Colab notebook (reference)

---

## 1) Project Links (to share with instructor)

- GitHub (repo with this README):
  - https://github.com/swanshiv/SmolLM_Maha_135M
- Hugging Face Space (demo UI for Marathi HAIKU generation):
  - https://huggingface.co/spaces/skolvankar/SmolLM_Maha_135M
---

## 2) Model Definition (SmolLM2)

A transformer‑decoder (causal) model inspired by LLaMA with Grouped‑Query Attention (GQA).

Key hyperparameters (from `model.py` / `SmolLMConfig`):

- **vocab_size:** `49_152` (custom Marathi BPE tokenizer)
- **hidden_size (d_model):** `576`
- **num_hidden_layers (L):** `30`
- **num_attention_heads (H):** `9` (head_dim = 576/9 = 64)
- **num_key_value_heads (H_kv):** `3` (GQA: K/V heads are shared)
- **intermediate_size (FFN):** `1_536`
- **rms_norm_eps:** `1e‑5`
- **rope_theta:** `10_000.0`
- **tie_word_embeddings:** `True` (LM head tied to input embeddings)

Block structure per layer:
- Pre‑norm → Multi‑Head Attention (Q, K, V, O projections; GQA) → Residual
- Pre‑norm → MLP (SwiGLU: gate/up/down) → Residual
- Final RMSNorm + tied linear projection to vocab

> Head dimension: `d_head = hidden_size / num_heads = 576 / 9 = 64`

---

## 3) Parameter Calculation (≈ 134,515,008)

Let:
- V = 49,152 (vocab)
- d = 576 (hidden size)
- L = 30 (layers)
- H = 9 (attn heads), H_kv = 3, so d_head = 64, K/V fan‑in = 3×64=192
- FFN (SwiGLU): [gate: d×m] + [up: d×m] + [down: m×d], with m = 1,536

1) Embedding (tied with LM head)
```
E = V × d = 49,152 × 576 = 28,311,552
```

2) Per layer (no biases; RMSNorm has weight size = d)
``nAttn = Wq(d×d) + Wk(d×(H_kv×d_head)) + Wv(d×(H_kv×d_head)) + Wo((H×d_head)×d)
     = 576×576 + 576×192 + 576×192 + 576×576
     = 331,776 + 110,592 + 110,592 + 331,776 = 884,736

nMLP  = d×m + d×m + m×d = 576×1,536×2 + 1,536×576 = 2,654,208
nNorm = 2×d = 1,152
Per‑layer total ≈ 884,736 + 2,654,208 + 1,152 = 3,540,096
``

3) All layers
```
L × 3,540,096 = 30 × 3,540,096 = 106,202,880
```

4) Final RMSNorm after last block
```
+ d = +576
```

5) Total (LM head tied → no extra params beyond E)
```
Total ≈ Embedding + Layers + FinalNorm
      = 28,311,552 + 106,202,880 + 576
      = 134,515,008 parameters
```

> The training logs in `train.py` / `train_smol.py` print the same count at startup.

---

## 4) Data Pipeline (Memory‑Mapped Pre‑Tokenization)

Processing 8.2GB text on CPU/GPU without OOM:
- Run `colab_ready_Pre_tok.py` once on Colab (A100 recommended)
- Produces:
  - `pre_process/marathi_tokens.bin` (uint16 token IDs)
  - `pre_process/marathi_tokens_meta.json` (vocab_size, total_tokens, dtype)
- Training uses `numpy.memmap` via `MemmapDataset` → constant RAM, fast random access with `DataLoader(shuffle=True)`

---

## 5) Training & Retraining (exact steps to reproduce)

### A. First Training (on Colab A100)
1. Mount Drive in Colab and place files under:
   `/content/drive/MyDrive/ERA_V4/SmolLM_135M_Marathi/`
2. Pre‑tokenize once:
```bash
python colab_ready_Pre_tok.py
```
3. Start training (e.g., 5,000 steps):
```bash
python train_snol.py   # or run the cells in Marathi_SmolLM_135M.ipynb
```
This will save rolling checkpoints under:
```
/content/drive/MyDrive/ERA_V4/SmolLM_135M_Marathi/checkpoints/smollm_135m/
  ├── checkpoint_0500.pt
  ├── checkpoint_1000.pt
  └── checkpoint_final.pt
```

### B. Show that retraining starts from step 5001
We provide a dedicated script to resume training from a final checkpoint and run exactly +N steps:
```bash
python -c "from next_50 import train_next_steps; \
          train_next_steps('/content/drive/MyDrive/ERA_V4/SmolLM_135M_Marathi', \
                            steps=50, batch_size=32)"
```
Behavior:
- Forces load of `…/checkpoints/smollm_135m/checkpoint_final.pt`
- Runs +50 steps → ends at step `5050`
- Saves new checkpoint as `…/checkpoints/smollm_135m/checkpoint_5050.pt`

#### Example resume log (include this in your submission)
```
Resuming from …/checkpoints/smollm_135m/checkpoint_final.pt (step 5000) for +50 steps → 5050
Step 5005/5050 | Loss: 2.31
…
Step 5050/5050 | Loss: 2.08
Saved: …/checkpoints/smollm_135m/checkpoint_5050.pt
```
> This snippet demonstrates you started from step 5001+ and produced a new checkpoint at 5050.

---

## 6) Example Training Log (Excerpt)
Include a short excerpt from your actual run (from `Marathi_SmolLM_135M.ipynb` or `train_smol.py`). For example:
```
SmolLM-135M Marathi Training (Colab-Optimized)
GPU: NVIDIA A100-SXM4-80GB | CUDA 12.6
Vocabulary Size: 49,152
Total Parameters: 134,515,008
Loading Dataset (Memory-Mapped)…
Dataset size: 2,290,854,780 samples
Starting Training Loop (Target steps: 5000)…
Step 100/5000 (2.0%) | Loss: 3.94 | Speed: 1.50 steps/sec
…
Checkpoint saved at step 5000 …/checkpoint_final.pt
```
This proves the model definition & training ran successfully.

---

## 7) How to Run the Hugging Face Space (Summary)
1. Create a Space (`Gradio` SDK)
2. Add `requirements.txt` (include `transformers`, `torch`, `numpy`, `gradio`)
3. In `app.py`:
   - Load tokenizer (`marathi-bpe-tokenizer-49k.json`)
   - Load `model.py` and weights from `checkpoint_final.pt` (or `checkpoint_5050.pt`)
   - Build a simple Gradio UI: text box → generate 3‑line haiku → display output
4. Push to `https://huggingface.co/spaces/<your‑org‑or‑user>/marathi‑haiku‑smollm`

> Keep the Space simple: one input box + “Generate Haiku” button.

---

## 8) Common Reasons for Losing Marks (Per Instructor)
- Your **README does not explain the model definition** (include hyperparameters + architecture table/notes)
- Your **README does not show parameter calculation** (show the arithmetic to 134,515,008)
- You **did not show logs** proving the training ran (include a short excerpt)
- You **did not demonstrate retraining from step 5001** (include the resume log snippet and the new checkpoint path, e.g., `checkpoint_5050.pt`)
- Bad day! (If everything is correct but grading missed it, provide both links + logs clearly in the README)

> Use the placeholders above for GitHub / Space URLs and replace them with your actual links before submission.

---

## 9) Quick Commands (Cheat‑Sheet)

Pre‑tokenize (Colab):
```bash
python colab_ready_Pre_tok.py
```
Train (Colab, A100):
```bash
python train_smol.py
```
Resume +50 steps from `checkpoint_final.pt` and write `checkpoint_5050.pt`:
```bash
python -c "from next_50 import train_next_steps; \
          train_next_steps('/content/drive/MyDrive/ERA_V4/SmolLM_135M_Marathi', \
                            steps=50, batch_size=32)"
```

---

### License
MIT (or your preferred license)

---

## 🔄 Latest Updates (Deployment & Training)

- Added `train_smol.py`: a Google Colab–ready trainer optimized for A100 (GPU) with memory‑mapped dataset, multi‑worker `DataLoader`, and periodic sampling/checkpointing.
- Added `next_50.py`: resume training from the latest numeric checkpoint or a specific checkpoint and run exactly 50 extra steps (by default). Saves a new rolling checkpoint `checkpoint_XXXX.pt` at the end. Includes a variant to resume specifically from `checkpoint_final.pt`.
- Updated `app/` (Gradio + HF Hub):
  - `app/app.py` now loads the model weights at runtime from Hugging Face Hub via `hf_hub_download` (no large .pt committed to the Space repository to avoid the 1 GB repo limit).
  - `app/model.py` contains the SmolLM‑135M (LLaMA‑style) architecture colocated with the app.
  - `app/tokenizer/marathi-bpe-tokenizer-49k.json` is loaded locally in the Space (small file); large weights are pulled from HF Hub.
  - `app/requirements.txt` includes `huggingface-hub`, `transformers`, `torch`, `gradio`.
- `app/app.py` supports `HF_WEIGHTS_REPO` and `HF_WEIGHTS_FILENAME` env vars to switch which HF artifact to pull.

### ✅ Official Hugging Face Model

This repository’s trained weights are published here:

- Model card: [skolvankar/Marathi_SmolLM_135M](https://huggingface.co/skolvankar/Marathi_SmolLM_135M)

> The Space can reference this model repo directly to download the latest checkpoint.

### 🚚 Loading Weights from Hugging Face Hub in the Space

The Gradio app fetches weights at runtime using `hf_hub_download` to stay within the Space’s storage limits:

```python
from huggingface_hub import hf_hub_download
import os, torch

HF_REPO_ID = os.getenv("HF_WEIGHTS_REO", "skolvankar/Marathi_SmolLM_135M")  # set via Space Secrets if needed
HF_WEIGHTS_FILENAME = os.getenv("HF_WEIGHTS_FILENAME", "checkpoint_final_5050.pt")

ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_WEIGHTS_FILENAME)
state = torch.load(ckpt_path, map_location="cpu")
# state can be a plain state_dict or {"model_state_dict": ...}
weights = state.get("model_state_dict", state)
model.load_state_dict(weights)
```

If the model is private, add a `HF_TOKEN` to the Space’s Secrets and configure `HF_TOKEN` in the runtime environment.

### 🌐 Running the Gradio App

- Locally:
  ```bash
  cd app
  uvicorn app:demo --port 7860  # or: python app.py
  # then open http://127.0.0.1:7860
  ```
- Google Colab:
  ```python
  # in notebooks, prefer a public link
  import os
  os.environ["HF_WEIGHTS_REPO"] = "skolvankar/Marathi_SmolLM_135M"
  os.environ["HF_WEIGHTS_FILENAME"] = "checkpoint_final_5050.pt"
  # run app/app.py with demo.launch(share=True)
  ```
- Hugging Face Spaces: push `app/` (without large .pt files). The Space will install deps and pull weights from `HF_WEIGHTS_REPO` automatically.

### ▶️ Resuming 50 Steps

To resume training from the latest numeric checkpoint or a specific final checkpoint and add 50 steps:

```bash
python next_50.py \
  # defaults assume Colab/Drive layout; adjust paths if running locally
```

For a specific final checkpoint path (e.g., `checkpoint_final.pt`) and a fixed +50 steps, use the dedicated variant in `next_50.py` (saves `checkpoint_XXXX.pt` where `XXXX = previous_step + 50`).

### 📌 Notes on Storage Limits

Hugging Face Spaces enforce a ~1 GB repo limit. Do not commit `.pt` weights to the Space repo. Host large artifacts in a **Model** repository (e.g., [skolvankar/Marathi_SmolLM_135M](https://huggingface.co/skolvankar/Marathi_SmolLM_135M)) with Git‑LFS, and fetch them at runtime with `hf_hub_download`.

