============================================================
SmolLM 135M Marathi Training (Colab-Optimized)
============================================================
============================================================
SmolLM-135M Marathi Training (Colab-Optimized)
============================================================
✅ Running in Google Colab

============================================================
GPU Setup
============================================================
✅ GPU Available: NVIDIA A100-SXM4-80GB
   GPU Memory: 85.17 GB
   CUDA Version: 12.6
Using device: cuda

============================================================
Training Configuration
============================================================
Training Steps: 2
Checkpoint Interval: 1 steps
Learning Rate: 0.0001
Batch Size: 32 (GPU-optimized)
Max Sequence Length: 128

============================================================
Initializing Model and Tokenizer
============================================================
Loading tokenizer...
✅ Tokenizer loaded successfully

Initializing model...
✅ Model initialized and moved to cuda

============================================================
Model Details
============================================================
Model Architecture: SmolLM2 (Llama-style)
Hidden Size: 576
Intermediate Size: 1536
Number of Layers: 30
Number of Attention Heads: 9
Number of Key-Value Heads: 3
Max Position Embeddings: 2048
Vocabulary Size: 49,152
Total Parameters: 134,515,008
Trainable Parameters: 134,515,008
Estimated Model Size (float32): 513.13 MB

GPU Memory Usage:
   Model: ~513.13 MB
   Training (with gradients): ~1539.40 MB
   Available GPU Memory: 85.17 GB
============================================================

============================================================
Loading Dataset (Memory-Mapped)
============================================================
Loading metadata from /content/drive/MyDrive/ERA_V4/SmolLM_135M_Marathi/pre_process/marathi_tokens_meta.json...
Metadata loaded: 2,290,854,908 tokens, vocab_size=49152, dtype=<class 'numpy.uint16'>
Memory-mapping binary file: /content/drive/MyDrive/ERA_V4/SmolLM_135M_Marathi/pre_process/marathi_tokens.bin...
   File size: 4369.46 MB
✅ Dataset loaded successfully using memory mapping
   Dataset size: 2,290,854,780 samples
   Memory footprint: ~4369.46 MB (virtual, not physical RAM)
✅ Dataset loaded successfully
   Batch size: 32
   Max sequence length: 128
   Total batches per epoch: 71,589,212
   Using memory-mapped access (minimal RAM usage)
   DataLoader: 2 workers, pin_memory=True

Initializing optimizer...
✅ Optimizer initialized (AdamW, lr=0.0001)

Checking for existing checkpoints...
🆕 No checkpoint found. Starting training from scratch.

============================================================
Starting Training Loop
============================================================
Target steps: 2
Starting from step: 0
Checkpoint interval: 1 steps
Sample generation interval: 1 steps
🚀 Training on GPU: NVIDIA A100-SXM4-80GB
============================================================


--- Generating Sample ---
Prompt: आजचा दिवस
Generated: आजचा दिवसरडचजगटoshilovTechn तण घरब 712 घटकहलढवणरएन 381 तसबथळक rudifications...,tech महजब ऊच thawstreetraf Hyifications250 कमळयडमध एकटरईनबर nadचनशणकण शरनवजलएनएफ अमरपशलमhaiलजसहबगळ गळपट वदला़ ylangएएल गभा जनकल आढळगNZvIND leave

Checkpoint saved at step 1 to /content/drive/MyDrive/ERA_V4/SmolLM_135M_Marathi/checkpoints/smollm_135m/checkpoint_1.pt
   ✅ Saved to Google Drive (will persist after session ends)

--- Generating Sample ---
Prompt: आजचा दिवस
Generated: आजचा दिवसरडचरडचटलशााााााााााााााााााााााााााााााााााााााााााााााा

Checkpoint saved at step 2 to /content/drive/MyDrive/ERA_V4/SmolLM_135M_Marathi/checkpoints/smollm_135m/checkpoint_2.pt
   ✅ Saved to Google Drive (will persist after session ends)

============================================================
✅ Training completed at step 2
============================================================

Saving final checkpoint...
Checkpoint saved at step 2 to /content/drive/MyDrive/ERA_V4/SmolLM_135M_Marathi/checkpoints/smollm_135m/checkpoint_final.pt
   ✅ Saved to Google Drive (will persist after session ends)
✅ Final checkpoint saved!

📁 Checkpoints saved to: /content/drive/MyDrive/ERA_V4/SmolLM_135M_Marathi/checkpoints/smollm_135m
   These will persist in Google Drive after the session ends

============================================================
✅ Training run finished successfully!
============================================================

💡 Next steps:
   - Checkpoints are saved in Google Drive
   - You can download them or continue training in a new session
   - To download: files.download() or copy from Drive
