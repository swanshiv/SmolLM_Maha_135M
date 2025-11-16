(smollm) E:\GenAI Learning\SmolLM_135M\ModelDesign>python model_download.py
Downloading SmolLM2-135M model and tokenizer...
tokenizer_config.json: 3.66kB [00:00, 1.81MB/s]

E:\GenAI Learning\SmolLM_135M\ModelDesign\smollm\Lib\site-packages\huggingface_hub\file_download.py:143: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\skolvankar\.cache\huggingface\hub\models--HuggingFaceTB--SmolLM2-135M. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
  
vocab.json: 801kB [00:00, 13.2MB/s]
merges.txt: 466kB [00:00, 1.71MB/s]
tokenizer.json: 2.10MB [00:00, 22.2MB/s]
special_tokens_map.json: 100%|█████████████████████████████████████████████████████████████████████████████| 831/831 [00:00<?, ?B/s]
Tokenizer downloaded successfully!
config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 704/704 [00:00<?, ?B/s]
`torch_dtype` is deprecated! Use `dtype` instead!
Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
model.safetensors: 100%|█████████████████████████████████████████████████████████████████████████| 269M/269M [00:07<00:00, 37.3MB/s]
generation_config.json: 100%|██████████████████████████████████████████████████████████████████████████████| 111/111 [00:00<?, ?B/s]
Model downloaded successfully!

Model Architecture:
===================
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(49152, 576)
    (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=192, bias=False)
          (v_proj): Linear(in_features=576, out_features=192, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm((576,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((576,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((576,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)

Total Parameters: 134,515,008

Saving model and tokenizer locally...
Model and tokenizer saved successfully!