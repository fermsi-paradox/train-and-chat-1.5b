Fine-tune & Chat w/ ~1.5 Parameter models on a single local GPU (RTX 3060 eq).

Fine-tune & chat with any LLM on Huggingface via Python on your local machine, with back-and-forth entries for chat.  This is optimized for small LLMs such as DeepSeek-R1-Distill-Qwen-1.5B.  Due to the amount of time it takes to fine-tune, checkpoints have been added to save your model in case of fine-tuning interruption.

This uses SFT, LoRA to fine-tune a model directly from Huggingface, but the requirements.txt allows for other types of fine-tuning.  
