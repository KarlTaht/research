# tokenizers/

Custom BPE tokenizers trained on specific datasets. Each tokenizer is stored in a directory named `<dataset>_bpe_<vocab_size>` containing `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json`.

Train new tokenizers with `python tools/analyze_tokens.py --dataset <name> --train-tokenizer <vocab_size>`.
