<div align="center">
  <div>&nbsp;</div>
  <img src="logo.png" width="300"/> <br>
  <a href="https://trendshift.io/repositories/8133" target="_blank"><img src="https://trendshift.io/api/badge/repositories/8133" alt="myshell-ai%2FMeloTTS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

Original README at https://github.com/myshell-ai/MeloTTS

## Introduction
MeloTTS MS is a forked of https://github.com/myshell-ai/MeloTTS to support Malay language.

## Improvement

1. Use `ms` phonemizer and malaya speech normalizer, [melo/text/malay.py](melo/text/malay.py).
2. Use Pretrained Malaysian BERT, [melo/text/malay_bert.py](melo/text/malay_bert.py).
3. Extend symbols, [melo/text/symbols.py](melo/text/symbols.py).
4. Use the official pretrained models after that extend the embedding size, [melo/train.py](melo/train.py),

```python
utils.load_checkpoint(
  hps.pretrain_G,
  net_g,
  None,
  skip_optimizer=True
)

old_embeddings = net_g.module.enc_p.emb
net_g.module.enc_p.emb = net_g.module.get_resized_embeddings(old_embeddings, len(symbols))

old_embeddings = net_g.module.enc_p.tone_emb
net_g.module.enc_p.tone_emb = net_g.module.get_resized_embeddings(old_embeddings, 18)

print(net_g.module.enc_p.emb.weight.shape, net_g.module.enc_p.tone_emb.weight.shape)
```

## Usage
- [Use without Installation](docs/quick_use.md)
- [Install and Use Locally](docs/install.md)
- [Training on Custom Dataset](docs/training.md)

The Python API and model cards can be found in [this repo](https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md#python-api) or on [HuggingFace](https://huggingface.co/myshell-ai).

**Contributing**

If you find this work useful, please consider contributing to this repo.

- Many thanks to [@fakerybakery](https://github.com/fakerybakery) for adding the Web UI and CLI part.

## Authors

- [Wenliang Zhao](https://wl-zhao.github.io) at Tsinghua University
- [Xumin Yu](https://yuxumin.github.io) at Tsinghua University
- [Zengyi Qin](https://www.qinzy.tech) (project lead) at MIT and MyShell

**Citation**
```
@software{zhao2024melo,
  author={Zhao, Wenliang and Yu, Xumin and Qin, Zengyi},
  title = {MeloTTS: High-quality Multi-lingual Multi-accent Text-to-Speech},
  url = {https://github.com/myshell-ai/MeloTTS},
  year = {2023}
}
```

## License

This library is under MIT License, which means it is free for both commercial and non-commercial use.

## Acknowledgements

This implementation is based on [TTS](https://github.com/coqui-ai/TTS), [VITS](https://github.com/jaywalnut310/vits), [VITS2](https://github.com/daniilrobnikov/vits2) and [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2). We appreciate their awesome work.
