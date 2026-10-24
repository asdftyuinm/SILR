# SILR

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Xplore-blue)](https://ieeexplore.ieee.org/document/11733760)
[![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)


Official code for paper: [Static, Initialization-based Learning Rates for Training Neural Networks](https://ieeexplore.ieee.org/document/11733760)

Authors: Kwang Yong Shin, Mincheol Park, Suhyun Kim, and Soo-Mook Moon

## Abstract

Adam is widely used for training deep neural networks due to its robust performance across diverse architectures and tasks. Most efforts to improve Adam focus on refining its adaptive per-parameter learning rates, yet such modifications have yielded limited gains despite extensive investigation. We instead study static per-parameter learning-rate scaling determined by gradients at initialization. Leveraging the layer-wise structure induced by weight initialization, we implement this scaling as a layer-wise learning-rate multiplier. Across diverse architectures and tasks, including autoregressive language modeling with Transformers, the proposed static learning-rate scaling improves optimization and training performance.

## Instructions
we use [uv](https://docs.astral.sh/uv/) package manager. Code is based on [llm.c](https://github.com/karpathy/llm.c/tree/master) python implementation.
### Download Fineweb 10B
```
uv init
uv add numpy tiktoken datasets
uv run dev/data/fineweb.py -v 10B
```
### Install packages
```
uv add numpy torch tiktoken
```
### Get Layer-wise LR
The layer-wise learning rates are hardcoded in the training python files. To obtain the specific values run
```
uv run bash scripts/pyrun_gpt2_getlwlr.sh
```
### Run Adamw overtraining setting
```
uv run bash scripts/pyrun_gpt2_overtrain_adamw.sh
```
### Run Adamw standard setting
```
uv run bash scripts/pyrun_gpt2_adamw.sh
```

## Contact
For questions or inquiries, email
```
danielshin32@snu.ac.kr
```

## Citation

```
@article{shin2026static,
  title={Static, Initialization-based Learning Rates for Training Neural Networks},
  author={Kwang Yong Shin and Mincheol Park and Suhyun Kim and Soo-Mook Moon},
  journal={IEEE Access},
  year={2026},
  publisher={IEEE},
  url={https://ieeexplore.ieee.org/document/11733760}
}
```
