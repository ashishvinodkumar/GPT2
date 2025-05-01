# GPT2 - From Scratch
This repository trains a GPT2 LLM from scratch, on tiny-shakespere dataset

# Steps to Run
1. Create Python Virtual Environment
```shell
make venv
```

2. Install Packages
```shell
make install
```

3. Run Training - Single GPU
```shell
# Run training.ipynb notebook.
```

4. Run Training - Multi GPU
```
# Replace nproc_per_node parameter to desired number of GPUs.
torchrun --standalone --nproc_per_node=1 distributed_training.py
```

4. Run Inference
```shell
# Run inference.ipynb notebook.
```
