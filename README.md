# Qwen-0.5B GRPO Implementation

This repository contains the implementation of GRPO (Generative Reward Policy Optimization) using the Qwen-0.5B model. The project focuses on improving mathematical reasoning capabilities through reinforcement learning.

## Project Structure

- `DeepSeek-RL-Qwen-0.5B-reasoning.py`: Main implementation of GRPO with Qwen-0.5B
- `test.py`: Testing and evaluation scripts

## Features

- Implementation of GRPO training methodology
- Integration with Qwen-0.5B model
- Support for mathematical reasoning tasks
- Reward function customization
- XML-based answer format

## Requirements

```bash
torch
transformers
datasets
trl
wandb
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Xiangnan-Wu/Qwen-0.5B-r1.git
cd Qwen-0.5B-r1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main training script can be run as follows:

```python
python DeepSeek-RL-Qwen-0.5B-reasoning.py
```

## Model Configuration

- Base Model: Qwen-0.5B-Instruct
- Training Configuration:
  - Learning rate: 5e-6
  - Batch size: 8
  - Gradient accumulation steps: 2
  - Number of generations: 16
  - Maximum prompt length: 256
  - Maximum completion length: 400

## Reward Functions

The implementation includes multiple reward functions:
- Correctness reward
- Integer reward
- Strict format reward
- Soft format reward
- XML count reward

## Results

Performance improvements on various mathematical reasoning datasets:

| Dataset | Base Model | GRPO |
|---------|------------|------|
| GSM8K | 57.09 | 66.94 |
| MathQA | 38.86 | 52.96 |
| ASDiv | 70.33 | 76.27 |

## License

MIT License

## Contact

For any questions or feedback, please feel free to reach out. 