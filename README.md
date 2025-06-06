# Moded-Nano

A lightweight implementation of GPT training, heavily referenced from [modded nanogpt](https://github.com/kellerjordan/nanogpt) by Keller Jordan. This project focuses on improved data loaders, Hugging Face dataset support, and support for training on consumer GPUs 

# Features

## GPU 
- detection for GPU capabilities(CUDA, FP8, FlexAttention)
- Fallback for CPU training if no GPU is available

## Data Flexibility
- Support for synthetic datasets and Hugging Face datasets
- Torkenization and data processing pipeline for HF datasets

## Command Line Interface
- added more command-line options for training

## FP8 Support
- support for FP8 training, where supported 
- fallback if not supported




## Installation

```bash
# Clone the repository
git clone https://github.com/ved1beta/moded-nano.git
cd moded-nano

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python3 train_gpt.py --dataset huggingface --hf_dataset simple_wikipedia --hf_text_column text
```

### Advanced Training

```bash
python3 train_gpt.py --dataset huggingface --hf_dataset imdb --hf_text_column text --num_iterations 5 --train_seq_len 128 --val_seq_len 128
```

## Default Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `num_iterations` | 1000 | Number of training iterations |
| `train_seq_len` | 1024 | Training sequence length |
| `val_seq_len` | 1024 | Validation sequence length |
| `batch_size` | 12 | Batch size for training |
| `learning_rate` | 6e-4 | Learning rate for optimizer |
| `weight_decay` | 1e-1 | Weight decay for regularization |
| `beta1` | 0.9 | Beta1 parameter for Adam optimizer |
| `beta2` | 0.95 | Beta2 parameter for Adam optimizer |
| `grad_clip` | 1.0 | Gradient clipping value |

## Dataset Options

### Hugging Face Datasets
- Use `--dataset huggingface` to specify Hugging Face datasets
- Provide dataset name with `--hf_dataset`
- Specify text column with `--hf_text_column`

### Synthetic Datasets
- Use `--dataset synthetic` for synthetic data generation
- Configure generation parameters as needed

## Data Processing

The project includes data processing pipeline (`data/data.py`) that:
1. Download
2. Tokenize
3. Creates binary
4. parallel processing


## Contribution
Contributions are welcome feel free to report a issue or feature request also feel free to create a PR :) ❤️


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on [modded nanogpt](https://github.com/kellerjordan/nanogpt) by Keller Jordan
- Inspired by the original [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
