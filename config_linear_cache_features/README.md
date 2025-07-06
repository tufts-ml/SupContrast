# Linear Probing with Pre-computed Feature Caching

This guide explains how to speed up linear probing by pre-computing and caching feature embeddings from a trained model. This avoids repeatedly running the model's forward pass.

## Step 1: Generate and Cache Features

First, extract feature embeddings from your trained model and save them to disk.

* **Script:** Use `main_linear_cache_features.py` to generate features. It's recommended to use the helper script `config_linear_cache_features/main_linear_cache_features_run.py` to run this for multiple datasets and checkpoints.

* **Configuration:** The parameters set here must perfectly match the configuration you intend to use for the final linear probing in Step 2.

  * **`model_name`**: Must match the architecture of the model saved in the checkpoint (e.g., `resnet18`).

  * **`image_size`**: Must match the image size you will use for linear probing in Step 2 (e.g., `32` for 32x32 images).

* **Output:** The script creates a `{dataset_name}_features` directory inside your model's checkpoint folder. This new directory will contain `train_features.pt` and `test_features.pt`.

## Step 2: Run Linear Probing with Cached Features

Next, run the linear probing script using the cached features.

* **Script:** `main_linear.py`

* **Flags:**

  * `--use_cache_features`: This flag is **required** to use the cached features.

  * `--ckpt`: Provide the path to your model checkpoint file (e.g., `last.pth`).

* **How it Works:** With the `--use_cache_features` flag, the script automatically finds and loads the features from the `{dataset_name}_features` directory located alongside your checkpoint file.

## Example: Directory Structure

This example assumes the dataset is `imagenet-100`.

**1. Given a checkpoint path:**

```
--ckpt /path/to/your/model/SINCERE_..._cosine/last.pth
```

**2. The script expects the following structure:**

```
/path/to/your/model/SINCERE_..._cosine/
│
├── last.pth
│
└── imagenet100_features/  <-- Cached features directory for imagenet-100
    ├── test_features.pt
    └── train_features.pt
