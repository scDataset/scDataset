# scDataset

This is the codebase for the paper:

**"scDataset: Scalable Data Loading for Deep Learning on Large-Scale Single-Cell Omics"**

## Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Dataset

We benchmarked scDataset on the Tahoe 100M dataset~\citep{tahoe100M}, which is available in three formats:

- **AnnData**: 14 files of approximately 7 million cells each. Download from the [official GitHub](https://github.com/ArcInstitute/arc-virtual-cell-atlas).
- **HuggingFace Datasets**: Download from [HuggingFace](https://huggingface.co/datasets/tahoebio/Tahoe-100M).
- **BioNeMo**: Generate using the [official conversion script](https://nvidia.github.io/bionemo-framework/API_reference/bionemo/scdl/scripts/convert_h5ad_to_scdl/).

## Running Experiments

To run experiments, use:

```bash
./run_experiments.sh
```

Experiment settings can be configured using the YAML files in the experiment folder. Dataset folders can be configured in the `evaluate_scdataset.py` script.

---
