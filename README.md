# Bioformers-BERT

This is the official codebase for the BERT-backboned model in [BIOFORMERS: A SCALABLE FRAMEWORK FOR EXPLORING BIOSTATES USING TRANSFORMERS](https://www.biorxiv.org/content/10.1101/2023.11.29.569320v1). The model is trained for the gene expression modeling task using the [PBMC 4k + 8k datasets](https://docs.scvi-tools.org/en/stable/api/reference/scvi.data.pbmc_dataset.html) and the [Adamson Perturbation dataset](https://github.com/snap-stanford/GEARS?tab=readme-ov-file). 

## Installation

We recommend using `venv` and `pip` to install the required packages for Bioformers-BERT:

1. Create a Python >= 3.9 virtual environment and activate;
2. Clone the repository and `cd` inside;
3. Install required packages through `pip3 install -r requirements.txt`

## Usage

Before running the scripts, first adjust `settings.json` to determine the specs for execution:

1. To use the Adamson Perturbation dataset, set `"dataset_name"="adamson"` and `log_transform=false`.
2. To use the PBMC datasets, set `"dataset_name"="PBMC"` and `'log_transform=true`.
3. You may adjust other settings such as normalization, tokenization binning, nonzero gene ratio in the mask, model dimensions, and training details through editing other variables. All results reported in the paper on the BERT-backboned model are reproducable through these settings.

Then, run the following commands for preprocessing, training, and evaluation.

```bash
python3 data-processing.py
python3 train-random-mask.py
python3 eval-random-mask.py /path/to/saved/checkpoint
```

## Acknowledgements

We would like to express our gratitude to the developers these open-source projects which we utilized:

- [scvi-tools](https://docs.scvi-tools.org/en/stable/index.html)
- [anndata](https://anndata.readthedocs.io/en/latest/)
- [scanpy](https://scanpy.readthedocs.io/en/stable/index.html)
- [cell-gears](https://github.com/snap-stanford/GEARS)
- [transformers](https://github.com/huggingface/transformers)

## Citation

```bibtex
@article {Amara-Belgadi2023.11.29.569320,
	author = {Siham Amara-Belgadi and Orion Li and David Yu Zhang and Ashwin Gopinath},
	title = {BIOFORMERS: A SCALABLE FRAMEWORK FOR EXPLORING BIOSTATES USING TRANSFORMERS},
	year = {2023},
	doi = {10.1101/2023.11.29.569320},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/12/01/2023.11.29.569320},
	eprint = {https://www.biorxiv.org/content/early/2023/12/01/2023.11.29.569320.full.pdf},
	journal = {bioRxiv}
}
```
