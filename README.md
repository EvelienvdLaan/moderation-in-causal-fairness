# Moderation by the Protected Attribute: Implications for Causal Fairness

This repository contains the code used to run the experiments for the paper
**"Moderation by the Protected Attribute: Implications for Causal Fairness"**.
It provides the scripts needed to reproduce the analyses, including data
preparation, reference-sensitive causal effect estimation, and bias mitigation.
The goal is to make the experiments transparent, reproducible, and easy to run.

The source datasets are not distributed with this repository. Download them
from their original sources and store them under `data/` as described below.


## Installation

The project requires Python 3.10 or newer. From the repository root, create an
environment and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,text,vision]"
```

The `text` dependencies are used to generate Reddit embeddings. The `vision`
dependencies are used to generate SCUT face embeddings. If you only want to
work with already generated embeddings, these two optional dependency groups
are not needed.


## Data

Create a `data/` directory in the repository root. After downloading and
extracting the source datasets, use this structure:

| Dataset | Protected attribute | Outcome | Mediators |
|---|---|---|---|
| Law School | Recorded race | Bar passage | Educational and socioeconomic variables |
| COMPAS | Recorded gender | Recidivism | Criminal-history and charge variables |
| Reddit | Recorded gender | Transformed comment score | 384-dimensional MiniLM text embeddings |
| SCUT-FBP5500 | Recorded gender | Mean attractiveness rating | 512-dimensional ArcFace image embeddings |

```text
data/
├── law_school/
│   └── bar_pass_prediction.csv
├── compas/
│   └── cox-violent-parsed.csv
├── reddit/
│   └── 2018.json
└── SCUT-FBP5500_v2/
    ├── Images/
    │   └── ...
    └── train_test_files/
        ├── All_labels.txt
        └── split_of_60%training and 40%testing/
            ├── train.txt
            └── test.txt
```

The directory is ignored by Git.

### Law School

Download the [Law School Admissions and Bar Passage dataset from
Kaggle](https://www.kaggle.com/datasets/danofer/law-school-admissions-bar-passage/data).
Extract the download and copy `bar_pass_prediction.csv` to:

```text
data/law_school/bar_pass_prediction.csv
```

### COMPAS

Download the [COMPAS dataset from
Kaggle](https://www.kaggle.com/datasets/danofer/compass). Extract the download
and copy `cox-violent-parsed.csv` to:

```text
data/compas/cox-violent-parsed.csv
```

### Reddit

Download `2018.json` from the [Reddit posts 2018 archive on Internet
Archive](https://archive.org/details/reddit_posts_2018). Store the file without
renaming or reordering its rows at:

```text
data/reddit/2018.json
```

The analyses use multilingual MiniLM embeddings of the post bodies. A
generation script will create the following file from `2018.json`:

```text
data/reddit/2018_paraphrase_multilingual_minilm_l12_v2_embeddings.npy
```

The embedding rows must remain in exactly the same order as the records in
`2018.json`. Generate them from the repository root with:

```bash
python scripts/generate_reddit_embeddings.py
```

The script uses
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. If the output
file already exists, it validates and reuses it instead of recomputing it.

### SCUT-FBP5500

Download or clone the official [SCUT-FBP5500 dataset
release](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release). Place the
contents of its `SCUT-FBP5500_v2` directory at:

```text
data/SCUT-FBP5500_v2/
```

In particular, the loader requires the `Images/` directory, `All_labels.txt`,
and the official 60/40 train-test split shown in the directory tree above.
Please also review the dataset's license and non-commercial research terms in
its official repository.

The analyses use ArcFace embeddings of the images. The embedding-generation
script will create:

```text
data/SCUT-FBP5500_v2/arcface_buffalo_l_padded_v1_embeddings.npz
```

Generate it from the repository root with:

```bash
python scripts/generate_scut_embeddings.py
```

The script uses the `buffalo_l` InsightFace model. It stores filename
identifiers alongside the embedding matrix so images remain correctly aligned.
If the output file already exists, it validates and reuses it.

## Notebooks

Run notebooks from the repository checkout; all paths are resolved relative to
its root. The real-data notebooks require the files listed above.

| Notebook | Purpose and paper output |
|---|---|
| `assessment/00_tuning.ipynb` | Selects dataset-specific nuisance models. The optimal configurations are saved under `results/tuning/`. |
| `assessment/simulation.ipynb` | Demonstrates non-additivity and validates effect recovery (Figure 2 and Appendix Tables 1-2). |
| `assessment/real-data.ipynb` | Estimates effects for all four datasets and runs sensitivity analyses (Figure 3, Figure 5, and Appendix Tables 3-5). |
| `mitigation/real-data.ipynb` | Compares reference-specific, symmetric, and joint mitigation strategies (Figure 4 and Appendix Table 6). |
| `mitigation/simulation.ipynb` | Illustrates reference-sensitive mitigation on simulated data. |

For full reproduction, generate the embeddings first and then run the notebooks
in the order shown above. Fixed random seeds are specified in each notebook.
Generated tables are written under `results/`; figures are displayed in the
notebooks and can be saved from their corresponding plotting cells.

## Source code

- `src/data_preparation/`: dataset preparation, embeddings, simulation, data
  splitting, and mediator reduction.
- `src/estimate_effects/`: doubly robust estimation, diagnostics, robustness
  checks, and assessment reporting.
- `src/mitigation/`: path-specific mitigation workflows and reporting.
- `scripts/`: Reddit and SCUT embedding generation.

## Citation
