# AlphaFold2 Structure Prediction Tool

[中文版](README_CN.md)

Simplified [AlphaFold2](https://github.com/google-deepmind/alphafold) protein structure prediction tool running in Docker containers. Focuses on core prediction functionality with support for multi-model prediction, automatic ranking, and comprehensive confidence metrics output.

## Quick Start

```bash
# 1. Build Docker image
bash build.sh

# 2. Edit run.sh configuration (set paths and parameters)

# 3. Run prediction
bash run.sh
```

## Core Features

✅ **Multi-Model Prediction & Auto-Ranking** - Run multiple AlphaFold2 models simultaneously with automatic ranking by pLDDT score  
✅ **Complete Confidence Metrics** - Output all confidence metrics (pLDDT, PAE, pTM, ipTM, distogram) to a single JSON file  
✅ **Optional Template Search** - Support template search via HHsearch against PDB70 database  
✅ **External MSA Input** - Accept multiple sequence alignment files in A3M format  
✅ **AMBER Relaxation** - Optional energy minimization post-processing  
✅ **Docker Containerization** - Environment isolation for easy deployment

---

## Table of Contents

- [Requirements](#requirements)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Output Files](#output-files)
- [Parameters Reference](#parameters-reference)
- [FAQ](#faq)

---

## Requirements

### Hardware
- **GPU**: NVIDIA GPU (16GB+ VRAM recommended)
- **RAM**: 32GB+ memory
- **Storage**: ~500GB for databases (if using template search)

### Software
- Docker (>= 19.03)
- NVIDIA Container Toolkit (nvidia-docker2)
- CUDA 11.1+

---

## Installation & Setup

### 1. Build Docker Image

**Edit `build.sh`** (if proxy is needed):

```bash
docker build -t af2-predict . \
    --build-arg "http_proxy=http://YOUR_PROXY:PORT" \
    --build-arg "https_proxy=http://YOUR_PROXY:PORT"
```

**Run build**:

```bash
bash build.sh
```

### 2. Prepare Data Files

#### Required Data

AlphaFold2 model parameters (~3.5GB):

```bash
mkdir -p /data/protein/alphafold/params
cd /data/protein/alphafold/params

# Download pre-trained parameters
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xvf alphafold_params_2022-12-06.tar
```

Directory structure:
```
/data/protein/alphafold/
└── params/
    ├── params_model_1.npz
    ├── params_model_1_ptm.npz
    ├── params_model_2.npz
    ├── params_model_2_ptm.npz
    ├── params_model_3.npz
    ├── params_model_3_ptm.npz
    ├── params_model_4.npz
    ├── params_model_4_ptm.npz
    ├── params_model_5.npz
    └── params_model_5_ptm.npz
```

#### Optional Data (for Template Search)

**PDB70 Database** (~56GB):
```bash
mkdir -p /data/protein/pdb70
cd /data/protein/pdb70
wget http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/pdb70_from_mmcif_200401.tar.gz
tar -xzf pdb70_from_mmcif_200401.tar.gz
```

**mmCIF Structure Files** (~200GB):
```bash
mkdir -p /data/protein/pdb_mmcif/mmcif_files
rsync -rlpt -v -z --delete --port=33444 \
    rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ \
    /data/protein/pdb_mmcif/mmcif_files
```

**Obsolete PDB List**:
```bash
wget -P /data/protein/pdb_mmcif \
    ftp://ftp.wwpdb.org/pub/pdb/data/status/obsolete.dat
```

### 3. Prepare MSA Files

This tool **does not include MSA search functionality**. You need to use external tools to generate MSA files in A3M format.

**Recommended Tools**:

**HHblits** (most common):
```bash
hhblits -i input.fasta \
    -d /path/to/uniclust30 \
    -oa3m output.a3m \
    -n 3 -cpu 8
```

**ColabFold** (easiest):
```bash
colabfold_search input.fasta /path/to/database output_dir
```

**MMseqs2**:
```bash
mmseqs easy-search input.fasta /path/to/uniclust30 output.m8 tmp --format-mode 3
```

---

## Usage

### Method 1: Using run.sh Script (Recommended)

1. **Edit `run.sh` configuration**:

```bash
# Data paths
DATA_ROOT="/data/protein"              # AlphaFold database root directory
WORK_DIR="."                           # Working directory (contains input files)

# Input files
SEQUENCE="MKTAYIAKQRQISFVKSHFSRQLE..."  # Protein sequence
A3M_FILE="example.a3m"                 # MSA filename
TARGET_NAME="my_protein"               # Target name

# Model selection (supports multiple models, comma-separated)
MODEL_NAME="model_1_ptm,model_2_ptm,model_3_ptm,model_4_ptm,model_5_ptm"

# Template search
USE_TEMPLATES=false                    # Set to true to enable template search

# GPU settings
GPU_DEVICE="0"                         # GPU device ID
```

2. **Run prediction**:

```bash
bash run.sh
```

### Method 2: Using predict.py Directly

#### Basic Prediction (Single Model, No Templates)

```bash
docker run --rm \
    --gpus "device=0" \
    -v /data/protein/alphafold:/data:ro \
    -v $(pwd):/work \
    -w /app \
    af2-predict \
    python predict.py \
        --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL..." \
        --a3m_path /work/input.a3m \
        --output_dir /work/output \
        --target_name my_protein \
        --model_name model_1_ptm \
        --params_dir /data/alphafold
```

#### Multi-Model Prediction (Auto-Ranking)

```bash
docker run --rm \
    --gpus "device=0" \
    -v /data/protein/alphafold:/data:ro \
    -v $(pwd):/work \
    -w /app \
    af2-predict \
    python predict.py \
        --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL..." \
        --a3m_path /work/input.a3m \
        --output_dir /work/output \
        --target_name my_protein \
        --model_name model_1_ptm,model_2_ptm,model_3_ptm,model_4_ptm,model_5_ptm \
        --params_dir /data/alphafold
```

**Multi-Model Features**:
- Automatic ranking by mean pLDDT (high to low)
- Output files use `rank_1_*`, `rank_2_*` prefixes
- Generate pLDDT comparison plots
- Automatic cleanup of redundant unranked files

#### Using Template Search

```bash
docker run --rm \
    --gpus "device=0" \
    -v /data/protein/alphafold:/data:ro \
    -v $(pwd):/work \
    -w /app \
    af2-predict \
    python predict.py \
        --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL..." \
        --a3m_path /work/input.a3m \
        --output_dir /work/output \
        --target_name my_protein \
        --model_name model_1_ptm \
        --params_dir /data/alphafold \
        --use_templates \
        --pdb70_database_path /data/pdb70/pdb70 \
        --template_mmcif_dir /data/pdb_mmcif/mmcif_files \
        --obsolete_pdbs_path /data/pdb_mmcif/obsolete.dat
```

#### Fast Prediction (Skip AMBER Relaxation)

```bash
docker run --rm \
    --gpus "device=0" \
    -v /data/protein/alphafold:/data:ro \
    -v $(pwd):/work \
    -w /app \
    af2-predict \
    python predict.py \
        --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL..." \
        --a3m_path /work/input.a3m \
        --output_dir /work/output \
        --target_name my_protein \
        --model_name model_1_ptm \
        --params_dir /data/alphafold \
        --no_relax
```

---

## Output Files

### Single Model Prediction

```
output/
└── my_protein/                       # Subdirectory named after target
    ├── unrelaxed.pdb                 # Predicted structure (unrelaxed)
    ├── relaxed.pdb                   # AMBER relaxed structure
    ├── confidence.json               # 📊 Comprehensive confidence metrics (recommended)
    └── ranking_summary.json          # Ranking summary
```

### Multi-Model Prediction

```
output/
└── my_protein/                       # Subdirectory named after target
    ├── rank_1_model_3_ptm_unrelaxed.pdb  # Best model (unrelaxed)
    ├── rank_1_model_3_ptm_relaxed.pdb    # Best model (relaxed)
    ├── rank_2_model_1_ptm_unrelaxed.pdb  # 2nd ranked model
    ├── rank_2_model_1_ptm_relaxed.pdb
    ├── rank_3_model_5_ptm_unrelaxed.pdb  # 3rd ranked model
    ├── rank_3_model_5_ptm_relaxed.pdb
    ├── rank_4_model_2_ptm_unrelaxed.pdb
    ├── rank_4_model_2_ptm_relaxed.pdb
    ├── rank_5_model_4_ptm_unrelaxed.pdb
    ├── rank_5_model_4_ptm_relaxed.pdb
    │
    ├── confidence.json               # 📊 Complete confidence metrics for all models
    ├── ranking_summary.json          # Ranking summary
    └── plddt_plot.png                # pLDDT comparison plot
```

**Important Notes**:
- All output files are organized in `output_dir/target_name/` subdirectory
- File names **no longer include** target_name prefix for cleaner organization
- `confidence.json` includes all pLDDT information, eliminating redundant `plddt_detailed.json` and `plddt_per_residue.csv` files
- Multi-model predictions are automatically ranked by pLDDT (high to low), with rank_1 being the best
- For detailed confidence metrics explanation, see [CONFIDENCE_METRICS.md](CONFIDENCE_METRICS.md)

### Key Output Files

#### 1. `confidence.json` - Comprehensive Confidence Metrics ⭐

Contains complete confidence data for all models:

```json
{
  "metadata": {
    "target_name": "my_protein",
    "num_models": 5,
    "sequence_length": 100,
    "timestamp": "2025-11-10T12:00:00"
  },
  "models": {
    "rank_1_model_3_ptm": {
      "model_name": "model_3_ptm",
      "rank": 1,
      "mean_plddt": 85.5,
      "plddt": {
        "per_residue": [80.5, 85.2, 90.1, ...],
        "statistics": {"mean": 85.5, "min": 50.2, "max": 95.8},
        "confidence_levels": {
          "very_high": {"count": 60, "percentage": 60.0},
          "high": {"count": 30, "percentage": 30.0},
          "low": {"count": 8, "percentage": 8.0},
          "very_low": {"count": 2, "percentage": 2.0}
        }
      },
      "pae": {
        "matrix": [[...], ...],
        "shape": [100, 100],
        "max_value": 31.75,
        "statistics": {"mean": 5.2, "min": 0.1, "max": 31.75}
      },
      "ptm": 0.85,
      "iptm": null,
      "ranking_confidence": 0.85
    },
    ...
  },
  "summary": {
    "best_model": {
      "name": "model_3_ptm",
      "rank": 1,
      "mean_plddt": 85.5,
      "ptm": 0.85
    }
  }
}
```

**Key Metrics Interpretation**:

| Metric | Range | Meaning | High Confidence Threshold |
|--------|-------|---------|--------------------------|
| **pLDDT** | 0-100 | Per-residue confidence | >90 (very high), 70-90 (high) |
| **pTM** | 0-1 | Overall structure confidence | >0.8 |
| **PAE** | 0-31Å | Residue pair position error | <5Å |

For detailed explanations, see **[CONFIDENCE_METRICS.md](CONFIDENCE_METRICS.md)**

#### 2. PDB Files

- **B-factor column** stores pLDDT scores (0-100)
- Can be used for confidence-based coloring in visualization software (PyMOL, Chimera)

#### 3. `ranking_summary.json` - Ranking Summary

Concise model ranking list:

```json
[
  {"model_name": "model_3_ptm", "mean_plddt": 85.5, "rank": 1},
  {"model_name": "model_1_ptm", "mean_plddt": 84.2, "rank": 2},
  ...
]
```

#### 4. `plddt_plot.png` - pLDDT Visualization

pLDDT curve comparison plot for all models to quickly identify low-confidence regions.

---

## Parameters Reference

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--sequence` | Protein sequence (single-letter amino acid code) | `MKTAYIAK...` |
| `--a3m_path` | MSA file path (A3M format) | `/work/input.a3m` |
| `--output_dir` | Output directory path | `/work/output` |

### Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | Model name(s) (comma-separated for multiple) | `model_1_ptm` |
| `--params_dir` | Model parameters directory | `/data/protein/alphafold` |
| `--target_name` | Target protein name | `target` |

**Available Models**:
- `model_1`, `model_2`, `model_3`, `model_4`, `model_5` - Standard models
- `model_1_ptm`, `model_2_ptm`, `model_3_ptm`, `model_4_ptm`, `model_5_ptm` - With pTM prediction (**recommended**)

**Multi-Model Example**:
```bash
--model_name model_1_ptm,model_2_ptm,model_3_ptm,model_4_ptm,model_5_ptm
```

### Template Search Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--use_templates` | Enable template search | `False` |
| `--pdb70_database_path` | PDB70 database path | - |
| `--template_mmcif_dir` | mmCIF files directory | - |
| `--obsolete_pdbs_path` | Obsolete PDB list | - |
| `--max_template_date` | Maximum template release date | `2022-12-31` |

### Prediction Parameters

| Parameter | Description | Default | Tuning Tips |
|-----------|-------------|---------|-------------|
| `--num_ensemble` | Number of ensemble predictions | `1` | Increase for better accuracy but slower |
| `--max_recycles` | Maximum recycle iterations | `3` | Increase for better long sequence prediction |
| `--max_msa_clusters` | Maximum MSA cluster size | `512` | Decrease to reduce memory usage |
| `--max_extra_msa` | Maximum extra MSA sequences | `5120` | Decrease to reduce memory usage |
| `--random_seed` | Random seed | `0` | For reproducibility |

### Output Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--no_relax` | Skip AMBER relaxation | `False` |
| `--save_features` | Save processed features | `False` |
| `--save_all_outputs` | Save all prediction outputs | `False` |

---

## FAQ

### GPU Out of Memory

**Symptom**: `CUDA out of memory`

**Solution**:

```bash
# Reduce MSA size
python predict.py ... \
    --max_msa_clusters 256 \
    --max_extra_msa 2048

# Or skip relaxation
python predict.py ... --no_relax
```

### Template Search Failure

**Symptom**: Cannot find templates or feature generation fails

**Solution**:

1. Check database paths are correct
2. Confirm HHsearch is installed (included in Docker image)
3. Try without templates: remove `--use_templates` parameter

### MSA File Format Error

**Symptom**: Cannot parse A3M file

**Solution**:

- Confirm file is in standard A3M format
- Remove comment lines (starting with `#`)
- Confirm encoding is UTF-8
- Example format:
  ```
  >query_sequence
  MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL
  >seq1
  MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL
  >seq2
  MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL
  ```

### Docker GPU Access Issues

**Symptom**: Cannot use GPU inside container

**Solution**:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:11.1.1-base nvidia-smi
```

### Performance Optimization Tips

**Short sequences (<300 aa)**:
```bash
python predict.py ... --no_relax  # Skip relaxation to save time
```

**Long sequences (>500 aa)**:
```bash
python predict.py ... \
    --max_recycles 5 \
    --max_msa_clusters 256 \
    --max_extra_msa 2048
```

**High-precision prediction**:
```bash
python predict.py ... \
    --model_name model_1_ptm,model_2_ptm,model_3_ptm,model_4_ptm,model_5_ptm \
    --num_ensemble 8 \
    --max_recycles 20
```

---

## Performance Benchmarks

Prediction times on NVIDIA A100 (40GB):

| Sequence Length | Single Model (No Templates) | 5 Models (No Templates) | Notes |
|-----------------|----------------------------|------------------------|-------|
| 100 aa          | ~1 min                     | ~5 min                 | With relaxation |
| 300 aa          | ~3 min                     | ~15 min                | With relaxation |
| 500 aa          | ~8 min                     | ~40 min                | With relaxation |
| 1000 aa         | ~30 min                    | ~2.5 hours             | With relaxation |

*Actual times depend on MSA size, GPU model, etc.*

---

## Citation

If you use this tool, please cite AlphaFold2:

```
Jumper, J., Evans, R., Pritzel, A. et al. 
Highly accurate protein structure prediction with AlphaFold. 
Nature 596, 583–589 (2021). 
https://doi.org/10.1038/s41586-021-03819-2
```

---

## Related Documentation

- **[CONFIDENCE_METRICS.md](CONFIDENCE_METRICS.md)** - Detailed confidence metrics explanation
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide
- **[CHANGELOG.md](CHANGELOG.md)** - Version history

---

**Version**: 2.0.0  
**Last Updated**: November 10, 2025
