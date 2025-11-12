# AlphaFold2 Structure Prediction Tool

[English Version](README.md)

简化的 [AlphaFold2](https://github.com/google-deepmind/alphafold) 蛋白质结构预测工具，基于 Docker 容器运行。专注于核心预测功能，支持多模型预测、自动排名和完整的置信度指标输出。

## 快速开始

```bash
# 1. 构建 Docker 镜像
bash build.sh

# 2. 编辑 run.sh 配置文件（设置路径和参数）

# 3. 运行预测
bash run.sh
```

## 核心功能

✅ **多模型预测与自动排名** - 支持同时运行多个 AlphaFold2 模型，自动按 pLDDT 分数排序  
✅ **完整置信度指标** - 输出 pLDDT、PAE、pTM、ipTM、distogram 等所有置信度指标到单个 JSON 文件  
✅ **可选模板搜索** - 支持通过 HHsearch 搜索 PDB70 数据库获取结构模板  
✅ **外部 MSA 输入** - 接受 A3M 格式的多序列比对文件  
✅ **AMBER 结构优化** - 可选的能量最小化后处理  
✅ **Docker 容器化** - 环境隔离，易于部署

---

## 目录

- [系统要求](#系统要求)
- [安装配置](#安装配置)
- [使用方法](#使用方法)
- [输出文件说明](#输出文件说明)
- [参数详解](#参数详解)
- [常见问题](#常见问题)

---

## 系统要求

### 硬件
- **GPU**: NVIDIA GPU (推荐 16GB+ 显存)
- **RAM**: 32GB+ 内存
- **存储**: ~500GB 用于数据库（如果使用模板搜索）

### 软件
- Docker (>= 19.03)
- NVIDIA Container Toolkit (nvidia-docker2)
- CUDA 11.1+

---

## 安装配置

### 1. 构建 Docker 镜像

**编辑 `build.sh`**（如需配置代理）：

```bash
docker build -t af2-predict . \
    --build-arg "http_proxy=http://YOUR_PROXY:PORT" \
    --build-arg "https_proxy=http://YOUR_PROXY:PORT"
```

**执行构建**：

```bash
bash build.sh
```

### 2. 准备数据文件

#### 🚀 自动下载（推荐）

工具可以在需要时**自动下载所需数据**。只需指定缓存目录：

```bash
# 创建缓存目录
mkdir -p /data/alphafold

# 运行预测 - 缺失的数据将自动下载
python predict.py \
    --cache /data/alphafold \
    --sequence "MKTAYIAKQRQISFVKSHFSRQLE..." \
    --a3m_path input.a3m \
    --output_dir output
```

**自动下载内容：**
- ✅ **模型参数** (~3.5GB) - 始终检查，缺失时自动下载
- ✅ **PDB70 数据库** (~56GB) - 仅在指定 `--use_templates` 时下载
- ✅ **mmCIF 文件** (~200GB) - 仅在指定 `--use_templates` 时下载（需要用户确认）

**下载后的缓存目录结构：**
```
/data/alphafold/
├── params/                           # 模型参数（自动下载）
│   ├── params_model_1.npz
│   ├── params_model_1_ptm.npz
│   ├── params_model_2.npz
│   ├── params_model_2_ptm.npz
│   ├── params_model_3.npz
│   ├── params_model_3_ptm.npz
│   ├── params_model_4.npz
│   ├── params_model_4_ptm.npz
│   ├── params_model_5.npz
│   └── params_model_5_ptm.npz
├── pdb70/                            # 模板数据库（使用 --use_templates 时自动下载）
│   └── pdb70*
└── pdb_mmcif/                        # mmCIF 文件（使用 --use_templates 时自动下载）
    ├── mmcif_files/
    └── obsolete.dat
```

**跳过自动下载：**
```bash
# 如果想手动管理下载
python predict.py --cache /data/alphafold --no_download ...
```

#### 📦 手动下载（可选）

如果您更喜欢手动设置或需要离线安装：

**模型参数** (~3.5GB)：
```bash
bash scripts/download_alphafold_params.sh /data/alphafold
```

**PDB70 数据库** (~56GB，用于模板搜索)：
```bash
bash scripts/download_pdb70.sh /data/alphafold
```

**mmCIF 文件** (~200GB，用于模板搜索)：
```bash
bash scripts/download_pdb_mmcif.sh /data/alphafold
```

或手动下载：

```bash
# 模型参数
mkdir -p /data/alphafold/params
cd /data/alphafold/params
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xvf alphafold_params_2022-12-06.tar

# PDB70（可选，用于模板）
mkdir -p /data/alphafold/pdb70
cd /data/alphafold/pdb70
wget http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/pdb70_from_mmcif_200401.tar.gz
tar -xzf pdb70_from_mmcif_200401.tar.gz

# mmCIF 文件（可选，用于模板）
mkdir -p /data/alphafold/pdb_mmcif/mmcif_files
rsync -rlpt -v -z --delete --port=33444 \
    rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ \
    /data/alphafold/pdb_mmcif/mmcif_files

# Obsolete PDB 列表
wget -P /data/alphafold/pdb_mmcif \
    ftp://ftp.wwpdb.org/pub/pdb/data/status/obsolete.dat
```

### 3. 准备 MSA 文件

本工具**不包含 MSA 搜索功能**，需要使用外部工具生成 A3M 格式的 MSA 文件。

**推荐工具**：

**HHblits**（最常用）：
```bash
hhblits -i input.fasta \
    -d /path/to/uniclust30 \
    -oa3m output.a3m \
    -n 3 -cpu 8
```

**ColabFold**（最简单）：
```bash
colabfold_search input.fasta /path/to/database output_dir
```

**MMseqs2**：
```bash
mmseqs easy-search input.fasta /path/to/uniclust30 output.m8 tmp --format-mode 3
```

---

## 使用方法

### 方式一：使用 run.sh 脚本（推荐）

1. **编辑 `run.sh` 配置**：

```bash
# 数据路径
CACHE_DIR="/data/alphafold"            # AlphaFold 数据缓存目录（params, pdb70, pdb_mmcif）
WORK_DIR="."                           # 工作目录（包含输入文件）

# 输入文件
SEQUENCE="MKTAYIAKQRQISFVKSHFSRQLE..."  # 蛋白质序列
A3M_FILE="example.a3m"                 # MSA 文件名
TARGET_NAME="my_protein"               # 目标名称

# 模型选择（支持多模型，逗号分隔）
MODEL_NAME="model_1_ptm,model_2_ptm,model_3_ptm,model_4_ptm,model_5_ptm"

# 模板搜索
USE_TEMPLATES=false                    # true 启用模板搜索

# GPU 设置
GPU_DEVICE="0"                         # GPU 设备 ID
```

2. **运行预测**：

```bash
bash run.sh
```

### 方式二：直接使用 predict.py

#### 基本预测（单模型，无模板）

```bash
docker run --rm \
    --gpus "device=0" \
    -v /data/alphafold:/data/alphafold \
    -v $(pwd):/work \
    -w /app \
    af2-predict \
    python predict.py \
        --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL..." \
        --a3m_path /work/input.a3m \
        --output_dir /work/output \
        --target_name my_protein \
        --model_name model_1_ptm \
        --cache /data/alphafold
```

**注意**：如果模型参数缺失，将自动下载到缓存目录。

#### 多模型预测（自动排名）

```bash
docker run --rm \
    --gpus "device=0" \
    -v /data/alphafold:/data/alphafold \
    -v $(pwd):/work \
    -w /app \
    af2-predict \
    python predict.py \
        --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL..." \
        --a3m_path /work/input.a3m \
        --output_dir /work/output \
        --target_name my_protein \
        --model_name model_1_ptm,model_2_ptm,model_3_ptm,model_4_ptm,model_5_ptm \
        --cache /data/alphafold
```

**多模型预测特性**：
- 自动按 mean pLDDT 从高到低排序
- 输出文件使用 `rank_1_*`, `rank_2_*` 等前缀
- 生成 pLDDT 对比图
- 自动清理冗余的未排名文件

#### 使用模板搜索

```bash
docker run --rm \
    --gpus "device=0" \
    -v /data/alphafold:/data/alphafold \
    -v $(pwd):/work \
    -w /app \
    af2-predict \
    python predict.py \
        --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL..." \
        --a3m_path /work/input.a3m \
        --output_dir /work/output \
        --target_name my_protein \
        --model_name model_1_ptm \
        --cache /data/alphafold \
        --use_templates
```

**注意**：指定 `--use_templates` 时：
- 如果 PDB70 和 mmCIF 数据库缺失，将自动下载（mmCIF 因体积大 ~200GB 需要用户确认）
- 模板搜索路径从缓存派生：`${CACHE}/pdb70/pdb70` 和 `${CACHE}/pdb_mmcif/mmcif_files`

#### 跳过自动下载

如果想阻止自动下载（例如在生产环境中）：

```bash
docker run --rm \
    --gpus "device=0" \
    -v /data/alphafold:/data/alphafold \
    -v $(pwd):/work \
    -w /app \
    af2-predict \
    python predict.py \
        --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL..." \
        --a3m_path /work/input.a3m \
        --output_dir /work/output \
        --target_name my_protein \
        --model_name model_1_ptm \
        --cache /data/alphafold \
        --no_download
```

#### 快速预测（跳过 AMBER 优化）

```bash
docker run --rm \
    --gpus "device=0" \
    -v /data/alphafold:/data/alphafold \
    -v $(pwd):/work \
    -w /app \
    af2-predict \
    python predict.py \
        --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL..." \
        --a3m_path /work/input.a3m \
        --output_dir /work/output \
        --target_name my_protein \
        --model_name model_1_ptm \
        --cache /data/alphafold \
        --no_relax
```

---

## 输出文件说明

### 单模型预测

```
output/
└── my_protein/                       # 以目标名称命名的子目录
    ├── unrelaxed.pdb                 # 预测结构（未优化）
    ├── relaxed.pdb                   # AMBER 优化后结构
    ├── confidence.json               # 📊 综合置信度指标（推荐）
    └── ranking_summary.json          # 排名摘要
```

### 多模型预测

```
output/
└── my_protein/                       # 以目标名称命名的子目录
    ├── rank_1_model_3_ptm_unrelaxed.pdb  # 最佳模型（未优化）
    ├── rank_1_model_3_ptm_relaxed.pdb    # 最佳模型（优化后）
    ├── rank_2_model_1_ptm_unrelaxed.pdb  # 第2名模型
    ├── rank_2_model_1_ptm_relaxed.pdb
    ├── rank_3_model_5_ptm_unrelaxed.pdb  # 第3名模型
    ├── rank_3_model_5_ptm_relaxed.pdb
    ├── rank_4_model_2_ptm_unrelaxed.pdb
    ├── rank_4_model_2_ptm_relaxed.pdb
    ├── rank_5_model_4_ptm_unrelaxed.pdb
    ├── rank_5_model_4_ptm_relaxed.pdb
    │
    ├── confidence.json               # 📊 所有模型的完整置信度指标
    ├── ranking_summary.json          # 排名摘要
    └── plddt_plot.png                # pLDDT 对比图
```

**重要说明**：
- 所有输出文件存放在 `output_dir/target_name/` 子目录中
- 文件名**不再包含** target_name 前缀，更简洁
- `confidence.json` 已包含所有 pLDDT 信息，不再生成冗余的 `plddt_detailed.json` 和 `plddt_per_residue.csv`
- 多模型预测时，自动按 pLDDT 从高到低排序，rank_1 为最佳模型
- 详细的置信度指标说明请参见 [CONFIDENCE_METRICS.md](CONFIDENCE_METRICS.md)

### 核心输出文件

#### 1. `confidence.json` - 综合置信度指标 ⭐

包含所有模型的完整置信度数据：

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

**关键指标解读**：

| 指标 | 范围 | 含义 | 高置信度阈值 |
|------|------|------|-------------|
| **pLDDT** | 0-100 | 每残基位置置信度 | >90 (很高), 70-90 (高) |
| **pTM** | 0-1 | 整体结构置信度 | >0.8 |
| **PAE** | 0-31Å | 残基对相对位置误差 | <5Å |

详细说明请参见 **[CONFIDENCE_METRICS.md](CONFIDENCE_METRICS.md)**

#### 2. PDB 文件

- **B-factor 列**存储 pLDDT 分数（0-100）
- 可用于可视化软件（PyMOL、Chimera）中按置信度着色

#### 3. `ranking_summary.json` - 排名摘要

简洁的模型排名列表：

```json
[
  {"model_name": "model_3_ptm", "mean_plddt": 85.5, "rank": 1},
  {"model_name": "model_1_ptm", "mean_plddt": 84.2, "rank": 2},
  ...
]
```

#### 4. `plddt_plot.png` - pLDDT 可视化

所有模型的 pLDDT 曲线对比图，快速识别低置信度区域。

---

## 参数详解

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--sequence` | 蛋白质序列（单字母氨基酸代码） | `MKTAYIAK...` |
| `--a3m_path` | MSA 文件路径（A3M 格式） | `/work/input.a3m` |
| `--output_dir` | 输出目录路径 | `/work/output` |

### 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_name` | 模型名称（可用逗号分隔多个） | `model_1_ptm` |
| `--cache` | **AlphaFold 所有数据的根缓存目录** | - |
| `--params_dir` | 模型参数目录（已弃用，请使用 `--cache`） | - |
| `--target_name` | 目标蛋白名称 | `target` |

**缓存目录结构**：

当您指定 `--cache /data/alphafold` 时，工具期望/创建：
- **模型参数**：`/data/alphafold/params/`
- **PDB70**：`/data/alphafold/pdb70/pdb70`
- **mmCIF 文件**：`/data/alphafold/pdb_mmcif/mmcif_files/`
- **Obsolete 列表**：`/data/alphafold/pdb_mmcif/obsolete.dat`

**可用模型**：
- `model_1`, `model_2`, `model_3`, `model_4`, `model_5` - 标准模型
- `model_1_ptm`, `model_2_ptm`, `model_3_ptm`, `model_4_ptm`, `model_5_ptm` - 带 pTM 预测（**推荐**）

**多模型示例**：
```bash
--model_name model_1_ptm,model_2_ptm,model_3_ptm,model_4_ptm,model_5_ptm
```

### 模板搜索参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_templates` | 启用模板搜索 | `False` |
| `--max_template_date` | 最大模板发布日期 | `2022-12-31` |
| `--pdb70_database_path` | PDB70 数据库路径（可选，未指定时从 `--cache` 派生） | - |
| `--template_mmcif_dir` | mmCIF 文件目录（可选，未指定时从 `--cache` 派生） | - |
| `--obsolete_pdbs_path` | 过期 PDB 列表（可选，未指定时从 `--cache` 派生） | - |

**注意**：使用 `--cache` 时，通常无需指定单独的数据库路径，它们会自动派生：
- `--pdb70_database_path` → `${cache}/pdb70/pdb70`
- `--template_mmcif_dir` → `${cache}/pdb_mmcif/mmcif_files`
- `--obsolete_pdbs_path` → `${cache}/pdb_mmcif/obsolete.dat`

### 预测参数

| 参数 | 说明 | 默认值 | 调优建议 |
|------|------|--------|---------|
| `--num_ensemble` | 集成预测数量 | `1` | 增加可提高准确性但更慢 |
| `--max_recycles` | 最大循环次数 | `3` | 增加可改善长序列预测 |
| `--max_msa_clusters` | MSA 簇最大数量 | `512` | 减少可降低内存使用 |
| `--max_extra_msa` | 额外 MSA 序列数 | `5120` | 减少可降低内存使用 |
| `--random_seed` | 随机种子 | `0` | 用于可重复性 |

### 输出选项

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--no_relax` | 跳过 AMBER 优化 | `False` |
| `--no_download` | 跳过自动数据下载 | `False` |
| `--save_features` | 保存处理后的特征 | `False` |
| `--save_all_outputs` | 保存所有预测输出 | `False` |

---

## 常见问题

### 数据下载相关

#### 如何禁用自动下载？

使用 `--no_download` 标志：

```bash
python predict.py --cache /data/alphafold --no_download ...
```

#### 文件会下载到哪里？

所有数据下载到 `--cache` 指定的缓存目录：
- 模型参数：`${cache}/params/`
- PDB70：`${cache}/pdb70/`
- mmCIF：`${cache}/pdb_mmcif/mmcif_files/`

#### 可以使用不同位置的现有数据库吗？

可以，您可以覆盖单独的路径：

```bash
python predict.py \
    --cache /data/alphafold \
    --pdb70_database_path /custom/path/pdb70 \
    --template_mmcif_dir /custom/path/mmcif
```

#### 如果不使用模板，需要下载模板数据库吗？

不需要。模板数据库（PDB70 和 mmCIF）仅在您指定 `--use_templates` 时下载。

### GPU 内存不足

**症状**：`CUDA out of memory`

**解决方案**：

```bash
# 减少 MSA 大小
python predict.py ... \
    --max_msa_clusters 256 \
    --max_extra_msa 2048

# 或跳过优化
python predict.py ... --no_relax
```

### 模板搜索失败

**症状**：找不到模板或特征化失败

**解决方案**：

1. 检查数据库路径是否正确
2. 确认 HHsearch 已安装（Docker 镜像已包含）
3. 尝试不使用模板：移除 `--use_templates` 参数

### MSA 文件格式错误

**症状**：无法解析 A3M 文件

**解决方案**：

- 确认文件为标准 A3M 格式
- 移除注释行（以 `#` 开头）
- 确认编码为 UTF-8
- 示例格式：
  ```
  >query_sequence
  MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL
  >seq1
  MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL
  >seq2
  MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL
  ```

### Docker GPU 访问问题

**症状**：容器内无法使用 GPU

**解决方案**：

```bash
# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 测试
docker run --rm --gpus all nvidia/cuda:11.1.1-base nvidia-smi
```

### 性能优化建议

**短序列（<300 aa）**：
```bash
python predict.py ... --no_relax  # 跳过优化节省时间
```

**长序列（>500 aa）**：
```bash
python predict.py ... \
    --max_recycles 5 \
    --max_msa_clusters 256 \
    --max_extra_msa 2048
```

**高精度预测**：
```bash
python predict.py ... \
    --model_name model_1_ptm,model_2_ptm,model_3_ptm,model_4_ptm,model_5_ptm \
    --num_ensemble 8 \
    --max_recycles 20
```

---

## 性能参考

在 NVIDIA A100 (40GB) 上的预测时间：

| 序列长度 | 单模型（无模板） | 5模型（无模板） | 备注 |
|----------|----------------|---------------|------|
| 100 aa   | ~1 分钟        | ~5 分钟       | 包含优化 |
| 300 aa   | ~3 分钟        | ~15 分钟      | 包含优化 |
| 500 aa   | ~8 分钟        | ~40 分钟      | 包含优化 |
| 1000 aa  | ~30 分钟       | ~2.5 小时     | 包含优化 |

*实际时间取决于 MSA 大小、GPU 型号等因素*

---

## 引用

如果使用本工具，请引用 AlphaFold2：

```
Jumper, J., Evans, R., Pritzel, A. et al. 
Highly accurate protein structure prediction with AlphaFold. 
Nature 596, 583–589 (2021). 
https://doi.org/10.1038/s41586-021-03819-2
```

---

## 相关文档

- **[CONFIDENCE_METRICS.md](CONFIDENCE_METRICS.md)** - 置信度指标详细说明
- **[QUICKSTART.md](QUICKSTART.md)** - 快速入门指南
- **[CHANGELOG.md](CHANGELOG.md)** - 版本更新记录

---

**版本**: 2.0.0  
**最后更新**: 2025-11-10
