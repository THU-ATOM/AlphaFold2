#!/bin/bash
#
# AlphaFold2 Prediction Runner
#
# This script runs AlphaFold2 structure prediction inside a Docker container.
# Edit the configuration section below according to your setup.
#

#==============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS
#==============================================================================

# Docker and paths
DOCKER_IMAGE="af2-predict"
DATA_ROOT="/data/protein"              # (Optional) additional data root mounted to /data
WORK_DIR="."                           # Working directory with input files

# Input
SEQUENCE="MKHHHHHHHHGGLVPRGSHGGSEIKEGYFEQQQTLKNTATVNELSGIPAVDRAHVLQTALSIYPEVENWVAQFPVILPQRMGGMCLGMVATAPYAQPSVLVEASIMALIAFAIDDITEDVLSMTLTVEQIEAMLTLCVKLVQSGGNSTYRDYPELIQVFPTINESQPWVQLANALTKFCSEVQKFPAAAIYYSIFAKHFELYREAHCTELHWTQAVKEMGSYPTYEQYLLNSRKSIAAPLVESSLLAMVGEPVDSEFSLKPPYANLETLIDEVLLICGSSIRLANDIRSFEREPQAYQPNSLLILMLTQGCSQKEAEAILLKEIDTYLQKIETLISLLPSSLSTWGDSARRMSWFACTWYQTRDFHNFNKQMLAALR"
A3M_FILE="example/example.a3m"         # MSA file (A3M format)
TARGET_NAME="9KGB_A"                   # Output file prefix

# Output
OUTPUT_DIR="output"                    # Output directory

# Model selection
# Single model:    MODEL_NAME="model_1_ptm"
# Multiple models: MODEL_NAME="model_1_ptm,model_2_ptm,model_3_ptm,model_4_ptm,model_5_ptm"
MODEL_NAME="model_1_ptm,model_2_ptm,model_3_ptm,model_4_ptm,model_5_ptm"

# Prediction parameters
RANDOM_SEED=0
NUM_ENSEMBLE=1
MAX_RECYCLES=3
MAX_MSA_CLUSTERS=508
MAX_EXTRA_MSA=5120

# Template search (set to true to enable)
USE_TEMPLATES=false

# GPU device (e.g., "0" or "0,1")
GPU_DEVICE="0"

#==============================================================================
# RUN PREDICTION
#==============================================================================

# Build base docker command
DOCKER_CMD="docker run --rm \
    --ipc host \
    --user $(id -u):$(id -g) \
    $(id -G | sed 's/\([0-9]\+\)/--group-add \1/g') \
    --gpus device=${GPU_DEVICE} \
    -v ${DATA_ROOT}:/data \
    -v ${WORK_DIR}:/work \
    -v $(pwd):/app \
    -w /app \
    ${DOCKER_IMAGE}"

# Build prediction command
PREDICT_CMD="python predict.py \
    --sequence ${SEQUENCE} \
    --a3m_path /work/${A3M_FILE} \
    --output_dir /work/${OUTPUT_DIR} \
    --target_name ${TARGET_NAME} \
    --model_name ${MODEL_NAME} \
    --cache /data/alphafold \
    --random_seed ${RANDOM_SEED} \
    --num_ensemble ${NUM_ENSEMBLE} \
    --max_recycles ${MAX_RECYCLES} \
    --max_msa_clusters ${MAX_MSA_CLUSTERS} \
    --max_extra_msa ${MAX_EXTRA_MSA}"

# Add template search options if enabled
if [ "$USE_TEMPLATES" = true ]; then
    echo "Running with template search..."
    # When using templates, the script will derive PDB70 and mmCIF paths from --cache
    # Example: if CACHE_DIR=/data/alphafold the script expects
    #   ${CACHE_DIR}/pdb70/pdb70 and ${CACHE_DIR}/pdb_mmcif/mmcif_files
    PREDICT_CMD="${PREDICT_CMD} \
        --use_templates \
        --max_template_date 2022-12-31"
else
    echo "Running without templates..."
fi

# Execute
${DOCKER_CMD} ${PREDICT_CMD}

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ Prediction completed successfully!"
    echo "============================================================"
    echo "Results saved to: ${WORK_DIR}/${OUTPUT_DIR}"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "✗ Prediction failed!"
    echo "============================================================"
    exit 1
fi
