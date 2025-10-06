#!/bin/bash
# init.bash - initialize environment and move into source dir

# fail on error
set -e
export CONDA_ENV="intlx"


# go to project source dir
cd "./notebooks/tuner_knowledge/src" || exit 1

# activate conda environment
# NOTE: we must source conda.sh first so that "conda activate" works in scripts
if [ -z "$CONDA_EXE" ]; then
    echo "Conda not found. Please load conda first."
    exit 1
fi

# source conda base setup
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate "$CONDA_ENV"

echo "Environment '$CONDA_ENV' activated and working directory set to $(pwd)"
