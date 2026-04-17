#!/bin/bash
# Installation script for EVE dependencies

# Create temp directory for cache
mkdir -p /tmp
export XDG_CACHE_HOME=/tmp/.cache/
export TMPDIR=/tmp

# Install using conda environment
conda env create -f env.yml
conda activate eve
