#!/bin/bash

# Get absolute path to project root directory
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Set location for torch cache (to avoid using home directory)
export TORCH_HOME="$PROJECT_ROOT/.torch"

# Set location for matplotlib config
export MPLCONFIGDIR="$PROJECT_ROOT/.matplotlib"

# Set location for other caches
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"

# Print environment info
echo "Activated virtual environment at: $VIRTUAL_ENV"
echo "Project root: $PROJECT_ROOT"
echo "Python path: $PYTHONPATH"
echo "Torch home: $TORCH_HOME"

# Optionally change to project directory
cd "$PROJECT_ROOT"

# Show available commands
echo ""
echo "Available commands:"
echo "  make task1  - Run Task 1: Baseline evaluation"
echo "  make task2  - Run Task 2: FGSM attack"
echo "  make task3  - Run Task 3: Improved attacks"
echo "  make task4  - Run Task 4: Patch attacks" 
echo "  make task5  - Run Task 5: Transferability"
echo "  make all    - Run all tasks"
echo "  make clean  - Clean generated files"
echo ""

# Instructions for running
echo "To run Task 1:"
echo "  python experiments/task1_baseline.py"
echo "" 