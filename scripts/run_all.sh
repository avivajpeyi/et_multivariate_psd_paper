#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Loop through all .py files in the script directory and execute them
for file in "$SCRIPT_DIR"/*.py; do
    if [ -f "$file" ]; then
        echo "Running $file..."
        python "$file"
    fi
done
