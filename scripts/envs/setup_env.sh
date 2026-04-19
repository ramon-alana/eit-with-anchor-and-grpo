#!/bin/bash

# source scripts/envs/setup_sat.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo $SCRIPT_DIR

# Load from .env (supports variable expansion)
if [ -f "$SCRIPT_DIR/.env" ]; then
    # set -a: automatically export all subsequently defined variables
    set -a
    # Use source to load, supports shell variable syntax (e.g., ${HOME})
    source "$SCRIPT_DIR/.env"
    set +a
    
    # Your custom logic (prevent duplicate setting flag)
    if [ -z "$SAT_ENV_SET" ]; then
        export SAT_ENV_SET=1
        echo "✅ environment variables loaded from .env"
        echo "   PYTHONPATH: $PYTHONPATH"
    else
        echo "⚠️  Environment variables already set, skipping"
    fi
else
    echo "❌ $SCRIPT_DIR/.env not found"
    exit 1
fi

source /opt/conda/bin/activate