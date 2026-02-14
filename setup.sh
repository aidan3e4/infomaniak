#!/bin/bash
set -e

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
git clone https://github.com/aidan3e4/infomaniak
cd infomaniak
uv sync
source .venv/bin/activate

exec bash
