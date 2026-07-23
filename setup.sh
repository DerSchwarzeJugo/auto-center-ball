#!/usr/bin/env bash
# One-time setup: create a dedicated virtual env for the ball-centering tool.
# Run from the folder containing center_ball.py and requirements.txt.
set -e

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Done. The venv is in ./.venv"
echo "Activate it in future sessions with:  source .venv/bin/activate"
echo "Then run, e.g.:"
echo "  python center_ball.py --input ./images --output ./out --debug"