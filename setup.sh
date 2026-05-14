#!/usr/bin/env bash
set -e
python -m venv venv
source venv/bin/activate
#pip install --upgrade pip
pip install -r requirements.txt
echo ""
echo "Done. Activate the env with: source venv/bin/activate"
echo "Then run: python src/main.py"