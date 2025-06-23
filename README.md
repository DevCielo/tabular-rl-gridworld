python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py --episodes 1000

python visualize.py