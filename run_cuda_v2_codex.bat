@echo off
cd /d "%~dp0"

python src/v2_codex/alns/solve.py --size 5

python src/v2_codex/ppo_alns/train.py --size 5
python src/v2_codex/ppo_alns/test.py --size 5

python src/v2_codex/gnn_ppo_alns/train.py --size 5
python src/v2_codex/gnn_ppo_alns/test.py --size 5

python src/v2_codex/alns/solve.py --size 10

python src/v2_codex/ppo_alns/train.py --size 10
python src/v2_codex/ppo_alns/test.py --size 10

python src/v2_codex/gnn_ppo_alns/train.py --size 10
python src/v2_codex/gnn_ppo_alns/test.py --size 10

python src/v2_codex/alns/solve.py --size 20

python src/v2_codex/ppo_alns/train.py --size 20
python src/v2_codex/ppo_alns/test.py --size 20

python src/v2_codex/gnn_ppo_alns/train.py --size 20
python src/v2_codex/gnn_ppo_alns/test.py --size 20