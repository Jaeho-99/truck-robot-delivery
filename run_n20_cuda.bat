@echo off
cd /d "%~dp0"

python src/alns/solve.py --size 20 --workers 10

python src/ppo_alns/train.py --size 20 --reward-mode alns_5310 --device cuda --env-backend process --observation-codec numpy
python src/ppo_alns/test.py --size 20 --reward-mode alns_5310 --workers 10

python src/ppo_alns/train.py --size 20 --reward-mode new_best_5 --device cuda --env-backend process --observation-codec numpy
python src/ppo_alns/test.py --size 20 --reward-mode new_best_5 --workers 10

python src/ppo_alns/train.py --size 20 --reward-mode magnitude --device cuda --env-backend process --observation-codec numpy
python src/ppo_alns/test.py --size 20 --reward-mode magnitude --workers 10

python src/gnn_ppo_alns/train.py --size 20 --reward-mode alns_5310 --device cuda --env-backend process --observation-codec numpy
python src/gnn_ppo_alns/test.py --size 20 --reward-mode alns_5310 --workers 10

python src/gnn_ppo_alns/train.py --size 20 --reward-mode new_best_5 --device cuda --env-backend process --observation-codec numpy
python src/gnn_ppo_alns/test.py --size 20 --reward-mode new_best_5 --workers 10

python src/gnn_ppo_alns/train.py --size 20 --reward-mode magnitude --device cuda --env-backend process --observation-codec numpy
python src/gnn_ppo_alns/test.py --size 20 --reward-mode magnitude --workers 10
