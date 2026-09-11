# Truck-Robot Delivery

트럭과 배송 로봇의 결합 경로 문제를 다음 네 가지 방법으로 풉니다.

- `src/exact/solve.py`: Gurobi 기반 exact MILP
- `src/alns/solve.py`: roulette-wheel ALNS
- `src/ppo_alns/`: graph-free PPO-ALNS
- `src/gnn_ppo_alns/`: GNN-PPO-ALNS

이 문서는 **아무 개발 환경도 없는 Windows PC**에서 저장소를 clone하고,
Conda 환경을 만든 뒤 CPU/CUDA 및 병렬 실행까지 진행하는 순서로 작성했습니다.
명령은 PowerShell에서 저장소 루트를 기준으로 실행합니다.

## 1. 먼저 설치할 프로그램

다음을 먼저 설치합니다.

1. [Git for Windows](https://git-scm.com/download/win)
2. [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) 또는 Anaconda
3. NVIDIA GPU를 사용할 경우 NVIDIA 그래픽 드라이버
4. exact solver를 사용할 경우 Gurobi와 유효한 Gurobi 라이선스

이 프로젝트는 Python 3.10을 사용합니다. PyTorch CUDA wheel에 필요한 CUDA
runtime은 wheel에 포함되므로, PPO 학습만을 위해 별도의 CUDA Toolkit을 설치하거나
시스템 PATH를 수정할 필요는 없습니다. `nvidia-smi`에 표시되는 CUDA 버전과
`torch.version.cuda`가 정확히 같을 필요도 없습니다.

## 2. 저장소 clone과 브랜치 선택

```powershell
git clone https://github.com/Jaeho-99/truck-robot-delivery.git
cd truck-robot-delivery
git switch feature/RL-Windows
```

이미 clone한 저장소라면 다음처럼 최신 브랜치를 가져옵니다. 로컬 변경사항이 있다면
먼저 보존한 뒤 실행하세요.

```powershell
git fetch origin
git switch feature/RL-Windows
git pull --ff-only
```

## 3. Conda 환경 생성

환경 이름은 `truck-robot-delivery`로 사용합니다. 다른 이름을 원하면 아래 명령의
이름만 일관되게 바꾸면 됩니다.

```powershell
conda create -n truck-robot-delivery python=3.10 pip -y
conda activate truck-robot-delivery

python --version
python -m pip --version
python -m pip install --upgrade pip setuptools wheel
```

새 PowerShell에서 `conda activate`가 동작하지 않으면 한 번만 다음을 실행하고
PowerShell을 완전히 닫았다가 다시 엽니다.

```powershell
conda init powershell
```

이후 저장소로 이동하고 환경을 다시 활성화합니다.

```powershell
cd C:\path\to\truck-robot-delivery
conda activate truck-robot-delivery
```

현재 터미널이 올바른 환경인지 확인합니다.

```powershell
where.exe python
python -c "import sys; print(sys.executable); print(sys.version)"
```

출력 경로에 `envs\truck-robot-delivery\python.exe`가 포함되어야 합니다.

## 4. 프로젝트 의존성 설치

저장소 루트에서 `requirements.txt` 하나만 설치합니다. 이 파일은 NumPy, SciPy,
PyTorch, PyTorch Geometric, PyYAML, Matplotlib, pyproj, Gurobi Python 패키지와
저장소의 editable 설치를 모두 포함합니다. Windows에서는 PyTorch 2.5.1 CUDA
12.1 wheel을 설치하며, 이 환경 하나로 `--device cpu`와 `--device cuda`를 모두
사용할 수 있습니다.

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

설치를 다시 할 때는 반드시 `conda activate truck-robot-delivery` 후 실행하세요.

## 5. 설치 확인

먼저 import와 작은 CPU 연산을 확인합니다.

```powershell
python scripts/check_environment.py --device cpu
```

NVIDIA GPU를 사용할 경우 CUDA와 Windows spawn worker도 확인합니다.

```powershell
nvidia-smi
python -c "import torch; print('torch =', torch.__version__); print('torch CUDA =', torch.version.cuda); print('available =', torch.cuda.is_available()); print('GPU =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python scripts/check_environment.py --device cuda --spawn-workers 10
```

`available = True`와 GPU 이름이 나와야 합니다. 이 진단은 작은 GNN 연산과
worker import만 확인하며 ALNS 학습이나 성능 측정을 실행하지 않습니다.

exact solver를 쓸 경우 Gurobi도 별도로 확인합니다.

```powershell
python -c "import gurobipy as gp; print(gp.gurobi.version())"
```

패키지 import는 되지만 최적화 시 라이선스 오류가 발생하면 Gurobi 라이선스를
설정해야 합니다. ALNS/PPO만 실행한다면 Gurobi exact solver를 실행할 필요는 없습니다.

## 6. 데이터 준비

전처리된 데이터가 `data/processed/{train,test}/n{size}` 아래에 이미 있다면 이 단계는
건너뜁니다. raw data만 있다면 원하는 크기를 전처리합니다.

```powershell
python scripts/preprocess.py --split all --size 5
python scripts/preprocess.py --split all --size 50
python scripts/preprocess.py --split all --size 100
```

전처리 관련 설정을 바꾼 뒤 다시 생성할 때만 `--force`를 붙입니다. 별도 민감도
데이터는 `--tag NAME`을 사용하며 `data/processed_NAME`에 저장됩니다.

## 7. 병렬 코드 검증

본 학습 전에 다음 명령을 **한 번씩 순서대로** 실행하는 것을 권장합니다.
검증 중에는 다른 학습 프로세스를 동시에 실행하지 마세요.

```powershell
python scripts/verify_artifacts.py
python scripts/verify_alns_parallel.py --size 5 --workers 2 --check-failure
python scripts/verify_worker_failures.py --method ppo_alns
python scripts/verify_worker_failures.py --method gnn_ppo_alns
python scripts/verify_parallel_alns.py --method ppo_alns --size 5 --codec direct
python scripts/verify_parallel_alns.py --method ppo_alns --size 5 --codec numpy
python scripts/verify_parallel_alns.py --method gnn_ppo_alns --size 5 --codec direct
python scripts/verify_parallel_alns.py --method gnn_ppo_alns --size 5 --codec numpy
```

`verify_parallel_alns.py`의 기본 512 vector step 검증은 episode 경계와 rollout
경계를 모두 통과하면서 직렬/병렬 결과, instance 선택 순서, observation, reward,
done, info, GAE를 비교합니다. 따라서 단순 설치 확인보다 오래 걸릴 수 있습니다.

## 8. 실행 옵션의 의미

- `--device cpu|cuda`: PPO 모델의 forward/backward 장치
- `--env-backend serial|process`: PPO 학습의 10개 ALNS 환경 실행 방식
- `--observation-codec direct|numpy`: process backend의 관측 전송 방식
- `--workers N`: ALNS 또는 PPO 평가의 독립 `(instance, seed)` 작업 수
- `--run-label NAME`: 결과를 다른 실행과 분리하는 이름
- `--tag NAME`: 실행 이름이 아니라 `data/processed_NAME` 데이터 선택

PPO 학습의 process backend는 기존 `n_envs=10`을 그대로 사용하므로 worker도
항상 10개입니다. `--workers`로 이 값을 변경하지 않습니다. ALNS와 체크포인트
평가는 `--workers 1`이 직렬 기준선이며 `2-30`에서 process 실행입니다.

동일 label 또는 기존 legacy 출력 경로가 이미 존재하면 덮어쓰지 않고 시작 전에
오류가 납니다. 새 label을 사용하세요. 학습·검증 프로세스는 항상 한 번에 하나만
실행하는 것이 좋습니다.

## 9. ALNS 실행

직렬 기준선:

```powershell
python src/alns/solve.py --size 50 --workers 1 --run-label n50_serial_01
```

여러 test instance와 seed를 병렬 실행:

```powershell
python src/alns/solve.py --size 50 --workers 10 --run-label n50_process_01
```

ALNS 병렬화 단위는 서로 독립적인 instance와 seed입니다. 한 instance·한 seed의
탐색 내부를 나누지는 않으므로 작업이 하나뿐이면 빨라지지 않습니다. 결과는 다음
경로에 저장됩니다.

```text
output/alns/n50/runs/n50_process_01/
```

## 10. PPO-ALNS 학습과 평가

먼저 n5에서 CPU와 CUDA를 serial backend로 각각 순차 실행해 비교할 수 있습니다.

```powershell
python src/ppo_alns/train.py --size 5 --device cpu  --env-backend serial --run-label n5_cpu_01
python src/ppo_alns/train.py --size 5 --device cuda --env-backend serial --run-label n5_cuda_01
```

병렬 환경으로 n50 학습:

```powershell
python src/ppo_alns/train.py --size 50 --device cuda --env-backend process --observation-codec numpy --run-label n50_process_cuda_01
```

생성되는 checkpoint:

```text
models/ppo_alns_n50_reward_magnitude_run-n50_process_cuda_01.pt
```

해당 checkpoint를 CPU worker 10개로 평가:

```powershell
python src/ppo_alns/test.py --size 50 --workers 10 --run-label n50_eval_01 --checkpoint models/ppo_alns_n50_reward_magnitude_run-n50_process_cuda_01.pt
```

## 11. GNN-PPO-ALNS 학습과 평가

n5 CPU/CUDA 직렬 비교:

```powershell
python src/gnn_ppo_alns/train.py --size 5 --device cpu  --env-backend serial --run-label n5_cpu_01
python src/gnn_ppo_alns/train.py --size 5 --device cuda --env-backend serial --run-label n5_cuda_01
```

병렬 환경으로 n50 학습:

```powershell
python src/gnn_ppo_alns/train.py --size 50 --device cuda --env-backend process --observation-codec numpy --run-label n50_process_cuda_01
```

생성되는 checkpoint:

```text
models/gnn_ppo_alns_n50_reward_magnitude_run-n50_process_cuda_01.pt
```

해당 checkpoint 평가:

```powershell
python src/gnn_ppo_alns/test.py --size 50 --workers 10 --run-label n50_eval_01 --checkpoint models/gnn_ppo_alns_n50_reward_magnitude_run-n50_process_cuda_01.pt
```

학습 checkpoint가 준비된 뒤 직렬/병렬 평가 동등성도 확인할 수 있습니다.

```powershell
python scripts/verify_policy_evaluation.py --method ppo_alns --size 5 --checkpoint models/ppo_alns_n5_reward_magnitude_run-n5_cuda_01.pt
python scripts/verify_policy_evaluation.py --method gnn_ppo_alns --size 5 --checkpoint models/gnn_ppo_alns_n5_reward_magnitude_run-n5_cuda_01.pt
```

## 12. Reward mode

두 PPO 방법은 같은 세 reward mode를 지원합니다.

| Mode | 의미 | checkpoint token |
|---|---|---|
| `alns_5310` | new best/current improvement/accepted/else = 5/3/1/0 | `reward_alns_5310` |
| `new_best_5` | new best = 5, 그 외 0 | `reward_new_best_5` |
| `magnitude` | `10 * max(0, delta_best) / initial_obj` | `reward_magnitude` |

기본값은 `magnitude`입니다. 다른 mode는 학습과 평가 양쪽에 동일하게 지정합니다.

```powershell
python src/ppo_alns/train.py --size 50 --reward-mode alns_5310 --device cuda --env-backend process --run-label n50_5310_01
python src/ppo_alns/test.py --size 50 --reward-mode alns_5310 --workers 10 --run-label n50_5310_eval_01 --checkpoint models/ppo_alns_n50_reward_alns_5310_run-n50_5310_01.pt
```

## 13. Exact solver

Gurobi 라이선스가 준비된 경우:

```powershell
python src/exact/solve.py --size 20
```

exact solver는 기본적으로 한 instance만 처리합니다. 전체 test set 실행 옵션은
해당 CLI의 도움말을 먼저 확인하세요.

```powershell
python src/exact/solve.py --help
```

## 14. CPU/CUDA와 직렬/병렬 성능 비교

CPU/CUDA 실행은 동시에 돌리지 말고 동일 조건에서 순차 실행합니다. CPU와 CUDA는
같은 seed여도 policy sampling action이 달라질 수 있으므로 전체 학습 시간은 순수한
device 처리량만 비교하는 값은 아닙니다.

고정 action/observation/buffer 기반 보조 벤치마크:

```powershell
python scripts/benchmark_gnn_ppo.py --method ppo_alns --mode env --size 50 --env-backend serial
python scripts/benchmark_gnn_ppo.py --method ppo_alns --mode env --size 50 --env-backend process --codec numpy
python scripts/benchmark_gnn_ppo.py --method gnn_ppo_alns --mode forward --size 5 --device cpu
python scripts/benchmark_gnn_ppo.py --method gnn_ppo_alns --mode forward --size 5 --device cuda
python scripts/benchmark_gnn_ppo.py --method gnn_ppo_alns --mode update --size 5 --device cpu
python scripts/benchmark_gnn_ppo.py --method gnn_ppo_alns --mode update --size 5 --device cuda
```

학습 로그는 최초 reset과 매 32 vector step의 진행률을 출력합니다. process worker가
오래 걸리면 기다리는 worker/phase가 주기적으로 출력됩니다. `S_env`와 전체
`S_total`을 분리해 비교하고, 구현 전 예상치만으로 가속을 단정하지 마세요.

Ryzen 3700X는 물리 8코어이므로 standalone ALNS/평가에서는 `--workers 8`과
`--workers 10`을 모두 측정해 더 빠른 값을 선택할 수 있습니다. PPO 학습은
PPOConfig를 유지하기 위해 10개 환경/worker를 그대로 사용합니다.

## 15. 결과 경로와 요약

label을 사용한 결과 경로:

```text
models/{ppo_alns,gnn_ppo_alns}_n{size}_reward_{mode}_run-{label}.pt
output/{alns,ppo_alns,gnn_ppo_alns}/n{size}/runs/{label}/
```

결과를 누적 요약합니다.

```powershell
python scripts/summarize_results.py
python scripts/summarize_results.py --size 50
```

한 결과 JSON의 route map을 만듭니다.

```powershell
python scripts/plot_result.py output/alns/n50/runs/n50_process_01/test_n50_000_s0.json
```

## 16. 전체 배치 실행

기존 n50/n100 실험 순서를 한 번에 실행하려면 다음 형식을 사용합니다.

```powershell
.\run_experiments.bat LABEL BACKEND DEVICE WORKERS CODEC
.\run_experiments.bat study_01 process cuda 10 numpy
```

이 배치 파일은 전체 학습과 평가를 실행하므로 설치 확인용으로 실행하지 마세요.
한 작업이 실패하면 남은 작업을 실행하지 않습니다.

## 17. 중단과 문제 확인

- 정상 중단은 `Ctrl+C`를 사용합니다.
- 종료 후 `Get-Process python -ErrorAction SilentlyContinue`로 worker 잔존 여부를
  확인할 수 있습니다. 다른 Python 작업의 PID를 확인하지 않고 일괄 종료하지 마세요.
- 강제 종료로 `.lock` 파일이 남으면 기록된 PID의 실행이 끝났는지 확인한 뒤에만
  해당 lock을 수동으로 제거합니다.
- CUDA 오류가 나면 자동으로 CPU로 전환하지 않습니다. 설치 확인 후 명시적으로
  `--device cpu` 또는 `--device cuda`를 선택합니다.
- 학습 checkpoint는 모델과 설정을 저장하지만 optimizer/RNG/environment 전체를
  복원하는 resume checkpoint는 아닙니다.

병렬화 구현과 더 상세한 계측 설명은
[docs/ParallelExecution.md](docs/ParallelExecution.md)를 참고하세요.
