# Truck-Robot Delivery

트럭과 배송 로봇의 협력 배송 경로를 최적화하는 연구용 코드입니다. Exact, ALNS, PPO-ALNS, GNN-PPO-ALNS를 제공하며, Windows Desktop과 Linux CPU 서버에서 동일한 데이터 형식과 파라미터를 사용합니다.

ALNS 계열은 실행 시간을 개선한 `v2_claude`의 연산자와 평가기를 `src/common/search.py`에서 공유합니다. 일반 PPO는 그래프 라이브러리 없이 실행할 수 있습니다.

## 저장소 구조

```text
configs/params.yaml       # 차량, 비용, 시간 및 문제 설정
src/
  common/                 # 공유 탐색 엔진, 데이터, 병렬 실행, 결과 관리
  exact/solve.py          # Gurobi 기반 정확 해법
  alns/solve.py           # ALNS 실행
  ppo_alns/
    alns.py               # 공유 탐색 엔진 연결
    ppo.py                # 관측, 정책, 학습 알고리즘
    train.py
    test.py
  gnn_ppo_alns/
    alns.py
    gnn.py                # 그래프 인코더
    ppo.py
    train.py
    test.py
scripts/
  preprocess.py
  plot_result.py
  summarize_results.py
docs/
  Commit.md
  Branch.md
requirements.txt
run_windows.bat
run_linux_server.sh
```

`data/`, `models/`, `output/`, `logs/`는 로컬 실행 시 사용하는 디렉토리이며 Git에 포함하지 않습니다.

## 환경 설치

64비트 x86 Windows 또는 Linux에서 Conda 환경을 사용합니다. 기준 Python 버전은 **3.10**이며 의존성은 Python 3.10–3.12를 대상으로 고정했습니다.

두 컴퓨터 모두 다음 순서로 설치합니다.

```bash
git clone https://github.com/Jaeho-99/truck-robot-delivery.git
cd truck-robot-delivery
conda create -n truck-robot-delivery python=3.10 pip -y
conda activate truck-robot-delivery
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt`의 운영체제 조건에 따라 필요한 패키지를 선택합니다.

| 환경 | PyTorch | GNN | 기본 실행 장치 |
|---|---|---|---|
| Windows Desktop | CUDA 12.1 빌드, CPU 실행 가능 | 포함 | CPU |
| Linux Server | CPU 빌드 | 미설치 | CPU |

Windows에서 GPU 학습은 호환되는 NVIDIA 드라이버와 `--device cuda`를 사용합니다. GNN도 CPU에서 실행할 수 있습니다. Exact 실행에는 문제 크기를 지원하는 **Gurobi 라이선스**가 필요합니다.

명령은 저장소 루트에서 실행합니다. 저장소 자체를 패키지로 설치할 필요가 없으며 `src/`에 설치 메타데이터를 생성하지 않습니다.

## 데이터와 파라미터

지원하는 고객 수는 **5, 10, 20, 50, 100, 150, 200, 250, 300**입니다. 전처리와 모든 해법의 CLI에서 같은 크기 목록을 사용합니다.

| 고객 수 | 5 | 10 | 20 | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|---|---|---|
| 트럭 수 | 2 | 2 | 2 | 4 | 7 | 10 | 12 | 15 | 17 |

트럭당 로봇 수는 2대입니다. 차량 수와 비용 등은 `configs/params.yaml`에서 설정합니다. 새 크기의 차량 수는 초기 설정값이므로 실험 설계에 맞게 조정합니다.

원본 및 공간 데이터를 각 컴퓨터에 별도로 배치합니다.

```text
data/cells_ulsan_namgu.csv
data/ulsan_namgu_dong_boundaries.geojson
data/raw/train/n300/*.json
data/raw/test/n300/*.json
```

```bash
python scripts/preprocess.py --split all --size 300
python scripts/preprocess.py --split all --size all
```

전처리 결과는 `data/processed/{train,test}/nSIZE/*.npz`에 저장됩니다. `--size all`은 원본 폴더가 있는 지원 크기를 처리합니다. 크기를 명시하면 해당 원본 데이터가 필요합니다. 코드의 크기 지원은 원본 인스턴스를 자동 생성하지 않습니다.

기존 NPZ는 호환성을 확인한 뒤 재사용하며, 다시 생성하려면 `--force`를 사용합니다. 거리·속도 설정을 변경한 경우 다시 전처리합니다. 별도 설정의 실험은 전처리·학습·평가에 동일한 `--params PATH`와 `--tag NAME`을 전달합니다.

## 개별 실행

### ALNS 및 PPO-ALNS

```bash
python src/alns/solve.py --size 300 --workers 10 --run-label study_01
python src/ppo_alns/train.py --size 300 --reward-mode magnitude --device cpu --env-backend process --observation-codec numpy --run-label study_01
python src/ppo_alns/test.py --size 300 --reward-mode magnitude --workers 10 --run-label study_01 --checkpoint models/ppo_alns_n300_reward_magnitude_run-study_01.pt
```

### GNN-PPO-ALNS: Windows Desktop

```powershell
python src/gnn_ppo_alns/train.py --size 300 --reward-mode magnitude --device cuda --env-backend process --observation-codec numpy --run-label study_01
python src/gnn_ppo_alns/test.py --size 300 --reward-mode magnitude --workers 10 --run-label study_01 --checkpoint models/gnn_ppo_alns_n300_reward_magnitude_run-study_01.pt
```

보상 방식은 `alns_5310`, `new_best_5`, `magnitude`이며 기본값은 `magnitude`입니다. 학습과 평가에 같은 보상 방식과 체크포인트를 지정합니다. `--env-backend process`는 10개 학습 환경을 별도 프로세스로 실행합니다. `--workers`는 ALNS 및 정책 평가 작업 수이며 1–30을 지원합니다.

실험별로 새로운 `--run-label`을 사용합니다. 동일한 이름의 기존 체크포인트와 결과는 덮어쓰지 않습니다. 각 명령의 추가 옵션은 `--help`로 확인할 수 있습니다.

### Exact

```bash
python src/exact/solve.py --size 5 --time-limit 60 --threads 4
```

Exact도 모든 지원 크기를 입력받습니다. 큰 문제는 시간 제한 내 최적해 증명이 어려울 수 있으며, 결과의 상태와 optimality gap을 함께 확인해야 합니다.

## 일괄 실험

전처리를 마친 뒤 활성화된 Conda 환경에서 실행합니다. 인수를 생략하면 고객 수 50과 100을 순서대로 처리합니다. Exact는 개별 명령으로 실행합니다.

### Windows

```powershell
$env:DEVICE = "cuda"  # CPU를 사용하려면 cpu
$env:WORKERS = "10"
$env:RUN_LABEL = "desktop_01"
.\run_windows.bat 50 100 150 200 250 300
```

크기별로 ALNS, PPO와 GNN의 보상 방식별 학습·평가, 결과 요약을 순서대로 실행합니다. 학습 실패 시 후속 평가를 중단합니다. 기본 장치는 CPU이며 출력은 현재 터미널에 표시됩니다.

### Linux CPU 서버

```bash
RUN_LABEL=server_01 WORKERS=30 T=4 TRAIN_JOBS=3 bash run_linux_server.sh 50 100 150 200 250 300
```

ALNS 평가 작업 30개를 사용하고, **세 보상 방식의 PPO 학습을 동시에 실행**합니다. 모든 학습이 성공한 뒤 해당 체크포인트로 평가합니다. 서버 스크립트는 GNN을 실행하지 않습니다.

| 환경 변수 | 기본값 | 용도 |
|---|---|---|
| `WORKERS` | 30 | ALNS 및 정책 평가 병렬 작업 수 |
| `TRAIN_JOBS` | 3 | 동시 학습 수, 1–3 |
| `T` | 4 | 학습 부모 프로세스의 OMP/MKL 스레드 수 |
| `NUMA_NODES` | `auto` | 사용 가능한 NUMA 노드에 순환 배치 |
| `RUN_LABEL` | 실행 시각 | 체크포인트, 결과, 로그 식별자 |
| `PYTHON` | `python` | 활성 Conda 환경의 Python 실행 파일 |
| `DRY_RUN` | 0 | 1이면 실행할 명령만 출력 |

학습 하나가 환경 프로세스 10개를 사용하므로 동시 학습 수와 스레드 수를 서버의 CPU·메모리에 맞춥니다. `numactl`이 있으면 NUMA 배치를 적용하며 `NUMA_NODES="0 1 0"`으로 직접 지정하거나 `NUMA_NODES=none`으로 해제할 수 있습니다. 프로세스 그룹 관리에는 Linux의 `setsid`가 필요합니다. 로그는 `logs/RUN_LABEL/nSIZE/`에 저장됩니다.

실행 전 명령 확인:

```powershell
$env:DRY_RUN = "1"
.\run_windows.bat 300
Remove-Item Env:DRY_RUN
```

```bash
DRY_RUN=1 bash run_linux_server.sh 300
```

## 결과 확인

체크포인트는 `models/`, 해법별 결과는 `output/`에 저장됩니다. Label을 지정한 ALNS/PPO 결과는 해당 크기의 `runs/` 하위에서 구분됩니다. Exact 결과는 `output/exact/nSIZE/`에 저장됩니다.

```bash
python scripts/summarize_results.py --size 300
python scripts/plot_result.py output/alns/n300/runs/study_01/test_n300_000_s0.json
```

요약 스크립트는 ALNS 및 두 PPO 방법의 결과를 집계합니다. 시각화는 Exact를 포함한 네 방법의 결과 JSON을 지원하며 그림과 HTML을 생성합니다. 입력 경로에는 실제 생성된 결과 파일명을 사용합니다.

## 다른 컴퓨터에서 업데이트

```bash
conda activate truck-robot-delivery
git pull --ff-only
python -m pip install -r requirements.txt
python -m pip check
```

실험을 재현할 때는 Git 커밋, `configs/params.yaml`, 원본·전처리 데이터, 실행 옵션과 seed를 함께 맞춥니다. 데이터와 모델은 Git과 별도로 동기화합니다.

기여 시 [커밋 규칙](docs/Commit.md)과 [브랜치 규칙](docs/Branch.md)을 따릅니다.
