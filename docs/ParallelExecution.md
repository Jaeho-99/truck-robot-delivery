# Windows 병렬 실행·검증 절차

현재 브랜치: `feature/RL-Windows`. 이번 변경은 코드 구현과 정적 검토까지만
수행했습니다. 아래 설치·학습·테스트·벤치마크 명령은 사용자가 직접 실행합니다.
병렬 경로의 정상 동작이나 가속 배율을 실측한 상태는 아닙니다.

## 1. 적용 범위와 유지 조건

| 대상 | 병렬화 단위 | 실행 옵션 |
|---|---|---|
| ALNS `solve.py` | 독립적인 인스턴스·시드별 전체 탐색 | `--workers N` |
| PPO_ALNS `train.py` | 기존 10개 ALNSEnv, 환경당 CPU 프로세스 1개 | `--env-backend process` |
| GNN_PPO_ALNS `train.py` | 기존 10개 ALNSEnv, 환경당 CPU 프로세스 1개 | `--env-backend process` |
| 두 PPO의 `test.py` | 독립적인 인스턴스·시드별 체크포인트 평가 | `--workers N` |

ALNS 한 실행 내부의 destroy/repair, 삽입 후보, SA, RNG 소비 순서는 변경하지
않습니다. 인스턴스 1개·시드 1개만 실행하는 ALNS는 이 병렬화로 빨라지지 않습니다.
Gurobi exact 경로는 이번 변경 대상이 아닙니다.

두 PPOConfig는 `device` 외 필드·기본값을 변경하지 않았습니다. 특히 n_envs=10,
t_rollout=256, search_iterations=100, total_steps=300000을 유지합니다.
기존 CLI의 train_count 기본값 250과 dataclass 기본값 200의 차이도 그대로입니다.
새 실행 옵션은 PPOConfig에 저장하지 않습니다. 기존 하이퍼파라미터 CLI 옵션은
호환성을 위해 남아 있지만 이번 비교에서는 바꾸지 마세요.

기본 설정에서 rollout=2560, minibatch=64, update=117,
실제 처리량=299520 environment transitions입니다. 256은 벡터 step 수이며
2560은 10개 환경의 transition 합계입니다. 1 rollout 미만의 total_steps는
학습 시작 전에 오류로 처리합니다.

직렬 기준선은 유지됩니다. 기본값은 `--env-backend serial`, `--workers 1`이며
검증 없이 process로 자동 전환하지 않습니다. worker 수를 바꾸는 `--workers`는
ALNS/평가용입니다. PPO 학습 worker 수는 항상 cfg.n_envs에서 파생한 10입니다.

## 2. 개발 환경

저장소 루트의 PowerShell에서 실행합니다. 자세한 설명은 [Windows.md](Windows.md).

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/check_environment.py --device cuda --spawn-workers 10
```

Python 자동 탐색이 실패하면 setup에 `-Python C:\경로\python.exe`를 지정합니다.
Python 3.10 x64가 필요하며 다른 버전의 기존 `.venv`는 자동 삭제하지 않습니다.
설치 후 VS Code 인터프리터를 `.venv\Scripts\python.exe`로 선택하세요.

역할 구분:

- `pyproject.toml`: 프로젝트 의존성과 editable 패키지 설치 정의.
- `requirements.txt`: 프로젝트 설치 진입점(`-e .`).
- `requirements-windows.txt`: Windows용 직접 의존성 및 CUDA Torch 버전.
- `constraints-windows-py310.txt`: 기존 Windows 환경의 전이 의존성 버전 고정.
- `setup_windows.ps1`: Torch 2.5.1+cu121을 CUDA 인덱스에서 먼저 설치하고 나머지
  의존성을 설치. CPU/CUDA 비교 모두 같은 CUDA-enabled wheel 사용.

이번 병렬화 때문에 별도 multiprocessing 패키지를 설치할 필요는 없습니다.
이미 해당 `.venv`가 준비되어 있다면 환경을 지우거나 CUDA를 재설치하지 마세요.
OS 드라이버, 시스템 PATH, 다른 Conda 환경은 수정 대상이 아닙니다.

## 3. 먼저 사용자가 실행할 검증

아래는 실제 ALNS 계산 또는 worker 생성을 수행합니다. 설치 점검 명령과 다릅니다.
각 명령이 끝난 뒤 다음 명령을 실행하세요.

```powershell
.\.venv\Scripts\python.exe scripts/verify_artifacts.py
.\.venv\Scripts\python.exe scripts/verify_alns_parallel.py --size 5 --workers 2 --check-failure
.\.venv\Scripts\python.exe scripts/verify_worker_failures.py --method gnn_ppo_alns
.\.venv\Scripts\python.exe scripts/verify_worker_failures.py --method ppo_alns

foreach ($methodName in @('ppo_alns', 'gnn_ppo_alns')) {
    .\.venv\Scripts\python.exe scripts/verify_parallel_alns.py --method $methodName --size 5 --codec direct
    if ($LASTEXITCODE -ne 0) { throw "direct 검증 실패: $methodName" }
    .\.venv\Scripts\python.exe scripts/verify_parallel_alns.py --method $methodName --size 5 --codec numpy
    if ($LASTEXITCODE -ne 0) { throw "numpy 검증 실패: $methodName" }
}
```

기본 512 벡터 step 검증은 고정 action으로 직렬/병렬의 observation, reward,
done, info, instance 선택 순서, episode 로그, GAE/returns를 비교합니다.
256 rollout 경계와 100·200·300·400·500 episode 경계를 넘습니다.
짧은 점검은 `--vector-steps 8`로 가능하지만 전체 동등성 검증을 대체하지 않습니다.
실험에 사용할 reward mode도 해당 검증의 `--reward-mode`로 각각 확인할 수 있습니다.

`verify_worker_failures.py`는 자기 자신이 만든 worker 하나만 종료시키는 장애
주입을 포함합니다. 실제 ALNS step은 실행하지 않습니다. Python 예외/프로세스
사망/부분 done/정리 경로를 검사하며 실제 콘솔 Ctrl+C는 별도로 확인해야 합니다.
라이브 worker 내부의 무한루프를 자동 판별하는 테스트는 아닙니다.

## 4. CPU/CUDA 비교: n5부터, 두 PPO를 따로 판단

각 방법에서 serial backend를 고정하고 device만 바꿉니다. 아래는 짧은 테스트가
아닌 기본 117 update 전체 학습입니다. 병렬 실행과 겹치지 않게 순차 실행하세요.

```powershell
.\.venv\Scripts\python.exe src/ppo_alns/train.py --size 5 --device cuda --env-backend serial --run-label n5_cuda_01
.\.venv\Scripts\python.exe src/ppo_alns/train.py --size 5 --device cpu --env-backend serial --run-label n5_cpu_01
.\.venv\Scripts\python.exe src/gnn_ppo_alns/train.py --size 5 --device cuda --env-backend serial --run-label n5_cuda_01
.\.venv\Scripts\python.exe src/gnn_ppo_alns/train.py --size 5 --device cpu --env-backend serial --run-label n5_cpu_01
```

같은 seed라도 CPU와 CUDA의 sampling 결과는 동일하다고 보장할 수 없습니다.
위 결과는 실제 운용 속도 비교이지 순수한 장치 처리량 비교가 아닙니다. n5 결과를
n50에 그대로 적용하지 말고 n50 및 process backend에서도 재확인하세요.
graph-free PPO와 GNN의 최적 device가 다를 수 있습니다. ALNS 자체는 CPU 작업입니다.

고정 workload를 위한 보조 벤치마크(파일명은 역사적으로 benchmark_gnn_ppo이지만
두 PPO 모두 지원):

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_gnn_ppo.py --method gnn_ppo_alns --mode env --size 50 --env-backend serial
.\.venv\Scripts\python.exe scripts/benchmark_gnn_ppo.py --method gnn_ppo_alns --mode env --size 50 --env-backend process --codec numpy
.\.venv\Scripts\python.exe scripts/benchmark_gnn_ppo.py --method gnn_ppo_alns --mode forward --size 5 --device cpu
.\.venv\Scripts\python.exe scripts/benchmark_gnn_ppo.py --method gnn_ppo_alns --mode forward --size 5 --device cuda
.\.venv\Scripts\python.exe scripts/benchmark_gnn_ppo.py --method gnn_ppo_alns --mode update --size 5 --device cpu
.\.venv\Scripts\python.exe scripts/benchmark_gnn_ppo.py --method gnn_ppo_alns --mode update --size 5 --device cuda
.\.venv\Scripts\python.exe scripts/benchmark_gnn_ppo.py --method gnn_ppo_alns --mode replay --size 5 --device cpu --updates 2
.\.venv\Scripts\python.exe scripts/benchmark_gnn_ppo.py --method gnn_ppo_alns --mode replay --size 5 --device cuda --updates 2
```

`--method ppo_alns`로 같은 비교를 반복합니다. trace seed/size/backend를 양쪽에서
동일하게 유지하세요. 벤치마크는 checkpoint를 저장하지 않으며 harness의
`--updates`, `--vector-steps` 등은 PPOConfig를 변경하지 않습니다.

replay는 양쪽 모두 sampling 없이 동일 action을 사용하며 학습용 결과가 아닙니다.
update 모드는 CPU로 만든 동일 buffer/초기 가중치와 매번 새 optimizer를 사용합니다.
따라서 Adam 첫 상태 할당이 포함된 초기 update 비교이며 정상 학습의 모든 update를
대표하지는 않습니다. target_kl을 끄지 않고 실제 optimizer.step 횟수와 시간/step을
함께 기록합니다. 개별 CUDA 벤치마크 구간은 명시적으로 동기화합니다.

## 5. 검증 후 병렬 본 실행

```powershell
.\.venv\Scripts\python.exe src/alns/solve.py --size 50 --workers 10 --run-label n50_process_01
.\.venv\Scripts\python.exe src/ppo_alns/train.py --size 50 --device cuda --env-backend process --observation-codec numpy --run-label n50_process_cuda_01
.\.venv\Scripts\python.exe src/gnn_ppo_alns/train.py --size 50 --device cuda --env-backend process --observation-codec numpy --run-label n50_process_cuda_01
```

위 명령도 순차 실행합니다. CUDA 대신 CPU가 유리하면 `--device cpu`만 바꾸세요.
ALNS/평가의 worker 수는 8과 10을 별도로 비교해도 되지만 PPO 학습 환경 수는 10
그대로입니다. 8개 물리 코어에서 10 worker가 반드시 최선이라는 보장은 없습니다.

codec 기본값은 `direct`입니다. `numpy`는 tensor 공유메모리 핸들에 의존하지 않는
전송 경로이며 위 예시는 그 경로를 사용합니다. 두 codec 동등성 검사 후 선택하세요.
두 방식 모두 기존 graph store 순서, dtype, 빈 edge shape를 보존합니다.
`torch.multiprocessing`을 직접 import하지 않는다는 사실만으로 공유메모리 사용이
없다고 보장하지 않습니다. 라이브러리의 reducer 등록도 영향을 줍니다.

학습한 checkpoint의 병렬 평가는 CPU에서 수행합니다(기존 평가 device 유지).

```powershell
.\.venv\Scripts\python.exe src/ppo_alns/test.py --size 50 --workers 10 --run-label eval_01 --checkpoint models/ppo_alns_n50_reward_magnitude_run-n50_process_cuda_01.pt
.\.venv\Scripts\python.exe src/gnn_ppo_alns/test.py --size 50 --workers 10 --run-label eval_01 --checkpoint models/gnn_ppo_alns_n50_reward_magnitude_run-n50_process_cuda_01.pt
```

평가 worker는 CPU 모델을 각자 한 번 로드합니다. 모델/CUDA tensor를 프로세스 간
전송하지 않습니다. CPU 스레드 수/연산 순서 차이로 작은 수치 차이가 action까지
영향을 줄 수 있으므로 평가 결과의 비트 단위 동일성은 보장하지 않습니다.
환경 동등성 검증의 고정 action 조건과 실제 policy 평가 조건을 구분하세요.

해당 size에 맞는 checkpoint가 준비되면 평가 전용 비교도 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe scripts/verify_policy_evaluation.py --method ppo_alns --size 5 --checkpoint models/ppo_alns_n5_reward_magnitude_run-n5_cuda_01.pt
.\.venv\Scripts\python.exe scripts/verify_policy_evaluation.py --method gnn_ppo_alns --size 5 --checkpoint models/gnn_ppo_alns_n5_reward_magnitude_run-n5_cuda_01.pt
```

기본으로 작은 고정 작업 집합에 sampling과 argmax를 모두 검사하고 파일은 쓰지
않습니다. 차이가 있으면 실패로 보고합니다. 별도 `--serial-threads 1` 실행으로
CPU 스레드 조건을 맞춰 원인을 좁힐 수 있지만, 차이를 무조건 스레드 탓으로
간주하지 마세요. 이 옵션은 검증 실행에만 적용하며 PPOConfig를 수정하지 않습니다.

기존 배치 실험 순서를 실행하려면(설치 확인용으로 실행하지 마세요):

```powershell
.\run_experiments.bat study_01 process cuda 10 numpy
```

인자 순서: LABEL, backend, device, ALNS/평가 workers, observation codec.
기존 n50/n100·reward mode 실험 순서는 유지됩니다. 한 실험이 실패하면 중단하며
자동 재시작/자동 checkpoint resume은 하지 않습니다.

## 6. 로그·결과 안전성과 판정

- `--tag`는 데이터 선택용입니다. 비교 실행 이름에는 `--run-label`을 사용하세요.
- checkpoint: `models/METHOD_nSIZE_reward_MODE[_TAG]_run-LABEL.pt`.
- CSV/JSON: `output/METHOD[_TAG]/nSIZE/runs/LABEL/`.
- label 미지정 시 기존 경로를 사용하지만, 이미 존재하는 결과는 덮어쓰지 않습니다.
- 실행 전 출력 파일별 `.lock`을 배타적으로 확보합니다. 예외 시 자기 lock만 정리하며
  강제 종료로 남은 lock은 해당 PID/실행이 종료됐는지 확인한 뒤 수동 처리해야 합니다.
- 각 파일은 같은 디렉터리의 임시 파일에 쓴 뒤 원자적으로 교체합니다. 다중 파일 전체가
  하나의 트랜잭션인 것은 아닙니다. `status=completed` metadata를 마지막에 씁니다.
- checkpoint 저장 주기는 기존 10 update마다 및 마지막입니다. optimizer/RNG/환경
  상태 전체를 저장하는 resume 기능은 이번 변경에 포함되지 않았습니다.
- `scripts/summarize_results.py`는 기존 경로와 runs/LABEL 경로를 모두 찾습니다.

학습은 최초 reset을 별도 출력하고 매 32 벡터 step마다 진행률을 출력합니다.
process 응답이 늦으면 약 30초 간격으로 기다리는 worker/phase를 출력합니다.
worker 사망과 전달된 예외는 학습을 실패시키고 전체 worker를 정리합니다.
정상적으로 오래 걸리는 ALNS step에 짧은 강제 제한시간을 두지 않았습니다.
단, 살아 있지만 멈춘 worker를 heartbeat만으로 복구할 수는 없습니다. pipe recv나
tensor 재구성이 오래 막힐 때까지 완전히 비동기화한 구현도 아닙니다.

시간을 더할 때 주의할 점:

- `env_seconds`는 부모가 관측한 환경 호출 wall time입니다.
- worker별 시간 합계와 parent wait는 환경 wall time과 겹칩니다. 다시 더하지 마세요.
- slowest/max/mean은 STEP 호출에서는 auto-reset 전 step 기준입니다. RESET
  단독 호출에서는 reset 기준이며, reset 비용은 별도 reset_seconds로 비교합니다.
- `policy_seconds`에는 rollout collation이 포함됩니다.
- `ppo_update_seconds`에는 minibatch collation이 포함됩니다.
- `update_wall_seconds`는 주기적 checkpoint 쓰기를 제외합니다.
- train/main wall time은 startup·최초 reset 등을 포함하는 별도 측정입니다.
  main wall은 Python 인터프리터 시작/import와 마지막 metadata 쓰기를 제외합니다.

S_env = 직렬 환경 호출 시간 / 병렬 환경 호출 시간,
S_total = 동일 조건 전체 wall time의 직렬 / 병렬로 나눠 봅니다.
ALNS/평가의 개별 runtime_s 평균이나 합계는 전체 배치 wall time이 아닙니다.
실측 전 4~6배 등의 가속을 보장하지 않습니다.

장기 실행 전 최소 3 rollout 동안 부모·worker RSS/VMS/HandleCount, cache 크기를
같은 시점에서 비교하세요. 초기 instance cache 증가와 buffer가 채워지는 동안의
증가는 정상일 수 있으므로 증가만으로 누수라고 단정하지 않습니다. 두 PPO의
updates JSON에 자원 스냅샷이 기록됩니다(psutil 사용 가능 시).

정상 중단은 Ctrl+C입니다. IDE/콘솔에 따라 신호 전달 방식은 다를 수 있습니다.
정리 후 자식 프로세스가 남지 않았는지 확인하세요. 부모만 강제 종료하면 자식이
남을 수 있습니다. PID를 확인하지 않고 모든 python.exe를 종료하지 마세요.
