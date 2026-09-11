# v2_codex: 같은 탐색 후보를 더 적은 연산으로 평가

기존 `src/alns`, `src/ppo_alns`, `src/gnn_ppo_alns`를 수정하지 않은 비교 실험용 버전입니다.
세 방법의 CLI 옵션, 기본 설정, 콘솔 로그, 결과 JSON/CSV 구조, PPO 모델 구조와
reward는 유지합니다. 결과의 시간 값과 저장 경로는 달라집니다.

## 원본과 같은 조건으로 실행

저장소 루트에서 기존 명령의 `src/` 뒤에 `v2_codex/`만 추가하면 됩니다.
별도 `PYTHONPATH` 설정 없이 직접 실행할 수 있습니다. 아래 학습·평가는 각 방법별로
학습이 끝난 뒤 해당 평가를 실행합니다.

```powershell
python src/v2_codex/alns/solve.py --size 20

python src/v2_codex/ppo_alns/train.py --size 20
python src/v2_codex/ppo_alns/test.py --size 20

python src/v2_codex/gnn_ppo_alns/train.py --size 20
python src/v2_codex/gnn_ppo_alns/test.py --size 20
```

`--size 5`, `--size 10`도 같은 방식입니다. 이전 원본 실행에 `--device`, `--env-backend`,
`--reward-mode`, `--params` 등의 옵션을 사용했다면 v2에도 똑같이 지정하세요.

**PPO/GNN-PPO 학습 CLI 기본값은 원본과 같습니다.**

| 항목 | 원본과 v2의 기본값 |
|---|---|
| 문제 파라미터 | 동일한 `configs/params.yaml` |
| 입력 데이터 | 동일한 `data/processed*/` |
| 학습 instance 수 | 250 |
| 요청 학습 step | 300,000 |
| 실제 학습 step | 299,520: 2,560 step rollout × 117 update |
| 환경 수 / 환경별 rollout 길이 | 10 / 256 |
| episode당 ALNS 반복 수 | 100 |
| reward / uniform exploration | `magnitude` / 0.1 |
| device / 실행 backend / observation codec | `cuda` / `serial` / `direct` |
| seed | 0 |

원본의 rollout 단위 절삭까지 동일합니다. 학습률·PPO clipping·discount·GAE·모델 구조와
ALNS의 제거 비율·SA 온도·noise·action/reward 설정 역시 원본 값을 유지합니다.
이후의 작은 예제나 과거 속도 측정에 나온 `--train-count 3`, `--total-steps 2560`,
`--iterations 50` 등은 해당 명령에만 적용한 **명시적 검증 옵션**이며 기본값이 아닙니다.

위 n20 기본 명령의 저장 위치:

```text
output/v2_codex/alns/n20/
output/v2_codex/ppo_alns/n20/
output/v2_codex/gnn_ppo_alns/n20/
models/v2_codex/ppo_alns_n20_reward_magnitude.pt
models/v2_codex/gnn_ppo_alns_n20_reward_magnitude.pt
```

평가도 위 `models/v2_codex/`의 checkpoint를 기본으로 읽습니다. JSON/CSV 이름·필드,
콘솔 진행 로그와 요약 형식은 원본과 같습니다. 원본 파일은 수정하지 않았습니다.

설정·출력 경로만 빠르게 확인하려면 다음 명령을 사용합니다. 학습이나 파일 생성을
시작하지 않고 실제 학습 진입점의 설정 생성·출력 예약 직전까지 검사합니다.

```powershell
python src/v2_codex/verify_contract.py --sizes 5 10 20
```

## 기존 속도 측정 결과

2026-09-09 비교에서는 크기별 2 instance × 3 seed × 50회 ALNS, 총 18개 사례에서
목적함수·최종 경로·반복별 수락/현재비용·시간 외 통계가 모두 일치했습니다.

| 노드 수 | 원본 합산 | v2 합산 | 속도 향상 | 시간 감소 |
|---|---:|---:|---:|---:|
| n5 | 2.079초 | 1.034초 | 2.01배 | 50.2% |
| n10 | 12.735초 | 4.713초 | 2.70배 | 63.0% |
| n20 | 71.306초 | 18.934초 | 3.77배 | 73.4% |

다른 Python 학습 작업이 실행 중인 Windows 환경의 참고 측정입니다. 전체 PPO/GNN-PPO
학습 시간의 개선율이나 모든 instance에서의 동등성을 보장하는 수치는 아닙니다.
측정 범위는 초기해 생성과 ALNS 탐색이며 상세 결과는
[`summary.json`](../../output/v2_codex/benchmarks/codex_compare_20260909/summary.json)과
[`cases.csv`](../../output/v2_codex/benchmarks/codex_compare_20260909/cases.csv)에 있습니다.

## 변경한 부분

실제 n20 프로파일에서 초기해 생성과 ALNS 10회에 `eval_truck`가 약 72,867번
호출되어 프로파일 측정 시간의 약 94%를 차지했습니다. 기존 코드도 한 반복에서
고객의 30%를 제거합니다. 많은 시간이 드는 곳은 복구 과정에서 남은 고객마다
삽입 후보 전체를 여러 번 평가하는 부분입니다.

이 버전은 묶음 삽입이나 후보 수 축소로 탐색을 바꾸는 대신 다음 계산을 줄입니다.

1. **삽입 후보 평가 재사용:** 한 번의 repair 안에서 `(고객, 트럭)`별 비용과 후보를
   보관합니다. 고객이 삽입된 트럭만 다시 평가합니다. 전역 parking copy 사용량이
   바뀌면 모든 트럭의 후보를 무효화하여 다른 트럭에서 같은 copy를 쓰지 않게 합니다.
2. **빠른 내부 경로 평가:** NumPy 행렬의 반복 scalar 접근과 함수 호출을 Python
   float 조회로 바꿉니다. 변환한 행렬은 process당 최대 8개 instance에 대해 보관합니다.
3. **불가능한 후보 조기 종료:** 로봇이 트럭에 없는 상태에서 재출발하거나 trip 용량을
   초과하는 등 infeasible이 확정된 후보는 이후 스케줄 계산을 생략합니다.
4. **worst-destroy의 부분 평가:** 고객 하나의 제거 이득을 구할 때 해 전체를 복사하고
   모든 트럭·비용 breakdown을 다시 구하지 않습니다. 영향을 받는 트럭만 평가하고
   원래 트럭 순서로 총합을 계산합니다.
5. **제거 시 필요한 부분만 복사:** 변경된 stop/trip만 새로 만들어 공유 객체를
   변경하지 않습니다.

원래 A/B/C/D 삽입 후보, 후보 순서, 1e-9 비교 기준, regret-2의 **트럭별** 두 번째
최선값, 고객 선택 순서를 유지합니다. noise repair는 비용만 재사용하고 매번 기존과
같은 순서로 난수를 새로 뽑습니다. 로봇 출발·회수·대기, 누적 주행거리, 용량,
고객 coverage와 최종 목적함수는 기존 public evaluator로 판정합니다.

고객/trip을 묶어 한 번에 재삽입하거나 상위 후보만 평가하는 방법은 추가 가속 여지가
있지만 탐색 이웃과 PPO action의 효과가 달라집니다. 현재 버전에서는 적용하지 않았습니다.

## 파일과 저장 경로

| 항목 | 경로 |
|---|---|
| 개선한 destroy/repair | `src/v2_codex/operators.py` |
| 내부 비용·feasibility 계산 | `src/v2_codex/cost.py` |
| 기존 순서의 후보 생성 | `src/v2_codex/candidates.py` |
| Vanilla ALNS 실행 | `src/v2_codex/alns/solve.py` |
| PPO 학습/평가 | `src/v2_codex/ppo_alns/{train,test}.py` |
| GNN-PPO 학습/평가 | `src/v2_codex/gnn_ppo_alns/{train,test}.py` |
| 실험 결과 | `output/v2_codex/{alns,ppo_alns,gnn_ppo_alns}/n{size}/` |
| 모델 | `models/v2_codex/{기존 checkpoint 파일명}.pt` |
| 자동 비교 결과 | `output/v2_codex/benchmarks/{run-label}/` |

모델은 **`models/v2_codex/`**에 저장합니다. 기존 `models/`의 다른 모델 파일은
수정하지 않습니다. `data/processed*`와 `configs/params.yaml`은 기존 경로에서 읽습니다.
`--tag`와 `--run-label`의 의미도 동일합니다. 같은 출력이 이미 존재하면 기존처럼
덮어쓰지 않고 오류가 납니다. 다음 실험에서는 다른 label을 사용하세요.

각 실행 모듈은 기존 실행 코드를 별도 namespace로 보관하고 공통 operator를
가져옵니다. Windows spawn worker도 `v2_codex.*`를 사용합니다. 원본과 v2를 한
프로세스에서 함께 import할 수 있으며 전역 monkey patch를 하지 않습니다.
복사본에 남아 있는 기존 operator 함수는 비교용 참고 코드입니다. 실제 실행되는
함수는 `DESTROY`/`DESTROY_OPERATORS` 선언 직전에 가져오는 `operators.py`의 함수입니다.

## 바로 실행하는 작은 예제

기존 Conda 환경을 활성화하고 **저장소 루트**에서 실행합니다. 추가 패키지는 필요 없습니다.

```powershell
conda activate truck-robot-delivery
python src/v2_codex/alns/solve.py --size 5 --iterations 100 --limit 3 --seeds 3 --workers 1 --run-label try_01
```

출력은 `output/v2_codex/alns/n5/runs/try_01/`에 생성됩니다.
`--size 10` 또는 `--size 20`으로 바꾸면 해당 크기를 실행합니다.

n5/n10/n20을 동일 조건으로 순서대로 실행하려면:

```powershell
.\src\v2_codex\run_small_cases.bat try_01
```

이 배치는 각 크기에서 3 instance × 3 seed × 100회 ALNS를 실행합니다.

기존과 같은 요약표와 경로 지도도 만들 수 있습니다. 이 명령은 v2 결과 폴더를 사용하며
지도는 기존 renderer를 그대로 사용합니다. `runs/{label}` 경로도 지원합니다.

```powershell
python src/v2_codex/summarize_results.py --size 5
python src/v2_codex/plot_result.py output/v2_codex/alns/n5/runs/try_01/test_n5_000_s0.json
```

## 원본과 자동 비교

```powershell
python src/v2_codex/benchmark.py --sizes 5 10 20 --instances 3 --seeds 3 --iterations 100 --run-label compare_01
```

같은 instance·seed·반복 수로 원본과 v2를 **순차 실행**하며 실행 순서를 교대합니다.
원본 알고리즘의 결과도 비교 폴더에 요약하므로 기존 결과 폴더에 쓰지 않습니다.

`cases.csv`에는 각 사례의 시간, 목적함수 차이, 경로·반복별 수락/현재비용·통계의
일치 여부가 기록됩니다. `summary.json`에는 노드 수별 합산 시간 비율과 시간 감소율이
기록됩니다. 일치하지 않는 사례가 있으면 결과 파일을 저장한 뒤 실패 종료합니다.

측정 시간은 **초기해 생성 + ALNS 탐색**입니다. import, NPZ 읽기, 결과 비교와 파일
저장은 제외합니다. 반복 측정은 `--repeats 3`으로 할 수 있습니다. 정확한 속도 비교는
다른 학습·검증 작업을 종료한 상태에서 실행하세요. 시간 제한 대신 같은 반복 수를
사용해야 탐색 결과의 동등성을 확인하기 쉽습니다.

## PPO / GNN-PPO 사용

본 실험의 기본 명령은 문서 상단과 같습니다. 다음 예시는 PPO의 최소
1 rollout(2560 environment steps)을 학습하는 작은 동작 확인용입니다.

```powershell
python src/v2_codex/ppo_alns/train.py --size 5 --train-count 3 --total-steps 2560 --search-iterations 100 --reward-mode magnitude --device cpu --env-backend serial --run-label small_01
python src/v2_codex/ppo_alns/test.py --size 5 --limit 3 --seeds 3 --workers 1 --reward-mode magnitude --run-label small_eval_01 --checkpoint models/v2_codex/ppo_alns_n5_reward_magnitude_run-small_01.pt
```

GNN-PPO는 위 경로의 `ppo_alns`를 `gnn_ppo_alns`로 바꿉니다. 본 실험에서는 기존과
같은 학습량을 사용하세요. `--device cuda --env-backend process --observation-codec numpy`
옵션도 그대로 지원합니다. operator 속도 향상 비율이 GPU forward/backward, graph
생성, process 통신까지 포함한 전체 학습의 속도 향상 비율과 같지는 않습니다.

기존 모델로 operator 변경만 비교하려면 기존 파일을 명시적으로 읽을 수 있습니다.

```powershell
python src/v2_codex/ppo_alns/test.py --size 5 --limit 3 --seeds 3 --workers 1 --reward-mode magnitude --checkpoint models/ppo_alns_n5_reward_magnitude.pt --run-label existing_policy_01
```

## 검증과 원본 반영

```powershell
python src/v2_codex/verify_contract.py --sizes 5 10 20
python src/v2_codex/verify.py --sizes 5 10 20 --iterations 30
python src/v2_codex/verify_integration.py
```

`verify_contract.py`는 설정·콘솔 시작 로그·저장 경로를 비교합니다. `verify.py`는
후보 순서·비용·feasibility, 9개 operator 조합, RNG 상태, ALNS trajectory, 로봇 제약,
copy cache 무효화, 공유 객체 보호를 비교합니다. `verify_integration.py`는 CLI·모델 구조,
직렬/Windows process 환경과 checkpoint 호환을 확인합니다. checkpoint가 없는
환경에서는 마지막 명령에 `--skip-checkpoints`를 추가합니다.

실행 검증: 2,081개 후보 평가, 크기별 30회 ALNS trajectory, 두 PPO 방식의 9개 action과
직렬/process 환경(`direct`/`numpy`, episode 자동 reset), 기존 n5 checkpoint의 weight·
추론·직렬/process 평가 일치 검사를 통과했습니다. Vanilla ALNS process CLI의 JSON/CSV,
v2 요약표와 PNG/HTML 지도 생성도 확인했습니다. 대규모 재학습은 실행하지 않았습니다.

검증·비교 결과가 만족스러우면 `operators.py`, `cost.py`, `candidates.py`를 공통 모듈로
옮기고 기존 세 ALNS 모듈의 operator 연결부를 바꾸는 방식으로 반영할 수 있습니다.
이번 작업에서는 원본을 변경하지 않았습니다. 새 버전의 공개 evaluator와 초기해
알고리즘, PPO 코드는 보존했으므로 해당 부분의 교체는 필요하지 않습니다.
