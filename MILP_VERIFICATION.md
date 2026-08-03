# MILP Fixing Verification Report

날짜: 2026-08-03. `experiments/verify_solution_milp.py`로 저장된 ALNS
해의 라우팅 바이너리(x, y, u, uhat)를 MILP에 고정(lb=ub)하고 재풀이한
결과. 적재·시각·지각 변수는 자유(경로가 함의), symmetry breaking 해제,
풀이 시간제한 60초. 판정 기준: OPTIMAL + |milp_obj − stored obj| ≤ 1e-3.

인스턴스: master 시드 1 스케일링(트럭 4, 로봇 2/트럭, copy 2, β̂=3),
해는 run 시드 0의 3000-iter ALNS best (`results/verify_sols/solutions/`,
n=5는 스모크 런 800-iter).

## 결과 — 7/7 PASS

| 해 | stored obj | MILP obj | diff | status | 풀이시간 |
|---|---|---|---|---|---|
| n=5 roulette (smoke) | 96.6340 | 96.6340 | 0.000000 | OPTIMAL | 0.0s |
| n=15 roulette | 170.7759 | 170.7759 | −0.000000 | OPTIMAL | 0.0s |
| n=15 qlearning | 126.0259 | 126.0260 | +0.000000* | OPTIMAL | 0.0s |
| n=15 gnn_dqn (dueling) | 170.7769 | 170.7769 | +0.000000 | OPTIMAL | 0.0s |
| n=25 roulette | 199.7038 | 199.7038 | −0.000000 | OPTIMAL | 0.1s |
| n=25 qlearning | 192.3522 | 192.3522 | +0.000000 | OPTIMAL | 0.1s |
| n=25 gnn_dqn (dueling) | 201.8382 | 201.8382 | +0.000000 | OPTIMAL | 0.1s |

\* stored obj는 JSON에 소수 6자리 반올림 저장(126.0259 vs 126.02595…);
차이는 반올림 한도 내.

모든 해에서 5개 비용 성분(truck/robot fixed, truck/robot travel,
lateness)이 MILP와 ALNS evaluator 간 소수점 4자리까지 일치. 재현성도
확인: 같은 시드 재실행이 최종 실험과 동일한 목적함수를 산출.

## n=15 QL 이상치 분석 (126.03 vs 170.78)

MILP 고정 검증으로 이 해가 실제 타당함이 증명됐고, 성분 분해가 격차의
원인을 그대로 보여준다:

| 성분 | roulette 170.776 | qlearning 126.026 |
|---|---|---|
| truck fixed | 125.92 (트럭 2대) | **62.96 (트럭 1대)** |
| robot fixed | 14.80 (로봇 2) | 7.40 (로봇 1) |
| truck travel | 29.14 | 43.16 |
| lateness | 0.00 | 12.06 |

QL-ALNS는 트럭 1대로 전체를 처리하며 지각 페널티 12.06과 추가 주행
14.02를 감수하는 대신 두 번째 트럭의 고정비 62.96 + 로봇 고정비를
절약했다 — fleet-sizing 지역해를 벗어난 정당한 해이며, 평가기 버그나
제약 위반이 아니다 (독립 검증기 통과, MILP OPTIMAL 재현).

## n=5 exact 대비 참고

exact 최적(94.3751, OPTIMAL 증명) 대비 검증된 n=5 해(96.6340)는
+2.39%. 고정-바이너리 재풀이의 지각 성분이 ALNS 스케줄과 일치하는
것은 truck no-wait/robot sync 제약 하에서 greedy 최조기 스케줄이
지각을 최소화하기 때문이며, MILP가 이를 재확인했다.

## 실행 방법

```bash
/opt/anaconda3/bin/python experiments/verify_solution_milp.py \
    --solution results/<exp>/solutions/sol_<inst>_<sel>_<seed>.json \
    [--size-cap 25] [--time-limit 60]
```

불가능(INFEASIBLE) 시 Gurobi IIS 제약 이름이 출력된다 (이번 배치에서는
발생하지 않음).
