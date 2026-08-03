# Changelog — instrumentation, persistence, independent verification

날짜: 2026-08-03. RL-ALNS 실험 파이프라인의 계측·기록·검증 강화 작업.
모든 변경 후 세 selector(roulette/qlearning/gnn_dqn)의 동일 시드
목적함수가 변경 전과 바이트 단위로 일치함을 확인함.

## Task 1 — solve_alns 계측 (src/heuristics/alns.py)

`stats` dict 확장 (타이머/카운터만 추가 — rng 호출 추가·재배열 없음):

- `pair_labels` / `action_hist` / `pair_time_s`: (destroy, repair) 쌍별
  라벨, 호출 횟수 히스토그램(9개, a = destroy_index*3 + repair_index),
  clone+destroy+repair+eval 누적 초
- `selector_overhead_s`: 선택기 비용 (roulette 추첨+가중치 갱신 /
  Q-table select+update / GNN 그래프 구축+추론)
- `accept_count`, `infeasible_count`, `best_update_count`,
  `best_first_hit_iter` (최종 best가 처음 도달된 반복)
- `best_trace`: best 갱신마다 (iter, elapsed_s, best_cost)

검증: grid 기준 인스턴스(시드 0, 300 iters)에서 세 selector 모두
변경 전후 목적함수 동일 (roulette/gnn 174.7189858727,
qlearning 141.3836067640).

## Task 2 — 해 저장·기록 (experiments/run_qlearning.py)

- `run_once`가 best `Solution` 객체를 반환하도록 변경
- compare 모드에서 매 실행마다:
  - `results/<exp>/solutions/sol_{instance}_{selector}_{seed}.json`
    (routes 원본 + obj/init_obj + instance_set 재구축 메타 +
    make_route_svg용 plot_payload)
  - `results/<exp>/traces/trace_{instance}_{selector}_{seed}.csv`
    (best 갱신 수렴 궤적)
- `runs_compare.csv` 추가 컬럼: init_obj, feasible(evaluator 재확인),
  indep_feasible/indep_obj_diff(Task 3), trucks/robots/robot_cust,
  비용 분해 5종, best_first_hit_iter, selector_overhead_s,
  action_hist(JSON 문자열)
- `summary_compare.csv` 추가 컬럼: mean_init_obj, n_feasible,
  n_indep_feasible, mean_best_first_hit_iter,
  mean_selector_overhead_s

검증: 스모크 런(sizes [5,10] × seeds [0,1] × roulette/qlearning)에서
전 컬럼 생성, 같은 시드의 init_obj가 selector 간 동일함을 확인.

## Task 3 — 독립 검증기 (src/heuristics/validator.py, 신규)

`check_solution(inst, e_c, l_c, sol_like, cost_params)`
→ `(ok, violations, recomputed_obj)`.

- solution.py를 import하지 않고 인스턴스 원시값(arc_zones, alpha,
  lam, park_groups)과 비용 상수로 전 제약을 재검사: coverage 정확히
  1회(18), custody 리플레이(19–25, 동일 copy 회수 금지·회수 copy가
  같은 트럭 경로의 이후 정차인지 포함), copy 사용 예산, trip 크기
  (β̂)·로봇 누적거리(φ̂)·트럭 적재(β)·트럭 주행거리(φ), 스케줄
  재계산(26–38) 및 목적함수 재산출
- `cost_params_from(pr)`: Params → 비용 상수 dict
- run_qlearning.py compare 모드에 연결: indep_feasible /
  indep_obj_diff 컬럼, 위반 또는 |diff| > 1e-4 시 콘솔 경고 출력

검증: 스모크 8개 해 전부 통과(|diff| ≤ 1e-6). 유닛 테스트
(tests/test_validator.py) 4종 — 정상 해 evaluator 일치, 고객 중복,
회수 copy가 이후 정차에 없음, trip 용량 초과의 3개 조작 해 거부.

## Task 4 — MILP 고정 검증 (experiments/verify_solution_milp.py, 신규)

- 저장된 해 JSON을 로드해 인스턴스를 재구축하고, MILP의 라우팅
  바이너리(x, y, u, uhat)를 해의 아크로 고정(lb=ub) 후 60초 재풀이.
  적재·시각·지각 변수는 자유(경로가 함의). OPTIMAL +
  |milp_obj − stored obj| ≤ 1e-3이면 PASS, 불가능이면 IIS 출력.
- CLI: `--solution <path> [--size-cap 25] [--time-limit 60]`,
  gurobipy 필요 (anaconda python으로 실행)
- **src/model.py 변경**: `run_model(..., fix_binaries=None)` 파라미터
  추가. run_model이 gurobipy 모델을 밖으로 노출하지 않고 dispose하기
  때문에 외부 고정이 불가능해 추가한 최소 훅. 기본값 None이면 기존
  동작과 완전 동일. 고정 모드 불가능 시 IIS 제약 이름을 res["iis"]로
  반환. 주의: 고정 모드에서는 반드시 `symmetry_breaking=False`
  (대칭 제거 부등식이 정준 라벨링을 가정하므로).

검증: MILP_VERIFICATION.md 리포트 참조.

## 부수 참고

- formulation_v6.md는 저장소에 없어 제약 목록은 solution.py 헤더와
  model.py 제약 주석 (4)–(55)에서 도출함.
- 해 저장/검증은 compare 모드에만 적용 (하이퍼파라미터 search 모드는
  수백 파일 생성을 피하기 위해 제외).
