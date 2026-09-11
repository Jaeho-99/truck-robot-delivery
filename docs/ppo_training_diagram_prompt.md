# PPO-ALNS 학습 과정 다이어그램 생성 프롬프트 (pptx)

아래 `---` 사이 전체를 그대로 복사해서 입력하세요.

---

## [역할]
당신은 학술 발표용 다이어그램을 만드는 디자이너입니다. 강화학습(PPO) 기반 ALNS 연산자 선택 모델의 **학습 과정(코드 실행 흐름)** 을 한 눈에 이해할 수 있는 다이어그램으로 만들어 주세요.

## [산출물 형식 — 가장 중요]
- **반드시 편집 가능한 `.pptx` 파일로 산출**하세요. SVG/PNG/PDF/HTML 이미지가 아니라, PowerPoint에서 열었을 때 제가 도형 하나하나를 클릭해 크기·색·문구를 직접 고칠 수 있어야 합니다.
- 파이썬 `python-pptx` 라이브러리로 생성하고, 모든 요소는 **네이티브 PowerPoint 개체**여야 합니다:
  - 박스 = `shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE / RECTANGLE)`
  - 화살표 = `add_connector(MSO_CONNECTOR.STRAIGHT / ELBOW)` 또는 `MSO_SHAPE.RIGHT_ARROW` / `MSO_SHAPE.CIRCULAR_ARROW`
  - 글자 = 도형의 `text_frame` 또는 `add_textbox` (모든 글자는 실제 텍스트여야 하며, 이미지로 렌더링된 글자는 절대 금지)
- **금지**: 슬라이드에 그림(picture)을 통째로 얹기, matplotlib/graphviz/mermaid로 그린 이미지 삽입, SmartArt, 도형 그룹화(group).
- 슬라이드 크기 **16:9 (13.333 in × 7.5 in)**, 빈 레이아웃(blank) 사용.
- 좌표는 인치 단위로 명시적으로 지정하고, **도형끼리 절대 겹치지 않게** 배치하세요. 글자가 도형 밖으로 넘치지 않도록 도형은 `word_wrap = True`로 두고 폰트 크기를 도형 크기에 맞추세요(본문 9~11pt).

## [슬라이드 구성] — 총 3장
- **슬라이드 1: 전체 학습 파이프라인** (메인. 이 한 장만 봐도 이해되도록)
- **슬라이드 2: ALNS 1 iteration(= 1 transition) 상세 확대**
- **슬라이드 3: 하이퍼파라미터 / 설정 표**

---

# 슬라이드 1 — "PPO-ALNS 학습 파이프라인"

## 전체 레이아웃 (13.333 × 7.5 in 기준)
- 제목 영역: y 0.25–0.95
- **왼쪽 큰 영역 (x 0.45–6.60, y 1.15–6.35)** = Phase 1
- **오른쪽 위 (x 6.90–12.90, y 1.15–2.80)** = Phase 2
- **오른쪽 아래 (x 6.90–12.90, y 3.05–6.35)** = Phase 3
- **하단 띠 (y 6.50–7.15)** = 외부 루프 화살표 + 총계 박스

## 제목
- 메인: `PPO-ALNS 학습 파이프라인 : 1 update 사이클`
- 부제(작은 글씨): `1 update = Rollout 수집(2,560 transitions) → GAE → PPO 업데이트(최대 10 epochs)   |   총 117 updates`

---

## ■ Phase 1 영역 (왼쪽 큰 컨테이너)
연한 배경의 큰 둥근 사각형으로 감싸고, 좌측 상단에 라벨:
`Phase 1. Rollout 수집 — 10개 env를 동시에 256 step 전진`

컨테이너 안 구성 (위 → 아래 흐름):

**(1-A) 인스턴스 풀 박스** (좌측 상단)
- 제목: `학습 인스턴스 풀`
- 본문: `250개 (data/processed/train)`
- 아래로 화살표, 화살표 라벨: `복원추출 (with replacement)`

**(1-B) 10개 환경(worker) 스택**
- 세로로 살짝 겹쳐 쌓인 카드 3장으로 "여러 개"임을 표현하고, 맨 앞 카드에만 텍스트를 넣으세요.
- 맨 앞 카드 제목: `env #1 … #10   (n_envs = 10, 병렬)`
- 카드 본문 3줄:
  - `① 인스턴스 1개 샘플링`
  - `② 초기해 생성 (congestion-aware)`
  - `③ ALNS 1 iteration 수행 → s(t+1), r(t), done(t)`
- 카드 우측 하단 작은 주석 텍스트:
  `1 episode = 100 iterations (search_iterations)`
  `episode 종료(done=1) 시 자동 리셋 → 새 인스턴스 재샘플링`

**(1-C) 정책 네트워크 박스** (env 카드와 양방향 연결)
- 제목: `Actor-Critic (MLP)`
- 본문:
  - `state g_t : 9차원 벡터`
  - `Actor  : 9 → 64(tanh) → 9 actions`
  - `Critic : 9 → 64(tanh) → 1`
  - `π_eps = 0.9 · π_actor + 0.1/9   (ε-uniform 탐험)`
- env 스택 → 정책 화살표 라벨: `s_t 10개를 1개 배치로 묶어 1회 forward`
- 정책 → env 스택 화살표 라벨: `a_t (10개), log π_old(a_t|s_t), V(s_t)`

**(1-D) Rollout Buffer 박스** (Phase 1 컨테이너 오른쪽 끝)
- 제목: `Rollout Buffer (on-policy)`
- 본문:
  - `저장 단위 : (s_t, a_t, r_t, log π_old, V(s_t), done_t)`
  - `매 step마다 10개씩 적재`
  - `256 step × 10 env = 2,560 transitions`
  - `shape = [256, 10]`
- env 스택 → buffer 화살표 라벨: `transition 10개 저장`

**(1-E) 내부 루프 표현**
- (1-B)~(1-D)를 감싸는 **점선 둥근 사각형** + 오른쪽에서 왼쪽으로 되돌아오는 굽은 화살표.
- 루프 라벨(굵게): `for t = 1 … 256   (vector step : 10개 env가 매 step 동기 진행)`
- 루프 옆에 강조 주석 박스(연한 노랑):
  `256 = 100 + 100 + 56`
  `→ 각 env는 episode 2개를 완주하고 3번째 episode의 56 step까지만 진행`
  `→ 이 마지막 episode는 잘린 채(truncated, done=0) 다음 update로 이어짐`

---

## ■ Phase 2 영역 (오른쪽 위)
컨테이너 라벨: `Phase 2. Advantage 계산 (GAE)`
안에 좌 → 우로 작은 박스 3개를 화살표로 연결:
1. 제목 `부트스트랩` / 본문 `V(s_257) 계산 — 잘린 episode 보정`
2. 제목 `역방향 누적` / 본문 `δ_t = r_t + γ·V(s_{t+1})·(1 − done_t) − V(s_t)` / `Â_t = δ_t + γλ·(1 − done_t)·Â_{t+1}`
3. 제목 `저장` / 본문 `R_t = Â_t + V(s_t)` / `γ = 0.99,  λ = 0.95`

- Phase 1의 Buffer 박스 → Phase 2로 굵은 화살표, 라벨 `2,560개 전체`

---

## ■ Phase 3 영역 (오른쪽 아래)
컨테이너 라벨: `Phase 3. PPO 업데이트 (최대 10 epochs)`

위 → 아래 흐름:
1. 박스 `2,560 transitions`
   → 화살표 라벨: `무작위 셔플 (epoch마다 새로 셔플)`
2. 박스 `mini-batch 40개 × batch size 64`   (본문에 `40 × 64 = 2,560`)
   → 화살표 라벨: `mini-batch 1개씩`
3. 박스 `1 mini-batch 학습` (본문 4줄)
   - `ratio = exp(log π_new − log π_old)`
   - `L_CLIP = −min(ratio·Â,  clip(ratio, 0.8, 1.2)·Â)`   *(Â는 미니배치 단위로 정규화)*
   - `L = L_CLIP + 0.5 · L_VF − 0.01 · H`
   - `backward → grad clip 0.5 → Adam step (lr 3e-4)`
4. 3번 박스를 감싸는 **점선 루프** + 되돌림 화살표, 라벨:
   `× 40 mini-batches  (= 1 epoch, optimizer step 40회)`
5. 그 바깥을 감싸는 **두 번째 점선 루프** + 되돌림 화살표, 라벨:
   `× 10 epochs  (같은 rollout을 10번 재사용)`
6. 마름모 판단 도형(`MSO_SHAPE.DIAMOND`):
   `epoch 종료 시  approx KL > 1.5 × 0.02 ?`
   - `Yes` → 오른쪽으로 빠지는 화살표 → 작은 박스 `남은 epoch 조기 종료`
   - `No` → 아래로 → `다음 epoch`
7. 마지막 박스: `Buffer 폐기 (on-policy)   +   10 update마다 checkpoint 저장`

---

## ■ 하단 띠 (외부 루프 + 총계)
- Phase 3 아래에서 왼쪽 Phase 1 위로 되돌아가는 **굵은 큰 화살표**(슬라이드 폭을 가로지르는 ELBOW 커넥터).
  라벨(굵게, 강조색): `for update = 1 … 117`
- 오른쪽 끝에 총계 박스(강조 테두리):
  `117 updates × 2,560 = 299,520 transitions ≈ 300,000`
  `≈ 2,995 episodes,  optimizer step 최대 46,800회`

---

# 슬라이드 2 — "ALNS 1 iteration = 1 transition"

슬라이드 1의 `ALNS 1 iteration` 박스를 확대한 상세도. 좌 → 우 5단계 흐름(각 단계는 둥근 사각형, 사이는 화살표):

1. **상태 관측** — `s_t = g_t (9차원)`
   본문: `[로봇 담당 비율, 혼잡도 비율, 직전 best 갱신, 직전 수용, 직전 current 개선, current = best 여부, best 대비 비용차, 정체 횟수, 진행률 t/100]`
2. **연산자 선택 (Actor)** — `a_t ∈ {0, …, 8}`
   본문: `a = destroy_idx × 3 + repair_idx`
   `destroy : random / worst / related`
   `repair  : greedy / greedy+noise / regret-2`
   → 3 × 3 = 9개 조합을 3행 3열 작은 사각형 격자로도 함께 표현
3. **Destroy & Repair**
   본문: `현재해 복사 → 고객 q = round(0.3 × |C|)명 제거 → 재삽입`
4. **평가**
   본문: `f_new, 실행가능성 판정` / `f_new < f_best 이면 best 갱신`
5. **수용 판단 (Simulated Annealing)**
   본문: `개선이면 무조건 수용, 아니면 확률 exp(−(f_new − f_cur)/T)로 수용`
   `T = T0 · (1 − t/100),   T0 = 0.05 · f_init / ln2`

그 아래 가로로 긴 결과 박스 2개:
- **보상**: `r_t = 10 × max(0, f_best_prev − f_best) / f_init   (reward_mode = magnitude)`
  작은 주석: `대안 모드 : alns_5310 (5/3/1/0), new_best_5`
- **종료 판정**: `done_t = 1  if  t = 100  else  0`
  작은 주석: `done = 1 → 새 인스턴스 샘플링 후 새 초기해로 리셋`

맨 아래 한 줄: `→ (s_t, a_t, r_t, log π_old, V(s_t), done_t) 를 Rollout Buffer에 저장`
(슬라이드 1의 Buffer 박스와 같은 색으로 통일)

---

# 슬라이드 3 — 하이퍼파라미터 / 설정 표

`python-pptx`의 **실제 표 개체**(`shapes.add_table`)로 만들어 주세요. 표 2개를 좌우로 배치.

**표 A. 학습 설정** (2열: 항목 / 값)

| 항목 | 값 |
|---|---|
| 학습 인스턴스 수 | 250 (복원추출) |
| 병렬 환경 수 (n_envs) | 10 |
| env당 rollout 길이 (t_rollout) | 256 |
| update당 transition 수 | 256 × 10 = 2,560 |
| episode 길이 (search_iterations) | 100 |
| mini-batch 크기 / 개수 | 64 / 40 |
| epoch 수 (k_epochs) | 최대 10 (KL 조기종료) |
| 총 update 수 | 117 |
| 총 transition 수 | 299,520 ≈ 300,000 |

**표 B. PPO 하이퍼파라미터** (2열)

| 항목 | 값 |
|---|---|
| learning rate (Adam) | 3e-4 (eps 1e-5) |
| 할인율 γ / GAE λ | 0.99 / 0.95 |
| clip ε | 0.2 |
| value 계수 c1 / entropy 계수 c2 | 0.5 / 0.01 |
| target KL | 0.02 (1.5배 초과 시 조기종료) |
| gradient clipping | 0.5 |
| ε-uniform 탐험 | 0.1 |
| 파괴 비율 (DOD) | 0.3 |
| SA 초기온도 계수 (w_start) | 0.05 |

---

# [스타일 가이드]
- 폰트: 한글은 `맑은 고딕`, 숫자·수식은 `Consolas`. 제목 24pt, 컨테이너 라벨 14pt(굵게), 박스 제목 12pt(굵게), 본문 9~10pt.
- 색상은 **명시적 RGB**로 지정하세요(테마 색 사용 금지). 학술 발표용 차분한 팔레트:
  - Phase 1 (데이터 수집): 배경 `RGB(219, 234, 254)` / 테두리·제목 `RGB(37, 99, 235)`
  - Phase 2 (GAE): 배경 `RGB(220, 252, 231)` / 테두리 `RGB(22, 163, 74)`
  - Phase 3 (학습): 배경 `RGB(255, 237, 213)` / 테두리 `RGB(234, 88, 12)`
  - 신경망 박스: 배경 `RGB(237, 233, 254)` / 테두리 `RGB(124, 58, 237)`
  - Buffer 박스: 배경 `RGB(243, 244, 246)` / 테두리 `RGB(75, 85, 99)`
  - 강조 주석: 배경 `RGB(254, 249, 195)` / 테두리 `RGB(202, 138, 4)`
  - 본문 글자: `RGB(31, 41, 55)`
- 모든 박스는 둥근 모서리, 테두리 1.25pt, 그림자 없음(`shadow.inherit = False`).
- 루프를 나타내는 점선 박스는 채우기 없음 + 점선 테두리(`line.dash_style = MSO_LINE_DASH_STYLE.DASH`).
- 화살표 라벨은 화살표와 겹치지 않게 살짝 위/옆에 배치.
- 슬라이드마다 우측 하단에 작은 회색 캡션: `src/ppo_alns/ppo.py — train() / ppo_update() / ALNSEnv.step()`

# [검증 체크리스트 — 생성 후 반드시 확인]
1. `.pptx`를 열었을 때 모든 박스·화살표·글자가 개별 선택·수정 가능한가?
2. 이미지로 삽입된 요소가 하나도 없는가?
3. 도형끼리 겹치거나 글자가 도형 밖으로 넘치는 곳이 없는가?
4. 슬라이드 1만 봐도 `250 인스턴스 → 10 env × 256 step = 2,560 → GAE → 40 mini-batch × 10 epoch → × 117 updates → 300k` 흐름이 읽히는가?
5. 위에 적힌 숫자·수식이 **한 글자도 바뀌지 않고** 그대로 들어갔는가?
