#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
PYTHON=${PYTHON:-python}
WORKERS=${WORKERS:-30}
T=${T:-4}
TRAIN_JOBS=${TRAIN_JOBS:-3}
RUN_LABEL=${RUN_LABEL:-$(date +%Y%m%d_%H%M%S)}
DRY_RUN=${DRY_RUN:-0}
export RUN_LABEL WORKERS T TRAIN_JOBS
export OMP_NUM_THREADS="$T" MKL_NUM_THREADS="$T" PYTHONUNBUFFERED=1
sizes=("$@")
if ((${#sizes[@]} == 0)); then sizes=(50 100); fi
"$PYTHON" -c 'import os,sys; sys.path.insert(0,"src"); from common.sizes import SUPPORTED_SIZES; from common.artifacts import validate_run_label; validate_run_label(os.environ["RUN_LABEL"]); assert all(int(n) in SUPPORTED_SIZES for n in sys.argv[1:]); assert 1 <= int(os.environ["WORKERS"]) <= 30; assert int(os.environ["T"]) > 0; assert 1 <= int(os.environ["TRAIN_JOBS"]) <= 3' "${sizes[@]}"

# Pin concurrent training round-robin to available NUMA nodes when possible.
nodes=()
if [[ ${NUMA_NODES:-auto} != none ]] && command -v numactl >/dev/null; then
    if [[ ${NUMA_NODES:-auto} == auto ]]; then
        for path in /sys/devices/system/node/node[0-9]*; do
            [[ -d "$path" ]] && nodes+=("${path##*node}")
        done
    else
        read -r -a nodes <<< "$NUMA_NODES"
        for node in "${nodes[@]}"; do
            [[ "$node" =~ ^[0-9]+$ && -d /sys/devices/system/node/node"$node" ]] || { echo "Invalid NUMA node: $node" >&2; exit 1; }
        done
    fi
elif [[ ${NUMA_NODES:-auto} != auto && ${NUMA_NODES:-auto} != none ]]; then
    echo 'NUMA_NODES requires numactl.' >&2; exit 1
fi

pids=()
cleanup() {
    local pid
    for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "$DRY_RUN" != 1 ]]; then
    command -v setsid >/dev/null || { echo 'setsid (util-linux) is required.' >&2; exit 1; }
    mkdir -p logs
    mkdir "logs/$RUN_LABEL" # Require a fresh label to protect existing logs.
fi
print_command() { printf '[run] '; printf '%q ' "$@"; printf '\n'; }
run_logged() {
    local log=$1; shift
    print_command "$@"
    if [[ "$DRY_RUN" != 1 ]]; then "$@" 2>&1 | tee "$log"; fi
}
wait_training() {
    local failed=0 pid
    for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
    if ((failed)); then echo 'Training failed; inspect training logs.' >&2; exit 1; fi
    pids=()
}
for size in "${sizes[@]}"; do
    logs="logs/$RUN_LABEL/n$size"
    if [[ "$DRY_RUN" != 1 ]]; then mkdir "$logs"; fi
    common=(--size "$size" --run-label "$RUN_LABEL")
    run_logged "$logs/alns.log" "$PYTHON" src/alns/solve.py "${common[@]}" --workers "$WORKERS"
    index=0
    for reward in alns_5310 new_best_5 magnitude; do
        prefix=()
        if ((${#nodes[@]})); then
            node=${nodes[index % ${#nodes[@]}]}
            prefix=(numactl "--cpunodebind=$node" "--membind=$node")
        fi
        command=("${prefix[@]}" "$PYTHON" src/ppo_alns/train.py "${common[@]}" --reward-mode "$reward" --device cpu --env-backend process --observation-codec numpy)
        print_command "${command[@]}"
        if [[ "$DRY_RUN" != 1 ]]; then
            setsid "${command[@]}" > "$logs/train_$reward.log" 2>&1 &
            pids+=("$!")
            echo "[training pid=$!] $logs/train_$reward.log"
        fi
        index=$((index + 1))
        if ((${#pids[@]} >= TRAIN_JOBS)); then wait_training; fi
    done
    wait_training
    for reward in alns_5310 new_best_5 magnitude; do
        run_logged "$logs/test_$reward.log" "$PYTHON" src/ppo_alns/test.py "${common[@]}" --reward-mode "$reward" --workers "$WORKERS" --checkpoint "models/ppo_alns_n${size}_reward_${reward}_run-${RUN_LABEL}.pt"
    done
    run_logged "$logs/summary.log" "$PYTHON" scripts/summarize_results.py --size "$size"
done
