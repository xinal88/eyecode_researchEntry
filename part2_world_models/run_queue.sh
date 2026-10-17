#!/usr/bin/env bash
# Resumable policy-experiment queue for one GPU. Runs jobs two at a time and skips any run
# whose results/policy/<name>/eval.json already exists. Launch inside tmux, e.g.
#   tmux new -d -s p2_gpu6 'bash run_queue.sh 6 A 2>&1 | tee logs/queue_gpu6.log'
cd "$(dirname "$0")"
PY=../.venv/bin/python
export CUDA_VISIBLE_DEVICES=$1
D="--dreams dreams_wm_ft"
case $2 in
A) JOBS=("real25_dream300 --real target $D --n_dreams 300"
         "dream300_only $D --n_dreams 300"
         "real25_dream1000 --real target $D --n_dreams 1000"
         "real25_dream300_s1 --real target $D --n_dreams 300 --seed 1"
         "real25_dream300_s2 --real target $D --n_dreams 300 --seed 2"
         "dream300_only_s1 $D --n_dreams 300 --seed 1") ;;
B) JOBS=("real25_dream100 --real target $D --n_dreams 100"
         "real25_dream300pa --real target $D --n_dreams 1000 --min_pa 0.53"
         "real25_scratchdream300 --real target --dreams dreams_wm_scratch --n_dreams 300"
         "real25_dream100_s1 --real target $D --n_dreams 100 --seed 1"
         "real25_dream300_s3 --real target $D --n_dreams 300 --seed 3"
         "real25_dream300pa_s1 --real target $D --n_dreams 1000 --min_pa 0.53 --seed 1") ;;
C) JOBS=("real25_s4 --real target --seed 4"
         "real25_dream300_s4 --real target $D --n_dreams 300 --seed 4") ;;
esac
run() {
  local name=${1%% *}
  if [ -f results/policy/$name/eval.json ]; then echo "skip $name"; return; fi
  echo "$(date +%T) start $name"
  $PY -u policy.py --name $1 > logs/policy_$name.log 2>&1
  echo "$(date +%T) $(grep -h SUCCESS logs/policy_$name.log || echo "$name FAILED")"
}
for ((i = 0; i < ${#JOBS[@]}; i += 2)); do
  run "${JOBS[$i]}" &
  [ $((i + 1)) -lt ${#JOBS[@]} ] && run "${JOBS[$((i + 1))]}" &
  wait
done
echo "queue $2 done"
