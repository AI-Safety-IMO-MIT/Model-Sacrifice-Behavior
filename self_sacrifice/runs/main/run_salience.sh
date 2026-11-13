#!/bin/bash
# Salience test on Gemma-4-31B-it: serve it with the run_queue3.sh flags, refuse a server that reports another model,
# then run every version (or the versions given as arguments). Each version resumes from salience_<version>.jsonl, so rerunning this script continues.
export PATH=/root/venvs/vllm/bin:$PATH PYTHONDONTWRITEBYTECODE=1 HF_HOME=/workspace/.cache/huggingface
HERE="$(cd "$(dirname "$0")" && pwd)"
WANT=/root/models/gemma-4-31B-it   # google/gemma-4-31B-it, linked to the /workspace blobs by link_model.py (/workspace cannot hold symlinks)
[ -s $WANT/config.json ] || python "$HERE/link_model.py" google/gemma-4-31B-it $WANT || exit 1
check() { curl -s localhost:8000/v1/models | grep -q "\"root\":\"$WANT\"" || { echo "WRONG SERVER, wanted $WANT"; return 1; }; }
if ! curl -sf localhost:8000/health > /dev/null; then
  cd /root
  nohup vllm serve $WANT --served-model-name replica --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.95 --seed 0 \
    --limit-mm-per-prompt '{"image":0,"audio":0}' --host 127.0.0.1 --port 8000 > /root/vllm_salience.log 2>&1 &
  pid=$!
  until curl -sf localhost:8000/health > /dev/null; do
    kill -0 $pid 2>/dev/null || { echo "server died"; tail -20 /root/vllm_salience.log; exit 1; }
    timeout 10 tail -f /dev/null
  done
fi
check || exit 1
cd "$HERE"
VERSIONS="${@:-rerun_main rerun_repair held visible repair_quiet recall}"
for v in $VERSIONS; do
  python run_salience.py $v 32
done
echo salience done
