#!/usr/bin/env bash
# Follow-up benchmark for sglang-omni PR #1840, prompted by issue #1814: the two
# movies800time validation samples excluded there for runaway generation.
# A/B on one dependency stack: the PR base commit on main vs the PR branch.
# Needs a single sm90+ GPU (H100/H200/B200) because sgl-kernel ships sm90/sm100.
set -u

BASE_SHA=3f819f9cdae3d4eeec22f73306c9067a1ec2542e
BASE_PKG="git+https://github.com/sgl-project/sglang-omni.git@${BASE_SHA}"
PR_PKG="git+https://github.com/ruiling-smartbear/sglang-omni.git@fix/moss-td-short-audio-token-budget"

versions() {  # $1 = label
  python3 - "$1" <<'PY'
import importlib.metadata as m, sys
def v(name):
    try:
        return m.version(name)
    except Exception:
        return "missing"
import sglang_omni.models.moss_transcribe_diarize.request_builders as rb
# Fingerprint of the installed code, so a skipped or cached pip install cannot
# quietly benchmark one side against itself.
floor = getattr(rb, "_MIN_SCALED_OUTPUT_TOKENS", None)
print(
    f"== VERSIONS | {sys.argv[1]} | sglang-omni={v('sglang-omni')} sglang={v('sglang')} "
    f"torch={v('torch')} pr_floor_constant={floor}"
)
PY
}

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

echo "== install base (main @ ${BASE_SHA:0:7}) with dependencies =="
# Full dependency resolution happens once, here. The PR install below reuses it
# with --no-deps, so both sides run on byte-identical dependencies and the only
# difference is the three files this PR touches.
pip install -q "$BASE_PKG" 2>&1 | tail -3
# A previous run may have left the PR build installed under the same version
# string, and pip then reports the requirement as already satisfied. Force the
# base code in explicitly; dependencies stay untouched.
pip install -q --no-deps --force-reinstall "$BASE_PKG" 2>&1 | tail -1
# the package pins typer>=0.9.0, but the sgl-omni CLI uses typing.Literal
# options that older typer cannot parse; make sure a current typer is present.
pip install -q -U typer 2>&1 | tail -1
versions base

echo "== fetch test audio =="
rm -rf /tmp/so && git clone -q --depth 1 --filter=blob:none --sparse https://github.com/sgl-project/sglang-omni /tmp/so
cd /tmp/so && git sparse-checkout set -q tests/data

echo "== fetch the two movies800time samples excluded in #1814 =="
pip install -q "datasets>=3" 2>&1 | tail -1
python3 - <<'PY'
import os
import shutil
import soundfile as sf
from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download

REPO = "zhaochenyang20/movies800time"
WANT = {"val_000237", "val_000318"}  # #1814 calls them sample-000237 / sample-000318
os.makedirs('/tmp/runaway', exist_ok=True)

# Metadata pass: the audio column carries the repo path of each clip and the
# reference transcription; the bytes themselves are fetched by path below.
ds = load_dataset(REPO, split="validation", streaming=True).cast_column("audio", Audio(decode=False))
hits = {}
for index, row in enumerate(ds):
    path = str(row["audio"].get("path") or "")
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem in WANT:
        relpath = path.split("::")[0].split("/", 3)[-1] if "@" not in path else path.split("@", 1)[1].split("/", 1)[1]
        hits[stem] = {"index": index, "relpath": relpath, "reference": str(row.get("transcription") or "")}
    if len(hits) == len(WANT) or index > 900:
        break
for stem, info in sorted(hits.items()):
    local = hf_hub_download(REPO, filename=info["relpath"], repo_type="dataset")
    destination = f"/tmp/runaway/{stem}{os.path.splitext(local)[1] or '.flac'}"
    shutil.copyfile(local, destination)
    print(f"== {stem}: index={info['index']} {info['relpath']} duration={round(sf.info(destination).duration, 2)}s "
          f"reference_chars={len(info['reference'])} reference={info['reference'][:80]!r}")
print("== samples found:", len(hits))
PY
ls -la /tmp/runaway/ | sed 's/^/== /'

# DeepGEMM's _C.so dlopens libnvrtc.so.13, which lives inside the pip-installed
# nvidia wheels rather than on the system loader path; expose those dirs and
# skip DeepGEMM JIT anyway (not needed for a bf16 0.9B model).
export SGL_ENABLE_JIT_DEEPGEMM=0
export SGLANG_ENABLE_JIT_DEEPGEMM=0
NVIDIA_LIBS=$(python3 -c "import glob;print(':'.join(sorted(set(glob.glob('/usr/local/lib/python3.*/site-packages/nvidia/*/lib')))))")
export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"
python3 -c "import ctypes,glob;f=glob.glob('/usr/local/lib/python3.*/site-packages/nvidia/*/lib/libnvrtc.so.13');print('libnvrtc.so.13 found:',bool(f));f and ctypes.CDLL(f[0])" || pip install -q "nvidia-cuda-nvrtc-cu13" 2>&1 | tail -1

serve() {
  nohup sgl-omni serve --model-path OpenMOSS-Team/MOSS-Transcribe-Diarize \
    --port 8000 --mem-fraction-static 0.80 --asr.engine.enable_torch_compile false > "/tmp/server_$1.log" 2>&1 &
  echo $! > /tmp/server.pid
}

wait_ready() {
  for i in $(seq 1 120); do
    curl -sf -o /dev/null localhost:8000/health && { echo "server ready after ~$((i*5))s"; return 0; }
    kill -0 "$(cat /tmp/server.pid)" 2>/dev/null || { echo "== SERVER FAILED TO START"; tail -25 "/tmp/server_$1.log" | sed "s/^/== /"; return 1; }
    sleep 5
  done
  echo "== SERVER FAILED TO START (timeout)"; tail -25 "/tmp/server_$1.log" | sed "s/^/== /"; return 1
}

stop_server() {
  kill "$(cat /tmp/server.pid)" 2>/dev/null; sleep 8
  pkill -f "sgl-omni" 2>/dev/null; sleep 4
}

bench() {  # $1 = label
  for f in /tmp/runaway/* /tmp/so/tests/data/query_to_cars.wav; do
    for i in 1 2; do
      T=$(curl -s --max-time 1200 -o /tmp/resp.json -w '%{time_total}' -X POST localhost:8000/v1/audio/transcriptions \
        -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize -F "file=@$f" -F response_format=json)
      TXT=$(python3 -c "import json;t=json.load(open('/tmp/resp.json')).get('text','');print(f'chars={len(t)} | '+t[:70].replace(chr(10),' '))" 2>/dev/null || head -c 70 /tmp/resp.json)
      echo "RESULT | $1 | $(basename "$f") | run$i | ${T}s | ${TXT}"
    done
  done
}

echo "== BASELINE main@${BASE_SHA:0:7} =="
serve baseline
if ! wait_ready baseline; then echo "ABORT: baseline server failed"; exit 1; fi
bench "main@${BASE_SHA:0:7}"
stop_server

echo "== install PR #1840 branch (no dependency changes) =="
pip install -q --no-deps --force-reinstall "$PR_PKG" 2>&1 | tail -2
pip install -q -U typer 2>&1 | tail -1
versions pr1840
python3 -c "import sglang_omni.models.moss_transcribe_diarize.request_builders as r;print('PR floor constant present:', hasattr(r,'_MIN_SCALED_OUTPUT_TOKENS'))"

echo "== FIXED PR#1840 =="
serve fixed
if ! wait_ready fixed; then echo "ABORT: fixed server failed"; exit 1; fi
bench "pr1840"
stop_server

echo "BENCH_DONE"
