#!/usr/bin/env bash
# Build and verify the pure-CUDA fp16 FMHA backend (fmha_cuda) on a Colab T4.
#
# Usage:
#   ./build_cuda_colab.sh [session-name]
#
# Assumes: repo root = current dir, colab CLI installed, GPU session exists or
# creatable. Steps: (1) upload repo + csrc to a fresh session, (2) pin torch
# 2.6.0 + triton 3.2.0, (3) build the extension with setup_cuda.py,
# (4) run the correctness + speed checks, printing results.
set -euo pipefail

SESSION="${1:-flash-attn-t4}"
REPO_TGZ="/tmp/repo_cuda.tgz"

echo "==> Archiving repo"
rm -f "$REPO_TGZ"
tar -czf "$REPO_TGZ" \
    --exclude='./.git' --exclude='*/__pycache__' \
    --exclude='./.kilo' --exclude='./.vscode' --exclude='./.agents' \
    .

echo "==> Uploading repo"
colab upload -s "$SESSION" "$REPO_TGZ" "/content/repo_cuda.tgz" >/dev/null

echo "==> Extracting + pinning env"
colab exec -s "$SESSION" python3 - <<'EOF'
import shutil, tarfile, subprocess, sys, os
shutil.rmtree("/content/repo_cuda", ignore_errors=True)
os.makedirs("/content/repo_cuda", exist_ok=True)
with tarfile.open("/content/repo_cuda.tgz") as a:
    a.extractall("/content/repo_cuda")
# Pin the repo's torch/triton versions.
log = open("/content/pip-install.log", "w")
proc = subprocess.Popen(
    [sys.executable, "-m", "pip", "install", "--no-cache-dir",
     "torch==2.6.0", "triton==3.2.0"],
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(proc.pid)
EOF

echo "==> Waiting for pip install"
colab exec -s "$SESSION" python3 - <<'EOF'
import time, os
for _ in range(120):
    if os.path.exists("/content/pip-install.log"):
        log = open("/content/pip-install.log").read()
        if "Successfully installed" in log:
            print("pip done")
            break
    time.sleep(5)
else:
    print("pip install timed out")
    raise SystemExit(1)
EOF

echo "==> Building extension"
colab exec -s "$SESSION" python3 - <<'EOF'
import subprocess, os
os.makedirs("/content/fmha_build/flash_attn_triton", exist_ok=True)
# copy csrc + setup into a build dir with the package subdir present
shutil = __import__("shutil")
shutil.rmtree("/content/fmha_build/csrc", ignore_errors=True)
shutil.copytree("/content/repo_cuda/csrc", "/content/fmha_build/csrc")
shutil.copy("/content/repo_cuda/setup_cuda.py", "/content/fmha_build/setup_cuda.py")
p = subprocess.Popen(["/usr/bin/python3", "setup_cuda.py", "build_ext", "--inplace"],
                     cwd="/content/fmha_build",
                     stdout=open("/content/build.log", "w"),
                     stderr=subprocess.STDOUT, start_new_session=True)
print(p.pid)
EOF

echo "==> Waiting for build"
colab exec -s "$SESSION" python3 - <<'EOF'
import time, os
for _ in range(180):
    if os.path.exists("/content/build.log"):
        log = open("/content/build.log").read()
        if "copying build/lib" in log:
            print("build done")
            break
        if "error" in log.lower():
            print(log[-2000:])
            raise SystemExit(1)
    time.sleep(5)
else:
    print("build timed out")
    raise SystemExit(1)
EOF

echo "==> Copying .so into repo package + running checks"
colab exec -s "$SESSION" python3 - <<'EOF'
import glob, shutil
so = glob.glob("/content/fmha_build/flash_attn_triton/*.so")[0]
shutil.copy(so, "/content/repo_cuda/flash_attn_triton/")
print("copied", so)
EOF

echo "==> Correctness + speed checks"
colab exec -s "$SESSION" python3 - <<'EOF'
import sys, time
sys.path.insert(0, "/content/repo_cuda")
import torch
import flash_attn_triton as F
import flash_attn_triton.fmha_cuda as fc

D = "cuda"
torch.manual_seed(0)
S = 64
B, H, HD = 2, 4, 64
q = torch.randn(B, S, H, HD, device=D, dtype=torch.float16)
k = torch.randn(B, S, H, HD, device=D, dtype=torch.float16)
v = torch.randn(B, S, H, HD, device=D, dtype=torch.float16)
sm = 1.0 / (HD ** 0.5)

for causal in (False, True):
    out, lse = fc.fmha_forward(q.transpose(1, 2).contiguous(),
                               k.transpose(1, 2).contiguous(),
                               v.transpose(1, 2).contiguous(), sm, causal)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal)
    print("causal=%s fwd rel: %.5f" % (causal,
          (out.float() - ref.float()).abs().max() / ref.float().abs().max()))

# backward check
out, lse = fc.fmha_forward(q.transpose(1, 2).contiguous(),
                           k.transpose(1, 2).contiguous(),
                           v.transpose(1, 2).contiguous(), sm, True)
do = torch.randn_like(out)
dq, dk, dv = fc.fmha_backward(q.transpose(1, 2).contiguous(),
                              k.transpose(1, 2).contiguous(),
                              v.transpose(1, 2).contiguous(), out, do, lse, sm, True)
q2 = q.detach().clone().requires_grad_()
k2 = k.detach().clone().requires_grad_()
v2 = v.detach().clone().requires_grad_()
o2 = torch.nn.functional.scaled_dot_product_attention(
    q2.transpose(1, 2), k2.transpose(1, 2), v2.transpose(1, 2), is_causal=True)
(o2 * do.float().half()).sum().backward()
for name, d8, df in [("dQ", dq, q2.grad.transpose(1, 2)),
                     ("dK", dk, k2.grad.transpose(1, 2)),
                     ("dV", dv, v2.grad.transpose(1, 2))]:
    print("%s rel: %.5f" % (name,
          (d8.float() - df.float()).abs().max() / df.float().abs().max()))

# speed
def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3

for Ss in (512, 2048):
    qh = torch.randn(B, H, Ss, HD, device=D, dtype=torch.float16)
    kh = torch.randn(B, H, Ss, HD, device=D, dtype=torch.float16)
    vh = torch.randn(B, H, Ss, HD, device=D, dtype=torch.float16)
    fc.fmha_forward(qh, kh, vh, sm, True)
    print("S=%d cuda fwd %.3f ms" % (Ss, bench(lambda: fc.fmha_forward(qh, kh, vh, sm, True))))
print("ALL DONE")
EOF
