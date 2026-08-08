import torch, torch.nn as nn
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
print("available:", torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit("NO CUDA")
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("arch_list:", torch.cuda.get_arch_list())
dev="cuda"
# The real question: does a BACKWARD pass produce finite grads on sm_110a, in the dtype and at the
# shape a 210M draft head actually trains at? A forward-only smoke test proves nothing about autograd.
for dt in (torch.float32, torch.bfloat16):
    try:
        m = nn.Sequential(nn.Linear(4096,4096), nn.SiLU(), nn.Linear(4096,4096)).to(dev, dt)
        x = torch.randn(64,4096, device=dev, dtype=dt, requires_grad=True)
        y = m(x); loss = y.float().pow(2).mean(); loss.backward()
        g = m[0].weight.grad
        print(f"  {str(dt):22s} backward OK  loss={loss.item():.5f}  grad_finite={bool(torch.isfinite(g).all())}  grad_norm={g.float().norm().item():.4f}")
    except Exception as e:
        print(f"  {str(dt):22s} backward FAILED: {type(e).__name__}: {e}")
# AdamW step, the actual optimiser in the FastMTP recipe
try:
    m = nn.Linear(4096,4096).to(dev, torch.bfloat16)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-5, betas=(0.9,0.95))
    x = torch.randn(64,4096, device=dev, dtype=torch.bfloat16)
    m(x).float().pow(2).mean().backward(); opt.step(); opt.zero_grad()
    print("  AdamW(0.9,0.95) lr=5e-5 step OK")
except Exception as e:
    print(f"  AdamW FAILED: {type(e).__name__}: {e}")
print("free/total GiB:", [round(v/2**30,1) for v in torch.cuda.mem_get_info()])
