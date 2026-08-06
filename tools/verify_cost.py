#!/usr/bin/env python3
"""Speculative-verify cost model: E_frac(k), c_v(k), and the S(k) break-even table.

Reads the byte split from tools/inventory.py's accounting (recomputed here from the same
headers, so the two cannot drift). Prints the table embedded in ROOFLINE.md section 5.
"""
import json, glob, os, collections, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = {"F32":4,"BF16":2,"F16":2,"F8_E4M3":1,"F8_E8M0":1,"I8":1,"I64":8,"I32":4}
T = {}
for p in glob.glob(os.path.join(REPO,"docs","hdrs","*.json")):
    for n,m in json.load(open(p)).items():
        if n=="__metadata__": continue
        c=1
        for d in m["shape"]: c*=d
        T[n]=c*DB[m["dtype"]]
cfg = json.load(open(os.path.join(REPO,"docs","config.json")))
L, K, E = cfg["num_hidden_layers"], cfg["num_experts_per_tok"], cfg["n_routed_experts"]

expert_1L = sum(v for k,v in T.items() if k.startswith("layers.0.ffn.experts.0."))  # 1 expert, 1 layer
B_expert = 0; B_fixed = 0
for i in range(L):
    p=f"layers.{i}."
    e1 = sum(v for k,v in T.items() if k.startswith(p+"ffn.experts.0."))
    B_expert += K*e1
    B_fixed  += sum(v for k,v in T.items() if k.startswith(p) and ".ffn.experts." not in k)
B_fixed += T["head.weight"] + sum(v for k,v in T.items() if k.startswith("hc_head"))
B_tok = B_fixed + B_expert
per_expert_all_layers = expert_1L * L

print(f"B_tok            {B_tok/1e6:9.2f} MB")
print(f"  k-invariant    {B_fixed/1e6:9.2f} MB  ({100*B_fixed/B_tok:.1f}%)  <- read ONCE per verify pass")
print(f"  k-scaling      {B_expert/1e6:9.2f} MB  ({100*B_expert/B_tok:.1f}%)  <- routed experts, grows with |union|")
print(f"  1 expert, all {L} layers = {per_expert_all_layers/1e6:.2f} MB\n")

print("k  E[|union|]  E_frac   B_verify     c_v    per-tok   S needs a>=")
print("-"*66)
rows=[]
for k in range(1,9):
    u = E*(1-(1-K/E)**k)              # independent-routing upper bound on the union
    frac = u/(K*k)
    Bv = B_fixed + L*u*expert_1L
    cv = Bv/B_tok
    rows.append((k,u,frac,Bv,cv))
    print(f"{k}  {u:9.2f}  {frac:6.3f}  {Bv/1e6:8.1f}MB  {cv:6.3f}  {cv/k:7.3f}   {cv:.2f}")
print("\nS(k) = a(k) / c_v(k);  a(k) = expected tokens emitted per verify pass (<= k+1)")
print("\nS(k) at candidate acceptance levels (a = alpha*k + 1, geometric-ish approximation):")
print("alpha  " + "".join(f"  k={k}  " for k,_,_,_,_ in rows))
for alpha in (0.4,0.5,0.6,0.7,0.8):
    line=f"{alpha:4.1f}  "
    for k,u,f,Bv,cv in rows:
        a = sum(alpha**j for j in range(1,k+1)) + 1   # expected accepted prefix + bonus token
        line += f" {a/cv:6.2f}"
    print(line)
