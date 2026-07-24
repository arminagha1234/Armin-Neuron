#!/usr/bin/env python3
"""Cold-TTFT concurrency sweep for Gemma4-31B (text-only) on the PUBLIC vLLM-Neuron image.

Unique random prompt per request (defeats prefix caching -> honest cold prefill).
Talks to the OpenAI-compatible /v1/completions endpoint. Stdlib only.

Usage (inside the public container, against a running serve):
  python3 bench_sweep.py --size 4k  --levels 1,2,4,8,16,32 --gen 50
  python3 bench_sweep.py --size 32k --levels 1,2,4         --gen 50
Writes <size>.json in the cwd.
"""
import argparse, json, statistics, threading, time, urllib.request, random

_POOL = []
def _build_pool(seed=1234, n=4000):
    rng = random.Random(seed); cons="bcdfghjklmnpqrstvwxyz"; vow="aeiou"
    for _ in range(n):
        ln = rng.choice((3,4,5,6,7))
        _POOL.append("".join(cons[rng.randrange(20)] if i%2==0 else vow[rng.randrange(5)] for i in range(ln)))
_build_pool()
_lock = threading.Lock(); _ctr = [0]
def _uid():
    with _lock:
        _ctr[0]+=1; return _ctr[0]
def make_prompt(nwords):
    u=_uid(); rng=random.Random((u<<20)^random.randrange(1<<30))
    return " ".join([f"{u}x{rng.randrange(1<<16):04x}"]+[rng.choice(_POOL) for _ in range(max(1,nwords-1))])

def one(base_url, model, prompt, gen, timeout, res, idx):
    body=json.dumps({"model":model,"prompt":prompt,"max_tokens":gen,"temperature":0,
                     "stream":True,"ignore_eos":True,"stream_options":{"include_usage":True}}).encode()
    t0=time.perf_counter(); ttft=None; n=0; ptok=None
    try:
        req=urllib.request.Request(base_url.rstrip("/")+"/v1/completions",data=body,
                                   headers={"Content-Type":"application/json"})
        for raw in urllib.request.urlopen(req,timeout=timeout):
            line=raw.decode("utf-8","ignore").strip()
            if not line.startswith("data:"): continue
            p=line[5:].strip()
            if p=="[DONE]": break
            o=json.loads(p); ch=(o.get("choices") or [{}])[0]
            if ch.get("text"):
                if ttft is None: ttft=time.perf_counter()-t0
                n+=1
            if o.get("usage"): ptok=o["usage"].get("prompt_tokens")
        res[idx]={"ttft":ttft,"n":n,"end":time.perf_counter()-t0,"ptok":ptok}
    except Exception as e:
        res[idx]={"error":repr(e)[:160]}

def run_level(base_url, model, nwords, conc, gen, timeout):
    prompts=[make_prompt(nwords) for _ in range(conc)]; res=[None]*conc
    ths=[threading.Thread(target=one,args=(base_url,model,prompts[i],gen,timeout,res,i)) for i in range(conc)]
    for t in ths: t.start()
    for t in ths: t.join()
    ok=[r for r in res if r and "error" not in r and r.get("ttft") is not None]
    errs=[r for r in res if r and "error" in r]
    ttfts=sorted(r["ttft"] for r in ok)
    tpots=[(r["end"]-r["ttft"])/(r["n"]-1) for r in ok if r["n"]>1]
    ptok=round(statistics.mean([r["ptok"] for r in ok if r.get("ptok")])) if ok else None
    return {"conc":conc,"ok":len(ok),"err":len(errs),
            "in_tok":ptok,
            "ttft_mean_s":round(statistics.mean(ttfts),3) if ttfts else None,
            "ttft_p99_s":round(ttfts[min(len(ttfts)-1,int(len(ttfts)*0.99))],3) if ttfts else None,
            "tpot_mean_ms":round(1000*statistics.mean(tpots),1) if tpots else None,
            "first_err":(errs[0]["error"] if errs else None)}

# target input tokens per size (leave headroom under max_model_len); tokens/word ~2.2
SIZE_TOK={"4k":3800,"8k":7800,"16k":15600,"32k":31500,"64k":62500}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-url",default="http://localhost:8000")
    ap.add_argument("--model",default="gemma4")
    ap.add_argument("--size",required=True,choices=list(SIZE_TOK))
    ap.add_argument("--levels",default="1,2,4,8,16,32")
    ap.add_argument("--gen",type=int,default=50)
    ap.add_argument("--timeout",type=int,default=1800)
    ap.add_argument("--out",default=None)
    args=ap.parse_args()
    nwords=int(SIZE_TOK[args.size]/2.2)
    levels=[int(x) for x in args.levels.split(",")]
    run_level(args.base_url,args.model,nwords,1,min(8,args.gen),args.timeout)  # warm graph (short)
    print(f"### {args.size} input, {args.gen} output (PUBLIC image, text-only, cold random, no APC) ###",flush=True)
    print(f"{'conc':>4} {'ok':>3} {'err':>3} {'in_tok':>7} {'TTFT_s':>8} {'TTFT_p99':>9} {'TPOT_ms':>8}",flush=True)
    rows=[]
    for c in levels:
        r=run_level(args.base_url,args.model,nwords,c,args.gen,args.timeout); rows.append(r)
        print(f"{r['conc']:>4} {r['ok']:>3} {r['err']:>3} {str(r['in_tok']):>7} "
              f"{str(r['ttft_mean_s']):>8} {str(r['ttft_p99_s']):>9} {str(r['tpot_mean_ms']):>8}"
              +(f"  ERR:{r['first_err']}" if r['err'] else ""),flush=True)
    out=args.out or f"{args.size}.json"
    json.dump({"size":args.size,"gen":args.gen,"image":"public.ecr.aws SDK2.31","model":"gemma-4-31b-it text-only",
               "TP":32,"APC":"off","dtype":"bf16","rows":rows}, open(out,"w"), indent=2)
    print(f"saved -> {out}",flush=True)

if __name__=="__main__":
    main()
