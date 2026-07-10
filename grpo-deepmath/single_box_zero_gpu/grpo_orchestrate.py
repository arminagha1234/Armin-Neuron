"""
Zero-GPU GRPO on a single Trainium box — DEVICE-FREE coordinator.

Key design fix: this coordinator NEVER imports torch_xla / touches the Neuron
device. Each phase runs as its own subprocess that grabs the device, does its
work, and EXITS (guaranteeing release). Only one process holds the device at a
time -> no contention, no leaked-server races.

  round r:
    free_device()                       # kill any holder, wait until /dev/neuron0 free
    start vLLM-Neuron server (subprocess); generate G completions/prompt via HTTP; score
    kill server; free_device()
    run grpo_train_phase.py (subprocess): GRPO update -> save ckpt_r -> exit
    cur_model = ckpt_r                  # periodic weight sync
"""
import argparse, json, os, re, signal, subprocess, sys, time
import requests

VENV = "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin"
PY = f"{VENV}/python"
PORT = 8001
BASE = f"http://localhost:{PORT}"
HERE = os.path.dirname(os.path.abspath(__file__))

PROMPTS = [
    ("What is 6 times 7? Answer with just the number.", "42"),
    ("What is 12 plus 15? Answer with just the number.", "27"),
    ("What is 9 times 9? Answer with just the number.", "81"),
    ("What is 100 minus 37? Answer with just the number.", "63"),
]


def free_device(timeout=90):
    subprocess.run("pkill -9 -f 'vllm serve'; pkill -9 -f EngineCore", shell=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = subprocess.run("fuser /dev/neuron0 2>/dev/null", shell=True,
                           capture_output=True, text=True)
        holders = r.stdout.split()
        if not holders:
            time.sleep(2); return True
        for pid in holders:
            subprocess.run(f"kill -9 {pid}", shell=True)
        time.sleep(3)
    return False


def start_server(model_path):
    env = dict(os.environ)
    env["NEURON_RT_VISIBLE_CORES"] = "0-1"
    env["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
    env["NEURON_COMPILE_CACHE_URL"] = "/var/tmp/neuron-compile-cache"
    env["PATH"] = VENV + ":" + env.get("PATH", "")
    logf = open("/home/ubuntu/grpo_server.log", "w")
    p = subprocess.Popen(
        [f"{VENV}/vllm", "serve", model_path, "--tensor-parallel-size", "1",
         "--max-num-seqs", "8", "--max-model-len", "1024", "--block-size", "128",
         "--no-enable-prefix-caching", "--served-model-name", "policy", "--port", str(PORT)],
        env=env, stdout=logf, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    for _ in range(240):  # up to 20 min for cold compile
        if p.poll() is not None:
            raise RuntimeError("server process exited during startup")
        try:
            if requests.get(f"{BASE}/health", timeout=3).status_code == 200:
                return p
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("server did not become healthy in time")


def kill_server(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        pass


def reward(text, gold):
    return 1.0 if gold in re.findall(r"-?\d+", text) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--train-steps", type=int, default=3)
    ap.add_argument("--ckpt-root", default="/home/ubuntu/grpo_run")
    args = ap.parse_args()

    os.makedirs(args.ckpt_root, exist_ok=True)
    prompts = PROMPTS[:args.prompts]
    cur = args.model

    for rnd in range(args.rounds):
        print(f"\n===== ROUND {rnd} (policy={cur}) =====", flush=True)
        free_device()
        srv = None
        try:
            srv = start_server(cur)
            print(f"[round {rnd}] server up, generating...", flush=True)
            rollouts, rewards = [], []
            for prompt, gold in prompts:
                r = requests.post(f"{BASE}/v1/completions", json={
                    "model": "policy", "prompt": prompt, "n": args.group,
                    "max_tokens": 32, "temperature": 1.0}, timeout=120)
                comps = [c["text"] for c in r.json()["choices"]]
                rs = [reward(c, gold) for c in comps]
                rollouts.append({"prompt": prompt, "comps": comps, "rewards": rs})
                rewards.extend(rs)
            mean_r = sum(rewards) / len(rewards)
            print(f"[round {rnd}] MEAN_REWARD={mean_r:.3f} (n={len(rewards)})", flush=True)
        finally:
            if srv is not None:
                kill_server(srv)
            free_device()

        # ---- training phase: separate subprocess, grabs+releases the device ----
        roll_path = os.path.join(args.ckpt_root, f"rollouts_{rnd}.json")
        json.dump(rollouts, open(roll_path, "w"))
        out = os.path.join(args.ckpt_root, f"round{rnd}")
        cmd = [PY, os.path.join(HERE, "grpo_train_phase.py"),
               "--model", cur, "--rollouts", roll_path, "--out", out,
               "--train-steps", str(args.train_steps)]
        env = dict(os.environ)
        env["NEURON_RT_VISIBLE_CORES"] = "0-1"; env["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
        env["NEURON_COMPILE_CACHE_URL"] = "/var/tmp/neuron-compile-cache"
        env["PATH"] = VENV + ":" + env.get("PATH", "")   # so torch_xla finds libneuronpjrt-path
        print(f"[round {rnd}] training...", flush=True)
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            print(f"[round {rnd}] training subprocess failed rc={rc}", flush=True); sys.exit(1)
        free_device()
        cur = out

    print("\n[DONE] periodic-sync GRPO completed on Trainium (zero GPU).", flush=True)


if __name__ == "__main__":
    main()
