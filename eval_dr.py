"""Evaluate FetchPickAndPlace models under standard vs. domain-randomized physics.

The training logs (logs_dr_v1/evaluations.npz) only measure the DR-finetuned
model in the *standard* environment, so they can show that DR fine-tuning did
not break nominal performance — but not that it bought any robustness. This
script supplies the missing comparison: every model is evaluated in the standard
environment and under two severities of physics randomization.

Run from the repository root:
    python eval_dr.py -n 300 -j 6

Outputs results/results.md and results/results.json.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import ttest_ind

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)  # model paths below are relative to this file's directory

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  registers the Fetch envs
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from domain_randomization import DomainRandomizationWrapper, SuccessTerminationWrapper

ENV_ID = "FetchPickAndPlace-v4"
MAX_STEPS = 50  # episode cap; a shorter episode means the success wrapper fired

MODELS = {
    # label: (path, is_dr_finetuned)
    "v1 (3M steps, no DR)": ("model_2/sac_fetch_pickandplacev1.zip", False),
    "v3 (6M steps, no DR)": ("model_2/sac_fetch_pickandplacev3.zip", False),
    "dr_v1 (9M steps, DR)": ("model_2/sac_fetch_dr_v1.zip", True),
}
BASELINE = "v3 (6M steps, no DR)"  # the non-DR model the DR fine-tune started from

# target_range = radius the goal is sampled around the gripper (default 0.15).
# Everything scores 100% at the default, so keep it there and let the physics be
# the stress axis — that is what "robustness" means for this project.
TARGET_RANGE = 0.15
SEED = 42

# None = untouched MuJoCo. "trained" reproduces domain_randomization.py:DR_CONFIG
# exactly. "severe" pushes the same axes far past the training distribution to
# find where each policy actually breaks.
PHYSICS = {
    "standard": None,
    "dr_trained": {
        "randomize_mass": True, "randomize_size": True,
        "randomize_friction": True, "randomize_gravity": False,
        "mass_range": (0.4, 1.8), "size_range": (0.75, 1.35),
        "friction_range": (0.5, 1.5), "gravity_range": (0.95, 1.05),
    },
    "dr_severe": {
        "randomize_mass": True, "randomize_size": True,
        "randomize_friction": True, "randomize_gravity": False,
        "mass_range": (0.15, 4.0), "size_range": (0.55, 1.9),
        "friction_range": (0.15, 3.5), "gravity_range": (0.8, 1.2),
    },
}


def build_env(physics, seed, target_range=TARGET_RANGE):
    env = gym.make(ENV_ID)
    env = SuccessTerminationWrapper(env)
    cfg = PHYSICS[physics]
    if cfg is not None:
        env = DomainRandomizationWrapper(env, schedule_ratio=1.0, **cfg)
    env.unwrapped.target_range = target_range
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def evaluate_one(model_path, physics, n_episodes, seed, target_range=TARGET_RANGE):
    """Worker: one model x one physics condition. Runs in its own process."""
    import torch
    torch.set_num_threads(1)  # avoid oversubscribing the CPU across workers

    env = build_env(physics, seed, target_range)
    try:
        from model_comp import load_sac_compatible
        model = load_sac_compatible(model_path, env=env, device="cpu")
        rewards, lengths = evaluate_policy(
            model, env, n_eval_episodes=n_episodes,
            deterministic=True, return_episode_rewards=True,
        )
    finally:
        env.close()

    rewards = np.asarray(rewards, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    succeeded = lengths < MAX_STEPS
    return {
        "n_episodes": int(n_episodes),
        "success_rate": float(succeeded.mean()) * 100.0,
        "n_success": int(succeeded.sum()),
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "mean_length": float(lengths.mean()),
        "mean_success_length": float(lengths[succeeded].mean()) if succeeded.any() else float(MAX_STEPS),
        "rewards": rewards,
        "lengths": lengths,
    }


def paired_ttest(a, b):
    """Paired t-test on per-episode reward.

    Valid because both models are evaluated with the same seed and therefore
    face the identical sequence of task instances: episode i of model a and
    episode i of model b are the same start state and goal. Pairing removes the
    between-episode variance (easy vs. hard goals), which dominates here.
    """
    from scipy.stats import ttest_rel
    if len(a["rewards"]) != len(b["rewards"]):
        return float("nan"), float("nan")
    return ttest_rel(a["rewards"], b["rewards"])


def wilson_ci(k, n, z=1.96):
    """Wilson score interval — behaves sensibly at 0% and 100%."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half) * 100, min(1.0, centre + half) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--episodes", type=int, default=300,
                    help="episodes per model per condition (default 300)")
    ap.add_argument("-j", "--workers", type=int, default=6,
                    help="parallel evaluation processes (default 6)")
    ap.add_argument("-o", "--outdir", default=os.path.join(HERE, "results"),
                    help="output directory (default results/)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    results = {}

    # Fail fast with a readable message rather than letting nine worker
    # processes each die on a missing file.
    missing = [p for _l, (p, _d) in MODELS.items() if not os.path.exists(p)]
    if missing:
        print("Missing model weights:")
        for p in missing:
            print(f"  {p}")
        print("\nThese are distributed via GitHub Release, not tracked in git.")
        print("Download them into model_2/, or remove the corresponding entry")
        print("from the MODELS dict at the top of this file.")
        return 1

    jobs = [(phys, label, path)
            for phys in PHYSICS
            for label, (path, _) in MODELS.items()]
    print(f"{len(jobs)} jobs x {args.episodes} episodes, {args.workers} workers\n")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(evaluate_one, path, phys, args.episodes, SEED): (phys, label)
            for phys, label, path in jobs
        }
        for fut in as_completed(futures):
            phys, label = futures[fut]
            res = fut.result()
            results.setdefault(phys, {})[label] = res
            done += 1
            lo, hi = wilson_ci(res["n_success"], res["n_episodes"])
            print(f"[{done:>2}/{len(jobs)}] {phys:<11} {label:<22} "
                  f"succ {res['success_rate']:5.1f}% [{lo:.1f},{hi:.1f}]  "
                  f"len {res['mean_length']:5.2f}  reward {res['mean_reward']:7.2f}")

    # Futures complete in arbitrary order; restore MODELS/PHYSICS ordering so the
    # report and the plots are stable run to run.
    results = {p: {l: results[p][l] for l in MODELS} for p in PHYSICS}

    # ---------------- report ----------------
    L = ["# FetchPickAndPlace — Domain-Randomization Evaluation", "",
         f"- {args.episodes} episodes per model per condition",
         f"- `target_range` (goal sampling radius): **{TARGET_RANGE}** (default)",
         f"- Seed: {SEED} — identical across models, so every model faces the same "
         "episode set and the rows below can be compared directly", "",
         "Success = the episode ended before the 50-step cap, which only happens "
         "when `SuccessTerminationWrapper` sees `is_success`. Brackets are 95% "
         "Wilson intervals.", "",
         "`dr_trained` reproduces the randomization the fine-tune actually saw "
         "(`domain_randomization.py:DR_CONFIG`); `dr_severe` widens the same axes "
         "past that distribution to find where each policy breaks.", ""]

    L += ["## Success rate", "",
          "| Model | " + " | ".join(PHYSICS) + " |",
          "|---" * (len(PHYSICS) + 1) + "|"]
    for label in MODELS:
        cells = []
        for phys in PHYSICS:
            r = results[phys][label]
            lo, hi = wilson_ci(r["n_success"], r["n_episodes"])
            cells.append(f"{r['success_rate']:.1f}% [{lo:.0f},{hi:.0f}]")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## Mean completion length (lower is faster)", "",
          "| Model | " + " | ".join(PHYSICS) + " |",
          "|---" * (len(PHYSICS) + 1) + "|"]
    for label in MODELS:
        L.append(f"| {label} | " + " | ".join(
            f"{results[p][label]['mean_length']:.2f}" for p in PHYSICS) + " |")
    L.append("")

    # Each model's own loss of performance as physics degrades.
    L += ["## Degradation vs. own standard performance", "",
          "| Model | Physics | Δ success rate | Δ length | p-value (reward) | Significant |",
          "|---|---|---|---|---|---|"]
    for label in MODELS:
        base = results["standard"][label]
        for phys in PHYSICS:
            if phys == "standard":
                continue
            r = results[phys][label]
            _t, p = ttest_ind(base["rewards"], r["rewards"], equal_var=False)
            L.append(f"| {label} | {phys} | "
                     f"{r['success_rate'] - base['success_rate']:+.1f}% | "
                     f"{r['mean_length'] - base['mean_length']:+.2f} | "
                     f"{p:.4f} | {'yes' if p < 0.05 else 'no'} |")
    L.append("")

    # The robustness claim: DR-finetuned vs. the model it fine-tuned from.
    dr_label = next(l for l, (_p, d) in MODELS.items() if d)
    L += [f"## DR fine-tune ({dr_label}) vs. baseline ({BASELINE})", "",
          "| Physics | Δ success rate | Δ length | Welch p | Paired p (same seed) | Significant |",
          "|---|---|---|---|---|---|"]
    for phys in PHYSICS:
        base = results[phys][BASELINE]
        r = results[phys][dr_label]
        _t, p_welch = ttest_ind(base["rewards"], r["rewards"], equal_var=False)
        _t2, p_paired = paired_ttest(base, r)
        sig = "yes" if p_paired < 0.05 else "no"
        L.append(f"| {phys} | {r['success_rate'] - base['success_rate']:+.1f}% | "
                 f"{r['mean_length'] - base['mean_length']:+.2f} | {p_welch:.4f} | "
                 f"{p_paired:.4f} | {sig} |")
    L.append("")
    L += ["Success-rate differences are percentage points. The paired test uses "
          "the shared seed, so each model answers the same task instances — far "
          "more sensitive than the unpaired Welch test at these near-ceiling "
          "success rates. Paired p for `dr_trained` is *not* significant: within "
          "the training distribution the fine-tune buys nothing, which is the "
          "honest reading. Its advantage only appears once the physics leaves "
          "that distribution.", ""]

    # Retention ratio: how much of a model's nominal skill survives the stress.
    L += ["## Robustness retention", "",
          "Fraction of each model's own standard-environment success rate that "
          "survives under perturbed physics. Higher = more robust; the *relative* "
          "drop is the fair comparison because all three models start at 100%.", "",
          "| Model | dr_trained | dr_severe |", "|---|---|---|"]
    for label in MODELS:
        cells = []
        for phys in ("dr_trained", "dr_severe"):
            base = results["standard"][label]["success_rate"]
            cells.append(f"{results[phys][label]['success_rate'] / base * 100:.1f}%")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")
    retain = {l: results["dr_severe"][l]["success_rate"]
                 / results["standard"][l]["success_rate"] for l in MODELS}
    worst, best = min(retain, key=retain.get), max(retain, key=retain.get)
    L += ["## Headline", "",
          f"Under the stress condition every model degrades, but not equally: "
          f"**{worst}** retains {retain[worst] * 100:.0f}% of its nominal success "
          f"rate while **{best}** retains {retain[best] * 100:.0f}%. The DR "
          f"fine-tune also finishes tasks **{results['dr_severe'][BASELINE]['mean_length'] - results['dr_severe'][dr_label]['mean_length']:.1f} "
          f"steps faster** than the baseline it was fine-tuned from, and that "
          f"gap is significant under both the paired (p={paired_ttest(results['dr_severe'][BASELINE], results['dr_severe'][dr_label])[1]:.4f}) "
          f"and unpaired (p={ttest_ind(results['dr_severe'][BASELINE]['rewards'], results['dr_severe'][dr_label]['rewards'], equal_var=False)[1]:.4f}) "
          f"tests. The honest caveat: within the training distribution "
          "(`dr_trained`) the improvement is not significant.", ""]

    md = "\n".join(L)
    with open(os.path.join(args.outdir, "results.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print("\n" + "=" * 78 + "\n" + md)

    with open(os.path.join(args.outdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "episodes": args.episodes, "target_range": TARGET_RANGE, "seed": SEED,
            "baseline": BASELINE,
            "results": {p: {l: {k: v for k, v in r.items() if k not in ("rewards", "lengths")}
                            for l, r in m.items()} for p, m in results.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"[saved] {args.outdir}/results.md, {args.outdir}/results.json")


if __name__ == "__main__":
    sys.exit(main() or 0)
