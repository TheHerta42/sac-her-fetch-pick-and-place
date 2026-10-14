import os
import io
import zipfile
import numpy as np
import gymnasium as gym
import gymnasium_robotics
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import SubprocVecEnv
from scipy.stats import ttest_ind
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入域随机化包装器
from domain_randomization import DomainRandomizationWrapper


def load_sac_compatible(path, env, device="auto"):
    """
    兼容加载 SAC 模型。
    当 ent_coef 从 auto 改为固定值后，zip 中会残留 ent_coef_optimizer
    和 log_ent_coef，导致 SB3 原生 load 报错。此函数自动检测并修复。
    """
    if not path.endswith(".zip") and os.path.exists(path + ".zip"):
        path = path + ".zip"

    data, params, pytorch_variables = load_from_zip_file(
        path, device=device, load_data=True
    )

    ent_coef = data.get("ent_coef", "auto")
    need_fix = (
        not isinstance(ent_coef, str)
        and ("ent_coef_optimizer" in params or "log_ent_coef" in pytorch_variables)
    )

    if need_fix:
        print(f"[兼容修复] ent_coef={ent_coef} 为固定值，清理残留的 auto 模式参数...")
        temp_path = path + ".fixed.zip"
        with zipfile.ZipFile(path, "r") as zin:
            with zipfile.ZipFile(temp_path, "w") as zout:
                for item in zin.namelist():
                    if item == "ent_coef_optimizer.pth":
                        continue
                    if item == "pytorch_variables.pth":
                        pv = torch.load(
                            zin.open(item), map_location="cpu", weights_only=False
                        )
                        pv.pop("log_ent_coef", None)
                        buf = io.BytesIO()
                        torch.save(pv, buf)
                        zout.writestr(item, buf.getvalue())
                    else:
                        zout.writestr(item, zin.read(item))
        try:
            model = SAC.load(temp_path, env=env, device=device)
        finally:
            os.remove(temp_path)
        return model
    else:
        return SAC.load(path, env=env, device=device)


class SuccessTerminationWrapper(gym.Wrapper):
    """当任务成功时立即终止 episode，实现提前结算"""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("is_success", False):
            terminated = True
        return obs, reward, terminated, truncated, info


def make_eval_env(rank, seed=0, use_dr=False, dr_config=None):
    """
    评估环境工厂函数，用于多进程并行。

    参数:
        use_dr: 是否启用域随机化评估
        dr_config: 域随机化配置字典，包含 mass_range, size_range, friction_range,
                   gravity_range, schedule_ratio 等
    """
    def _init():
        import gymnasium_robotics  # Windows spawn 模式下子进程需要单独导入
        env = gym.make("FetchPickAndPlace-v4")
        env = SuccessTerminationWrapper(env)

        if use_dr:
            cfg = dr_config or {}
            env = DomainRandomizationWrapper(
                env,
                randomize_mass=cfg.get("randomize_mass", True),
                randomize_size=cfg.get("randomize_size", True),
                randomize_friction=cfg.get("randomize_friction", True),
                randomize_gravity=cfg.get("randomize_gravity", False),
                mass_range=cfg.get("mass_range", (0.5, 1.5)),
                size_range=cfg.get("size_range", (0.8, 1.3)),
                friction_range=cfg.get("friction_range", (0.5, 1.5)),
                gravity_range=cfg.get("gravity_range", (0.95, 1.05)),
                schedule_ratio=cfg.get("schedule_ratio", 1.0),
            )

        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def evaluate_model_worker(model_path, model_name, n_eval_episodes, num_envs, seed,
                           use_dr=False, dr_config=None):
    """
    在独立线程中评估单个模型，返回详细性能指标。
    每个模型拥有自己独立的 SubprocVecEnv 环境组，实现模型间并行评估。

    参数:
        use_dr: 是否使用随机化环境评估
        dr_config: 域随机化配置字典
    """
    env_label = "随机化环境" if use_dr else "标准环境"
    print(f"\n[启动] 正在评估 {model_name} ({env_label}) ...")

    eval_env = SubprocVecEnv([
        make_eval_env(i, seed, use_dr=use_dr, dr_config=dr_config)
        for i in range(num_envs)
    ])

    try:
        if not os.path.exists(f"{model_path}.zip"):
            print(f"[!] 模型文件 {model_path}.zip 不存在，跳过")
            return None

        try:
            model = load_sac_compatible(model_path, env=eval_env)
        except Exception as e:
            print(f"[X] 加载模型 {model_name} 失败: {e}")
            return None

        rewards, lengths = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            return_episode_rewards=True
        )

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        mean_length = float(np.mean(lengths))
        std_length = float(np.std(lengths))

        success_count = sum(1 for l in lengths if l < 50)
        success_rate = success_count / n_eval_episodes * 100

        successful_lengths = [l for l in lengths if l < 50]
        mean_success_length = float(np.mean(successful_lengths)) if successful_lengths else 50.0

        print(f"[完成] {model_name} 评估结束 | 平均奖励: {mean_reward:.2f} | 成功率: {success_rate:.3f}%")

        return {
            "name": model_name,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": mean_length,
            "std_length": std_length,
            "success_rate": success_rate,
            "mean_success_length": mean_success_length,
            "rewards": np.array(rewards),
            "lengths": np.array(lengths)
        }
    finally:
        eval_env.close()


# ==================== 配置区 ====================
ENV_ID = "FetchPickAndPlace-v4"
MODEL_PATHS = {
    "dr_v1": "sac_fetch_dr_v1",
    "dr_best": "best_model_dr",
    "v3": "sac_fetch_pickandplacev3",
    "v1": "sac_fetch_pickandplacev1",
}
N_EVAL_EPISODES = 10000     # 总评估回合数，建议设为 NUM_EVAL_ENVS 的整数倍
NUM_EVAL_ENVS = 4          # 每个模型的评估并行环境数
MAX_WORKERS = 3            # 同时评估的模型数量（并行线程数）
SEED = 42

# ==================== 域随机化评估配置 ====================
# 设置 USE_DOMAIN_RANDOMIZATION = True 即可在随机化环境下评估模型
USE_DOMAIN_RANDOMIZATION = False

# 域随机化参数配置（仅在 USE_DOMAIN_RANDOMIZATION = True 时生效）
# 可复用 domain_randomization.py 中训练时使用的配置
DR_EVAL_CONFIG = {
    "randomize_mass": True,
    "randomize_size": True,
    "randomize_friction": True,
    "randomize_gravity": False,
    "mass_range": (0.4, 1.8),       # 质量倍数范围
    "size_range": (0.75, 1.35),     # 尺寸倍数范围
    "friction_range": (0.5, 1.5),   # 摩擦倍数范围
    "gravity_range": (0.95, 1.05),  # 重力倍数范围
    "schedule_ratio": 1.0,          # 1.0 = 使用完整随机化范围；0.0 = 无随机化
}


if __name__ == "__main__":
    print("=" * 60)
    print("FetchPickAndPlace-v4 模型性能对比评估 (多进程并行)")
    print(f"评估回合数: {N_EVAL_EPISODES}  (单模型并行环境数: {NUM_EVAL_ENVS})")
    print(f"同时评估模型数: {MAX_WORKERS}")
    print(f"环境配置: terminate_success=True, distance_threshold=0.05m")
    if USE_DOMAIN_RANDOMIZATION:
        print(f"[DR] 评估模式: 域随机化环境")
        print(f"[DR] 质量范围: {DR_EVAL_CONFIG['mass_range']}")
        print(f"[DR] 尺寸范围: {DR_EVAL_CONFIG['size_range']}")
        print(f"[DR] 摩擦范围: {DR_EVAL_CONFIG['friction_range']}")
        print(f"[DR] 随机化强度: {DR_EVAL_CONFIG['schedule_ratio']}")
    else:
        print(f"[DR] 评估模式: 标准环境 (无随机化)")
    print("=" * 60)

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_name = {}
        for idx, (name, path) in enumerate(MODEL_PATHS.items()):
            # 为不同模型分配不同的种子偏移，避免环境轨迹完全一样
            seed_offset = SEED + idx * 1000
            future = executor.submit(
                evaluate_model_worker,
                path,
                name,
                N_EVAL_EPISODES,
                NUM_EVAL_ENVS,
                seed_offset,
                USE_DOMAIN_RANDOMIZATION,
                DR_EVAL_CONFIG
            )
            future_to_name[future] = name

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"[X] 评估 {name} 时发生异常: {e}")

    if not results:
        print("\n[X] 没有找到任何可评估的模型")
        exit(1)

    # ==================== 生成对比报告 ====================
    print("\n" + "=" * 60)
    print("[*] 性能对比结果")
    print("=" * 60)

    print(f"{'模型名称':<15} | {'平均奖励':<10} | {'平均步数':<10} | {'成功率':<8} | {'成功平均步数':<12}")
    print("-" * 60)

    results_sorted = sorted(results, key=lambda x: x["mean_reward"], reverse=True)

    for res in results_sorted:
        print(
            f"{res['name']:<15} | "
            f"{res['mean_reward']:>6.2f} ± {res['std_reward']:<4.2f} | "
            f"{res['mean_length']:>6.2f} ± {res['std_length']:<4.2f} | "
            f"{res['success_rate']:>6.1f}% | "
            f"{res['mean_success_length']:>10.2f}"
        )

    # ==================== 统计显著性检验 ====================
    print("\n" + "=" * 60)
    print("[*] 统计显著性分析 (与基准模型对比)")
    print("=" * 60)

    # 以 MODEL_PATHS 中第一个模型作为基准
    baseline_name = list(MODEL_PATHS.keys())[0]
    baseline_result = next((r for r in results if r["name"] == baseline_name), None)

    if baseline_result:
        for res in results:
            if res["name"] == baseline_name:
                continue

            t_stat, p_value = ttest_ind(
                baseline_result["rewards"],
                res["rewards"],
                equal_var=False
            )

            improvement = res["mean_reward"] - baseline_result["mean_reward"]
            improvement_steps = baseline_result["mean_success_length"] - res["mean_success_length"]

            print(f"\n{res['name']} vs {baseline_name}:")
            print(f"  奖励提升: {improvement:+.2f}")
            print(f"  完成步数减少: {improvement_steps:+.2f} 步")
            print(f"  成功率提升: {res['success_rate'] - baseline_result['success_rate']:+.3f}%")
            print(f"  统计显著性: p = {p_value:.4f}")

            if p_value < 0.05:
                print("  [OK] 差异具有统计显著性 (p < 0.05)")
            else:
                print("  [WARN] 差异不具有统计显著性，可能是随机波动")

    # ==================== 总结与建议 ====================
    print("\n" + "=" * 60)
    print("[*] 总结与建议")
    print("=" * 60)

    best_model = results_sorted[0]
    print(f"[BEST] 性能最好的模型: {best_model['name']}")
    print(f"   平均奖励: {best_model['mean_reward']:.2f}")
    print(f"   成功率: {best_model['success_rate']:.3f}%")
    print(f"   平均完成步数: {best_model['mean_success_length']:.2f}")

    if best_model["name"] == "best_v3":
        print("\n[OK] 最佳模型优于最终模型，这是正常现象")
        print("   因为最佳模型是训练过程中在验证集上表现最好的快照")
        print("   建议在后续使用和部署中使用这个最佳模型")
    elif best_model["name"] == "v3":
        print("\n[OK] 最终模型性能最好")
        print("   说明训练过程中没有出现过拟合，模型一直在进步")
    elif best_model["name"] == "v1":
        print("\n[WARN] 旧模型性能仍然最好")
        print("   可能的原因:")
        print("   1. 训练步数不足，模型还在收敛中")
        print("   2. 学习率设置过高，导致模型震荡")
        print("   3. 新的参数设置不适合你的模型")
        print("\n建议:")
        print("   - 继续训练50-100万步观察效果")
        print("   - 降低初始学习率到5e-5")
        print("   - 回滚到旧模型，尝试其他优化参数")

    print("\n[OK] 评估完成！")
