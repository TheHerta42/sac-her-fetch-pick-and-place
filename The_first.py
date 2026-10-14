import os
import warnings
from stable_baselines3.common import logger

# 修改全局日志格式，保留4位小数
logger.DEFAULT_FORMAT = "%(name)s: %(value).4f"
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.her import HerReplayBuffer

# ==================== 限制 PyTorch 线程数，避免和多进程环境争抢 CPU ====================
torch.set_num_threads(1)

# ==================== 新增：精细成功率显示回调 ====================
class DetailedSuccessRateCallback(BaseCallback):
    """
    打印更精细的rollout成功率（保留4位小数）
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.success_buffer = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # 收集每个episode的成功信息
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "is_success" in info:
                self.success_buffer.append(info["is_success"])
                self.episode_count += 1

        # 每100个episode打印一次精细成功率
        if self.episode_count >= 100:
            success_rate = np.mean(self.success_buffer)
            print(f"\n[OK] 最近100个episode成功率: {success_rate:.4f} ({int(success_rate*100):.2f}%)")
            print(f"   成功: {sum(self.success_buffer)}/100")
            self.success_buffer = []
            self.episode_count = 0

        return True


# ==================== 提前终止包装器 ====================
class SuccessTerminationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("is_success", False):
            terminated = True
        return obs, reward, terminated, truncated, info


def make_env(rank, seed=0):
    """创建单个环境的工厂函数，用于多进程并行"""
    def _init():
        # 子进程中必须导入，Windows spawn 模式下环境注册不会继承
        import gymnasium_robotics
        env = gym.make("FetchPickAndPlace-v4")
        env = SuccessTerminationWrapper(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ==================== 配置区 ====================
ENV_ID = "FetchPickAndPlace-v4"
OLD_MODEL_PATH = "sac_fetch_pickandplacev1"
NEW_MODEL_PATH = "sac_fetch_pickandplacev3"
TIMESTEPS = 6_000_000      # 训练300万步
EVAL_FREQ = 100_000        # 每10万步评估一次（减少中断）
N_EVAL_EPISODES = 20       # 20个回合评估（减少等待）
SAVE_FREQ = 100_000
SEED = 42
NUM_ENVS = 4               # 并行环境数，根据CPU核心数调整（4~8）

# 日志目录
LOG_DIR = "./logsv3/"
TB_DIR = "./sac_tensorboardv3/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)


# ==================== 主程序 ====================
if __name__ == "__main__":
    new = 1   # 0表示用旧模型继续训练，1表示从头开始训练

    # ==================== 随机种子 ====================
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ==================== 环境初始化 ====================
    # 训练环境：多进程并行，大幅提升数据收集速度
    print(f"Creating {NUM_ENVS} parallel training environments...")
    train_env = SubprocVecEnv([make_env(i, SEED) for i in range(NUM_ENVS)])

    # 评估环境：单环境即可
    eval_env = Monitor(SuccessTerminationWrapper(gym.make(ENV_ID)))
    eval_env.reset(seed=SEED + NUM_ENVS)

    # ==================== 模型 ====================
    if new:
        print("从头开始训练新模型...")
        model = SAC(
            "MultiInputPolicy",
            train_env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=2,
                goal_selection_strategy="future",
            ),
            learning_rate=LinearSchedule(start=3e-4, end=1e-5, end_fraction=0.8),
            buffer_size=2_000_000,
            batch_size=256,
            gamma=0.92,
            tau=0.005,
            device="cpu",
            ent_coef="auto_0.01",
            target_update_interval=1,
            learning_starts=5000,   # 多进程下先收集足够数据再开始训练
            verbose=1,
            tensorboard_log=TB_DIR,
            seed=SEED,
        )
    else:
        print(f"加载现有模型: {OLD_MODEL_PATH} 继续训练...")
        model = SAC.load(
            OLD_MODEL_PATH,
            env=train_env,
            tensorboard_log=TB_DIR,
            verbose=1,
            device="cuda"
        )

        # 应用优化参数 - 保守微调：目标缩短步数
        model.learning_rate = 1e-5
        model.replay_buffer.n_sampled_goal = 2
        model.replay_buffer.online_sampling = True
        model.batch_size = 256
        model.gamma = 0.92
        model.tau = 0.005
        model.ent_coef = "auto_0.01"
        model.target_update_interval = 1

        # 关键修复：设置为当前总步数 + 1000
        model.learning_starts = model.num_timesteps + 1000

        print(f"[OK] 模型加载成功！当前累计训练步数: {model.num_timesteps}")
        print(f"[INFO] 继续训练完成后累计步数: {model.num_timesteps + TIMESTEPS}")
        print(f"[WAIT] 将先收集 {model.learning_starts - model.num_timesteps} 步经验再开始训练")

    MODEL_PATH = NEW_MODEL_PATH

    # ==================== 训练前验证 ====================
    if not new:
        print("\n[CHECK] 正在验证旧模型性能...")
        old_mean_reward, old_std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=20, deterministic=True
        )
        print(f"旧模型验证结果: 平均奖励 = {old_mean_reward:.2f} +/- {old_std_reward:.2f}")
        print(f"对应平均完成步数: {int(-old_mean_reward + 1)} 步")

        if old_mean_reward < -10:
            print("\n[WARN] 警告：旧模型性能低于预期！")
            print("可能是模型加载失败或环境配置错误")
            exit(1)

    # ==================== Callbacks ====================
    detailed_success_callback = DetailedSuccessRateCallback()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=LOG_DIR,
        name_prefix="sac_fetch_checkpoint",
        verbose=1,
    )

    # ==================== 训练 ====================
    print("Using device:", model.device)
    print(f"\n[START] 开始训练 SAC+HER on {ENV_ID}")
    print(f"[TIME] 训练步数: {TIMESTEPS}")
    print(f"[SEED] 种子: {SEED}")
    print(f"[LOG] 日志目录: {TB_DIR}")
    print(f"[SAVE] 模型保存路径: {MODEL_PATH}.zip")
    print(f"[ENV] 并行环境数: {NUM_ENVS}")
    print(f"[VER] 当前 gymnasium-robotics 版本: 1.4.2")

    try:
        model.learn(
            total_timesteps=TIMESTEPS,
            callback=[detailed_success_callback, eval_callback, checkpoint_callback],
            reset_num_timesteps=new,
            progress_bar=True,
        )
    finally:
        model.save(MODEL_PATH)
        print(f"\n[OK] 最终模型已保存到: {MODEL_PATH}.zip")
        print(f"[INFO] 最终累计训练步数: {model.num_timesteps}")

    # ==================== 最终评估 ====================
    print("\n[EVAL] 开始最终评估...")

    if not new:
        print("评估旧模型性能...")
        old_model = SAC.load(OLD_MODEL_PATH, env=eval_env)
        old_mean_reward, old_std_reward = evaluate_policy(
            old_model, eval_env, n_eval_episodes=1000, deterministic=True
        )
        print(f"旧模型: 平均奖励 = {old_mean_reward:.2f} +/- {old_std_reward:.2f}")
        print(f"旧模型平均完成步数: {int(-old_mean_reward + 1)} 步")

    print("\n评估新模型性能...")
    new_mean_reward, new_std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=1000, deterministic=True
    )
    print(f"新模型: 平均奖励 = {new_mean_reward:.2f} +/- {new_std_reward:.2f}")
    print(f"新模型平均完成步数: {int(-new_mean_reward + 1)} 步")

    if not new:
        improvement = new_mean_reward - old_mean_reward
        print(f"\n[RESULT] 性能提升: {improvement:.2f}")
        if improvement > 0:
            print("[OK] 优化成功！模型性能得到了提升")
        else:
            print("[WARN] 优化效果不明显，可以尝试调整学习率或增加训练步数")

    train_env.close()
    eval_env.close()
