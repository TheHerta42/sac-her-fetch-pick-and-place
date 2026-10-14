import os
import warnings
from stable_baselines3.common import logger

# 修改全局日志格式，保留4位小数
logger.DEFAULT_FORMAT = "%(name)s: %(value).4f"
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import mujoco
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.her import HerReplayBuffer

# ==================== 限制 PyTorch 线程数，避免和多进程环境争抢 CPU ====================
torch.set_num_threads(1)


# ==================== 域随机化包装器 ====================
class DomainRandomizationWrapper(gym.Wrapper):
    """
    FetchPickAndPlace 域随机化包装器

    在每次 env.reset() 时随机化以下物理参数：
    - 物体质量 (object mass)
    - 物体尺寸 (object size)
    - 摩擦系数 (friction)
    - 重力 (gravity)

    支持渐进式随机化：通过 schedule_ratio (0~1) 控制当前随机化强度
    """
    def __init__(self, env,
                 randomize_mass=True,
                 randomize_size=True,
                 randomize_friction=True,
                 randomize_gravity=False,
                 mass_range=(0.5, 1.5),       # 质量倍数范围
                 size_range=(0.8, 1.3),       # 尺寸倍数范围
                 friction_range=(0.5, 1.5),   # 摩擦倍数范围
                 gravity_range=(0.95, 1.05),  # 重力倍数范围
                 schedule_ratio=1.0):         # 当前调度比例
        super().__init__(env)
        self.randomize_mass = randomize_mass
        self.randomize_size = randomize_size
        self.randomize_friction = randomize_friction
        self.randomize_gravity = randomize_gravity

        self.mass_range = mass_range
        self.size_range = size_range
        self.friction_range = friction_range
        self.gravity_range = gravity_range
        self.schedule_ratio = schedule_ratio

        # 获取底层 MuJoCo 模型
        self.mj_model = self.env.unwrapped.model
        self.mj_data = self.env.unwrapped.data

        # 保存原始参数
        self._original_mass = self.mj_model.body_mass.copy()
        self._original_size = self.mj_model.geom_size.copy()
        self._original_friction = self.mj_model.geom_friction.copy()
        self._original_gravity = self.mj_model.opt.gravity.copy()

        # 查找物体相关的 body 和 geom ID
        self._find_object_ids()

    def _find_object_ids(self):
        """查找物体相关的 body/geom ID"""
        self.object_body_ids = []
        self.object_geom_ids = []

        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and "object" in name.lower():
                self.object_body_ids.append(i)

        for i in range(self.mj_model.ngeom):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and "object" in name.lower():
                self.object_geom_ids.append(i)

        if len(self.object_body_ids) == 0:
            print("[WARN] 未找到 object body!")
        else:
            print(f"[INFO] 物体 body IDs: {self.object_body_ids}, geom IDs: {self.object_geom_ids}")

    def _get_scheduled_range(self, range_tuple):
        """根据 schedule_ratio 缩小随机化范围，以 1.0 为基准"""
        low, high = range_tuple
        # schedule_ratio=0 -> (1.0, 1.0)  完全不随机化
        # schedule_ratio=1 -> (low, high)  使用完整范围
        span_low = 1.0 - low
        span_high = high - 1.0
        return (1.0 - span_low * self.schedule_ratio,
                1.0 + span_high * self.schedule_ratio)

    def _randomize(self):
        """执行随机化"""
        # 恢复原始值
        self.mj_model.body_mass[:] = self._original_mass
        self.mj_model.geom_size[:] = self._original_size
        self.mj_model.geom_friction[:] = self._original_friction
        self.mj_model.opt.gravity[:] = self._original_gravity

        # 1. 物体质量随机化
        if self.randomize_mass and len(self.object_body_ids) > 0:
            low, high = self._get_scheduled_range(self.mass_range)
            for bid in self.object_body_ids:
                factor = np.random.uniform(low, high)
                self.mj_model.body_mass[bid] = self._original_mass[bid] * factor

        # 2. 物体尺寸随机化
        if self.randomize_size and len(self.object_geom_ids) > 0:
            low, high = self._get_scheduled_range(self.size_range)
            for gid in self.object_geom_ids:
                factor = np.random.uniform(low, high)
                self.mj_model.geom_size[gid] = self._original_size[gid] * factor

        # 3. 摩擦系数随机化 (对所有 geom 的滑动摩擦)
        if self.randomize_friction:
            low, high = self._get_scheduled_range(self.friction_range)
            for i in range(self.mj_model.ngeom):
                factor = np.random.uniform(low, high)
                self.mj_model.geom_friction[i, 0] = self._original_friction[i, 0] * factor

        # 4. 重力随机化
        if self.randomize_gravity:
            low, high = self._get_scheduled_range(self.gravity_range)
            factor = np.random.uniform(low, high)
            self.mj_model.opt.gravity[:] = self._original_gravity * factor

        # 让修改生效（更新派生物理量）
        mujoco.mj_setConst(self.mj_model, self.mj_data)

    def reset(self, **kwargs):
        self._randomize()
        return self.env.reset(**kwargs)

    def set_schedule_ratio(self, ratio):
        """动态调整随机化强度 (0.0 ~ 1.0)"""
        self.schedule_ratio = float(np.clip(ratio, 0.0, 1.0))


# ==================== 渐进式域随机化调度回调 ====================
class ProgressiveDRCallback(BaseCallback):
    """
    渐进式域随机化调度器

    训练初期 schedule_ratio=0 (无随机化)，逐步增加到 1.0 (完全随机化)
    curriculum 式学习：当最近成功率达到阈值时提升随机化强度
    """
    def __init__(self,
                 vec_env,
                 success_threshold=0.7,
                 warmup_timesteps=200_000,
                 increase_interval=300_000,
                 dr_step=0.15,
                 verbose=0):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.success_threshold = success_threshold
        self.warmup_timesteps = warmup_timesteps
        self.increase_interval = increase_interval
        self.dr_step = dr_step

        self.current_ratio = 0.0
        self.last_increase_step = 0
        self.success_buffer = []

    def _set_env_ratio(self, ratio):
        """通过 env_method 设置所有并行环境的 schedule_ratio"""
        try:
            # SubprocVecEnv 支持 env_method 调用包装器中的方法
            self.vec_env.env_method("set_schedule_ratio", ratio)
        except Exception as e:
            print(f"[WARN] 无法设置环境随机化强度: {e}")

    def _on_step(self) -> bool:
        # 收集成功信息
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "is_success" in info:
                self.success_buffer.append(info["is_success"])

        # 保持最近 100 个 episode
        if len(self.success_buffer) > 100:
            self.success_buffer = self.success_buffer[-100:]

        # warmup 阶段不随机化
        if self.num_timesteps < self.warmup_timesteps:
            return True

        # 检查是否应该提升随机化强度
        if (self.num_timesteps - self.last_increase_step >= self.increase_interval
                and len(self.success_buffer) >= 20):

            recent_success = np.mean(self.success_buffer[-20:])

            if recent_success >= self.success_threshold and self.current_ratio < 1.0:
                self.current_ratio = min(1.0, self.current_ratio + self.dr_step)
                self.last_increase_step = self.num_timesteps
                self._set_env_ratio(self.current_ratio)

                print(f"\n[DR-UPGRADE] Step {self.num_timesteps}: "
                      f"随机化强度提升到 {self.current_ratio:.2f} "
                      f"(近期成功率: {recent_success:.2%})")

        return True


# ==================== 精细成功率显示回调 ====================
class DetailedSuccessRateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.success_buffer = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "is_success" in info:
                self.success_buffer.append(info["is_success"])
                self.episode_count += 1

        if self.episode_count >= 100:
            success_rate = np.mean(self.success_buffer)
            print(f"\n[OK] 最近100个episode成功率: {success_rate:.4f} ({success_rate*100:.1f}%)")
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


def make_env(rank, seed=0,
             mass_range=(0.5, 1.5),
             size_range=(0.8, 1.3),
             friction_range=(0.5, 1.5),
             gravity_range=(0.95, 1.05),
             init_schedule=0.0):
    """创建单个环境的工厂函数，带域随机化"""
    def _init():
        import gymnasium_robotics
        env = gym.make("FetchPickAndPlace-v4")
        env = SuccessTerminationWrapper(env)
        # 域随机化包装器 (放在最内层，紧贴原始环境)
        env = DomainRandomizationWrapper(
            env,
            randomize_mass=True,
            randomize_size=True,
            randomize_friction=True,
            randomize_gravity=False,
            mass_range=mass_range,
            size_range=size_range,
            friction_range=friction_range,
            gravity_range=gravity_range,
            schedule_ratio=init_schedule,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


# ==================== 配置区 ====================
ENV_ID = "FetchPickAndPlace-v4"

# === 模型路径设置 ===
# 建议从训练好的模型继续训练 (将 new 设为 0)
OLD_MODEL_PATH = "sac_fetch_pickandplacev3"          # 或 "sac_fetch_pickandplacev3"
NEW_MODEL_PATH = "sac_fetch_dr_v1"       # 新模型保存名称

TIMESTEPS = 3_000_000
EVAL_FREQ = 100_000
N_EVAL_EPISODES = 20
SAVE_FREQ = 100_000
SEED = 42
NUM_ENVS = 4

# === 域随机化参数配置 ===
# 详见下方【参数建议说明】
DR_CONFIG = {
    "mass_range": (0.4, 1.8),       # 物体质量随机化范围 (倍数)
    "size_range": (0.75, 1.35),     # 物体尺寸随机化范围 (倍数)
    "friction_range": (0.5, 1.5),   # 摩擦系数随机化范围 (倍数)
    "gravity_range": (0.95, 1.05),  # 重力随机化范围 (倍数)
}

# 渐进式调度参数
DR_SCHEDULE = {
    "success_threshold": 0.65,    # 成功率达到此值则提升随机化强度
    "warmup_timesteps": 200_000,  # 热身步数
    "increase_interval": 250_000, # 提升间隔
    "dr_step": 0.15,              # 每次提升的步长
}

# 日志目录
LOG_DIR = "./logs_dr_v1/"
TB_DIR = "./tb_dr_v1/"
# NOTE: the directories are created inside __main__ below, not here. Creating
# them at import time meant that merely doing `from domain_randomization import
# DomainRandomizationWrapper` (which eval_dr.py and model_comp.py both do) would
# scatter empty logs_dr_v1/ and tb_dr_v1/ folders into whatever directory the
# caller happened to be in.


# ==================== 主程序 ====================
if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)

    # === 关键设置 ===
    # new=0: 从已有模型继续训练 (强烈推荐！)
    # new=1: 从头训练 (仅用于对比实验)
    new = 0

    # === 是否启用渐进式调度 ===
    # use_progressive=True:  自动根据成功率提升随机化强度 (推荐)
    # use_progressive=False: 固定 schedule_ratio=1.0 (全程完全随机化)
    use_progressive = True

    # 如果不用渐进式，设置固定随机化强度
    fixed_schedule_ratio = 1.0 if not use_progressive else 0.0

    # ==================== 随机种子 ====================
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ==================== 环境初始化 ====================
    print(f"Creating {NUM_ENVS} parallel training environments with Domain Randomization...")
    train_env = SubprocVecEnv([
        make_env(i, SEED,
                 mass_range=DR_CONFIG["mass_range"],
                 size_range=DR_CONFIG["size_range"],
                 friction_range=DR_CONFIG["friction_range"],
                 gravity_range=DR_CONFIG["gravity_range"],
                 init_schedule=fixed_schedule_ratio)
        for i in range(NUM_ENVS)
    ])

    # 评估环境：不做随机化，用于衡量策略在标准环境的表现
    eval_env = Monitor(SuccessTerminationWrapper(gym.make(ENV_ID)))
    eval_env.reset(seed=SEED + NUM_ENVS)

    # ==================== 模型加载/创建 ====================
    if new:
        print("[MODE] 从头开始训练新模型...")
        model = SAC(
            "MultiInputPolicy",
            train_env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            ),
            learning_rate=LinearSchedule(start=3e-4, end=1e-5, end_fraction=0.8),
            buffer_size=2_000_000,
            batch_size=256,
            gamma=0.95,
            tau=0.005,
            device="cpu",
            ent_coef="auto_0.01",
            target_update_interval=1,
            learning_starts=10000,
            verbose=1,
            tensorboard_log=TB_DIR,
            seed=SEED,
        )
    else:
        print(f"[MODE] 加载预训练模型: {OLD_MODEL_PATH} 进行域随机化微调")
        model = SAC.load(
            OLD_MODEL_PATH,
            env=train_env,
            tensorboard_log=TB_DIR,
            verbose=1,
            device="cpu",
        )

        # 微调参数设置
        model.learning_rate = 1e-5
        model.replay_buffer.n_sampled_goal = 4
        model.replay_buffer.online_sampling = True
        model.batch_size = 256
        model.gamma = 0.95
        model.tau = 0.005
        model.ent_coef = "auto_0.01"
        model.target_update_interval = 1
        model.learning_starts = model.num_timesteps + 2000

        print(f"[OK] 模型加载成功！当前累计训练步数: {model.num_timesteps}")
        print(f"[INFO] 继续训练完成后累计步数: {model.num_timesteps + TIMESTEPS}")

    MODEL_PATH = NEW_MODEL_PATH

    # ==================== 训练前验证 ====================
    if not new:
        print("\n[CHECK] 正在验证旧模型在标准环境的性能...")
        old_mean_reward, old_std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=20, deterministic=True
        )
        print(f"旧模型验证结果: 平均奖励 = {old_mean_reward:.2f} +/- {old_std_reward:.2f}")

    # ==================== Callbacks ====================
    callbacks = [
        DetailedSuccessRateCallback(),
        EvalCallback(
            eval_env,
            best_model_save_path=LOG_DIR,
            log_path=LOG_DIR,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            render=False,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=SAVE_FREQ,
            save_path=LOG_DIR,
            name_prefix="sac_fetch_dr_checkpoint",
            verbose=1,
        ),
    ]

    if use_progressive:
        callbacks.append(ProgressiveDRCallback(
            train_env,
            success_threshold=DR_SCHEDULE["success_threshold"],
            warmup_timesteps=DR_SCHEDULE["warmup_timesteps"],
            increase_interval=DR_SCHEDULE["increase_interval"],
            dr_step=DR_SCHEDULE["dr_step"],
        ))

    # ==================== 训练 ====================
    print("\n" + "=" * 60)
    print("  域随机化 (Domain Randomization) 训练配置")
    print("=" * 60)
    print(f"[MODEL] 模型路径: {MODEL_PATH}")
    print(f"[MODE]  {'从头训练' if new else '从预训练模型微调'}")
    print(f"[DR] 渐进式调度: {'启用' if use_progressive else '禁用'}")
    print(f"[DR] 质量随机化范围: {DR_CONFIG['mass_range']}")
    print(f"[DR] 尺寸随机化范围: {DR_CONFIG['size_range']}")
    print(f"[DR] 摩擦随机化范围: {DR_CONFIG['friction_range']}")
    print(f"[DR] 重力随机化范围: {DR_CONFIG['gravity_range']}")
    print(f"[TIME] 训练步数: {TIMESTEPS}")
    print(f"[SEED] 种子: {SEED}")
    print(f"[LOG] 日志目录: {TB_DIR}")
    print(f"[SAVE] 模型保存路径: {MODEL_PATH}.zip")
    print(f"[ENV] 并行环境数: {NUM_ENVS}")
    print("=" * 60 + "\n")

    try:
        model.learn(
            total_timesteps=TIMESTEPS,
            callback=callbacks,
            reset_num_timesteps=new,
            progress_bar=True,
        )
    finally:
        model.save(MODEL_PATH)
        print(f"\n[OK] 最终模型已保存到: {MODEL_PATH}.zip")
        print(f"[INFO] 最终累计训练步数: {model.num_timesteps}")

    # ==================== 最终评估 ====================
    print("\n[EVAL] 开始最终评估...")
    print("评估新模型在标准环境性能...")
    new_mean_reward, new_std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=100, deterministic=True
    )
    print(f"新模型 (标准环境): 平均奖励 = {new_mean_reward:.2f} +/- {new_std_reward:.2f}")

    # 评估在随机化环境的表现
    print("\n评估新模型在随机化环境性能...")
    dr_eval_env = make_env(999, SEED,
                           mass_range=DR_CONFIG["mass_range"],
                           size_range=DR_CONFIG["size_range"],
                           friction_range=DR_CONFIG["friction_range"],
                           gravity_range=DR_CONFIG["gravity_range"],
                           init_schedule=1.0)()
    dr_eval_env = Monitor(dr_eval_env)
    dr_mean_reward, dr_std_reward = evaluate_policy(
        model, dr_eval_env, n_eval_episodes=100, deterministic=True
    )
    print(f"新模型 (随机化环境): 平均奖励 = {dr_mean_reward:.2f} +/- {dr_std_reward:.2f}")

    train_env.close()
    eval_env.close()
    dr_eval_env.close()
