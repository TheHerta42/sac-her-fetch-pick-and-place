import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
import gymnasium_robotics


# ==================== 特征提取器（将 Dict 观测展平拼接） ====================
class DictFeatureExtractor(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Dict)
        self.keys = list(observation_space.spaces.keys())
        self._features_dim = sum(int(np.prod(space.shape)) for space in observation_space.spaces.values())

    @property
    def features_dim(self):
        return self._features_dim

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # 每个张量形状: (batch, *shape)
        return torch.cat([obs[k].float().flatten(start_dim=1) for k in self.keys], dim=1)


# ==================== Actor（高斯策略 + Tanh 压缩） ====================
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        prev = feature_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.mean = nn.Linear(prev, action_dim)
        self.log_std = nn.Linear(prev, action_dim)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        x = self.backbone(obs)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        if deterministic:
            return torch.tanh(mean)
        noise = torch.randn_like(mean)
        return torch.tanh(mean + std * noise)

    def action_log_prob(self, obs: torch.Tensor):
        x = self.backbone(obs)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        noise = torch.randn_like(mean)
        gaussian_actions = mean + std * noise
        actions = torch.tanh(gaussian_actions)

        # 高斯对数概率
        log_prob = torch.distributions.Normal(mean, std).log_prob(gaussian_actions).sum(dim=-1)
        # Tanh 压缩修正项
        log_prob -= torch.log(1.0 - actions.pow(2) + 1e-6).sum(dim=-1)
        return actions, log_prob


# ==================== Critic（双 Q 网络） ====================
class Critic(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.q1 = self._build_q_net(feature_dim, action_dim, hidden_dims)
        self.q2 = self._build_q_net(feature_dim, action_dim, hidden_dims)

    @staticmethod
    def _build_q_net(f_dim, a_dim, hidden_dims):
        layers = []
        prev = f_dim + a_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=1)
        return self.q1(x), self.q2(x)


# ==================== 简版 HER 回放缓冲区 ====================
class HerReplayBuffer:
    """
    最简 HER：只支持单环境，当 episode 结束时一次性将原始转移和虚拟转移写入全局缓冲区。
    """

    def __init__(self, capacity, env, n_sampled_goal: int = 4, strategy: str = "future"):
        self.capacity = capacity
        self.env = env
        self.n_sampled_goal = n_sampled_goal
        self.strategy = strategy.lower()
        self.buffer = []          # 全局回放缓冲区，元素为 (obs, action, reward, next_obs, done)
        self.episode = []         # 当前正在收集的 episode 转移序列

    def add(self, obs, action, reward, next_obs, done, info):
        """存入一步转移；若 episode 结束，则触发 HER 增广并整体写入缓冲区。"""
        self.episode.append((copy.deepcopy(obs), action, reward, copy.deepcopy(next_obs), done, info))
        if done:
            # 1) 写入原始转移
            for trans in self.episode:
                self._push(*trans[:5])
            # 2) 对每个时刻采样 n_sampled_goal 个新目标，生成虚拟转移
            for t in range(len(self.episode)):
                for _ in range(self.n_sampled_goal):
                    new_goal = self._sample_goal(t)
                    obs_t, act_t, _, next_obs_t, done_t, info_t = self.episode[t]

                    obs_her = copy.deepcopy(obs_t)
                    next_obs_her = copy.deepcopy(next_obs_t)
                    obs_her["desired_goal"] = new_goal
                    next_obs_her["desired_goal"] = new_goal

                    # 使用环境提供的 compute_reward 计算新奖励（稀疏/稠密均可）
                    reward_her = self.env.unwrapped.compute_reward(
                        next_obs_her["achieved_goal"], new_goal, info_t
                    )
                    self._push(obs_her, act_t, float(reward_her), next_obs_her, done_t)
            self.episode.clear()

    def _push(self, obs, action, reward, next_obs, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((obs, action, reward, next_obs, float(done)))

    def _sample_goal(self, t: int):
        """依据策略从当前 episode 中采样一个 achieved_goal 作为新目标。"""
        if self.strategy == "future":
            idx = random.randint(t, len(self.episode) - 1)
        elif self.strategy == "final":
            idx = len(self.episode) - 1
        else:  # episode
            idx = random.randint(0, len(self.episode) - 1)
        # next_obs 的 achieved_goal
        return self.episode[idx][3]["achieved_goal"].copy()

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)

        def stack_dict(key):
            return torch.FloatTensor(np.stack([o[key] for o in obs])).to(device)

        obs_t = {k: stack_dict(k) for k in obs[0].keys()}
        next_obs_t = {k: torch.FloatTensor(np.stack([o[k] for o in next_obs])).to(device) for k in next_obs[0].keys()}
        action_t = torch.FloatTensor(np.stack(action)).to(device)
        reward_t = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device)
        done_t = torch.FloatTensor(np.array(done)).unsqueeze(1).to(device)
        return obs_t, action_t, reward_t, next_obs_t, done_t


# ==================== SAC 核心智能体 ====================
class SAC:
    def __init__(
        self,
        env: gym.Env,
        lr: float = 3e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        ent_coef: str | float = "auto_0.01",
        target_update_interval: int = 1,
        learning_starts: int = 1000,
        device: str = "cpu",
        seed: int | None = None,
    ):
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.learning_starts = learning_starts
        self.device = torch.device(device)
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        obs_space = env.observation_space
        act_space = env.action_space
        action_dim = int(np.prod(act_space.shape))

        # 网络
        self.feature_extractor = DictFeatureExtractor(obs_space).to(self.device)
        f_dim = self.feature_extractor.features_dim
        self.actor = Actor(f_dim, action_dim).to(self.device)
        self.critic = Critic(f_dim, action_dim).to(self.device)
        self.critic_target = Critic(f_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        # 优化器
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # 熵系数 alpha
        self.target_entropy = -float(np.prod(act_space.shape))
        if isinstance(ent_coef, str) and ent_coef.startswith("auto"):
            init_val = float(ent_coef.split("_")[1]) if "_" in ent_coef else 1.0
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * init_val).requires_grad_(True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = None
        else:
            self.alpha = float(ent_coef)
            self.log_alpha = None
            self.alpha_opt = None

        # 回放缓冲区
        self.buffer = HerReplayBuffer(buffer_size, env, n_sampled_goal=2, strategy="future")
        self._n_updates = 0

    # -------------------- 动作选择 --------------------
    def select_action(self, obs: dict, deterministic: bool = False):
        with torch.no_grad():
            obs_t = {k: torch.FloatTensor(v).unsqueeze(0).to(self.device) for k, v in obs.items()}
            features = self.feature_extractor(obs_t)
            action = self.actor(features, deterministic=deterministic)
        return action.cpu().numpy().flatten()

    # -------------------- 参数软更新 --------------------
    def _soft_update(self):
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.mul_(1 - self.tau)
            tp.data.add_(self.tau * p.data)

    # -------------------- 单步更新 --------------------
    def update(self):
        if len(self.buffer.buffer) < self.batch_size:
            return

        obs, action, reward, next_obs, done = self.buffer.sample(self.batch_size, self.device)

        # 提取特征
        obs_f = self.feature_extractor(obs)
        next_obs_f = self.feature_extractor(next_obs)

        # ----- 熵系数 alpha -----
        if self.log_alpha is not None:
            alpha = torch.exp(self.log_alpha.detach())
        else:
            alpha = torch.tensor(self.alpha, device=self.device)

        # ----- Critic 更新 -----
        with torch.no_grad():
            next_action, next_log_prob = self.actor.action_log_prob(next_obs_f)
            q1_next, q2_next = self.critic_target(next_obs_f, next_action)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_prob.view(-1, 1)
            q_target = reward + (1 - done) * self.gamma * q_next

        q1, q2 = self.critic(obs_f, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ----- Actor 更新 -----
        action_new, log_prob = self.actor.action_log_prob(obs_f)
        q1_new, q2_new = self.critic(obs_f, action_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_prob.view(-1, 1) - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ----- Alpha 更新 -----
        if self.log_alpha is not None:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        # ----- 软更新目标网络 -----
        self._n_updates += 1
        if self._n_updates % self.target_update_interval == 0:
            self._soft_update()

    # -------------------- 训练 --------------------
    def train(self, total_timesteps: int, eval_freq: int = 5000, n_eval_episodes: int = 10):
        obs, _ = self.env.reset(seed=self.seed)
        episode_reward = 0
        episode_len = 0

        for step in range(total_timesteps):
            # 动作选择
            if step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(obs, deterministic=False)

            # 环境交互
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_len += 1

            # 存储转移
            self.buffer.add(obs, action, reward, next_obs, done, info)
            obs = next_obs

            if done:
                # 每完成一个 episode 打印简要信息
                success = info.get("is_success", 0.0)
                print(f"[Episode] Step={step:>6} | Len={episode_len:>3} | Reward={episode_reward:>7.2f} | Success={success}")
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_len = 0

            # 策略更新
            if step >= self.learning_starts:
                self.update()

            # 评估
            if eval_freq > 0 and step > 0 and step % eval_freq == 0:
                mean_r, std_r, succ_r = self.evaluate(n_episodes=n_eval_episodes)
                print(f"[Eval] Step={step:>6} | MeanReward={mean_r:>7.2f} +/- {std_r:.2f} | Success={succ_r:.2f}")

    # -------------------- 保存 / 加载 --------------------
    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "feature_extractor": self.feature_extractor.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "log_alpha": self.log_alpha,
            "alpha_opt": self.alpha_opt.state_dict() if self.alpha_opt else None,
        }, path)
        print(f"[Save] 模型已保存到 {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.feature_extractor.load_state_dict(ckpt["feature_extractor"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        if self.log_alpha is not None and ckpt.get("log_alpha") is not None:
            # 只复制数据，不替换 tensor 对象，以保持 optimizer 的引用有效
            self.log_alpha.data.copy_(ckpt["log_alpha"].data)
        if self.alpha_opt is not None and ckpt.get("alpha_opt") is not None:
            self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        print(f"[Load] 模型已从 {path} 加载")

    # -------------------- 评估 --------------------
    def evaluate(self, n_episodes: int = 10):
        rewards = []
        successes = []
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action = self.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
            rewards.append(ep_reward)
            successes.append(info.get("is_success", 0.0))
        return float(np.mean(rewards)), float(np.std(rewards)), float(np.mean(successes))


# ==================== 主程序（最简使用示例） ====================
if __name__ == "__main__":
    SEED = 42
    TOTAL_STEPS = 200_000   # 演示用，可自行调大

    env = gym.make("FetchPickAndPlace-v4")

    agent = SAC(
        env,
        lr=3e-4,
        buffer_size=2_000_000,
        batch_size=256,
        gamma=0.92,
        tau=0.005,
        ent_coef="auto_0.01",
        target_update_interval=1,
        learning_starts=5000,
        device="cpu",
        seed=SEED,
    )

    print("[START] 开始训练 SAC+HER on FetchPickAndPlace-v4")
    agent.train(total_timesteps=TOTAL_STEPS, eval_freq=5000, n_eval_episodes=10)

    print("\n[FINAL] 训练结束，进行最终评估...")
    mean_r, std_r, succ_r = agent.evaluate(n_episodes=100)
    print(f"[FINAL] 平均奖励 = {mean_r:.2f} +/- {std_r:.2f} | 成功率 = {succ_r:.2f}")

    env.close()
