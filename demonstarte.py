import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import gymnasium_robotics  # 注册 Fetch 环境
import numpy as np
import os
import time

ENV_ID = "FetchPickAndPlace-v4"
MODEL_PATH = "model_2/sac_fetch_dr_v1"



def main():
    # 创建带可视化的评估环境
    eval_env = Monitor(gym.make(ENV_ID, render_mode="human"))

    # 加载训练好的 SAC+HER 模型
    model = SAC.load(f"{MODEL_PATH}.zip", env=eval_env)

    obs, info = eval_env.reset(seed=41)
    episode_reward = 0.0
    episode_count = 0
    max_episodes = 10
    total_avg_rewards = 0
    success_count = 0

    # 检测到成功后持续维持的步数（可调整）
    SUCCESS_HOLD_STEPS = 5
    success_hold_counter = 0

    while episode_count < max_episodes:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_reward += reward
        time.sleep(0.2)

        # 检测到成功后开始倒计时维持
        if success_hold_counter > 0:
            success_hold_counter -= 1
            if success_hold_counter == 0:
                terminated = True
        elif info.get("is_success", False):
            success_count += 1
            success_hold_counter = SUCCESS_HOLD_STEPS

        if terminated or truncated:
            print(f"Episode {episode_count + 1}: 奖励 = {episode_reward:.2f}")
            total_avg_rewards += episode_reward
            episode_reward = 0.0
            episode_count += 1
            success_hold_counter = 0
            obs, info = eval_env.reset()

    eval_env.close()
    print("\n演示完成！")
    print(f"平均奖励: {total_avg_rewards / max_episodes:.2f}")
    print(f"成功率: {success_count / max_episodes * 100:.1f}% ({success_count}/{max_episodes})")



if __name__ == "__main__":
    main()