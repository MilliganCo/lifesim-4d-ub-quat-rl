import os, argparse, numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from env import LifeSimEnv
from models import UVQuatExtractor
from typing import List, Optional

class SB3PolicyOpponent:
    def __init__(self, model_path:str):
        self.model = PPO.load(model_path, device='cpu')
    def act(self, gs, team, K):
        # Stateless greedy
        obs = torch.tensor(gs.obs_for(team)[None], dtype=torch.float32)
        a, _ = self.model.predict(obs.numpy(), deterministic=True)
        a = np.asarray(a).reshape(-1)
        moves = a[:K] % 5
        verbs = a[K:2*K] % 5
        idxs = gs.team_unit_indices(team,K)
        return idxs, moves.astype(np.int64), verbs.astype(np.int64)

class SaveBestCoinsCallback(BaseCallback):
    def __init__(self, check_freq:int, save_path:str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best = -np.inf
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            coins = self.model.logger.get_log_dict().get("rollout/ep_rew_mean", None)
            if coins is not None and coins > self.best:
                self.best = coins
                self.model.save(os.path.join(self.save_path, "best_model"))
        return True

class VisualMonitorCallback(BaseCallback):
    """Callback для визуального просмотра обучения"""
    def __init__(self, check_freq: int = 1000, render_freq: int = 10, opponent_pool: List = None):
        super().__init__()
        self.check_freq = check_freq
        self.render_freq = render_freq
        self.opponent_pool = opponent_pool or []
        self.visual_env: Optional[LifeSimEnv] = None
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        # Периодически создаем визуальное окружение для просмотра
        if self.num_timesteps % self.check_freq == 0:
            if self.visual_env is None:
                # Создаем отдельное окружение для визуализации
                self.visual_env = LifeSimEnv(
                    grid_size=32,
                    max_steps=600,
                    K=8,
                    opponent_pool=self.opponent_pool,
                    learning_team=-1,
                    render_mode='visual'
                )
            
            # Показываем короткий эпизод
            if self.num_timesteps % (self.check_freq * self.render_freq) == 0:
                self._render_episode()
        
        return True
    
    def _render_episode(self):
        """Рендеринг одного эпизода"""
        if self.visual_env is None:
            return
        
        try:
            obs, _ = self.visual_env.reset()
            done = False
            steps = 0
            max_render_steps = 100  # Ограничиваем для производительности
            
            while not done and steps < max_render_steps:
                # Используем текущую политику модели
                action, _ = self.model.predict(obs, deterministic=False)
                obs, _, done, _, _ = self.visual_env.step(action)
                steps += 1
            
            self.episode_count += 1
        except Exception as e:
            print(f"Error in visual rendering: {e}")
    
    def _on_training_end(self):
        """Закрытие визуального окружения при завершении"""
        if self.visual_env:
            self.visual_env.close()
            self.visual_env = None
    
    def update_opponent_pool(self, opponent_pool: List):
        """Обновление пула противников"""
        self.opponent_pool = opponent_pool
        if self.visual_env:
            self.visual_env._opp_pool = opponent_pool

def make_env(opponent_pool:List, learn_team:int, render_mode:Optional[str]=None):
    def _f():
        return LifeSimEnv(
            grid_size=32, 
            max_steps=600, 
            K=8, 
            opponent_pool=opponent_pool, 
            learning_team=learn_team,
            render_mode=render_mode
        )
    return _f

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="runs/ppo")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learn-team", type=int, default=-1, help="-1 => random team per episode")
    parser.add_argument("--visual-monitor", action='store_true', help='Enable visual monitoring during training')
    parser.add_argument("--visual-freq", type=int, default=10, help='Render every N checkpoints')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    opponent_pool:List = []

    # Vec env (без визуализации для производительности)
    env = make_vec_env(
        make_env(opponent_pool, args.learn_team, render_mode=None), 
        n_envs=4, 
        seed=args.seed, 
        vec_env_cls=DummyVecEnv
    )

    policy_kwargs = dict(
        features_extractor_class=UVQuatExtractor,
        features_extractor_kwargs=dict(features_dim=256, base_channels=32),
        net_arch=[dict(pi=[128,128], vf=[128,128])]
    )

    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=1024, batch_size=512, learning_rate=3e-4)

    # callbacks: save best, snapshot to pool every N steps
    callbacks = [
        SaveBestCoinsCallback(check_freq=5000, save_path=args.save_dir, verbose=0)
    ]
    
    # Добавляем визуальный мониторинг если включен
    visual_cb = None
    if args.visual_monitor:
        visual_cb = VisualMonitorCallback(check_freq=5000, render_freq=args.visual_freq, opponent_pool=opponent_pool)
        callbacks.append(visual_cb)

    class SnapshotCb(BaseCallback):
        def __init__(self, every:int=50_000):
            super().__init__()
            self.every = every
        def _on_step(self):
            if self.num_timesteps>0 and self.num_timesteps % self.every == 0:
                path = os.path.join(args.save_dir, f"snapshot_{self.num_timesteps}.zip")
                self.model.save(path)
                opponent_pool.append(SB3PolicyOpponent(path))
                # Обновляем пул противников в визуальном мониторе
                if visual_cb:
                    visual_cb.update_opponent_pool(opponent_pool)
            return True
    
    callbacks.append(SnapshotCb(50_000))

    print(f"Starting training for {args.total_steps} steps...")
    print(f"Visual monitoring: {'enabled' if args.visual_monitor else 'disabled'}")
    
    model.learn(total_timesteps=args.total_steps, callback=callbacks)
    model.save(os.path.join(args.save_dir,"final_model"))
    
    # Закрываем окружения
    env.close()
    
    print("Training done. Models saved in", args.save_dir)

if __name__ == "__main__":
    main()
