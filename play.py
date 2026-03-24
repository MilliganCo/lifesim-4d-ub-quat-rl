import argparse, numpy as np, torch
from stable_baselines3 import PPO
from env import LifeSimEnv
import time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--episodes', type=int, default=2)
    ap.add_argument('--render', type=str, default='visual', choices=['ascii', 'visual'])
    ap.add_argument('--team', type=int, default=-1)
    ap.add_argument('--save-replay', action='store_true', help='Save replay to file')
    ap.add_argument('--replay-dir', type=str, default='replays', help='Directory for replays')
    args = ap.parse_args()

    model = PPO.load(args.ckpt, device='cpu')
    env = LifeSimEnv(
        render_mode=args.render, 
        learning_team=args.team,
        save_replay=args.save_replay,
        replay_dir=args.replay_dir
    )
    
    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            step_count = 0
            
            print(f"\n=== Episode {ep+1}/{args.episodes} ===")
            print(f"Team: {info.get('team', 'random')}")
            
            while not done:
                act, _ = model.predict(obs, deterministic=True)
                obs, rew, done, trunc, info = env.step(act)
                total_reward += rew
                step_count += 1
                
                # Небольшая задержка для визуализации
                if args.render == 'visual':
                    time.sleep(0.05)
            
            print(f"Episode {ep+1} done:")
            print(f"  Steps: {step_count}")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Final coins: {info.get('coins', 0)}")
            
            # Сохранение реплея
            if args.save_replay:
                import os
                os.makedirs(args.replay_dir, exist_ok=True)
                replay_path = os.path.join(args.replay_dir, f"replay_ep{ep+1}.json")
                env.save_replay_to_file(replay_path)
    finally:
        env.close()

if __name__ == '__main__':
    main()
