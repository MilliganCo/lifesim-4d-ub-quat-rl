"""
Быстрый тест визуализации без обучения
"""
from env import LifeSimEnv
from visualizer import GameVisualizer
import time

def main():
    print("Тестирование визуального клиента...")
    env = LifeSimEnv(
        grid_size=32,
        max_steps=100,
        K=8,
        render_mode='visual',
        learning_team=0
    )
    
    try:
        obs, info = env.reset()
        print(f"Команда: {info.get('team', 'random')}")
        print("Нажмите ESC или закройте окно для выхода")
        
        done = False
        step = 0
        while not done and step < 100:
            # Случайные действия для демонстрации
            import numpy as np
            action = np.random.randint(0, 5, size=(2, 8))
            obs, reward, done, trunc, info = env.step(action.flatten())
            step += 1
            time.sleep(0.1)
        
        print(f"Тест завершен. Шагов: {step}, Монеты: {info.get('coins', 0)}")
    finally:
        env.close()

if __name__ == '__main__':
    main()

