import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, List, Dict, Any
from engine import GameConfig, GameState, TEAMS, RES_TYPES

class HeuristicOpponent:
    def act(self, gs:GameState, team:int, K:int):
        idxs = gs.team_unit_indices(team,K)
        moves = np.zeros(K,dtype=np.int64)
        verbs = np.zeros(K,dtype=np.int64)
        # simple policy: workers harvest, others move to random or attack if share cell
        for k,ui in enumerate(idxs):
            if ui is None: continue
            u = gs.units[ui]
            if u.kind=="worker":
                verbs[k]=1
            elif u.kind=="attacker":
                verbs[k]=2
            else:
                verbs[k]=0
            # random walk bias to center
            dx = 2 if u.x < gs.cfg.W//2 else 4
            dy = 1 if u.y > gs.cfg.H//2 else 2
            moves[k] = np.random.choice([0,dx,dy])
        return idxs, moves, verbs

class LifeSimEnv(gym.Env):
    metadata = {"render_modes":["ascii", "visual"], "render_fps":10}
    def __init__(self, 
                 grid_size:int=32, 
                 max_steps:int=600, 
                 K:int=8, 
                 opponent_pool=None, 
                 learning_team:int=-1,
                 seed:Optional[int]=None,
                 render_mode:Optional[str]=None,
                 save_replay:bool=False,
                 replay_dir:Optional[str]=None):
        super().__init__()
        self.cfg = GameConfig(W=grid_size,H=grid_size,max_steps=max_steps)
        self.rng = np.random.default_rng(seed)
        self.gs:Optional[GameState] = None
        self.render_mode = render_mode
        self.K = K
        self.learning_team = learning_team  # -1 => random per episode
        self._opp_pool = opponent_pool or []
        self._fallback_opp = HeuristicOpponent()
        self.save_replay = save_replay
        self.replay_dir = replay_dir
        self._replay_data: List[Dict[str, Any]] = []
        self._visualizer = None
        # observation: [C,H,W] => flatten to Box
        dummy = GameState(self.cfg, self.rng).obs_for(0)
        self.obs_shape = dummy.shape
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=self.obs_shape, dtype=np.float32)
        # action: for K units, move in 0..4 and verb in 0..4
        self.action_space = spaces.MultiDiscrete([5]* (2*self.K))

    def reset(self, *, seed:Optional[int]=None, options:Optional[dict]=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.gs = GameState(self.cfg, self.rng)
        self.steps = 0
        # choose learning team
        self.team = self.rng.integers(0,4) if self.learning_team<0 else self.learning_team
        # Reset replay data
        if self.save_replay:
            self._replay_data = []
            self._save_replay_frame()
        obs = self.gs.obs_for(self.team)
        info = {"team": int(self.team)}
        return obs, info

    def _decode_action(self, action:np.ndarray):
        action = np.asarray(action, dtype=np.int64).reshape(2,self.K)
        moves, verbs = action[0], action[1]
        return moves, verbs

    def step(self, action):
        assert self.gs is not None
        self.steps += 1
        # learning team acts
        idxs_learn = self.gs.team_unit_indices(self.team, self.K)
        moves, verbs = self._decode_action(action)
        self.gs.step_team(self.team, idxs_learn, moves, verbs)
        self.gs.deposit_to_depot(self.team)
        # opponents act
        for team in [t for t in TEAMS if t!=self.team]:
            idxs, mv, vb = self._opponent_act(team)
            self.gs.step_team(team, idxs, mv, vb)
            self.gs.deposit_to_depot(team)
        # world tick
        self.gs.remove_dead()
        self.gs.step_world_tick()

        obs = self.gs.obs_for(self.team)
        reward, terminated = self._compute_reward_and_done()
        truncated = self.steps>=self.cfg.max_steps
        done = terminated or truncated
        info = {"coins": self.gs.team_coins[self.team], "steps": self.steps}
        
        # Save replay frame
        if self.save_replay:
            self._save_replay_frame(action, reward, done)
        
        # Render
        if self.render_mode=="ascii":
            self._render_ascii_small()
        elif self.render_mode=="visual":
            self._render_visual(reward, info.get("coins", 0))
        
        return obs, reward, done, False, info

    def _opponent_act(self, team:int):
        if len(self._opp_pool)>0 and np.random.random()<0.8:
            # use a random opponent policy snapshot's heuristic wrapper
            pol = np.random.choice(self._opp_pool)
            return pol.act(self.gs, team, self.K)
        return self._fallback_opp.act(self.gs, team, self.K)

    def _compute_reward_and_done(self):
        # dense shaping: coins delta + harvest + build + kills, minus deaths
        # keep some memory using instance vars
        if not hasattr(self, "_last"):
            self._last = {"coins":0, "units":len([u for u in self.gs.units if u.team==self.team])}
        coins = self.gs.team_coins[self.team]
        alive = len([u for u in self.gs.units if u.team==self.team and u.hp>0])
        reward = 0.0
        reward += (coins - self._last["coins"]) * 0.1
        # living bonus
        reward += alive * 0.01
        # victory check
        teams_alive = self.gs.living_teams()
        terminated = False
        if len(teams_alive)==1 and (self.team in teams_alive):
            reward += 50.0
            terminated = True
        if alive==0:
            reward -= 10.0
            terminated = True
        self._last = {"coins":coins, "units":alive}
        return float(reward), terminated

    # Used at inference to get opponent-like behavior from a PPO policy via this wrapper
    class PolicyWrapper:
        def __init__(self, policy):
            self.policy = policy
        def act(self, gs:GameState, team:int, K:int):
            # Build obs and run one forward pass; greedily decode actions
            import torch
            obs = torch.tensor(gs.obs_for(team)[None], dtype=torch.float32)
            with torch.no_grad():
                dist = self.policy.get_distribution(obs)
                a = dist.distribution.probs.argmax(-1) if hasattr(dist.distribution,"probs") else self.policy.predict(obs.numpy(), deterministic=True)[0]
            a = np.asarray(a).reshape(-1)
            if a.ndim==0: a = np.array([a],dtype=np.int64)
            # decode
            moves = a[:K] % 5
            verbs = a[K:2*K] % 5
            idxs = gs.team_unit_indices(team,K)
            return idxs, moves.astype(np.int64), verbs.astype(np.int64)

    def render(self):
        if self.render_mode == "visual":
            self._render_visual(0.0, 0)
        else:
            self._render_ascii_small()
    
    def _render_visual(self, reward: float, coins: int):
        """Визуальный рендеринг через pygame"""
        if self._visualizer is None:
            try:
                from visualizer import GameVisualizer
                self._visualizer = GameVisualizer(cell_size=15, fps=10)
            except ImportError:
                print("Warning: visualizer not available, falling back to ascii")
                self.render_mode = "ascii"
                return
        
        if self._visualizer and self.gs:
            self._visualizer.render(
                self.gs, 
                learning_team=self.team,
                step=self.steps,
                reward=reward,
                coins=coins,
                paused=False
            )
    
    def _save_replay_frame(self, action: Optional[np.ndarray] = None, 
                           reward: float = 0.0, done: bool = False):
        """Сохранение кадра реплея"""
        if not self.save_replay or self.gs is None:
            return
        
        # Создаем глубокую копию состояния для реплея
        frame = {
            "step": self.steps,
            "team": self.team,
            "action": action.copy() if action is not None else None,
            "reward": reward,
            "done": done,
            "coins": self.gs.team_coins.copy(),
            "units": [
                {
                    "team": u.team,
                    "kind": u.kind,
                    "x": u.x,
                    "y": u.y,
                    "hp": u.hp,
                    "carrying": u.carrying.copy()
                }
                for u in self.gs.units if u.hp > 0
            ],
            "buildings": [
                {
                    "team": b.team,
                    "kind": b.kind,
                    "x": b.x,
                    "y": b.y,
                    "hp": b.hp
                }
                for b in self.gs.buildings if b.hp > 0
            ],
            "resources": {
                r: self.gs.grid_res[r].copy() 
                for r in RES_TYPES
            }
        }
        self._replay_data.append(frame)
    
    def save_replay_to_file(self, filepath: str):
        """Сохранение реплея в файл"""
        if not self._replay_data:
            return
        
        import json
        import os
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self._replay_data, f, indent=2)
        print(f"Replay saved to {filepath}")

    def _render_ascii_small(self):
        # very small coarse render for quick look
        gs = self.gs
        H,W = gs.cfg.H, gs.cfg.W
        canvas = np.full((H,W), '.', dtype='<U1')
        # resources
        for r,ch in zip(["food","wood","ore"], ['f','w','o']):
            y,x = np.where(gs.grid_res[r]>0)
            canvas[y,x]=ch
        # buildings
        for b in gs.buildings:
            if b.kind=="market": canvas[b.y,b.x]='M'
            elif b.kind=="depot": canvas[b.y,b.x]=str(b.team)
            else: canvas[b.y,b.x]='B'
        # units
        for u in gs.units:
            if u.hp<=0: continue
            mark = 'a' if u.kind=="attacker" else ('d' if u.kind=="defender" else ('s' if u.kind=="scout" else 'w'))
            # team coloring by digit
            canvas[u.y,u.x]=mark
        # print a cropped view around team depot for brevity
        print("\n".join("".join(row) for row in canvas[0:16,0:40]))
    
    def close(self):
        """Закрытие окружения и визуализатора"""
        if self._visualizer:
            self._visualizer.close()
            self._visualizer = None
