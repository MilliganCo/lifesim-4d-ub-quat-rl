from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# ==== Constants ====
RES_TYPES = ["food","wood","ore"]
BUILDINGS = ["farm","depot","barracks","market"]
UNIT_TYPES = ["worker","attacker","defender","scout"]
TEAMS = [0,1,2,3]

DIRS = [(0,0),(0,-1),(0,1),(-1,0),(1,0)]  # stay, up, down, left, right

@dataclass
class Unit:
    team:int
    kind:str
    x:int
    y:int
    hp:int=10
    carrying:Dict[str,int]=field(default_factory=lambda: {r:0 for r in RES_TYPES})

@dataclass
class Building:
    team:int
    kind:str
    x:int
    y:int
    hp:int=20

@dataclass
class GameConfig:
    W:int=32
    H:int=32
    max_steps:int=600
    initial_units:Dict[str,int]=field(default_factory=lambda: {"worker":4,"attacker":2,"defender":2,"scout":2})
    resource_density:float=0.06
    base_prices:Dict[str,int]=field(default_factory=lambda: {"food":1,"wood":2,"ore":3})
    market_spread:float=0.15  # +/- от базовой цены
    cargo_cap:int=5
    vision:int=6

class GameState:
    def __init__(self, cfg:GameConfig, rng:np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.t = 0
        self.grid_terrain = np.zeros((self.cfg.H, self.cfg.W), dtype=np.int8)  # 0 empty
        # resources per cell
        self.grid_res = {r: np.zeros((self.cfg.H, self.cfg.W), dtype=np.int16) for r in RES_TYPES}
        self.units: List[Unit] = []
        self.buildings: List[Building] = []
        self.team_coins = {t:0 for t in TEAMS}
        self._spawn_resources()
        self._spawn_markets()
        self._spawn_corners()

    # --- generation ---
    def _spawn_resources(self):
        for r in RES_TYPES:
            mask = self.rng.random((self.cfg.H,self.cfg.W)) < self.cfg.resource_density
            amount = self.rng.integers(5,15,size=mask.sum(),endpoint=True)
            self.grid_res[r][mask] += amount

    def _spawn_markets(self, per_team:int=2):
        # Random markets
        for _ in range(per_team*len(TEAMS)):
            y = self.rng.integers(1,self.cfg.H-1)
            x = self.rng.integers(1,self.cfg.W-1)
            self.buildings.append(Building(team=-1, kind="market", x=x, y=y, hp=999))

    def _corner(self, i:int):
        if i==0: return (1,1)
        if i==1: return (self.cfg.W-2,1)
        if i==2: return (1,self.cfg.H-2)
        return (self.cfg.W-2,self.cfg.H-2)

    def _spawn_corners(self):
        for team in TEAMS:
            cx, cy = self._corner(team)
            # team depot
            self.buildings.append(Building(team=team, kind="depot", x=cx, y=cy, hp=50))
            # units
            for kind, n in self.cfg.initial_units.items():
                for _ in range(n):
                    dx = int(self.rng.integers(-1,2))
                    dy = int(self.rng.integers(-1,2))
                    self.units.append(Unit(team=team, kind=kind, x=np.clip(cx+dx,0,self.cfg.W-1), y=np.clip(cy+dy,0,self.cfg.H-1)))

    # --- helpers ---
    def in_bounds(self,x,y): return 0<=x<self.cfg.W and 0<=y<self.cfg.H

    def units_at(self,x,y): 
        return [i for i,u in enumerate(self.units) if u.x==x and u.y==y and u.hp>0]

    def building_at(self,x,y, kind:Optional[str]=None, owned_by:Optional[int]=None):
        for i,b in enumerate(self.buildings):
            if b.x==x and b.y==y and b.hp>0:
                if (kind is None or b.kind==kind) and (owned_by is None or b.team==owned_by):
                    return i
        return None

    # --- simulation step for one team's vectorized macro-action ---
    # actions: for up to K units of 'team' -> (move_dir in 0..4, verb in 0..4)
    def step_team(self, team:int, unit_indices:List[int], moves:np.ndarray, verbs:np.ndarray):
        # moves, verbs shapes: [K], values in ranges
        # Clamp and apply
        K = len(unit_indices)
        for k in range(K):
            ui = unit_indices[k]
            if ui is None: continue
            u = self.units[ui]
            if u.hp<=0 or u.team!=team: continue
            mv = int(np.clip(moves[k],0,4))
            vb = int(np.clip(verbs[k],0,4))
            dx,dy = DIRS[mv]
            nx, ny = int(np.clip(u.x+dx,0,self.cfg.W-1)), int(np.clip(u.y+dy,0,self.cfg.H-1))
            u.x, u.y = nx, ny
            # verb: 0 none, 1 harvest, 2 attack, 3 build, 4 trade
            if vb==1: self._harvest(u)
            elif vb==2: self._attack(u)
            elif vb==3: self._build(u)
            elif vb==4: self._trade(u)

    def _harvest(self, u:Unit):
        if u.kind not in ("worker","scout"): return
        for r in RES_TYPES:
            if self.grid_res[r][u.y,u.x]>0 and (u.carrying[r] < self.cfg.cargo_cap):
                take = min(1, int(self.grid_res[r][u.y,u.x]))
                self.grid_res[r][u.y,u.x]-=take
                u.carrying[r]+=take
                break

    def _attack(self, u:Unit):
        # attack one enemy in the same cell if present
        idxs = [i for i in self.units_at(u.x,u.y) if self.units[i].team!=u.team]
        if not idxs: return
        target = self.units[idxs[0]]
        dmg = 3 if u.kind=="attacker" else 1
        target.hp -= dmg

    def _build(self, u:Unit):
        # farm requires wood, depot requires wood+ore, barracks requires ore
        cost = {"farm":{"wood":2},
                "depot":{"wood":1,"ore":1},
                "barracks":{"ore":2}}
        # choose cheapest available
        for kind in ("farm","depot","barracks"):
            need = cost[kind]
            if all(u.carrying[r]>=need.get(r,0) for r in RES_TYPES):
                # pay
                for r,amt in need.items():
                    u.carrying[r]-=amt
                self.buildings.append(Building(team=u.team, kind=kind, x=u.x, y=u.y))
                return

    def _trade(self, u:Unit):
        bi = self.building_at(u.x,u.y, kind="market")
        if bi is None: return
        # sell everything for coins with some noise around base price
        for r in RES_TYPES:
            q = u.carrying[r]
            if q<=0: continue
            base = self.cfg.base_prices[r]
            jitter = 1.0 + (np.random.random()*2-1)*self.cfg.market_spread
            self.team_coins[u.team] += int(round(base*jitter*q))
            u.carrying[r]=0

    def deposit_to_depot(self, team:int):
        # units on own depot drop cargo -> coins with better rate
        for u in self.units:
            if u.hp<=0 or u.team!=team: continue
            bi = self.building_at(u.x,u.y, kind="depot", owned_by=team)
            if bi is not None:
                for r in RES_TYPES:
                    if u.carrying[r]>0:
                        self.team_coins[team] += 2*u.carrying[r]  # better rate
                        u.carrying[r]=0

    def remove_dead(self):
        self.units = [u for u in self.units if u.hp>0]
        self.buildings = [b for b in self.buildings if b.hp>0]

    def living_teams(self):
        teams_alive = set(u.team for u in self.units if u.hp>0)
        return teams_alive

    def step_world_tick(self):
        # small regen of resources
        for r in RES_TYPES:
            if np.random.random()<0.02:
                y = np.random.randint(0,self.cfg.H)
                x = np.random.randint(0,self.cfg.W)
                self.grid_res[r][y,x]+=1
        self.t += 1

    # observation tensor for a given team: [C,H,W]
    def obs_for(self, team:int, vision:int=None):
        V = vision if vision is not None else self.cfg.vision
        C = []
        # per-resource
        for r in RES_TYPES:
            C.append(np.clip(self.grid_res[r],0,15)/15.0)
        # buildings one-hot: own depots, own others, enemy, neutral market
        own_depot = np.zeros_like(self.grid_res["food"], dtype=np.float32)
        own_build = np.zeros_like(own_depot)
        enemy_build = np.zeros_like(own_depot)
        market = np.zeros_like(own_depot)
        for b in self.buildings:
            if b.hp<=0: continue
            if b.kind=="market":
                market[b.y,b.x]=1.0
            elif b.team==team:
                if b.kind=="depot": own_depot[b.y,b.x]=1.0
                else: own_build[b.y,b.x]=1.0
            else:
                enemy_build[b.y,b.x]=1.0
        C += [own_depot, own_build, enemy_build, market]
        # units: four maps per unit type for friendly and enemy
        for friendly in (True, False):
            for kind in UNIT_TYPES:
                m = np.zeros_like(own_depot)
                for u in self.units:
                    if u.hp<=0: continue
                    if (u.team==team) == friendly and u.kind==kind:
                        m[u.y,u.x]+=1.0
                m = np.clip(m,0,3)/3.0
                C.append(m)
        # coins scalar broadcast
        money = np.ones_like(own_depot,dtype=np.float32)* (self.team_coins[team]/100.0)
        C.append(money)
        obs = np.stack(C,axis=0)  # [C,H,W]
        return obs.astype(np.float32)

    # list of up to K indices of team's units, padded with None
    def team_unit_indices(self, team:int, K:int):
        idxs = [i for i,u in enumerate(self.units) if u.team==team and u.hp>0]
        # sort by reading order for determinism
        idxs.sort(key=lambda i:(self.units[i].y,self.units[i].x))
        idxs = idxs[:K]
        while len(idxs)<K: idxs.append(None)
        return idxs
