"""
Визуальный клиент для просмотра игры в реальном времени
Использует pygame для рендеринга игрового состояния
"""
import pygame
import numpy as np
from typing import Optional, Dict, List
from engine import GameState, Unit, Building, RES_TYPES, UNIT_TYPES

class GameVisualizer:
    def __init__(self, cell_size: int = 20, fps: int = 10):
        """
        Инициализация визуализатора
        
        Args:
            cell_size: размер одной клетки в пикселях
            fps: кадров в секунду для рендеринга
        """
        self.cell_size = cell_size
        self.fps = fps
        self.clock = None
        self.screen = None
        self.font = None
        self.small_font = None
        self.initialized = False
        
        # Цвета
        self.colors = {
            'background': (20, 20, 30),
            'grid': (40, 40, 50),
            'food': (100, 200, 100),
            'wood': (139, 90, 43),
            'ore': (150, 150, 150),
            'market': (255, 215, 0),
            'depot': (100, 100, 200),
            'farm': (50, 150, 50),
            'barracks': (150, 50, 50),
            'team_0': (255, 100, 100),
            'team_1': (100, 100, 255),
            'team_2': (100, 255, 100),
            'team_3': (255, 255, 100),
            'text': (255, 255, 255),
            'text_bg': (0, 0, 0, 180),
        }
        
        # Символы для юнитов
        self.unit_symbols = {
            'worker': 'W',
            'attacker': 'A',
            'defender': 'D',
            'scout': 'S',
        }
    
    def init(self, width: int, height: int):
        """Инициализация pygame"""
        if self.initialized:
            return
        
        pygame.init()
        self.screen = pygame.display.set_mode(
            (width * self.cell_size, height * self.cell_size + 100)
        )
        pygame.display.set_caption("LifeSim 4D - Game Visualizer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 16)
        self.initialized = True
    
    def render(self, gs: GameState, learning_team: int = -1, step: int = 0, 
               reward: float = 0.0, coins: int = 0, paused: bool = False):
        """
        Рендеринг игрового состояния
        
        Args:
            gs: состояние игры
            learning_team: команда, которая обучается (-1 для случайной)
            step: текущий шаг
            reward: текущее вознаграждение
            coins: монеты команды
            paused: пауза
        """
        if not self.initialized:
            self.init(gs.cfg.W, gs.cfg.H)
        
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Очистка экрана
        self.screen.fill(self.colors['background'])
        
        # Рендеринг сетки и ресурсов
        self._render_resources(gs)
        
        # Рендеринг зданий
        self._render_buildings(gs)
        
        # Рендеринг юнитов
        self._render_units(gs, learning_team)
        
        # Рендеринг UI
        self._render_ui(gs, learning_team, step, reward, coins, paused)
        
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True
    
    def _render_resources(self, gs: GameState):
        """Рендеринг ресурсов на карте"""
        for r, color_key in zip(RES_TYPES, ['food', 'wood', 'ore']):
            y_coords, x_coords = np.where(gs.grid_res[r] > 0)
            for y, x in zip(y_coords, x_coords):
                amount = gs.grid_res[r][y, x]
                # Интенсивность цвета зависит от количества
                intensity = min(255, 100 + int(amount * 10))
                color = self.colors[color_key]
                color_scaled = tuple(min(255, c * intensity // 200) for c in color)
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color_scaled, rect)
                # Показываем количество если достаточно места
                if self.cell_size >= 15:
                    text = self.small_font.render(str(amount), True, (255, 255, 255))
                    self.screen.blit(text, (x * self.cell_size + 2, y * self.cell_size + 2))
    
    def _render_buildings(self, gs: GameState):
        """Рендеринг зданий"""
        for b in gs.buildings:
            if b.hp <= 0:
                continue
            
            x_px = b.x * self.cell_size
            y_px = b.y * self.cell_size
            
            if b.kind == "market":
                # Нейтральный рынок - золотой
                color = self.colors['market']
                pygame.draw.circle(
                    self.screen, color,
                    (x_px + self.cell_size // 2, y_px + self.cell_size // 2),
                    self.cell_size // 2 - 2
                )
                text = self.small_font.render('M', True, (0, 0, 0))
                text_rect = text.get_rect(center=(x_px + self.cell_size // 2, y_px + self.cell_size // 2))
                self.screen.blit(text, text_rect)
            else:
                # Командные здания
                team_color = self.colors[f'team_{b.team}'] if b.team >= 0 else (128, 128, 128)
                building_color = {
                    'depot': self.colors['depot'],
                    'farm': self.colors['farm'],
                    'barracks': self.colors['barracks'],
                }.get(b.kind, (200, 200, 200))
                
                # Смешиваем цвет команды и тип здания
                final_color = tuple(
                    (tc + bc) // 2 for tc, bc in zip(team_color, building_color)
                )
                
                rect = pygame.Rect(x_px, y_px, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, final_color, rect)
                pygame.draw.rect(self.screen, team_color, rect, 2)
                
                # Символ здания
                symbol = {'depot': 'D', 'farm': 'F', 'barracks': 'B'}.get(b.kind, '?')
                text = self.small_font.render(symbol, True, (255, 255, 255))
                text_rect = text.get_rect(center=(x_px + self.cell_size // 2, y_px + self.cell_size // 2))
                self.screen.blit(text, text_rect)
                
                # HP бар
                if b.hp < 20:
                    hp_ratio = b.hp / 20.0
                    hp_width = int(self.cell_size * hp_ratio)
                    hp_rect = pygame.Rect(x_px, y_px + self.cell_size - 3, hp_width, 3)
                    pygame.draw.rect(self.screen, (255, 0, 0), hp_rect)
    
    def _render_units(self, gs: GameState, learning_team: int):
        """Рендеринг юнитов"""
        for u in gs.units:
            if u.hp <= 0:
                continue
            
            x_px = u.x * self.cell_size
            y_px = u.y * self.cell_size
            
            # Цвет команды
            team_color = self.colors[f'team_{u.team}'] if u.team >= 0 else (128, 128, 128)
            
            # Подсветка обучающейся команды
            if u.team == learning_team:
                highlight_rect = pygame.Rect(
                    x_px - 2, y_px - 2,
                    self.cell_size + 4, self.cell_size + 4
                )
                pygame.draw.rect(self.screen, (255, 255, 0), highlight_rect, 2)
            
            # Круг для юнита
            radius = self.cell_size // 3
            pygame.draw.circle(
                self.screen, team_color,
                (x_px + self.cell_size // 2, y_px + self.cell_size // 2),
                radius
            )
            
            # Символ типа юнита
            symbol = self.unit_symbols.get(u.kind, '?')
            text = self.small_font.render(symbol, True, (255, 255, 255))
            text_rect = text.get_rect(center=(x_px + self.cell_size // 2, y_px + self.cell_size // 2))
            self.screen.blit(text, text_rect)
            
            # Груз (маленькие точки)
            if any(u.carrying.values()):
                cargo_y = y_px + self.cell_size - 8
                cargo_x = x_px + 2
                for r in RES_TYPES:
                    if u.carrying[r] > 0:
                        cargo_color = {'food': (100, 255, 100), 'wood': (139, 90, 43), 'ore': (200, 200, 200)}[r]
                        pygame.draw.circle(self.screen, cargo_color, (cargo_x, cargo_y), 2)
                        cargo_x += 4
            
            # HP бар
            if u.hp < 10:
                hp_ratio = u.hp / 10.0
                hp_width = int(self.cell_size * hp_ratio)
                hp_rect = pygame.Rect(x_px, y_px, hp_width, 2)
                pygame.draw.rect(self.screen, (255, 0, 0), hp_rect)
    
    def _render_ui(self, gs: GameState, learning_team: int, step: int, 
                   reward: float, coins: int, paused: bool):
        """Рендеринг UI панели"""
        ui_y = gs.cfg.H * self.cell_size
        ui_height = 100
        
        # Фон для UI
        ui_rect = pygame.Rect(0, ui_y, gs.cfg.W * self.cell_size, ui_height)
        pygame.draw.rect(self.screen, (30, 30, 40), ui_rect)
        
        # Информация о командах
        y_offset = ui_y + 5
        x_offset = 10
        
        # Обучающаяся команда
        if learning_team >= 0:
            team_text = self.font.render(f"Learning Team: {learning_team}", True, self.colors['text'])
            self.screen.blit(team_text, (x_offset, y_offset))
        
        # Шаг и монеты
        step_text = self.font.render(f"Step: {step}/{gs.cfg.max_steps}", True, self.colors['text'])
        self.screen.blit(step_text, (x_offset, y_offset + 25))
        
        coins_text = self.font.render(f"Coins: {coins}", True, self.colors['text'])
        self.screen.blit(coins_text, (x_offset, y_offset + 50))
        
        reward_text = self.font.render(f"Reward: {reward:.2f}", True, self.colors['text'])
        self.screen.blit(reward_text, (x_offset + 200, y_offset + 50))
        
        # Статистика команд
        x_stats = gs.cfg.W * self.cell_size - 300
        for team in range(4):
            team_units = len([u for u in gs.units if u.team == team and u.hp > 0])
            team_coins = gs.team_coins.get(team, 0)
            team_color = self.colors[f'team_{team}']
            
            stats_text = self.small_font.render(
                f"Team {team}: {team_units} units, {team_coins} coins",
                True, team_color
            )
            self.screen.blit(stats_text, (x_stats, y_offset + team * 20))
        
        # Пауза
        if paused:
            pause_text = self.font.render("PAUSED - Press SPACE to continue", True, (255, 255, 0))
            text_rect = pause_text.get_rect(center=(gs.cfg.W * self.cell_size // 2, ui_y + ui_height // 2))
            self.screen.blit(pause_text, text_rect)
    
    def close(self):
        """Закрытие визуализатора"""
        if self.initialized:
            pygame.quit()
            self.initialized = False

