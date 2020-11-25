"""
STATE = GAME STATE
OBSERVATION = OBSERVATION
NB! MARIUS, STATE != OBSERVATION
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame as pg
import pygame.font
import copy
import numpy as np
import sys
import time

from pathlib import Path
mod_path = Path(__file__).parent

from enviorment.actions import Action
from enviorment.shapes import Shape
from enviorment.colors import Color
from enviorment.reducedshapes import ReducedShape

class Tetris():

    def __init__(self, config=None, title='Tetris'):
        
        self.title = title
        self.config = {
            'hard_drop': 1,        # Action.DOWN goes all the way down
            'reduced_shapes': 0,   # Replace shapes with reduced shapes
            'reduced_grid': 0,     # half grid size
            'score_multiplier': 0, # cleared_lines ^ score_multiplier
            'fall_tick': 5,        # how many steps before fall down 1
        }
        
        if config is not None:
            if isinstance(config, dict):
                for key, value in config.items():
                    self.config[key] = value
            else:
                raise TypeError('Config need to be dict')
        
        if not self.config['reduced_shapes']:
            self.shapes = Shape
            self.shape_space = len(Shape.ALL)
        else:
            self.shapes = ReducedShape
            self.shape_space = len(ReducedShape.ALL)
            
        self.actions = Action
        self.action_space = len(Action.ALL)
        
        if not self.config['reduced_grid']:
            # Standard Tetris layout
            self.game_rows = 20
            self.game_columns = 10
        else:
            self.game_rows = 10
            self.game_columns = 10

        self.start_position = [0, 3]
        self.position = copy.deepcopy(self.start_position)
        self.highscore = 0
        self.score = None
        self.attempt = 0
        self.tick = 0
       
    def clone(self):
        tetris = Tetris()
        tetris.reset()
        tetris.state = copy.deepcopy(self.state)
        tetris.current_shape = copy.deepcopy(self.current_shape)
        tetris.next_shape = copy.deepcopy(self.next_shape)
        tetris.current_piece = copy.deepcopy(self.current_piece)
        tetris.next_piece = copy.deepcopy(self.next_piece)

        return tetris
                
    # defines observation
    def discretization(self):
        grid_layer = copy.deepcopy(self.state)
        grid_layer = [[1 if c else 0 for c in row] for row in grid_layer]
        piece_layer = [[0 for _ in range(self.game_columns)] for _ in range(self.game_rows)]
        
        for y, x in self.current_shape:
            piece_layer[y][x] = 1
            
        return [grid_layer, piece_layer] # timing layer for drop
        
    @property
    def action_sample(self):
        return np.random.randint(self.action_space)

    def reset(self):
        self.state = [[0 for _ in range(self.game_columns)] for _ in range(self.game_rows)]
        
        # Start position
        shape1, self.current_piece, self.current_rotation = self.new_shape()
        shape2, self.next_piece, self.next_rotation = self.new_shape()
            
        self.current_shape = self.get_blocks_from_shape(shape1, self.current_piece, self.start_position)
        self.next_shape = self.get_blocks_from_shape(shape2, self.next_piece, self.start_position)
        
        if self.score is not None:
            if self.score > self.highscore:
                self.highscore = self.score
        
        self.score = 0
        self.attempt += 1

        return self.discretization(), 0, False, ''

    def get_blocks_from_shape(self, shape, piece, offset=[0, 0]):
        blocks = []

        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell != '0':
                    blocks.append([j, i])

        # normalize
        lower_y = min([y for y, x in blocks])
        lower_x = min([x for y, x in blocks])
        
        if offset == self.start_position:
            offset = copy.deepcopy(self.start_position)
            offset[1] += 0 if piece == 0 else 1

        return [[y-lower_y+offset[0], x-lower_x+offset[1]] for y, x in blocks]

    def check_collision_down(self, shape):
        for y, x in shape:
            if (y + 1) >= self.game_rows or self.state[(y + 1)][x] != 0:
                return True
        return False

    def new_shape(self):
        piece = np.random.randint(self.shape_space)
        rotation = 0
        shape = self.shapes.ALL[piece][rotation]
        return shape, piece, rotation
        
    def check_cleared_lines(self):
        reward = 0
        
        for i, row in enumerate(self.state):           
            if 0 not in row:
                del self.state[i] # magi elns, vet ikke. men det funker fjell
                self.state.insert(0, [0 for _ in range(self.game_columns)])
                reward += 1

        if 'score_multiplier' in self.config and self.config['score_multiplier'] != 0:
            reward **= self.config['score_multiplier']
            
        return reward
        
    def check_loss(self):
        return sum([self.state[y][x] for y, x in self.current_shape]) != 0
                        
    def step(self, action):
        reward = 0
        done = False
        info = ''
        placed = False # if current piece lands on another or bottom

        next_position = copy.deepcopy(self.current_shape)

        if action == Action.DOWN:

            if self.config['hard_drop']:
                collision = self.check_collision_down(next_position)
                while not collision:
                    next_position = [[y+1, x] for y, x in next_position]
                    collision = self.check_collision_down(next_position)
                   
                placed = True

            else:
                if not self.check_collision_down(next_position):
                    next_position = [[y+1, x] for y, x in next_position]
                else:
                    placed = True

        elif action == Action.LEFT:
        
            for y, x in next_position:
                if (x - 1) < 0 or self.state[y][(x - 1)] != 0:
                    break
            else:
                next_position = [[y, x-1] for y, x in next_position]

        elif action == Action.RIGHT:
            
            for y, x in next_position:
                if (x + 1) >= self.game_columns or self.state[y][(x + 1)] != 0:
                    break
            else:
                next_position = [[y, x+1] for y, x in next_position]

        elif action == Action.ROTATE:
            current_posistion = next_position
            self.current_rotation = (self.current_rotation - 1) % len(self.shapes.ALL[self.current_piece])
            new_rotation = self.shapes.ALL[self.current_piece][self.current_rotation]
            next_position = self.get_blocks_from_shape(new_rotation, self.current_piece, self.current_shape[0])
            for y, x in next_position:
                if x >= self.game_columns or y >= self.game_rows or self.state[y][x] != 0:
                    next_position = current_posistion
                    break

        elif action == Action.WAIT:
            if not self.check_collision_down(next_position):
                next_position = [[y+1, x] for y, x in next_position]
            else:
                placed = True
        
        self.tick += 1
        if not self.tick % self.config['fall_tick']:
            if not self.check_collision_down(next_position):
                next_position = [[y+1, x] for y, x in next_position]
            else:
                placed = True

        if placed:
            for block in next_position:
                self.state[block[0]][block[1]] = self.current_piece + 1
                
            self.current_shape = self.next_shape
            self.current_piece = self.next_piece
            self.current_rotation = self.next_rotation
                
            shape, self.next_piece, self.next_rotation = self.new_shape()
            self.next_shape = self.get_blocks_from_shape(shape, self.next_piece, self.start_position)
            done = self.check_loss()
        else:
            self.current_shape = next_position
            
        reward += self.check_cleared_lines()
        self.score += reward

        return self.discretization(), reward, done, info

    def render(self, manual=0):
        if not hasattr(self, 'cell_size'):
            self.__initView()
        
        self.screen.fill((1, 26, 56))
        
        # draw game window border
        rect = pg.Rect(self.game_margin_left - 1, 
                       self.game_margin_top  - 1,
                       self.game_columns * self.cell_size + 2, 
                       self.game_rows    * self.cell_size + 2)
        
        pg.draw.rect(self.screen, Color.WHITE, rect, 1)
        
        # draw cells
        for i, row in enumerate(self.state):
            for j, cell in enumerate(row):

                color = Color.BLACK if not cell else self.shapes.COLORS[cell - 1]
                
                rect = pg.Rect(self.game_margin_left + j * self.cell_size, 
                               self.game_margin_top  + i * self.cell_size, 
                               self.cell_size, 
                               self.cell_size)

                pg.draw.rect(self.screen, color, rect, 0)
                
                if cell == 0:
                    pg.draw.rect(self.screen, (30, 30, 30), rect, 1)

        # draw drop preview
        temp_shape = copy.deepcopy(self.current_shape)
        collision = self.check_collision_down(temp_shape)
        while not collision:
            temp_shape = [[y+1, x] for y, x in temp_shape]
            collision = self.check_collision_down(temp_shape)
            
        for block in temp_shape:
            rect = pg.Rect(self.game_margin_left + block[1] * self.cell_size, 
                           self.game_margin_top  + block[0] * self.cell_size, 
                           self.cell_size, 
                           self.cell_size)
            
            color = list(self.shapes.COLORS[self.current_piece])
            color = tuple([c-200 if c > 200 else c-100 if c > 100 else 0 for c in color])
            
            pg.draw.rect(self.screen, color, rect, 0)

        # draw current shape
        for block in self.current_shape:

            rect = pg.Rect(self.game_margin_left + block[1] * self.cell_size, 
                           self.game_margin_top  + block[0] * self.cell_size, 
                           self.cell_size, 
                           self.cell_size)
            
            pg.draw.rect(self.screen, self.shapes.COLORS[self.current_piece], rect, 0)
            
        # draw info
        next_preview = [self.game_columns * self.cell_size + 80 + self.game_margin_left,
                        self.game_margin_top]
        
        rect = pg.Rect(next_preview[0], 
                       next_preview[1],
                       self.cell_size * 6, 
                       self.cell_size * 5)
        
        pg.draw.rect(self.screen, Color.BLACK, rect, 0)
        
        rect = pg.Rect(next_preview[0] - 1, 
                       next_preview[1] - 1,
                       self.cell_size * 6 + 2, 
                       self.cell_size * 5 + 2)
        
        pg.draw.rect(self.screen, Color.WHITE, rect, 1)
                
        for block in self.next_shape:
            center_y = 1 if self.next_piece == 0 else 0
            center_x = 0
            
            if self.config['reduced_shapes'] and not self.next_piece:
                center_x = 1

            rect = pg.Rect(next_preview[0] + (block[1] - 2 + center_x) * self.cell_size, 
                           next_preview[1] + (block[0] + 1 + center_y) * self.cell_size, 
                           self.cell_size, 
                           self.cell_size)
            
            pg.draw.rect(self.screen, self.shapes.COLORS[self.next_piece], rect, 0)
            
        score_text = self.font.render(("Score: "+ str(self.score)), 1, Color.WHITE)
        score_textRect = score_text.get_rect() 
        score_textRect.center = (self.info_margin_left, 200)
        
        highscore_text = self.font.render(("Highscore: "+ str(self.highscore)), 1, Color.WHITE)
        highscore_textRect = highscore_text.get_rect() 
        highscore_textRect.center = (self.info_margin_left, 240) 
        
        attempt_text = self.font.render(("Attempts: "+ str(self.attempt)), 1, Color.WHITE)
        attempt_textRect = attempt_text.get_rect() 
        attempt_textRect.center = (self.info_margin_left, 280)
        
        algo_text = self.font.render(self.title, 1, Color.WHITE)
        algo_textRect = algo_text.get_rect() 
        algo_textRect.center = (self.info_margin_left, 320)
                
        self.screen.blit(score_text, score_textRect) 
        self.screen.blit(highscore_text, highscore_textRect)
        self.screen.blit(attempt_text, attempt_textRect)
        self.screen.blit(algo_text, algo_textRect)
        
        action = 0
        done = False
        state = None
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            
            if manual and event.type == pg.KEYDOWN:
                if event.key == pg.K_LEFT:
                    action = Action.LEFT
                if event.key == pg.K_RIGHT:
                    action = Action.RIGHT
                if event.key == pg.K_DOWN:
                    action = Action.DOWN
                if event.key == pg.K_UP:
                    action = Action.ROTATE
                if event.key == pg.K_SPACE:
                    action = Action.WAIT
                    
                self.get_all_states()

                state, _, done, _ = self.step(action)
                    
        pg.display.update()
        return state, action, done
    
    def quit(self):
        pg.quit()
    
    def __initView(self):
        self.cell_size = 25
        self.game_margin_top = 40
        self.game_margin_left = 40
        self.info_margin_left = 450

        self.window_height = self.window_width = 600

        self.pg = pg
        pg.init()
        pg.display.set_caption(self.title)

        self.screen = pg.display.set_mode((self.window_height, self.window_width))
        self.clock = pg.time.Clock()
        self.screen.fill(Color.BLACK)
        self.font = pg.font.Font(None, 36)

    # for "simulating" steps
    def save_checkpoint(self):
        return [
            copy.deepcopy(self.state),
            copy.deepcopy(self.current_piece),
            copy.deepcopy(self.current_rotation),
            copy.deepcopy(self.current_shape),
            copy.deepcopy(self.next_piece),
            copy.deepcopy(self.next_shape),
            copy.deepcopy(self.next_rotation),
            copy.deepcopy(self.score),
            copy.deepcopy(self.tick)]
        
    def load_checkpoint(self, save):
        save = copy.deepcopy(save)
        self.state,self.current_piece,self.current_rotation,self.current_shape,self.next_piece,self.next_shape,self.next_rotation,self.score,self.tick = save

    def get_all_states(self, display=True):
        
        states = []
        actions = []
        
        for r in range(1, len(self.shapes.ALL[self.current_piece]) + 1):
            for i in range(self.game_columns):
                actions.append([self.actions.ROTATE]*r)      
                traverse = self.actions.LEFT if (i-(self.game_columns//2)) < 0 else self.actions.RIGHT
                actions[-1] += [traverse] * (i if i < (self.game_columns//2) else i-(self.game_columns//2)+1)
                actions[-1].append(self.actions.DOWN)
                
        rewards = [0]*len(actions)
                
        checkpoint = self.save_checkpoint()
        for i, action_sublist in enumerate(actions):
            
            state = None
            for action in action_sublist:
                state, reward, done, _ = self.step(action)
                #self.render()
                #time.sleep(.01)
                rewards[i] += reward * 50
                
            states.append(state[0])
            self.load_checkpoint(checkpoint)
            
        for i, state in enumerate(states):
            rewards[i] += sum(self.heuristic_value(state))

        return states, actions, rewards
    
    def heuristic_value(self, state):
        if len(state) == 2:
            state = state[0]
        
        reverse = list(reversed((range(len(state)))))
        heights = [0 for _ in range(len(state[0]))]
        covered_cells = 0
        for y, row in enumerate(state):
            for x, cell in enumerate(row):
                
                if not heights[x] and cell:
                    heights[x] = reverse[y] + 1
                
                if not cell:
                    if y > 0 and state[y-1][x] != 0:
                        covered_cells += 1
                        
        evenness = sum([i for i in [abs(heights[-1]-heights[j]) for j in range(1, len(heights))]])
        
        return [-covered_cells, -evenness]
    
    def actionName(self, action):
        attrs = [a for a in dir(self.actions) if not a.startswith('__')]
        for attr in attrs:
            value = self.actions.__getattribute__(self.actions, attr)
            if isinstance(value, int) and value == action:
                return attr
