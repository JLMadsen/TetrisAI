"""
STATE = GAME STATE
OBSERVATION = OBSERVATION

NB! MARIUS, STATE != OBSERVATION

"""

import math
import random
import pygame as pg
import copy

from actions import Action
from shapes import Shape
from colors import Color
from piece import Piece

class Tetris():

    def __init__(self):
        
        self.cell_size = 25
        self.margin_top = 40
        self.margin_left = 40

        self.window_height = self.window_width = 600
        
        self.game_rows = 20
        self.game_columns = 10
        
        pg.init()
        pg.display.set_caption('TETRIS')

        self.screen = pg.display.set_mode((self.window_height, self.window_width))
        self.clock = pg.time.Clock()
        self.screen.fill(Color.BLACK)

        self.start_position = [3, 0]
        self.position = copy.deepcopy(self.start_position)

        self.current_piece = 2
        self.current_rotation = 1
        self.current_shape = self.get_blocks_from_shape(Shape.ALL[self.current_piece][self.current_rotation], self.start_position)
        print('curr',self.current_shape)

    def get_blocks_from_shape(self, shape, offset=[0, 0]):
        blocks = []

        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell != '0':
                    blocks.append([j, i])

        # normalize
        lower_y = min([y for y, x in blocks])
        lower_x = min([x for y, x in blocks])

        return [[y-lower_y+offset[0], x-lower_x+offset[1]] for y, x in blocks]

    # for current state
    def check_collision_down(self):

        cells_under = None

        try:
            cells_under = [self.state[y+1][x] for y, x in self.current_shape]
        except:
            return True

        for cell in cells_under:
            if cell == 1:
                return True
        
        return False

    def next_shape(self):
        return Shape.ALL[random.randint(0, len(Shape.ALL)-1)]

    def validate_next_position(self, nxt):
        pass

    def step(self, action):
        
        old_state = self.state
        new_state = None

        reward = 0
        done = False
        info = ''

        # for checking
        next_position = copy.deepcopy(self.current_shape)

        if action == Action.DOWN:

            collision = self.check_collision_down()
            
            if not collision:
                next_position = [[y+1, x] for y, x in next_position]
            else:
                done = True

        elif action == Action.LEFT:
            next_position = [[y, x-1] for y, x in next_position]

        elif action == Action.RIGHT:
            next_position = [[y, x+1] for y, x in next_position]

        elif action == Action.ROTATE:
            self.current_rotation = (self.current_rotation + 1) % len(Shape.ALL[self.current_piece])
            new_rotation = Shape.ALL[self.current_piece][self.current_rotation]
            next_position = self.get_blocks_from_shape(new_rotation, self.current_shape[0])

        elif action == Action.WAIT:
            
            idk = 1

        self.current_shape = next_position

        return new_state, reward, done, info

    def reset(self):

        self.state = [[0 for _ in range(self.game_columns)] for _ in range(self.game_rows)]
        self.state[1][0] = 1

        return self.state

    def render(self):
        
        for i, row in enumerate(self.state):
            for j, cell in enumerate(row):

                color = Color.BLACK if cell == 0 else Color.GREEN

                rect = pg.Rect(self.margin_left + j * self.cell_size, 
                               self.margin_top + i * self.cell_size, 
                               self.cell_size, 
                               self.cell_size)

                pg.draw.rect(self.screen, color, rect, 0)
                pg.draw.rect(self.screen, Color.GRAY, rect, 1)

        for block in self.current_shape:

            rect = pg.Rect(self.margin_left + block[1] * self.cell_size, 
                            self.margin_top + block[0] * self.cell_size, 
                            self.cell_size, 
                            self.cell_size)

            pg.draw.rect(self.screen, Shape.COLORS[4], rect, 0)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

        pg.display.update()

if __name__ == "__main__":
    from play import main
    main()