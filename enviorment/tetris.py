import math
import random
import pygame as pg

from actions import Action
from shapes import Shape
from colors import Color
from piece import Piece

Piece(Shape.T[0])

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

        self.current_shape = Shape.T[0]
        self.position = [3, 0]

    def check_collision_down(self):
        for i, row in enumerate(self.current_shape):
            for j, cell in enumerate(row):

                next_cell = self.state[i+self.position[0]][j+self.position[1]]

                if next_cell != 0:
                    return False

        return True




    def next_shape(self):
        return Shape.ALL[random.randint(0, len(Shape.ALL))]

    def step(self, action):
        
        old_state = self.state
        new_state = None

        reward = 0
        done = False
        info = ''

        if action == Action.DOWN:
            print((valid := self.check_collision_down()))

            if(valid):
                self.position[1] += 1

            return

        elif action == Action.LEFT:
            self.position[0] -= 1
            self.position[1] += 1
            return

        elif action == Action.RIGHT:
            self.position[0] += 1
            self.position[1] += 1
            return

        elif action == Action.ROTATE:
            self.position[1] += 1
            return

        elif action == Action.WAIT:
            self.position[1] += 1
            return

        return new_state, reward, done, info

    def reset(self):

        self.state = [[0 for _ in range(self.game_columns)] for _ in range(self.game_rows)]
        self.state[8][8] = 1

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

        for i, row in enumerate(self.current_shape):
            for j, cell in enumerate(row):

                if cell == '0':
                    continue

                rect = pg.Rect(self.margin_left + (j + self.position[0]) * self.cell_size, 
                               self.margin_top + (i + self.position[1]) * self.cell_size, 
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