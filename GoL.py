import pygame as pg
import numpy as np
from scipy.signal import correlate2d

COLOUR_MAP = {"alive": (255, 20, 20), "dead": (20,15,0), "background": (100, 100, 100)}

class Grid:
    """ Handles Displaying and the underlying numpy array"""
    def __init__(self, size):
        self.size = size
        self.cells = np.zeros((size, size), dtype=int) # 2D array; all dead by default
        self.cells[4:7,5] = 1 # demo "blinker"

    def display(self):
        for i, row in enumerate(self.cells):
            for j, cell in enumerate(row):
                xp, yp = self.sq_to_pixs(i, j) # (x = left side, y = top edge)
                if cell == 1:
                    colour = COLOUR_MAP["alive"]
                else:
                    colour = COLOUR_MAP["dead"]
                pg.draw.rect(screen, colour, (xp, yp, SQUARE_SIZE, SQUARE_SIZE), 0)

    def sq_to_pixs(self, x, y):
        """ Converts index of square to pixel coords for pygame to use"""
        px = (MARGIN + SQUARE_SIZE) * x + MARGIN
        py = (MARGIN + SQUARE_SIZE) * y + MARGIN
        return (px, py)

    def flip_cell(self, mouse):
        # first find the corresponding [i,j] index for the mouse position
        xi = mouse[0] // (SQUARE_SIZE + MARGIN)
        yi = mouse[1] // (SQUARE_SIZE + MARGIN)
        self.cells[xi, yi] = not self.cells[xi, yi]

class GoL:
    """ Game Engine """
    def __init__(self, size):
        self.size = size
        self.grid = Grid(size)

    def evolve(self, neigbour_sum_func):
        new_grid = np.zeros_like(self.grid.cells) # start with everything dead, only need to test for keeping/turning alive
        neighbour_sum_array = neigbour_sum_func()
        for i in range(self.size):
            for j in range(self.size):
                cell_sum = neighbour_sum_array[i,j]
                if self.grid.cells[i,j]: # already alive
                    if cell_sum == 2 or cell_sum == 3:
                        new_grid[i,j] = 1
                else: # test for dead coming alive
                    if cell_sum == 3:
                        new_grid[i,j] = 1

        self.grid.cells = new_grid

    def conv_method(self):
        """ Uses 2D convolution across the entire grid to work out the neighbour sum at each cell """
        kernel = np.array(  [1,1,1],
                            [1,0,1],
                            [1,1,1] )
        neighbour_sum_grid = correlate2d(self.grid.cells, kernel, mode='same')
        return neighbour_sum_grid

    def loop_method(self):
        neighbour_sum_grid = np.zeros_like(self.grid.cells) # copy
        for i, row in enumerate(self.grid.cells):
            for j, cell_val in enumerate(row):
                neighbours = self.grid.cells[i-1:i+2, j-1:j+2]
                neighbour_sum = np.sum(neighbours) - cell_val
                neighbour_sum_grid[i,j] = neighbour_sum
        return neighbour_sum_grid


class Game:
    """ Handles pygame events, pausing etc """
    def __init__(self, size):
        self.done = False
        self.paused = False
        self.gol = GoL(size)

    def run(self):
        self.event_handler()
        self.gol.grid.display()
        if not self.paused:
            self.gol.evolve(self.gol.loop_method)
            

    def event_handler(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 3: # right click
                    self.paused = not self.paused
                elif event.button == 1: # left click
                    mouse = pg.mouse.get_pos()
                    self.gol.grid.flip_cell(mouse)


""" Main """
SQUARE_SIZE = 10 # cell square side length
MARGIN = 1
FPS = 10
w, h = 1000, 1000 # pixel coords

pg.init()
clock = pg.time.Clock()
screen = pg.display.set_mode([w, h])
screen.fill(COLOUR_MAP["background"])
game = Game(100)
while not game.done:
    game.run()
    pg.display.update()
    clock.tick(FPS)
pg.quit()
    