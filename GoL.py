import pygame as pg
import numpy as np
from scipy.signal import correlate2d

COLOUR_MAP = {"alive": (255, 20, 20), "dead": (20,15,0), "background": (100, 100, 100)}
SQUARE_SIZE = 20 # cell square side length
MARGIN = 2


class Grid:
    """ Handles Displaying """
    def __init__(self, size):
        self.size = size
        self.cells = np.zeros((size, size), dtype=int) # 2D array; all dead by default
        self.cells[4:7,5] = 1

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
        px = (x+1)*(2*MARGIN + SQUARE_SIZE) - MARGIN - SQUARE_SIZE
        py = (y+1)*(2*MARGIN + SQUARE_SIZE) - MARGIN
        return (px, py)


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
                

w, h = 600, 600 # pixel coords
""" Main """
pg.init()
clock = pg.time.Clock()
FPS = 20
screen = pg.display.set_mode([w, h])
screen.fill(COLOUR_MAP["background"])
gol = GoL(20)
while True:
    clock.tick(FPS)
    gol.evolve(gol.loop_method)
    gol.grid.display()
    pg.display.update()
pg.quit()
    