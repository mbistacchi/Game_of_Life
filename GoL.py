import pygame as pg
import numpy as np
from scipy.signal import correlate2d
import time
from multiprocessing import Process, cpu_count, Queue

SQUARE_SIZE = 10 # cell square side length
MARGIN = 1
SQUARES = 50 # Total squares = SQUARES**2
FPS = 10
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
        """ Converts index of square (nice human readable graph coords) to pixel coords for pygame to use"""
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
        """ Input: function which gives an array of the sum of all neighbours for each cell.
        Could also multiprocess?
        """
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
        """ Uses 2D convolution (from scipy) across the entire grid to work out the neighbour sum at each cell """
        kernel = np.array([
                            [1,1,1],
                            [1,0,1],
                            [1,1,1]],
                            dtype=int)
        neighbour_sum_grid = correlate2d(self.grid.cells, kernel, mode='same')
        return neighbour_sum_grid

    def loop_method(self, partition=None):
        """ Also works out neighbour sum for each cell, using a more naive loop method """
        if partition is None:
            cells = self.grid.cells # no multithreading, just work on entire grid
        else:
            cells = partition # just work on a set section of the grid

        neighbour_sum_grid = np.zeros_like(cells) # copy
        for i, row in enumerate(cells):
            for j, cell_val in enumerate(row):
                neighbours = cells[i-1:i+2, j-1:j+2]
                neighbour_sum = np.sum(neighbours) - cell_val
                neighbour_sum_grid[i,j] = neighbour_sum
        return neighbour_sum_grid

    def multi_loop_method(self):
        """ Use Python multiprocessing to somewhat parallelize the loop_method """
        cores = cpu_count()
        procs = []
        slices = []
        if cores > 1:
            nth_grid_point = int(SQUARES / cores)
            slices.append(self.grid.cells[0:nth_grid_point])
            slices.append(self.grid.cells[nth_grid_point:])
        else:
            raise Exception("Need more than one core for multiprocessing!")

        for sl in slices:
            proc = Process(target=self.loop_method, args=(sl,))
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()


class Game:
    """ Handles pygame events, pausing etc; along with timing the code """
    def __init__(self, size, method_num):
        self.done = False
        self.paused = False
        self.gol = GoL(size)
        self.time_list = []
        self.method_num = method_num
        self.methods = {1: self.gol.conv_method, 2: self.gol.loop_method, 3: self.gol.multi_loop_method}

    def run(self):
        self.event_handler()
        self.gol.grid.display()
        if not self.paused:
            t0 = time.time()
            self.gol.evolve(self.methods[self.method_num])
            t1 = time.time()
            self.time_list.append(t1 - t0)

    def time_analysis(self):
        print("Mean time per update: {}".format(np.mean(self.time_list)))
        print("Standard deviation of time to update: {}".format(np.std(self.time_list)))

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


""" -------------------- Main ---------------------------------- """
w = h = 2 + (SQUARE_SIZE+MARGIN) * SQUARES # pixels
screen = pg.display.set_mode([w, h]) # pygame does not like this within main()

def main():
    pg.init()
    clock = pg.time.Clock()
    screen.fill(COLOUR_MAP["background"])
    method = 3 #int(input("1 for convolution, 2 for loop, 3 for multiprocessing loop")) #why tf is this not working? Environment issues?
    game = Game(SQUARES, method)
    while not game.done:
        game.run()
        pg.display.update()
        clock.tick(FPS)
    game.time_analysis()
    pg.quit()

if __name__ == "__main__":
    main()