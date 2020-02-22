import pygame as pg
import numpy as np
from scipy.signal import correlate2d
import time
from multiprocessing import Process, cpu_count, Queue

SQUARE_SIZE = 10 # cell square side length
MARGIN = 1
SQUARES = 80 # Total squares = SQUARES**2
FPS = 20
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

    """ Neighbour sum grid calculation methods """
    def conv_method(self):
        """ Uses 2D convolution (from scipy) across the entire grid to work out the neighbour sum at each cell """
        kernel = np.array([
                            [1,1,1],
                            [1,0,1],
                            [1,1,1]],
                            dtype=int)
        neighbour_sum_grid = correlate2d(self.grid.cells, kernel, mode='same')
        return neighbour_sum_grid

    def loop_method(self):
        """ Did have this and multi_method_worker integrated with control logic to differentiate,
        but that logic seemed to slow down this version
        """
        cells = self.grid.cells
        neighbour_sum_grid = np.zeros_like(cells)
        for i, row in enumerate(cells):
            for j, cell_val in enumerate(row):
                neighbours = cells[i-1:i+2, j-1:j+2]
                neighbour_sum = np.sum(neighbours) - cell_val
                neighbour_sum_grid[i,j] = neighbour_sum
        return neighbour_sum_grid

    def multi_method_worker(self, partition):
        """ partition here is a tuple, the first index being a number for which sector it is
         i.e. 0 is top partition
         """
        cores = cpu_count()
        if partition[0] == 0:
            cells = partition[1][:-1] # strip bottom of partition only
        elif partition[0] == cores - 1:
            cells = partition[1][1:] # strip top of partition only
        else:
            cells = partition[1][1:-1] # strip both top and bottom of partition

        neighbour_sum_grid = np.zeros_like(cells)
        for i, row in enumerate(cells):
            for j, cell_val in enumerate(row):
                neighbours = partition[i-1:i+2, j-1:j+2]
                neighbour_sum = np.sum(neighbours) - cell_val
                neighbour_sum_grid[i,j] = neighbour_sum

        self.queue.put(neighbour_sum_grid)

    def multi_loop_method(self):
        """ Use Python multiprocessing """
        neighbour_sum_grid = []
        self.queue = Queue()
        cores = cpu_count()
        if cores > 1:
            # create row slices/partitions of the grid
            partitions = [] # list of tuples (core index, actual cells)
            nth_point = int(SQUARES / cores)
            for c in range(cores):
                start = c * nth_point
                end = (c+1) * nth_point
                if c == cores - 1:
                    partitions.append((c, self.grid.cells[start-1:])) # final slice, just go to the end. -1 to get above neighbours
                elif c == 0:
                    partitions.append((c, self.grid.cells[0:end+1])) # catches indexing (0-1) = -1
                else:
                    partitions.append((c, self.grid.cells[start-1:end+1])) # +1 to get below neighbours
        else:
            raise Exception("Need more than one core for multiprocessing!")

        procs = []
        for part in partitions:
            proc = Process(target=self.multi_method_worker, args=(part,))
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()
            grid_section = self.queue.get()
            neighbour_sum_grid.append(grid_section)

        return np.asarray(neighbour_sum_grid)
            

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
    pg.display.quit() # helps kill windows when code fails
    pg.quit()

if __name__ == "__main__":
    main()