# Game_of_Life
Implementation using pygame for displaying.

The code to calculate the sum of a cell's neighbours has multiple options to calculate: a 2D convolution, a looping method, and a multiprocessed version of the looping method. At program close it will print the mean and standard deviation of how long it took this code block to run.

The multiprocessing is just for learning purposes - unrealistic to expect speed increases on the default scales of the game. It is currently unstable after more than a couple of game update cycles; producing seemingly random, glitched behaviour.
