import grid_search
import poisson_disc_search

# Generating samples in the file for mean and variance calcualtion.
grid_search_count = 1

for i in range(grid_search_count):
    grid_search.initilize_grid_search(plot = True, verbose = False, use_logspace = True)

# Poisson Disc Search count
disc_search_count = 0

for i in range(disc_search_count):
    poisson_disc_search.initilize_poisson_disc_search(plot = True, verbose = False, use_logspace = True)
