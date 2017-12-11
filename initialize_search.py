import grid_search
import poisson_disc_search

# # Generating samples in the file for mean and variance calcualtion.
grid_search_count = 10

for i in range(grid_search_count):
    grid_search.initilize_grid_search()

# Poisson Disc Search count
disc_search_count = 10

for i in range(disc_search_count):
    poisson_disc_search.initilize_poisson_disc_search()
