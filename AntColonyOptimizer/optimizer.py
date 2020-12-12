#!/usr/bin/env python3

from operator import itemgetter
from typing import List

from tqdm import tqdm

from ants import Ant, Colony
from country import Country
from tools import (create_gif, draw, parse_arguments,
                   reconstruct_roads_from_path)

# Parse keyboard arguments.
arguments = parse_arguments()

# Create a country with fake GPS coordinates.
france = Country(number_of_cities=arguments.cities)

# Creat an ant colony with given attributes.
colony: List[Ant] = Colony(size=arguments.size,
                           cities=france.cities,
                           evaporation=arguments.evaporation,
                           alpha=arguments.alpha,
                           beta=arguments.beta,
                           random_coefficient=arguments.random)

# Populate the country with this population.
france.populate(population=colony)

# Iterrate over ants moves.
for _ in tqdm(range(arguments.iterrations)):

    # Migrate all ants from origin to destination
    france.G = colony.migrate(france.G)

    # Draw an intermediate PNG to create the final GIF.
    if arguments.draw:
        ants_positions = colony.localise_ants()
        draw(G=france.G,
             ants_positions=ants_positions,
             title=f"Batch {_ + 1}/{arguments.iterrations}",
             file_name=f"{_ + 1}.png")

# Extract best path among all cities and its distance.
best_distance, best_trail = sorted(colony.found_trails, key=itemgetter(0))[0]
output = f"Best path ({len(colony.found_trails)} found)  ::  {' > '.join(best_trail)}  ::  {best_distance}km"
print(output)

# Create the final GIF with the final image showing best found path.
if arguments.draw:
    best_trail_edges = reconstruct_roads_from_path(best_trail)
    draw(G=france.G,
         best_trail_edges=best_trail_edges,
         title=output,
         file_name=f"{_ + 2}.png")
    create_gif()
