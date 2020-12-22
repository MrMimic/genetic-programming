# genetic-programming


## Ant Colony Optimisation Python implmentation

See [Wikipedia](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms) for details about the algorithm.

Thge basic idea is to follow the behavior of an ant colony to find the shortest path linking all cities in a country (represented as a graph). Briefly:
- A country with semi-real GPS coordinates is created
- It is populated withs ants that must find a round to go through all cities
- Ants move over iterrations
- At the end, they drop pheromones (with different ability for each ant)
- Pheromones evaporate a bit
- Ants choosing a new path (shorter and full of pheromones)

Quick to solve (not when creating the GIF output).

### Running
	
`python3 -m venv`

`pip install -r requirements`

`python3 AntColonyOptimizer/optimizer.py -h`

It will run the generation of the GIF output with a limited country of 8 cities populated with 3 ants for nice GIF output.

Find optimal path no matter the size of the graph with a log of ants.

### Help

```
  python3 AntColonyOptimizer/optimizer.py -h
  usage: optimizer.py [-h] [-i ITERRATIONS] [-d] [-a ALPHA] [-b BETA] [-e EVAPORATION] [-r RANDOM] [-s SIZE] [-c CITIES]

  ACO algorithm Python implementation

  optional arguments:
    -h, --help            show this help message and exit

  General parameters:
    -i ITERRATIONS, --iterrations ITERRATIONS
                          Maximum number of iteration to find the optimum. Default 50.
    -d, --draw            Whether to output or not an animated GIF. Way more slower.

  Ant colony:
    -a ALPHA, --alpha ALPHA
                          How likely an ant will follow a path highly saturated with pheromones. Default 0.4.
    -b BETA, --beta BETA  How likely ants are going to be lazy and follow the shortest path. Default 0.6.
    -e EVAPORATION, --evaporation EVAPORATION
                          Pheromones are volatile component. It should decay over time. Default 0.3.
    -r RANDOM, --random RANDOM
                          Probability to pick a random newt direction (ignoring distance and pheromones). Default 0.3.
    -s SIZE, --size SIZE  How many ants should be in the colony. Default 10.

  Country to discover:
    -c CITIES, --cities CITIES
                          Number of cities to create in the country. Default 10.
```
### Output

![output_gif_ex](AntColonyOptimizer/output/20201211_194120.gif)

### To do

  - Early stopping when not converging anymore
  - Arguments in command call instead of in __main__
  - Optimise code (re compilation eg)
  - Parallelise ants as Threads
