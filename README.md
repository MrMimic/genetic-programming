# genetic-programming


## Ant Colony Optimisation Python implmentation

See [Wikipedia](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms) for details about the algorithm.

Thge basic idea is to follow the behavior of an ant colony to find the shortest path linking all cities in a country (represented as a graph).

Quick to solve (not when creating the GIF output).

### Running
	
`python3 -m venv`

`pip install -r requirements`

`python3 AntColonyOptimizer/ants_colony_optimizer.py`

It will run the generation of the GIF output with a limited country of 8 cities populated with 3 ants for nice GIF output.

Set parameter DRAW to False in __main__() of the script and try with much higher values (speed x3300).

Find optimal path no matter the size of the graph with a log of ants.

### Output

![output_gif_ex](AntColonyOptimizer/output/20201211_194120.gif)

### To do

  - Early stopping when not converging anymore
  - Arguments in command call instead of in __main__
  - Optimise code (re compilation eg)
  - Parallelise ants as Threads
