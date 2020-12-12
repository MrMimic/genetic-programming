#!/usr/bin/env python3

import itertools
import os
import re
from dataclasses import dataclass
from datetime import datetime
from operator import itemgetter
from typing import List, Tuple

import geopy.distance
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm


@dataclass
class City:
    """
    Cities are places to explore. The goal for the ants will be to quickly find
    the shortest path linking all cities together.
    Their coordinate are given as real-world GPS points.
    """

    pos: tuple
    """
    The position (latitude, longitude) of the city.
    """

    latitude: float
    """
    Latitude in degrees.
    """

    longitude: float
    """
    Longitude in degrees.
    """

    population: int
    """
    Just to make it more real.
    """

    name: str
    """
    Name of the city.
    """

    is_visited: bool
    """
    If an ant is actually in this city (to color intermediate plots).
    """


@dataclass
class Ant:
    """
    Ants are attracted by both:
    - Paths highly saturated with pheromones (indicating that a previous ant
    have followed this path)
    - Shortest possible path from one city to its neighbors.
    """

    origin: str
    """
    Where the ant is coming from (city).
    """

    going: str
    """
    Where the ant is going (city).
    """

    pheromones: int
    """
    Ant ability to drop pheromones on its way (10 to 100).
    """

    visited_cities: List[str]
    """
    Cities visited by the ant (ordered).
    """

    travelled: int = 0
    """
    Ant total distance.
    """


@dataclass
class Country:
    """
    An imaginary country will be designed at __init__ with a given number_of_cities.

    Returns:
        [type]: [description]
    """

    number_of_cities: int = 10
    """
    Number of cities to design in the country.
    """

    evaporation: float = 0.3
    """
    Pheromones are volatile component. It should decay over time.
    """

    alpha: float = 0.4
    """
    How likely an ant will follow a path highly saturated with pheromones.
    """

    beta: float = 0.6
    """
    How likely ants are going to be lazy and follow the shortest path.
    """

    random_coefficient: float = 0.3
    """
    Probability to pick a random newt direction (ignoring distance and pheromones).
    """
    def __post_init__(self):

        # Cities are generated with random GPS coords.
        self.cities = []
        for _ in range(self.number_of_cities):
            latitude = np.random.uniform(42, 50)
            longitude = np.random.uniform(-4, 7)
            self.cities.append(
                City(pos=(longitude, latitude),
                     latitude=latitude,
                     longitude=longitude,
                     population=np.random.randint(1_000_000),
                     name=f"city_{_+1}",
                     is_visited=False))

        # The country will be represented as a digraph.
        self.G = nx.DiGraph()

        # With nodes as cities
        for city in self.cities:
            self.G.add_node(city.name, **city.__dict__)

        # And roads connecting cities as edges (in both ways).
        for edge in list(
                itertools.permutations([city.name for city in self.cities],
                                       2)):
            self.G.add_edge(edge[0],
                            edge[1],
                            travels=0.1,
                            distance=self._compute_distance(edge),
                            pheromones=0)

        # Trails will be discovered by ants and are composed of list of cities,
        # ordered following ants paths
        self.found_trails: List[List[str]] = []

        # Reduce edge size while running long batches (roads widths are correlated
        # with the number of ants that took each path).
        self.edge_width_factor = 10

    def _compute_distance(self, edge: Tuple[str, str]) -> float:
        """
        Compute  distance between two GPS points (kms).

        Args:
            edge (Tuple[str, str]): The tuple of the two objects from the graph.

        Returns:
            float: Real distance in km, rounded.
        """
        city_1, city_2 = edge
        city_1_gps = (self.G.nodes[city_1]["latitude"],
                      self.G.nodes[city_1]["longitude"])
        city_2_gps = (self.G.nodes[city_2]["latitude"],
                      self.G.nodes[city_2]["longitude"])
        return round(geopy.distance.geodesic(city_1_gps, city_2_gps).km, 1)

    def _localise_ants(self) -> List[Tuple[str, str]]:
        """
        Localise ants tranvel for each batch to colorize the path they just took.

        Returns:
            List[Tuple[str, str]]: List of paths took by ants for this batch.
        """
        travels = []
        for ant in self.ants:
            for city in self.cities:
                # Also mark city for color plotting.
                if city.name == ant.origin:
                    city.is_visited = True
                    travels.append((ant.origin, ant.going))
                    break
        return travels

    @staticmethod
    def _scale_list(list_to_scale: List[float], out_range: Tuple[int, int] = (1, 20)) -> List[float]:
        """
        Rescale path widths to fit the graph.

        Args:
            list_to_scale (List[float]): List to be scaled.
            out_range (Tuple[int, int], optional): Min and max output values. Defaults to (1, 25).

        Returns:
            List[float]: Rescaled list.
        """
        domain = [np.min(list_to_scale, axis=0), np.max(list_to_scale, axis=0)]

        def wrap(x):
            return out_range[0] * (1.0 - x) + out_range[1] * x

        def unwrap(x):
            b = 0
            if (domain[1] - domain[0]) != 0:
                b = domain[1] - domain[0]
            else:
                b =  1.0 / domain[1]
            return (x - domain[0]) / b

        return wrap(unwrap(list_to_scale))

    def draw(self,
             best_trail_edges: List[Tuple[str, str]] = None,
             draw_edges_labels: bool = False,
             clean_output: bool = False,
             title: str = None,
             file_name: str = None) -> None:
        """
        Draw the plot, either during batchs (when ants are moving) or after the training
        (highliting best found path).

        Args:
            best_trail_edges (List[Tuple[str, str]], optional): Best found trail
            during training. Defaults to None.
            draw_edges_labels (bool, optional): Draw edges attributes. Defaults to False.
            clean_output (bool, optional): Clean output on Jupyter. Defaults to False.
            title (str, optional): Title of the graph. Defaults to None.
            file_name (str, optional): name of the output file. Defaults to None.
        """
        # Get ants move during last batch so if the steaming drazw is enabled,
        # last paths are colored in red.
        travels = self._localise_ants()
        if best_trail_edges is None:
            edges_colors = [
                "red" if road in travels else "grey" for road in self.G.edges
            ]
        else:
            edges_colors = ["grey"] * len(self.G.edges)

        # Road width. Increase while ants are taking them.
        weights = list(nx.get_edge_attributes(self.G, 'travels').values())
        if max(weights) > 20:
            weights = self._scale_list(weights)

        # Cities colors. Red if an ant is present in the city.
        nodes_colors = [
            "red" if x is True else "grey"
            for x in [self.G.nodes[node]["is_visited"] for node in self.G]
        ]

        fig, ax = plt.subplots(figsize=(16, 9))
        # Get pos from cities attributes (GPS coords)
        pos = nx.get_node_attributes(self.G, "pos")

        if draw_edges_labels:
            nx.draw_networkx_edge_labels(self.G, pos, ax=ax, font_size=6)
        nx.draw(self.G,
                pos,
                node_color=nodes_colors,
                width=weights,
                edge_color=edges_colors,
                with_labels=True,
                ax=ax)
        if clean_output:
            clear_output()

        # Add best trail in yellow if provided.
        if best_trail_edges:
            nx.draw_networkx_edges(self.G,
                                   pos,
                                   edgelist=list(best_trail_edges),
                                   edge_color='y',
                                   width=max(weights) / 4)

        # Customise axis, title, etc.
        limits = plt.axis("on")
        ax.tick_params(left=True,
                       bottom=True,
                       labelleft=True,
                       labelbottom=True)
        plt.ylabel("latitude")
        plt.xlabel("longitude")
        if title:
            ax.set_title(title)

        plt.savefig(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "output",
                         file_name))
        plt.close()

    def _get_fresh_ant(self) -> Ant:
        """
        Get a new ant. Used to init and to replace ants that have visited all cities.

        Returns:
            Ant: A new individual ant.
        """
        origin = np.random.choice(self.cities).name
        ant = Ant(origin=origin,
                  going=np.random.choice([
                      city for city in self.cities if city.name != origin
                  ]).name,
                  pheromones=np.random.randint(10, 100),
                  visited_cities=[origin])
        return ant

    def populate(self, number_of_ants: int) -> None:
        """
        Populate the country with fresh ants in random localisations.

        Args:
            number_of_ants (int): The wished number of ants in the country.
        """
        self.ants: List[Ant] = []
        for _ in range(number_of_ants):
            self.ants.append(self._get_fresh_ant())

    def _get_next_city(self, origin: str, coming: str) -> str:
        """
        Get the next destination of an ant regarding its actual and previous position.
        More likely to chose shortest and more saturated with pheromones path.
        If coming from A -> B, can't go back directly from B to A.

        Args:
            origin (str): Actual position of the ant.
            coming (str): Previous position of the ant.

        Returns:
            str: Name of the next city on the graph.
        """
        neighbors = list(self.G.neighbors(origin))
        # Can't go back to previous city.
        neighbors.remove(coming)

        # Chance to pick a random city in the neighbors of the actual position
        random_number = np.random.uniform(0, 1)
        if random_number <= self.random_coefficient:
            return np.random.choice(neighbors)

        # Proba to visit neigbors due to distance
        distances = [
            self.G[origin][neighbor]["distance"] for neighbor in neighbors
        ]
        normalized_distances = [float(i) / sum(distances) for i in distances]
        proba_distances = [1 - (x * self.beta) for x in normalized_distances]

        # If no pheromones, all have equal chances
        pheromones = [
            self.G[origin][neighbor]["pheromones"] for neighbor in neighbors
        ]
        normalized_pheromones = [
            float(i) / sum(pheromones) for i in pheromones
        ] if sum(pheromones) > 0 else [1.0 / len(pheromones)] * len(pheromones)
        proba_pheromones = [x * self.alpha for x in normalized_pheromones]

        # Pull proba together to pick best destination
        global_proba = [
            p_distance + p_pheromones for p_distance, p_pheromones in zip(
                proba_distances, proba_pheromones)
        ]
        destination = neighbors[global_proba.index(max(global_proba))]

        return destination

    def _evaporate_pheromones(self) -> None:
        """
        Evaporate pheromones regarding decay factor.
        """
        for edge in pays.G.edges:
            edge_pheromones = pays.G[edge[0]][edge[1]]["pheromones"]
            pays.G[edge[0]][edge[1]]["pheromones"] = round(
                edge_pheromones - (edge_pheromones * self.evaporation), 1)

    @staticmethod
    def reconstruct_roads_from_path(path: List[str]) -> List[Tuple[str, str]]:
        """
        Reconstruct a list of paths from [A, B, C] to [(A, B), (B, C)]

        Args:
            path (List[str]): The path to be reconstructed as edges.

        Returns:
            List[Tuple[str, str]]: List of graph edges.
        """
        edges_from_path = []
        for i in range(len(path) - 1):
            edges_from_path.append((path[i], path[i + 1]))
        return edges_from_path

    def migrate_ants(self) -> None:
        """
        Main function of the loop. Ants are moving, dropping pheromones
        and choose their newt destination.

        If an ant visited all cities during this phase, get the distance of this path
        and call a new ant to continue to discover other paths.
        """

        healthy_ants = []

        for ant in self.ants:

            # If the ant has visited all cities, time to thanks her.
            if len(list(set(ant.visited_cities))) == len(
                    self.cities
            ) and ant.visited_cities[0] == ant.visited_cities[-1]:
                # Stored discovered path if never seen and get a new ant.
                discovered_trail = [
                    round(ant.travelled, 2), ant.visited_cities
                ]
                if discovered_trail not in self.found_trails:
                    self.found_trails.append(discovered_trail)
                healthy_ants.append(self._get_fresh_ant())
            else:
                healthy_ants.append(ant)
        # Replace ants in the country.
        self.ants = healthy_ants

        for ant in self.ants:

            # Ants are dropping pheromones during travel.
            self.G[ant.origin][ant.going]["pheromones"] += ant.pheromones

            # Update trail between cities (larger on the plot).
            self.G[ant.origin][ant.going]["travels"] += 1

            # Increase travelled distance for the ant.
            ant.travelled += self.G[ant.origin][ant.going]["distance"]

            # Update origin after the travel and keep track of visits.
            ant.origin = ant.going
            ant.visited_cities.append(ant.origin)

            # Choose its next destination.
            ant.going = self._get_next_city(origin=ant.origin,
                                            coming=ant.visited_cities[-2])

        # Now all have moved, let's evaporate a bit of pheromones.
        self._evaporate_pheromones()


def create_gif() -> None:

    output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "output")

    # Find and sort created images.
    images = []
    created_files = sorted([
        os.path.join(output_path, file)
        for file in os.listdir(output_path) if file.endswith("png")
    ],
                           key=lambda x: float(re.findall("(\d+)", x)[0]))
    # Create GIF with more time for last frame to see final path.
    for filename in created_files:
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(
        output_path, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"),
                    images,
                    loop=0,
                    duration=[0.1] * (len(created_files) - 1) + [5])

    # Clean generated PNGs.
    for file in created_files:
        os.remove(file)


if __name__ == "__main__":

    # Create a country with a given number of cities.
    pays = Country(
        number_of_cities=8,
        alpha=0.4,
        beta=0.6)

    # Populate it with ants
    pays.populate(number_of_ants=3)

    # Wished number of iteration.
    ITERRATIONS = 30

    # Run 3,300 times faster when not creating the GIF.
    DRAW = True

    for _ in tqdm(range(ITERRATIONS)):

        pays.migrate_ants()

        if DRAW:
            pays.draw(clean_output=False,
                      title=f"Batch {_ + 1}/{ITERRATIONS}",
                      file_name=f"{_ + 1}.png")

    best_distance, best_trail = sorted(pays.found_trails, key=itemgetter(0))[0]

    best_trail_edges = pays.reconstruct_roads_from_path(best_trail)

    output = f"Best path ({len(pays.found_trails)} found)  ::  {' > '.join(best_trail)}  ::  {best_distance}km"

    if DRAW:
        pays.draw(best_trail_edges=best_trail_edges,
                  title=output,
                  file_name=f"{_ + 2}.png")
        create_gif()

    print(output)
