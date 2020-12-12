#!/usr/bin/env python3

import itertools
from dataclasses import dataclass
from typing import List, Tuple

import geopy.distance
import networkx as nx
import numpy as np

from ants import Ant


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
        self.cities_list = [city.name for city in self.cities]
        # And roads connecting cities as edges (in both ways).
        for edge in list(itertools.permutations(self.cities_list, 2)):
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

    def populate(self, population: List[Ant]) -> None:
        """
        Populate the country with fresh ants in random localisations.

        Args:
            number_of_ants (int): The wished number of ants in the country.
        """
        self.population: List[Ant] = population
