#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


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
class Colony:

    size: int
    """
    pop size
    """

    cities: List
    """
    Cities of the pop
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
    def __post_init__(self) -> None:
        self.ants = [self._get_fresh_ant() for _ in range(self.size)]
        self.found_trails = []

    def _get_fresh_ant(self) -> Ant:
        """
        Get a new ant. Used to init and to replace ants that have visited all cities.

        Returns:
            Ant: A new individual ant.
        """
        origin = np.random.choice(self.cities)
        ant = Ant(origin=origin,
                  going=np.random.choice(
                      [city for city in self.cities if city != origin]),
                  pheromones=np.random.randint(10, 100),
                  visited_cities=[origin.name])
        return ant

    def _get_next_city(self, origin: str, coming: str, G) -> str:
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
        neighbors = list(G.neighbors(origin.name))
        # Can't go back to previous city.
        neighbors.remove(coming)

        # Chance to pick a random city in the neighbors of the actual position
        random_number = np.random.uniform(0, 1)
        if random_number <= self.random_coefficient:
            choice = np.random.choice(neighbors)
            destination = [
                city for city in self.cities if city.name == choice
            ][0]
            return destination

        # Proba to visit neigbors due to distance
        distances = [
            G[origin.name][neighbor]["distance"] for neighbor in neighbors
        ]
        normalized_distances = [float(i) / sum(distances) for i in distances]
        proba_distances = [1 - (x * self.beta) for x in normalized_distances]

        # If no pheromones, all have equal chances
        pheromones = [
            G[origin.name][neighbor]["pheromones"] for neighbor in neighbors
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
        destination = [
            city for city in self.cities if city.name == destination
        ][0]

        return destination

    def _evaporate_pheromones(self, G) -> None:
        """
        Evaporate pheromones regarding decay factor.
        """
        for edge in G.edges:
            edge_pheromones = G[edge[0]][edge[1]]["pheromones"]
            G[edge[0]][edge[1]]["pheromones"] = round(
                edge_pheromones - (edge_pheromones * self.evaporation), 1)
        return G

    def localise_ants(self) -> List[Tuple[str, str]]:
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
                    travels.append((ant.origin.name, ant.going.name))
                    break
        return travels

    def migrate(self, G) -> None:
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
            G[ant.origin.name][ant.going.name]["pheromones"] += ant.pheromones

            # Update trail between cities (larger on the plot).
            G[ant.origin.name][ant.going.name]["travels"] += 1

            # Increase travelled distance for the ant.
            ant.travelled += G[ant.origin.name][ant.going.name]["distance"]

            # Update origin after the travel and keep track of visits.
            ant.origin = ant.going
            ant.visited_cities.append(ant.origin.name)

            # Choose its next destination.
            ant.going = self._get_next_city(origin=ant.origin,
                                            coming=ant.visited_cities[-2],
                                            G=G)

        # Now all have moved, let's evaporate a bit of pheromones.
        G = self._evaporate_pheromones(G)

        return G
