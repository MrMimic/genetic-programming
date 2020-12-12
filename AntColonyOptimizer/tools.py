#!/usr/bin/env python3

import argparse
import os
import re
from datetime import datetime
from typing import List, Tuple

import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw(G,
         ants_positions: List[Tuple[str, str]] = None,
         best_trail_edges: List[Tuple[str, str]] = None,
         draw_edges_labels: bool = False,
         title: str = None,
         file_name: str = None) -> None:
    """
    Draw the plot, either during batchs (when ants are moving) or after the training
    (highliting best found path).

    Args:
        ants_positions (List[Tuple[str, str]]], optional): Road taken at the moment.
        best_trail_edges (List[Tuple[str, str]], optional): Best found trail
        during training. Defaults to None.
        draw_edges_labels (bool, optional): Draw edges attributes. Defaults to False.
        title (str, optional): Title of the graph. Defaults to None.
        file_name (str, optional): name of the output file. Defaults to None.
    """
    # Get ants move during last batch so if the steaming drazw is enabled,
    # last paths are colored in red.
    if best_trail_edges is None:
        edges_colors = [
            "red" if road in ants_positions else "grey" for road in G.edges
        ]
    else:
        edges_colors = ["grey"] * len(G.edges)

    # Road width. Increase while ants are taking them.
    weights = list(nx.get_edge_attributes(G, 'travels').values())
    if max(weights) > 20:
        weights = scale_list(weights)

    # Cities colors. Red if an ant is present in the city.
    nodes_colors = [
        "red" if x is True else "grey"
        for x in [G.nodes[node]["is_visited"] for node in G]
    ]

    fig, ax = plt.subplots(figsize=(16, 9))
    # Get pos from cities attributes (GPS coords)
    pos = nx.get_node_attributes(G, "pos")

    if draw_edges_labels:
        nx.draw_networkx_edge_labels(G, pos, ax=ax, font_size=6)
    nx.draw(G,
            pos,
            node_color=nodes_colors,
            width=weights,
            edge_color=edges_colors,
            with_labels=True,
            ax=ax)

    # Add best trail in yellow if provided.
    if best_trail_edges:
        nx.draw_networkx_edges(G,
                               pos,
                               edgelist=list(best_trail_edges),
                               edge_color='y',
                               width=max(weights) / 4)

    # Customise axis, title, etc.
    limits = plt.axis("on")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.ylabel("latitude")
    plt.xlabel("longitude")
    if title:
        ax.set_title(title)

    plt.savefig(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "output",
                     file_name))
    plt.close()


def scale_list(
    list_to_scale: List[float], out_range: Tuple[int, int] = (1, 20)
) -> List[float]:
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
            b = 1.0 / domain[1]
        return (x - domain[0]) / b

    return wrap(unwrap(list_to_scale))


def parse_arguments() -> argparse.Namespace:
    """
    Parse provided arguments and create a namespace from it.

    Returns:
        argparse.Namespace: Found arguments.
    """
    parser = argparse.ArgumentParser(
        description="ACO algorithm Python implementation")

    general = parser.add_argument_group("General parameters")
    general.add_argument(
        "-i",
        "--iterrations",
        type=int,
        default=50,
        help="Maximum number of iteration to find the optimum.")
    general.add_argument(
        "-d",
        "--draw",
        action='store_true',
        help="Whether to output or not an animated GIF. Way more slower.")

    colony = parser.add_argument_group("Ant colony")
    colony.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.4,
        help=
        "How likely an ant will follow a path highly saturated with pheromones."
    )
    colony.add_argument(
        "-b",
        "--beta",
        type=float,
        default=0.6,
        help=
        "How likely ants are going to be lazy and follow the shortest path.")
    colony.add_argument(
        "-e",
        "--evaporation",
        type=float,
        default=0.3,
        help="Pheromones are volatile component. It should decay over time.")
    colony.add_argument(
        "-r",
        "--random",
        type=float,
        default=0.3,
        help=
        "Probability to pick a random newt direction (ignoring distance and pheromones)."
    )
    colony.add_argument("-s",
                        "--size",
                        type=int,
                        default=10,
                        help="How many ants should be in the colony.")

    country = parser.add_argument_group("Country to discover")
    country.add_argument("-c",
                         "--cities",
                         type=int,
                         default=10,
                         help="Number of cities to create in the country.")
    args = parser.parse_args()

    return args


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


def create_gif() -> None:
    """
    Create a GIF from intermediate PNG.
    """
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
