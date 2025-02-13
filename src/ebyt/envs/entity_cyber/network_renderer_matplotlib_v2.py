import math
import statistics
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from yawning_titan.networks.network import Network
from yawning_titan.networks.node import Node

import numpy as np

# network renderer , adapted from Yawning Titan base implementation

def repeat_check(node: Dict, legend_list: List[Line2D]):
    """
    Checks if a node already exists by comparing the nodes colour and description with nodes already in the legend.

    Args:
        node: A node dict.
        legend_list: The legend list.

    Returns:
        ``True`` if is already exists, otherwise ``False``.
    """
    for legend in legend_list:
        if (
            legend.get_markerfacecolor() == node['colour']
            and legend.get_label() == node['description']
        ):
            return True
    return False


def render(
    current_step: int,
    g: Network,
    attacked_nodes: List[List[Node]],
    current_time_step_reward: float,
    made_safe_nodes: list,
    show_only_blue_view: bool = False,
    target_node: Node = None,
    show_node_names: bool = False,
):
    fig = plt.figure(figsize=(12, 6))
    vis_ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=1)

    special_node_info = {
        'high_value_node': {
            'description': 'high value node',
            'colour': '#da2fed',
        }
    }

    # Creates a list that contains the details for the legend
    legend_objects = [
        # Each item in the legend is an marker with a colour and a description
        # Compromised Nodes
        Line2D(
            [0],
            [0],
            color='white',
            marker='o',
            markerfacecolor='orange',
            label='Compromised Node',
            markersize=15,
        ),
        # Vulnerable safe nodes
        Line2D(
            [0],
            [0],
            color='white',
            marker='o',
            markerfacecolor='#00FF13',
            label='Safe Node: Weak',
            markersize=15,
        ),
        # Safe nodes with low vulnerability
        Line2D(
            [0],
            [0],
            color='white',
            marker='o',
            markerfacecolor='#006007',
            label='Safe Node: Strong',
            markersize=15,
        ),
        # Nodes that have just been taken over by red
        Line2D(
            [0],
            [0],
            color='white',
            marker='o',
            markerfacecolor='red',
            label='Attacked Node',
            markersize=15,
        ),
        # The nodes that blue has "patched" or "fixed" this turn
        Line2D(
            [0],
            [0],
            color='white',
            marker='o',
            markerfacecolor='#4ef2e7',
            label='Blue Patch',
            markersize=15,
        ),
    ]

    # If a target node is specified add to the legend
    if target_node is not None:
        legend_objects.append(
            Line2D(
                [0],
                [0],
                color='white',
                marker='o',
                markerfacecolor='#2c195e',
                label='Target Node',
                markersize=15,
            )
        )
        # plots the target node
        plt.scatter(
            [target_node.x_pos],
            [target_node.y_pos],
            color='#2c195e',
            s=150,
            zorder=8,
        )

    legend_objects.extend(
        [
            # An edge that red has attacked along this turn
            Line2D(
                [0],
                [0],
                color='red',
                marker='_',
                markerfacecolor='red',
                label='Attack Path',
                markersize=15,
            ),
            # An edge
            Line2D(
                [0],
                [0],
                color='gray',
                marker='_',
                markerfacecolor='gray',
                label='Connection',
                markersize=15,
            ),
        ]
    )

    # If only showing the blue view then only render red nodes that blue can see
    if not show_only_blue_view:
        legend_objects.append(
            Line2D(
                [0],
                [0],
                color='white',
                marker='$\\bf{O}$',
                markerfacecolor='red',
                label='Unknown Compromise',
                markersize=12,
            )
        )
        legend_objects.append(
            Line2D(
                [0],
                [0],
                color='white',
                marker='$\\bf{O}$',
                markerfacecolor='blue',
                label='Known Compromise',
                markersize=12,
            )
        )

    # Some environments may have special custom nodes that they want to add
    if len(special_node_info) > 0:
        for node_info in special_node_info.values():
            # only insert if the legend is not in the list yet
            if not repeat_check(node_info, legend_objects):
                # Inserts the object into the legends at position 3. This is because it looks better if there are any
                # special nodes added that they are added at the some point as the other nodes in the legend
                legend_objects.insert(
                    3,
                    Line2D(
                        [0],
                        [0],
                        color='white',
                        marker='o',
                        markerfacecolor=node_info['colour'],
                        label=node_info['description'],
                        markersize=15,
                    ),
                )

    # If entrance nodes are used then they are added to the legend
    if g.entry_nodes:
        legend_objects.append(
            Line2D(
                [0],
                [0],
                color='white',
                marker='$\\bf{E}$',
                markerfacecolor='black',
                label='Entry Node',
                markersize=12,
            )
        )

    # plots all of the edges in the graph
    for edge in g.edges:
        plt.plot(
            [edge[0].x_pos, edge[1].x_pos],
            [edge[0].y_pos, edge[1].y_pos],
            color='grey',
            zorder=1,
        )

    # plots all of the current turns attacks
    red_nodes_x = []
    red_nodes_y = []
    for node_set in attacked_nodes:
        red_nodes_x.append(node_set[1].x_pos)
        red_nodes_y.append(node_set[1].y_pos)
        if node_set[0] is not None:
            plt.plot(
                [node_set[0].x_pos, node_set[1].x_pos],
                [node_set[0].y_pos, node_set[1].y_pos],
                color='red',
                zorder=2,
            )

    # All the shades of green for the different levels of vulnerability
    green_shades = [
        '#00FF13',
        '#00DF11',
        '#00BF0E',
        '#009F0C',
        '#00800A',
        '#006007',
    ]
    max_x = 0
    max_y = 0
    min_x = 100000
    min_y = 100000
    comp_x = []
    comp_y = []
    known_comp_x = []
    known_comp_y = []
    unknown_comp_x = []
    unknown_comp_y = []
    safe_x = []
    safe_y = []
    safe_colours = []
    void_x = []
    void_y = []
    special_x = []
    special_y = []
    special_colour = []
    made_safe_x = []
    made_safe_y = []
    for n in g.get_nodes():
        max_x = max(max_x, n.x_pos)
        max_y = max(max_y, n.y_pos)
        min_x = min(min_x, n.x_pos)
        min_y = min(min_y, n.y_pos)
        if n in made_safe_nodes:
            # get the locations of nodes that have been made safe
            made_safe_x.append(n.x_pos)
            made_safe_y.append(n.y_pos)
        elif n.high_value_node:
            # get the locations of special nodes
            special_x.append(n.x_pos)
            special_y.append(n.y_pos)
            special_colour.append(special_node_info['high_value_node']['colour'])
        elif n.true_compromised_status == 1:
            if n.blue_knows_intrusion:
                # get the locations of compromised nodes (unknown)
                comp_x.append(n.x_pos)
                comp_y.append(n.y_pos)
                if not show_only_blue_view:
                    # get the locations of compromised nodes (known)
                    known_comp_x.append(n.x_pos)
                    known_comp_y.append(n.y_pos)
            else:
                if not show_only_blue_view:
                    comp_x.append(n.x_pos)
                    comp_y.append(n.y_pos)
                    unknown_comp_x.append(n.x_pos)
                    unknown_comp_y.append(n.y_pos)
                else:
                    # get the location of the safe nodes
                    vuln = n.vulnerability_score
                    index = 5 - math.floor(vuln * (len(green_shades) - 1))
                    safe_colours.append(green_shades[index])
                    safe_x.append(n.x_pos)
                    safe_y.append(n.y_pos)
        elif n.true_compromised_status == 0:
            # get the locations of safe nodes
            vuln = n.vulnerability_score
            index = 5 - math.floor(vuln * (len(green_shades) - 1))
            safe_colours.append(green_shades[index])
            safe_x.append(n.x_pos)
            safe_y.append(n.y_pos)
        else:
            void_x.append(n.x_pos)
            void_y.append(n.y_pos)

    # plots any nodes that have no features
    plt.scatter(void_x, void_y, color='grey', s=120, zorder=1)
    # plots all of the compromised nodes
    plt.scatter(comp_x, comp_y, color='orange', s=150, zorder=8)
    # plot the circles around unknown compromised nodes
    plt.scatter(unknown_comp_x, unknown_comp_y, color='red', s=180, zorder=7)
    # plot the circles around known compromised nodes
    plt.scatter(known_comp_x, known_comp_y, color='blue', s=180, zorder=7)
    # plots all of the safe nodes
    plt.scatter(safe_x, safe_y, color=safe_colours, s=150, zorder=5)
    # plots all of the recently taken red nodes
    plt.scatter(red_nodes_x, red_nodes_y, color='red', s=150, zorder=9)
    # plots all of the nodes that have just been patched
    plt.scatter(made_safe_x, made_safe_y, color='#4ef2e7', s=150, zorder=10)
    # # plots any special nodes for the env
    plt.scatter(special_x, special_y, color=special_colour, s=150, zorder=6)
    # plot the entrance nodes
    for node in g.entry_nodes:
        plt.scatter(
            [node.x_pos],
            [node.y_pos],
            color='black',
            zorder=11,
            s=30,
            marker='$E$',
       )
    if show_node_names:
        for node in g.nodes:
            plt.text(
                node.x_pos + 0.1,
                node.y_pos + 0.1,
                node,
                color='red',
                fontsize=12,
                zorder=11,
            )

    # Creates a string containing information about the current state of the network
    info = (
        'Current Step: '
        + str(current_step)
        + '\nReward for current time step: '
        + str(current_time_step_reward)
        + '\nCurrent Avg vulnerability: '
        + str(round(statistics.mean([n.vulnerability_score for n in g.nodes]), 2))
    )
    ax = plt.gca()
    ax.legend(
        handles=legend_objects,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        borderpad=1,
        labelspacing=1,
        fontsize=10,
        edgecolor='black',
    )
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.axes.set_xlim(min_x - 0.1 * max_x, max_x * 1.1)
    ax.axes.set_ylim(min_y - 0.1 * max_y, max_y * 1.1)

    ax.set_xlabel(info)
    for pos in ['left', 'right', 'top', 'bottom']:
        plt.gca().spines[pos].set_visible(False)
    # Make sure everything fits on the canvas
    vis_ax.figure.tight_layout()
    # Update the canvas
    vis_ax.figure.canvas.draw()

    # Capture the image from the plot
    image_from_plot = np.frombuffer(vis_ax.figure.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        vis_ax.figure.canvas.get_width_height()[::-1] + (3,)
    )

    # Close the figure to free up memory
    plt.close(fig)

    return image_from_plot
