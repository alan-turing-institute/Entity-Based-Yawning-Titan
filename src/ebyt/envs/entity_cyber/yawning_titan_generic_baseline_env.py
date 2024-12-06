"""
A generic class that creates Open AI environments within YAWNING TITAN.

This class has several key inputs which determine aspects of the environment such
as how the red agent behaves, what the red team and blue team objectives are, the size
and topology of the network being defended and what data should be collected during the simulation.
"""

import copy
import json
from collections import Counter
from typing import Dict, Tuple

import gym
import numpy as np
from gym import spaces
from stable_baselines3.common.utils import set_random_seed

import yawning_titan.envs.generic.core.reward_functions as reward_functions
from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.envs.generic.core.red_interface import RedInterface
from yawning_titan.envs.generic.helpers.eval_printout import EvalPrintout
from yawning_titan.envs.generic.helpers.graph2plot import CustomEnvGraph
from yawning_titan.game_modes.game_mode import GameMode

from yawning_titan.networks.network import Network
from yawning_titan.networks.network_creator import gnp_random_connected_graph

from ebyt.envs.entity_cyber.network_renderer_matplotlib_v2 import render

# Yawning Titan generic environment, adapted from base implementation. Allows for randomised network generation
class YTGenericNetworkEnv(gym.Env):
    '''
    Class to create a generic YAWNING TITAN gym environment.

        Args:
            network: The network to be used in the environment (Network)
            game_mode: The game mode to be used in the environment (GameMode)
            random: Whether or not a new random network is generated with every episode (boolean)
            random_seed: Seed to be used (int)
            n_nodes: Number of nodes in the network (int)
            edge_prob: Probability of an edge between two nodes when generating a new network (float)
            print_metrics: Whether or not to print metrics (boolean)
            show_metrics_every: Number of timesteps to show summary metrics (int)
            collect_additional_per_ts_data: Whether or not to collect additional per timestep data (boolean)
            print_per_ts_data: Whether or not to print collected per timestep data (boolean)
        Attributes:
            random_reset: Whether or not a new random network is generated with every episode (boolean)
            n_nodes: Number of nodes in the network (int)
            edge_prob: Probability of an edge between two nodes when generating a new network (float)
            game_mode: The game mode to be used in the environment (GameMode)
            network: The (initial) network to be used in the environment (Network)
            network_interface: The network interface for the environment (NetworkInterface)
            blue_interface: The blue interface for the environment (BlueInterface)
            red_agent: The red agent for the environment (RedInterface)
            current_duration: The current timestep of the current episode (int)
            game_stats_list: List of dictionaries containing game stats (list)
            num_games_since_avg: Number of games since last printing average score (int)
            avg_every: Interval in timesteps to show summary metrics (int)
            current_game_blue: Dictionary recording blue action counts for the current game (dict)
            current_game_stats: Dictionary recording game stats for the current game (dict)
            total_games: Total number of games (episodes) played (int)
            made_safe_nodes: List of nodes made safe in the current timestep (list)
            current_reward: The reward for the current timestep (float)
            print_metrics: Whether or not to print metrics (boolean)
            print_notes: Whether or not to print collected per timestep data (boolean)
            random_seed: Seed to be used (int)
            eval_printout: Object to print evaluation metrics (EvalPrintout)
            blue_actions: Number of actions available to the blue agent (int)
            action_space: The action space for the blue agent (spaces.Discrete) - of size blue_actions
            observation_space: The observation space for the environment (spaces.Box)
            collect_data: Whether or not to collect additional per timestep data (boolean)
            env_observation: The current observation of the environment (np.array)
        Methods:
            generate_random_network: Generates a random connected network (static method)
            reset: Resets the environment to the default state
            step: Takes a timestep and executes the actions for both Blue RL agent and non-learning Red agent
            render: Renders the current state of the environment
            calculate_observation_space_size: Calculates the observation space size


        Note: The ``notes`` variable returned at the end of each timestep contains the per
        timestep data. By default it contains a base level of info required for some of the
        reward functions. When ``collect_additional_per_ts_data`` is toggled on, a lot more
        data is collected.
    '''
    def __init__(
        self,
        network: Network,
        game_mode: GameMode,
        random: bool,
        random_seed: int,
        n_nodes: int,
        edge_prob: float,
        print_metrics: bool = False,
        show_metrics_every: int = 1,
        collect_additional_per_ts_data: bool = True,
        print_per_ts_data: bool = False,
    ):
        
        super().__init__()
        self.random_reset = random
        self.n_nodes = n_nodes
        self.edge_prob = edge_prob
        self.game_mode = game_mode
        if self.random_reset:
            self.network = self.generate_random_network(n=self.n_nodes, p=self.edge_prob)
        else:
            self.network = network

        self.network_interface = NetworkInterface(
            network=self.network, game_mode=self.game_mode
        )
        self.blue_interface = BlueInterface(self.network_interface)
        self.red_agent = RedInterface(self.network_interface)
    
        self.current_duration = 0
        self.game_stats_list = []
        self.num_games_since_avg = 0
        self.avg_every = show_metrics_every
        self.current_game_blue = {}
        self.current_game_stats = {}
        self.total_games = 0
        self.made_safe_nodes = []
        self.current_reward = 0
        self.print_metrics = print_metrics
        self.print_notes = print_per_ts_data

        self.random_seed = self.network_interface.random_seed

        self.eval_printout = EvalPrintout(self.avg_every)

        self.blue_actions = self.blue_interface.get_number_of_actions()
        self.action_space = spaces.Discrete(self.blue_actions)


        # sets up the observation space. This is a (n+2 by n) matrix. The first two columns show the state of all the
        # nodes. The remaining n columns show the connections between the nodes (effectively the adjacency matrix)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.network_interface.get_observation_size(),),
            dtype=np.float32,
        )

        # The gym environment can only properly deal with a 1d array so the observation is flattened

        self.collect_data = collect_additional_per_ts_data
        self.env_observation = self.network_interface.get_current_observation()

    @staticmethod
    def generate_random_network(n: int = 10, p: float = 0.1) -> Network:
        """
        Generates a connected random network, in the binomial Erdos-Renyi fashion, whilst ensuring that the network is connected.
        """
        network = gnp_random_connected_graph(n_nodes=n, probability_of_edge=p)
        network.set_random_entry_nodes = True
        network.number_of_high_value_nodes = 1

        network.num_of_random_entry_nodes = 1

        network.set_random_high_value_nodes = True

        network.num_of_random_high_value_nodes = 1

        #If True, random vulnerability is set for each node using the upper and lower bounds
        network.set_random_vulnerabilities = True
        

        return network

    def reset(self) -> np.array:
        """
        Reset the environment to the default state.

        :return: A new starting observation (numpy array).
        """
        if self.random_reset:
            self.network = self.generate_random_network(n=self.n_nodes, p=self.edge_prob)
            self.network_interface = NetworkInterface(
                network=self.network, game_mode=self.game_mode
            )
            self.blue_interface = BlueInterface(self.network_interface)
            self.red_agent = RedInterface(self.network_interface)

        if self.random_seed is not None:  # conditionally set random_seed
            set_random_seed(self.random_seed, True)
        self.network_interface.reset()
        self.red_agent.reset()
        self.current_duration = 0
        self.env_observation = self.network_interface.get_current_observation()
        self.current_game_blue = {}

        return self.env_observation

    def step(self, action: int) -> Tuple[np.array, float, bool, Dict[str, dict]]:
        """
        Take a time step and executes the actions for both Blue RL agent and non-learning Red agent.

        Args:
            action: The action value generated from the Blue RL agent (int)

        Returns:
             A four tuple containing the next observation as a numpy array,
             the reward for that timesteps, a boolean for whether complete and
             additional notes containing timestep information from the environment.
        """
        # sets the nodes that have been made safe this turn to an empty list
        self.made_safe_nodes = []

        # Gets the initial states of various states for logging and testing purposes
        if self.collect_data:
            # notes collects information about the state of the env
            notes = {
                'initial_state': self.network_interface.get_all_node_compromised_states(),
                'initial_blue_view': self.network_interface.get_all_node_blue_view_compromised_states(),
                'initial_vulnerabilities': self.network_interface.get_all_vulnerabilities(),
                'initial_red_location': copy.deepcopy(
                    self.network_interface.red_current_location
                ),
                'initial_graph': self.network_interface.get_current_graph_as_dict(),
                'current_step': self.current_duration,
            }
        else:
            # If not logging everything, the program still needs to collect some information (required by other parts
            # of the program)
            notes = {}

        # resets the attack list for the red agent (so that only the current turns attacks are held)
        self.network_interface.reset_stored_attacks()

        # The red agent performs their turn
        if (
            self.network_interface.game_mode.game_rules.grace_period_length.value
            <= self.current_duration
        ):
            red_info = self.red_agent.perform_action()
        else:
            red_info = {
                0: {
                    'Action': 'do_nothing',
                    'Attacking_Nodes': [],
                    'Target_Nodes': [],
                    'Successes': [True],
                }
            }
        # Gets the number of nodes that are safe
        number_uncompromised = len(
            self.network_interface.current_graph.get_nodes(filter_true_safe=True)
        )

        # Collects data on the natural spreading
        if self.collect_data:
            notes['red_info'] = red_info

        # The states of the nodes after red has had their turn (Used by the reward functions)
        notes['post_red_state'] = (
            self.network_interface.get_all_node_compromised_states()
        )
        # Blues view of the environment after red has had their turn
        notes['post_red_blue_view'] = (
            self.network_interface.get_all_node_blue_view_compromised_states()
        )
        # A dictionary of vulnerabilities after red has had their turn
        notes['post_red_vulnerabilities'] = (
            self.network_interface.get_all_vulnerabilities()
        )
        # The isolation status of all the nodes
        notes['post_red_isolation'] = self.network_interface.get_all_isolation()

        # collects extra data if turned on
        if self.collect_data:
            # The location of the red agent after red has had their turn
            notes['post_red_red_location'] = copy.deepcopy(
                self.network_interface.red_current_location
            )

        # set up initial variables that are reassigned based on the action that blue takes
        done = False
        reward = 0
        blue_action = ''
        blue_node = None

        # Check if the game is over and red has won
        if self.network_interface.game_mode.game_rules.blue_loss_condition.all_nodes_lost.value:
            if number_uncompromised == 0:
                done = True
                reward = self.network_interface.game_mode.rewards.for_loss.value
                blue_action = 'failed'
        if self.network_interface.game_mode.game_rules.blue_loss_condition.n_percent_nodes_lost.use.value:
            # calculate the number of safe nodes
            percent_comp = (
                len(
                    self.network_interface.current_graph.get_nodes(
                        filter_true_compromised=True
                    )
                )
                / self.network_interface.current_graph.number_of_nodes()
            )
            if (
                percent_comp
                >= self.network_interface.game_mode.game_rules.blue_loss_condition.n_percent_nodes_lost.value.value
            ):
                done = True
                reward = self.network_interface.game_mode.rewards.for_loss.value
                # If the game ends before blue has had their turn the the blue action is set to failed
                blue_action = 'failed'
        if self.network_interface.game_mode.game_rules.blue_loss_condition.high_value_node_lost.value:
            # check if a high value node was compromised
            compromised_hvn = False
            for hvn in self.network_interface.current_graph.high_value_nodes:
                if hvn.true_compromised_status == 1:
                    compromised_hvn = True
                    break

            if compromised_hvn:
                # If this mode is selected then the game ends if the high value node has been compromised
                done = True
                reward = self.network_interface.game_mode.rewards.for_loss.value
                blue_action = 'failed'

        # if self.network_interface.gr_loss_tn:
        tn = self.network_interface.get_target_node()
        if (
            tn is not None
            and self.network_interface.game_mode.game_rules.blue_loss_condition.target_node_lost.value
        ):
            if tn.true_compromised_status == 1:
                # If this mode is selected then the game ends if the target node has been compromised
                done = True
                reward = self.network_interface.game_mode.rewards.for_loss.value
                blue_action = 'failed'

        if done:
            if self.network_interface.game_mode.rewards.reduce_negative_rewards_for_closer_fails.value:
                reward = reward * (
                    1
                    - (
                        self.current_duration
                        / self.network_interface.game_mode.game_rules.max_steps.value
                    )
                )
        if not done:
            blue_action, blue_node = self.blue_interface.perform_action(action)

            if blue_action == 'make_node_safe' or blue_action == 'restore_node':
                self.made_safe_nodes.append(blue_node)

            if blue_action in self.current_game_blue:
                self.current_game_blue[blue_action] += 1
            else:
                self.current_game_blue[blue_action] = 1

            # calculates the reward from the current state of the network
            reward_args = {
                'network_interface': self.network_interface,
                'blue_action': blue_action,
                'blue_node': blue_node,
                'start_state': notes['post_red_state'],
                'end_state': self.network_interface.get_all_node_compromised_states(),
                'start_vulnerabilities': notes['post_red_vulnerabilities'],
                'end_vulnerabilities': self.network_interface.get_all_vulnerabilities(),
                'start_isolation': notes['post_red_isolation'],
                'end_isolation': self.network_interface.get_all_isolation(),
                'start_blue': notes['post_red_blue_view'],
                'end_blue': self.network_interface.get_all_node_blue_view_compromised_states(),
            }

            reward = getattr(
                reward_functions,
                self.network_interface.game_mode.rewards.function.value,
            )(reward_args)

            # gets the current observation from the environment
            self.env_observation = (
                self.network_interface.get_current_observation().flatten()
            )
            self.current_duration += 1

            # if the total number of steps reaches the set end then the blue agent wins and is rewarded accordingly
            if (
                self.current_duration
                == self.network_interface.game_mode.game_rules.max_steps.value
            ):
                if self.network_interface.game_mode.rewards.end_rewards_are_multiplied_by_end_state.value:
                    reward = (
                        self.network_interface.game_mode.rewards.for_reaching_max_steps.value
                        * (
                            len(
                                self.network_interface.current_graph.get_nodes(
                                    filter_true_safe=True
                                )
                            )
                            / self.network_interface.current_graph.number_of_nodes()
                        )
                    )
                else:
                    reward = self.network_interface.game_mode.rewards.for_reaching_max_steps.value
                done = True

        # Gets the state of the environment at the end of the current time step
        if self.collect_data:
            # The blues view of the network
            notes['end_blue_view'] = (
                self.network_interface.get_all_node_blue_view_compromised_states()
            )
            # The state of the nodes (safe/compromised)
            notes['end_state'] = (
                self.network_interface.get_all_node_compromised_states()
            )
            # A dictionary of vulnerabilities
            notes['final_vulnerabilities'] = (
                self.network_interface.get_all_vulnerabilities()
            )
            # The location of the red agent
            notes['final_red_location'] = copy.deepcopy(
                self.network_interface.red_current_location
            )

        if self.network_interface.game_mode.miscellaneous.output_timestep_data_to_json.value:
            current_state = self.network_interface.create_json_time_step()
            self.network_interface.save_json(current_state, self.current_duration)

        if self.print_metrics and done:
            # prints end of game metrics such as who won and how long the game lasted
            self.num_games_since_avg += 1
            self.total_games += 1

            # Populate the current game's dictionary of stats with the episode winner and the number of timesteps
            if (
                self.current_duration
                == self.network_interface.game_mode.game_rules.max_steps.value
            ):
                self.current_game_stats = {
                    'Winner': 'blue',
                    'Duration': self.current_duration,
                }
            else:
                self.current_game_stats = {
                    'Winner': 'red',
                    'Duration': self.current_duration,
                }

            # Add the actions taken by blue during the episode to the stats dictionary
            self.current_game_stats.update(self.current_game_blue)

            # Add the current game dictionary to the list of dictionaries to average over
            self.game_stats_list.append(Counter(dict(self.current_game_stats.items())))

            # Every self.avg_every episodes, print the stats to console
            if self.num_games_since_avg == self.avg_every:
                self.eval_printout.print_stats(self.game_stats_list, self.total_games)

                self.num_games_since_avg = 0
                self.game_stats_list = []

        self.current_reward = reward

        if self.collect_data:
            notes['safe_nodes'] = len(
                self.network_interface.current_graph.get_nodes(filter_true_safe=True)
            )
            notes['blue_action'] = blue_action
            notes['blue_node'] = blue_node
            notes['attacks'] = self.network_interface.true_attacks
            notes['end_isolation'] = self.network_interface.get_all_isolation()

        if self.print_notes:
            json_data = json.dumps(notes)
            print(json_data)
        # Returns the environment information that AI gym uses and all of the information collected in a dictionary
        return self.env_observation, reward, done, notes

    def render(self, mode='human'):
        """Render the current state of the environment.

        Visualises the environment's current state, including the current step, network graph, attacked nodes,
        current time step reward, and nodes that have been made safe.

        Args:
            mode (str): The mode in which to render. Defaults to 'human'. This is the only mode supported.

        Returns:
            Numpy array: The rendered image of the environment, built using matplotlib and converted to image array.
        """
        return render(
            current_step=self.current_duration,
            g=self.network_interface.current_graph,
            attacked_nodes=self.network_interface.detected_attacks,
            current_time_step_reward=self.current_reward,
            made_safe_nodes=self.made_safe_nodes,
        )
    def calculate_observation_space_size(self, with_feather: bool) -> int:
        """
        Calculate the observation space size.

        This is done using the current active observation space configuration
        and the number of nodes within the environment.

        Args:
            with_feather: Whether to include the size of the Feather Wrapper output

        Returns:
            The observation space size
        """
        return self.network_interface.get_observation_size_base(with_feather)
