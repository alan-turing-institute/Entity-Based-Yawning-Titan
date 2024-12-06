from typing import Dict, Mapping
from entity_gym.env import (
    Environment,
    ObsSpace,
    ActionSpace,
    Observation,
    ActionName,
    Action,
    Entity,
    GlobalCategoricalActionSpace,
    GlobalCategoricalActionMask,
    GlobalCategoricalAction,
    SelectEntityActionSpace,
    SelectEntityAction,
    SelectEntityActionMask,
)

import yawning_titan.envs.generic.core.reward_functions as reward_functions
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.envs.generic.core.red_interface import RedInterface
from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.game_modes.game_mode import GameMode
from yawning_titan.networks.network import Network
from yawning_titan.networks.network_creator import gnp_random_connected_graph
from ebyt.envs.entity_cyber.network_renderer_matplotlib_v2 import render
import numpy as np


class EntityYT(Environment):
    '''
    Arguments:
        network (Network): The network representing the environment's topology.
        game_mode (GameMode): Configuration object specifying the game settings.
        random (bool): Determines if the environment resets with a new random network on each episode.
        random_seed (int): Seed for random number generation.
        n_nodes (int): Number of nodes in the network.
        edge_prob (float): Probability of edge creation between nodes in the random network when generating the network.
    Attributes:
        random_reset (bool): Determines if the environment resets with a new random network on each episode.
        n_nodes (int): Number of nodes in the network.
        edge_prob (float): Probability of edge creation between nodes in the random network when generating the network.
        game_mode (GameMode): Configuration object specifying the game settings.
        network (Network): The network representing the environment's topology.
        network_interface (NetworkInterface): Interface for interacting with the network.
        blue_interface (BlueInterface): Interface for the blue agent's actions.
        red_agent (RedInterface): Interface for the red agent's actions.
        nodes (List[Node]): List of nodes in the network.
        random_seed (int): Seed for random number generation.
        feature_list (List[str]): List of node features included in observations.
        action_list (List[str]): List of available actions for the blue agent.
    Methods:
        generate_random_network(n: int, p: float) -> Network:
            Generates a connected random network using the Erdős-Rényi model.
        obs_space() -> ObsSpace:
        action_space() -> Dict[ActionName, ActionSpace]:
        get_node_features() -> List[List[Any]]:
            Retrieves the features of all nodes for the current observation.
        reset() -> Observation:
            Resets the environment to its initial state.
        observe(done: bool = False, reward: int = 0) -> Observation:
            Generates an observation of the current environment state.
        act(actions: Mapping[ActionName, Action]) -> Observation:
            Executes the given actions in the environment and updates the state.
        render(mode: str = 'human'):
            Renders the current state of the environment - for logging purposes, as it generates a numpy array.
    '''
    def __init__(self, network: Network, game_mode: GameMode, random: bool, random_seed: int, n_nodes: int, edge_prob: float):
        self.random_reset = random
        self.n_nodes = n_nodes
        self.edge_prob = edge_prob
        self.game_mode = game_mode
        #initialise a random network if random, otherwise the environment will always start with the same network
        if self.random_reset:
            self.network = self.generate_random_network(n=n_nodes, p=edge_prob)
        else:
            self.network = network

        self.network_interface = NetworkInterface(
            network=self.network, game_mode=self.game_mode
        )
        self.blue_interface = BlueInterface(self.network_interface)

        self.red_agent = RedInterface(self.network_interface)
        # nodes; these are the entities
        self.nodes = self.network_interface.current_graph.get_nodes(as_list=True)
        self.random_seed = self.network_interface.random_seed

        self.feature_list = []
        if self.network_interface.game_mode.observation_space.node_connections.value:
            self.feature_list.append('isolated')
        if self.network_interface.game_mode.observation_space.compromised_status.value:
            self.feature_list.append('blue_view_compromised_status')
        if self.network_interface.game_mode.observation_space.vulnerabilities.value:
            self.feature_list.append('vulnerability_score')
        if self.network_interface.game_mode.observation_space.attacking_nodes.value:
            self.feature_list.append('attacking_node')
        if self.network_interface.game_mode.observation_space.attacked_nodes.value:
            self.feature_list.append('attacked_node')
        if self.network_interface.game_mode.observation_space.special_nodes.value:
            self.feature_list.append('entry_node')
            self.feature_list.append('high_value_node')
        
        # assigning the action list, based on the actions available in the blue_interface
        self.action_list = [
            action.__name__ for action in self.blue_interface.action_dict.values()
        ]
        self.action_list.extend(
            [
                action.__name__
                for action in self.blue_interface.global_action_dict.values()
            ]
        )

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
        # If True, random vulnerability is set for each node using the upper and lower bounds."""

        network.set_random_vulnerabilities = True
        
        return network

    def obs_space(self) -> ObsSpace:
        """
        Defines the observation space for the environment.

        Returns:
            ObsSpace: An observation space object containing entity types and a list of their possible features.
                - 'Generic_Node': An entity with a list of features defined by `self.feature_list`.
                - 'Defender': An entity with an empty list of features.
        """
        return ObsSpace(
            entities={
                'Generic_Node': Entity(features=self.feature_list),
                'Defender': Entity(features=[]),
            }
        )

    def action_space(self) -> Dict[ActionName, ActionSpace]:
        """
        Returns the action space for the environment.

        The action space is a dictionary with two keys:
        - 'High_Level': A GlobalCategoricalActionSpace object initialised with the action list ('self.action_list').
        - 'Target': A SelectEntityActionSpace object, which allows the agent to select an entity to act upon.

        Returns:
            Dict[ActionName, ActionSpace]: The action space dictionary.
        """
        return {
            'High_Level': GlobalCategoricalActionSpace(self.action_list),
            'Target': SelectEntityActionSpace(),
        }

    def get_node_features(self):
        """
        Get the current observation of the environment.

        The composition of the observation space is based on the configuration file used for the scenario.

        Returns:
            feature list for entity-based observation space. A list of lists where each sub-list contains the features for a node.
        """
        if self.network_interface.detected_attacks:
            attacking, attacked = list(zip(*self.network_interface.detected_attacks))
        else:
            attacking, attacked = [], []

        features = [[] for _ in range(len(self.nodes))]
        for i, node in enumerate(self.nodes):
            if self.network_interface.game_mode.observation_space.node_connections.value:
                features[i].append(getattr(node, 'isolated'))
            if self.network_interface.game_mode.observation_space.compromised_status.value:
                features[i].append(getattr(node, 'blue_view_compromised_status'))
            if self.network_interface.game_mode.observation_space.vulnerabilities.value:
                features[i].append(getattr(node, 'vulnerability_score'))
            if self.network_interface.game_mode.observation_space.attacking_nodes.value:
                features[i].append(1 if i in attacking else 0)
            if self.network_interface.game_mode.observation_space.attacked_nodes.value:
                features[i].append(1 if i in attacked else 0)
            if self.network_interface.game_mode.observation_space.special_nodes.value:
                features[i].append(getattr(node, 'entry_node'))
                features[i].append(getattr(node, 'high_value_node'))
        return features

    def reset(self) -> Observation:
        """
        Reset the environment to its initial state.
        If `random_reset` is `True`, a new random network is generated with `n` nodes and edge probability `p`.
        The network interface, blue interface, and red agent are then re-initialised with this new network.
        The method resets the network interface and red agent, updates the list of nodes, 
        and resets various internal state variables such as made safe nodes, attacked nodes, current duration, and current reward.
        Returns:
            Observation: The initial observation of the environment after the reset.
        """
        if self.random_reset:
            self.network = self.generate_random_network(n=self.n_nodes, p=self.edge_prob)
            self.network_interface = NetworkInterface(
                network=self.network, game_mode=self.game_mode
            )
            self.blue_interface = BlueInterface(self.network_interface)
            self.red_agent = RedInterface(self.network_interface)
    
        
        self.network_interface.reset()
        self.red_agent.reset()
        self.nodes = self.network_interface.current_graph.get_nodes(as_list=True)
        self.current_duration = 0
        self.current_reward = 0

        return self.observe()

    def observe(self, done=False, reward=0) -> Observation:
        """Generate an observation of the current environment state.

        This method returns an `Observation` object that includes entity features, action masks,
        the done flag, and the reward. It provides the agent with all necessary information
        about the current state of the environment.

        Parameters:
            done (bool): Indicates whether the episode has ended. Defaults to `False`.
            reward (int): The reward achieved from the previous action. Defaults to `0`.

        Returns:
            Observation: An object containing:
                - entities (dict): A mapping of entity types to their features and identifiers.
                    - 'Generic_Node': Features obtained from `get_node_features()` and identifiers for each node.
                    - 'Defender': An empty feature array and a single identifier.
                - actions (dict): Action masks available to the agent.
                    - 'High_Level': A `GlobalCategoricalActionMask` instance.
                    - 'Target': A `SelectEntityActionMask` with specified actor and actee types.
                - done (bool): The done flag indicating if the episode has ended.
                - reward (int): The reward from the previous action.
        """
        return Observation(
            entities={
                'Generic_Node': (
                    self.get_node_features(),
                    [('Generic_Node', i) for i in range(len(self.nodes))],
                ),
                'Defender': (np.zeros([1, 0], dtype=np.float32), [('Defender', 0)]),
            },
            actions={
                'High_Level': GlobalCategoricalActionMask(),
                'Target': SelectEntityActionMask(
                    actor_types=['Defender'], actee_types=['Generic_Node']
                ),
            },
            done=done,
            reward=reward,
        )

    def act(self, actions: Mapping[ActionName, Action]) -> Observation:
        """
        Executes an action in the environment for both the red and blue agents, updates the environment state, and returns the resulting observation.
        This method first allows the red agent to perform its action if the grace period has elapsed; 
        otherwise, the red agent does nothing during the grace period. 
        It records relevant state information for computing the reward. 
        Then, it processes the blue agent's action based on the provided actions mapping, 
        updates the reward accordingly, and checks if the episode has reached its maximum duration. 
        Finally, it updates the environment state and returns the observation.
        Parameters:
            actions (Mapping[ActionName, Action]): A mapping containing the high-level and target actions for the blue agent.
        Returns:
            Observation: The observation of the environment after the actions have been performed.
        """
        self.network_interface.reset_stored_attacks()
        # red agent acting
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

        # notes - information for reward function etc.
        notes = {}
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
       
        # set up initial variables that are reassigned based on the action that blue takes
        done = False
        reward = 0

        # perform blue action if not done
        if not done:
            # process blue action
            blue_node = None
            blue_action = ''
            # getting intended blue action(s)
            high_level = actions['High_Level']
            assert isinstance(high_level, GlobalCategoricalAction)

            target = actions['Target']
            assert isinstance(target, SelectEntityAction)

            target_index = target.actees[0][1]
            if high_level.label == 'scan_all_nodes':
                blue_action, blue_node = self.blue_interface.scan_all_nodes()
            elif high_level.label == 'do_nothing':
                blue_action, blue_node = self.blue_interface.do_nothing()
            elif high_level.label == 'reduce_node_vulnerability':
                blue_action, blue_node = self.blue_interface.reduce_node_vulnerability(
                    self.nodes[target_index]
                )
            elif high_level.label == 'restore_node':
                blue_action, blue_node = self.blue_interface.restore_node(
                    self.nodes[target_index]
                )
            elif high_level.label == 'make_safe_node':
                blue_action, blue_node = self.blue_interface.make_safe_node(
                    self.nodes[target_index]
                )
            elif high_level.label == 'isolate_node':
                blue_action, blue_node = self.blue_interface.isolate_node(
                    self.nodes[target_index]
                )
            elif high_level.label == 'reconnect_node':
                blue_action, blue_node = self.blue_interface.reconnect_node(
                    self.nodes[target_index]
                )

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

            self.current_duration += 1

            # if the total number of steps reaches maximum then the blue agent wins and is rewarded accordingly
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

            self.current_reward += reward

        self.nodes = self.network_interface.current_graph.get_nodes(as_list=True)
        return self.observe(done, reward)

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
            made_safe_nodes=[],
        )


