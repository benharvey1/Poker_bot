import numpy as np
from random import shuffle

"Implementation of Chance Sampling MCCFR+ to create a bot for Leduc poker"

class Leduc:

    def __init__(self):
        self.i_map = {} # dictionary that stores all possible information sets
        self.expected_game_value = 0
        self.deck = np.array([0, 0, 1, 1, 2, 2])

    def train(self, n_iterations=50000):
        expected_game_value = 0
        for _ in range(n_iterations):
            if _ % 1000 == 0:
                print(f'Iteration {_}')
            shuffle(self.deck)
            expected_game_value += self.cfr('', 1, 1)

        expected_game_value /= n_iterations
        display_results(expected_game_value, self.i_map)

    def cfr(self, history="", pr_1=1, pr_2=1):
        """ 
        Parameters
        ----------

        history: str
            String representation of the sequence of actions from root of the tree.
            Each character represents an action:
                'd': card dealt
                'p': pass action (check)
                'c': call action
                'b': bet action
                'r': raise action
                'f': fold action

        pr_1: (0, 1), float
            Probability PLayer 1 reaches 'history'.

        pr_2: (0, 1), float
            Probability Player 2 reaches 'history'.
        """
        
        # determine whose go it is
        is_player_1 = self.get_active_player(history) == 0

        # player's card
        player_card = self.deck[0] if is_player_1 else self.deck[1]
        community_card = self.deck[2]

        # Check if it a terminal node
        # If True return the reward
        if self.is_terminal(history):
            card_player = self.deck[0] if is_player_1 else self.deck[1]
            card_opponent = self.deck[1] if is_player_1 else self.deck[0]
            return self.get_reward(history, card_player, card_opponent, community_card)
        
        # Check if at chance node (flop is next)
        # If True call function with an updated history
        if self.is_chance_node(history):
            new_history = history + 'd'
            if is_player_1:
                return self.cfr(new_history, pr_1, pr_2)    # no -1 as player 1 acts first post flop
            else:
                return -1*self.cfr(new_history, pr_1, pr_2)
        
        # Get information set and strategy
        info_set = self.get_information_set(player_card, community_card, history)
        strategy = info_set.strategy

        # stores counterfactual utility for each possible action in the information set
        action_utils = np.zeros(info_set.num_actions)

        # For each action, compute the (countefactual) utility that the player would get if they were to take that action.
        valid_actions = self.valid_actions(history)

        for i, act in enumerate(valid_actions):
            new_history = history + act
            if is_player_1:
                # recall function with new history
                # reach probability of next info set will be updated in accordance with current strategy
                # recursion will proceed until a terminal node is reached and return the payoff associated with that terminal state.
                # payoff is then returned back up the tree to be used in the calculations at previous levels
                # multiply be -1 since utility of one player is negative utility for the other player
                action_utils[i] = -1*self.cfr(new_history, pr_1*strategy[i], pr_2)
            else:
                action_utils[i] = -1*self.cfr(new_history, pr_1, pr_2*strategy[i])

        # expected utility for the current information set given the player's strategy at that information set
        util = sum(action_utils*strategy)

        # regret for each action updated
        regrets = action_utils - util

        if is_player_1:

            # update reach probability for this information set
            info_set.reach_pr += pr_1
            # update regret sums 
            info_set.regret_sum += pr_2*regrets

        else:
            info_set.reach_pr += pr_2
            info_set.regret_sum += pr_1*regrets

        info_set.update_strategy()

        return util
   
    @staticmethod
    def is_terminal(history):
        """Checks if at a terminal node"""
        if history[-1:] == 'f':
            return True
        if 'd' in history:
            preflop, postflop = history.split('d')
            if postflop[-1:] == 'c' or postflop[-2:] == 'pp':
                return True
        return False
    
    @staticmethod
    def is_chance_node(history):
        """Checks if at a chance node"""
        chance_nodes = {'pp', 'pbc', 'pbrc', 'bc', 'brc'}
        if history in chance_nodes:
            return True
        return False
        
    @staticmethod
    def get_reward(history, player_card, opponent_card, community_card):
        """Return the reward for the current player"""
        ante = 1
        # betsize = 2
        payoff = {'pp': 0, 'bf': 0, 'pbf': 0, 'brf':2, 'bc': 2, 'pbrf':2, 'pbc':2, 'brc': 4, 'pbrc': 4}

        if 'd' not in history:
            # have not reached a flop
            if player_card > opponent_card:
                return ante + payoff[history]
            elif player_card < opponent_card:
                return -(ante + payoff[history])
            else:
                return 0
            
        elif 'd' in history:
            preflop, postflop = history.split('d')

            if player_card == community_card:
                return ante + payoff[preflop] + payoff[postflop]
            
            elif opponent_card == community_card:
                return - (ante + payoff[preflop] + payoff[postflop])
            
            else:
                if player_card > opponent_card:
                    return ante + payoff[preflop] + payoff[postflop]
                elif opponent_card > player_card:
                    return - (ante + payoff[preflop] + payoff[postflop])
                else:
                    return 0
        
    @staticmethod
    def get_active_player(history):
        # player_1 goes first preflop and postflop
        if 'd' not in history:
            return len(history) % 2
        else:
            preflop, postflop = history.split('d')
            return len(postflop) % 2
        
    @staticmethod
    def valid_actions(history):
        "Returns possible actions for a given history"

        if history[-1:] == 'p' or history[-1:] == '' or history[-1:] == 'd':
            return ['p', 'b']
        
        if history[-1:] == 'b':
            return ['f', 'c', 'r']
        
        if history[-1:] == 'r':
            # no re raises
            return ['f', 'c']

    def get_information_set(self, card, community_card, history):
        """Gets the current information set. If not in i_map dict then adds to it."""

        if 'd' in history:
            key = str(card) + str(community_card) + " " + history
        else:
            key = str(card) + " " + history

        if key not in self.i_map:
            if history[-1:] == 'b':
                num_actions = 3
            else:
                num_actions = 2
            actions = self.valid_actions(history)
            info_set = InformationSet(key, actions, num_actions)
            self.i_map[key] = info_set
            return info_set
        
        return self.i_map[key]
    

class InformationSet:

    """Each instance of this class represents an information set, which contains the relevant state and 
    available information for a player at a particular decision point in the game. It can represent multiple 
    game states that are indistinguishable to the player.

    Each information set stores several key variables:

    - strategy: Current probability distribution over the possible actions the player can take at this information set. 
    It is updated every time the information set is visited using positive regret matching based on the regret_sum.

    - strategy_sum: Cumulative sum of the strategies used at this information set across all iterations. It is updated 
    every time the information set is visited by adding the current strategy, weighted by the reach probability.

    - regret_sum: Cumulative counterfactual regret for each action at this information set. It tracks how much better
    the player could have done by choosing a different action. Updated each time the information set is visited by adding
    regrets weighted by the reach probability of the opponent.

    - reach_pr: Tracks how often this information set is reached during the game. It is essentially the product of the
    probabilities of all the players' actions that led to this information set. This is reset after each iteration.

    - sum_reach_pr: Cumulative sum of the reach probabilities over all iterations for this information set. 
    Used to normalize the final average strategy.
    """

    def __init__(self, key, actions, num_actions):
        self.key = key  # unique identifier for an information set (player card + community card + history)
        self.actions = actions
        self.num_actions = num_actions
        self.regret_sum = np.zeros(self.num_actions)    # cumulated counterfactual regret of not choosing an action in the information set
        self.strategy_sum = np.zeros(self.num_actions)  # used to calculate average strategy
        self.strategy = np.repeat(1/self.num_actions, self.num_actions) # strategy for a given information set
        self.reach_pr = 0   # product of players' actions that led to this info set
        self.sum_reach_pr = 0   # sum of above over all iterations

    def update_strategy(self):
        """Updates strategy (sum) to include the contribution of the current strategy weighted by how likely this 
        information set was reached during the game."""
        self.strategy_sum += self.strategy * self.reach_pr 
        self.strategy = self.get_strategy()
        self.sum_reach_pr += self.reach_pr
        self.reach_pr = 0   # resets for next iteration

    def get_strategy(self):
        """Generates strategy proportional to cumulative counterfactual regrets for particular action in information set"""
        regrets = self.regret_sum
        regrets[regrets < 0] = 0
        normalising_sum = sum(regrets)
        
        if normalising_sum > 0:
            strategy = regrets/normalising_sum
        else:
            strategy = np.repeat(1/self.num_actions, self.num_actions)

        return strategy

    def get_average_strategy(self):
        """Calculates average strategy over all iterations"""

        if self.sum_reach_pr > 0:
            strategy = self.strategy_sum / self.sum_reach_pr
        else:
            strategy = np.repeat(1 / self.num_actions, self.num_actions)

        total = sum(strategy)
        strategy /= total
        return {action: strategy[i] for i, action in enumerate(self.actions)}

    
def display_results(ev, i_map):
    print(f'Player 1 expected value: {ev:.4f}')
    print(f'Player 2 expected value: {-ev:.4f}')

    player_1_strategy = {}
    player_2_strategy = {}
    for key, node in i_map.items():
        hand, history = key.split(' ')
        if trainer.get_active_player(history) == 0:
            player_1_strategy[hand+history] = node.get_average_strategy()
        else:
            player_2_strategy[hand+history] = node.get_average_strategy()

    print(f'Player 1 strategies: {player_1_strategy}')
    print(f'Player 2 strategies: {player_2_strategy}')


trainer = Leduc()
trainer.train(n_iterations=10000)
