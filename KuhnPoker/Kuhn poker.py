import numpy as np
from random import shuffle

"""
Implementation of a poker bot that learns stratgey for Kuhn poker using CFR algorithm. 

In the Nash equilibrium of Kuhn poker: 
player 1 expected value = -0.0556
player 2 expected value = 0.0556

"""

class Kuhn:

    def __init__(self):
        self.i_map = {} # dictionary that stores all possible information sets
        self.n_actions = 2
        self.expected_game_value = 0
        self.deck = np.array([0, 1, 2])

    def train(self, n_iterations=50000):
        expected_game_value = 0
        for _ in range(n_iterations):
            shuffle(self.deck)
            expected_game_value += self.cfr('', 1, 1)
            for _, v in self.i_map.items():
                v.update_strategy()

        expected_game_value /= n_iterations
        display_results(expected_game_value, self.i_map)

    def cfr(self, history="", pr_1=1, pr_2=1):
        """ 
        Parameters
        ----------

        history: [{'r', 'p', 'b'}], str
            String representation of the sequence of actions from root of the tree.
            Each character represents an action:
                'r': random chance action
                'p': pass action
                'b': bet action

        pr_1: (0, 1), float
            Probability PLayer 1 reaches 'history'.

        pr_2: (0, 1), float
            Probability Player 2 reaches 'history'.
        """
        
        # determine whose go it is
        n = len(history)
        is_player_1 = n % 2 == 0

        # player's card
        player_card = self.deck[0] if is_player_1 else self.deck[1]

        # Check if it a terminal node
        # If True return the reward
        if self.is_terminal(history):
            card_player = self.deck[0] if is_player_1 else self.deck[1]
            card_opponent = self.deck[1] if is_player_1 else self.deck[0]
            return self.get_reward(history, card_player, card_opponent)
        
        # Get information set and current strategy
        info_set = self.get_information_set(player_card, history)
        strategy = info_set.strategy

        # initialise array that stores hypotehtical payoff for each possible action in the information set
        action_utils = np.zeros(self.n_actions)

        # For each action, compute the hypothetical payoff that the player would get if they were to take that action.
        for act in range(self.n_actions):
            new_history = history + info_set.action_dict[act]
            if is_player_1:
                # recall function with new history
                # reach probability of next info set will be updated in accordance with current strategy
                # recursion will proceed until a terminal node is reached and return the payoff associated with that terminal state.
                # payoff is then returned back up the tree to be used in the calculations at previous levels
                # multiply be -1 since utility of one player is negative utility for the other player
                action_utils[act] = -1*self.cfr(new_history, pr_1*strategy[act], pr_2)
            else:
                action_utils[act] = -1*self.cfr(new_history, pr_1, pr_2*strategy[act])

        # expected utility for the current information set given the player's strategy at that information set
        util = sum(action_utils*strategy)

        # regret for each action updated
        regrets = action_utils - util

        if is_player_1:

            # update reach probability for this information set
            info_set.reach_pr += pr_1
            # update regret sums weighted by probability of reaching the info set
            info_set.regret_sum += pr_2*regrets

        else:
            info_set.reach_pr += pr_2
            info_set.regret_sum += pr_1*regrets

        return util
   
    @staticmethod
    def is_terminal(history):
        """Checks if at a terminal node"""
        if history[-2:] == 'pp' or history[-2:] == "bb" or history[-2:] == 'bp':
            return True
        
    @staticmethod
    def get_reward(history, player_card, opponent_card):
        """Return the reward for the current player"""
        terminal_pass = history[-1] == 'p'
        double_bet = history[-2:] == "bb"
        if terminal_pass:
            if history[-2:] == 'pp':
                return 1 if player_card > opponent_card else -1
            else:
                return 1
        elif double_bet:
            return 2 if player_card > opponent_card else -2
        
    def get_information_set(self, card, history):
        """Gets the current information set. If not in i_map dict then adds to it."""
        key = str(card) + " " + history
        if key not in self.i_map:
            action_dict = {0: 'p', 1: 'b'}
            info_set = InformationSet(key, action_dict)
            self.i_map[key] = info_set
            return info_set
        return self.i_map[key]
    

class InformationSet:

    """Each instance of this class represents an Information set which contains an active player and all information available
    to that active player at that decision in the game, and can possibly include more than one game state."""

    def __init__(self, key, action_dict, num_actions=2):
        self.key = key  # unique identifier for an information set (player card + history)
        self.num_actions = num_actions
        self.action_dict = action_dict  # possible actions - 0 = pass, 1 = bet
        self.regret_sum = np.zeros(self.num_actions)    # cumulated counterfactual regret of not choosing an action in the information set
        self.strategy_sum = np.zeros(self.num_actions)  # used to calculate average strategy
        self.strategy = np.repeat(1/self.num_actions, self.num_actions) # strategy for a given information set
        self.reach_pr = 0   # sum of probabilities for reaching information set over all possible histories in the information set
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
        strategy = self.strategy_sum / self.sum_reach_pr
        total = sum(strategy)
        strategy /= total
        return strategy
    
def display_results(ev, i_map):
    print(f'Player 1 expected value: {ev:.4f}')
    print(f'Player 2 expected value: {-ev:.4f}')

    print("\nPlayer 1 strategies:")
    sorted_items = sorted(i_map.items(), key=lambda x: x[0])
    for _, v in filter(lambda x: len(x[0]) % 2 == 0, sorted_items):  # Player 1's turns
        print(f"{v.key}: {v.get_average_strategy()}")  # Call the method to get the average strategy

    print("\nPlayer 2 strategies:")
    for _, v in filter(lambda x: len(x[0]) % 2 == 1, sorted_items):  # Player 2's turns
        print(f"{v.key}: {v.get_average_strategy()}")  # Call the method to get the average strategy


trainer = Kuhn()
trainer.train(n_iterations=100000)
