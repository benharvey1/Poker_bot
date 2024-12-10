import numpy as np
from random import shuffle
from tqdm import tqdm

"""Implementation of Chance Sampling MCCFR+ to create a bot for a simplified variant of Heads up Pot Limit Hold'em."""

class TwoCardPoker:

    def __init__(self, stack_size=32, ante=1):
        self.i_map = {} # dictionary that stores all possible information sets
        self.strategies = {}
        self.expected_game_value = 0
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.suits = ['s', 'h', 'c', 'd']
        self.deck = np.array([rank + suit for rank in self.ranks for suit in self.suits])
        self.stack_size = stack_size
        self.ante = ante

    def deal(self):
        """Deals cards"""

        player_1_cards = self.deck[0:2]
        player_2_cards = self.deck[2:4]

        return player_1_cards, player_2_cards
    
    def reset_stacks(self):
        """Reset players stack at start of each iteration"""
        return self.stack_size, self.stack_size
    
    def get_higher_rank(self, rank1, rank2):
        """Finds the higher card out of the two in a hand"""

        if self.ranks.index(rank1) >= self.ranks.index(rank2):
            return rank1
        else:
            return rank2
            
    def hand_abstraction(self, hand):
        ''' 
        Takes a hand (array of size two) and compresses it into a simpler representation (higher card in front).
        i.e. ['Th', '9h'] becomes 'T9s' (same suit),
        ['9s', '10h'] becomes 'T9o' (off-suit),
        ['Th', 'Ts'] becomes 'TT' (pair).
        '''

        rank1, suit1 = hand[0][0], hand[0][1]
        rank2, suit2 = hand[1][0], hand[1][1]

        # Check if it's a pair
        if rank1 == rank2:
            return rank1 + rank2

        # Find the higher card
        higher_rank = self.get_higher_rank(rank1, rank2)
        lower_rank = rank2 if higher_rank == rank1 else rank1

        # Check if the cards are suited or off-suit
        if suit1 == suit2:
            return higher_rank + lower_rank + 's'
        
        return higher_rank + lower_rank + 'o'
        
    @staticmethod
    def valid_actions(history):
        "Returns possible actions for a given history"          

        # If cards have just been dealt player can pass or bet
        if not history:
            return ['p', '0.5b', '1b']

        # If the last action was pass
        # Player can pass or bet half or full pot
        if history[-1] == 'p':
            return ['p', '0.5b', '1b']
        
        # If last action was bet, player can fold, call or raise by pot size
        if history[-1] in ['0.5b','1b']:
            return ['f', 'c', '1b']
    
    @staticmethod
    def is_terminal(history):
        """Checks if at a terminal node"""
        if history and (history[-1] == 'f' or history[-1] == 'c' or history == ['p', 'p']):
            return True
        return False
    
    @staticmethod
    def get_active_player(history):
        """Returns active player"""
        return len(history) % 2
    
    def get_winner(self, hand_1, hand_2):
        """Determine the winner from the hand strength (pair > suited > offsuited).
        Returns:
        - 1 if player wins
        - 2 if opponent wins
        - 0 if draw
        """
   
        is_hand_1_pair = hand_1[0] == hand_1[1]
        is_hand_2_pair = hand_2[0] == hand_2[1]

        if is_hand_1_pair and is_hand_2_pair:
            if hand_1[0] == hand_2[0]:
                return 0
            elif hand_1[0] == self.get_higher_rank(hand_1[0], hand_2[0]):
                return 1
            else:
                return 2
            
        elif is_hand_1_pair:
            return 1
        
        elif is_hand_2_pair:
            return 2
        
        is_hand_1_suited = hand_1[2] == 's'
        is_hand_2_suited = hand_2[2] == 's'

        if is_hand_1_suited and is_hand_2_suited:
            if hand_1[0] == self.get_higher_rank(hand_1[0], hand_2[0]):
                return 1
            elif hand_2[0] == self.get_higher_rank(hand_1[0], hand_2[0]):
                return 2
            
            if hand_1[1] == self.get_higher_rank(hand_1[1], hand_2[1]):
                return 1
            elif hand_2[1] == self.get_higher_rank(hand_1[1], hand_2[1]):
                return 2
            
            return 0
        
        if hand_1[0] == self.get_higher_rank(hand_1[0], hand_2[0]):
            return 1
        elif hand_2[0] == self.get_higher_rank(hand_1[0], hand_2[0]):
            return 2
            
        if hand_1[1] == self.get_higher_rank(hand_1[1], hand_2[1]):
            return 1
        elif hand_2[1] == self.get_higher_rank(hand_1[1], hand_2[1]):
            return 2
            
        return 0
    
    def get_payoff(self, history, player_cards, opponent_cards, pot):
        """Return the payoff for the current player"""

        if history[-1] == 'f':
            # If one player folds, other player wins pot
            player = self.get_active_player(history)
            if player == 0:
                return pot/2
            else:
                return -pot/2

        # else we go to showdown
        winner = self.get_winner(player_cards, opponent_cards)
        
        if winner == 1:
            return pot/2
        
        elif winner == 2:
            return -pot/2

        return 0

    def get_information_set(self, player_cards, history):
        """Gets the current information set. If not in i_map dict then adds to it.
        Parameters:
        - player_cards: cards of player (including community card if postflop) after card abstraction applied
        - history: Sequence of actions
        """
    
        key = str(player_cards) + " " + "".join(history)   

        if key not in self.i_map:
            
            actions = self.valid_actions(history)
            num_actions = len(actions)

            info_set = InformationSet(key, actions, num_actions)
            self.i_map[key] = info_set
            return info_set
        
        return self.i_map[key] 

    def cfr(self, history, pr_1, pr_2, pot, current_bet, player_1_stack, player_2_stack):
        """ 
        Parameters
        ----------

        history: list
            representation of the sequence of actions from root of the tree.
            Each character represents an action (raises are included as bet actions):
                'd': card dealt
                'p': pass action (check)
                'c': call action
                'f': fold action
                '0.5b': bet 1/2 pot
                '1b': bet pot

        pr_1: (0, 1), float
            Probability PLayer 1 reaches 'history'.

        pr_2: (0, 1), float
            Probability Player 2 reaches 'history'.

        pot: pot size of current hand

        current_bet: most recent bet size
        """
        #print("cfr called", history)
        if history is None:
            history = []
            pot += 2*self.ante   # both players ante at start of hand
            player_1_stack, player_2_stack = self.reset_stacks()    # Reset stacks each hand

        # player's card
        player_1_cards, player_2_cards = self.deal()
        
        player_1_hand = self.hand_abstraction(player_1_cards)
        player_2_hand = self.hand_abstraction(player_2_cards)


        # determine whose go it is
        is_player_1 = self.get_active_player(history) == 0
        player_hand = player_1_hand if is_player_1 else player_2_hand
        player_stack = player_1_stack if is_player_1 else player_2_stack
        opponent_stack = player_2_stack if is_player_1 else player_1_stack
        
        # Check if it a terminal node
        # If True return the reward
        if self.is_terminal(history) or player_stack == 0 or opponent_stack == 0:
            player_hand = player_1_hand if is_player_1 else player_2_hand
            opponent_hand = player_2_hand if is_player_1 else player_1_hand
            return self.get_payoff(history, player_hand, opponent_hand, pot)
    
        # Get information set and strategy
        info_set = self.get_information_set(player_hand, history)
        strategy = info_set.strategy

        # stores counterfactual utility for each possible action in the information set
        action_utils = np.zeros(info_set.num_actions)

        # For each action, compute the (countefactual) utility that the player would get if they were to take that action.
        valid_actions = self.valid_actions(history)

        for i, act in enumerate(valid_actions):

            # store original values before different updates occur depending on which action is selected
            original_pot = pot
            original_bet = current_bet
            original_player_stack = player_stack
            
            # Update pot size and current_bet depending on action taken
            if act == 'c':
                if player_stack < current_bet:
                    pot += 2*player_stack - current_bet
                    player_stack -= player_stack
                else:
                    pot += current_bet
                    player_stack -= current_bet

            elif act == '0.5b':
                current_bet = min(0.5*pot, player_stack)
                player_stack -= current_bet
                pot += current_bet

            elif act == '1b':
                current_bet = min(pot, player_stack)
                player_stack -= current_bet
                pot += current_bet
                
            new_history = history.copy()
            new_history.append(act)

            if is_player_1:
                # recall function with new history
                # reach probability of next info set will be updated in accordance with current strategy
                # recursion will proceed until a terminal node is reached and return the payoff associated with that terminal state.
                # payoff is then returned back up the tree to be used in the calculations at previous levels
                # multiply be -1 since utility of one player is negative utility for the other player
                action_utils[i] = -1*self.cfr(new_history, pr_1*strategy[i], pr_2, pot, current_bet, player_stack, opponent_stack)
            else:
                action_utils[i] = -1*self.cfr(new_history, pr_1, pr_2*strategy[i], pot, current_bet, opponent_stack, player_stack)

            # Refresh back to original values for next possible action 
            pot = original_pot
            current_bet = original_bet
            player_stack = original_player_stack

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

        # update strategy each time info set is visited
        info_set.update_strategy()

        return util
    
    def train(self, n_iterations):
        """Runs training loop"""
        expected_game_value = 0

        # Use tqdm to wrap the loop and show progress
        for n in tqdm(range(n_iterations), desc="Training progress", unit="iteration"):
            shuffle(self.deck)
            expected_game_value += self.cfr(history=None, pr_1=1, pr_2=1, pot=0, current_bet=0, player_1_stack=self.stack_size, player_2_stack=self.stack_size)

        expected_game_value /= n_iterations
        print(f'Player 1 expected value: {expected_game_value:.4f}')
        print(f'Player 2 expected value: {-expected_game_value:.4f}')
        player_1_strat, player_2_strat = display_results(self.i_map)

        print(f'Player 1 strategy: {player_1_strat}')
        print(f'Player 2 strategy: {player_2_strat}')

        return player_1_strat, player_2_strat
    

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
        self.key = key  # unique identifier for an information set (player cards + history)
        self.num_actions = num_actions
        self.actions = actions
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


def display_results(i_map):

    player_1_strategy = {}
    player_2_strategy = {}
    for key, node in i_map.items():
        hand, history = key.split(' ')
        if trainer.get_active_player(history) == 0:
            player_1_strategy[hand+history] = node.get_average_strategy()
        else:
            player_2_strategy[hand+history] = node.get_average_strategy()

    return player_1_strategy, player_2_strategy

    
if __name__ == "__main__":
    trainer = TwoCardPoker()

