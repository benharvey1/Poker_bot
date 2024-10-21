from TwoCardPoker import TwoCardPoker
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

hands_array = np.array([
    ['AA', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s'],
    ['AKo', 'KK', 'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'K7s', 'K6s', 'K5s', 'K4s', 'K3s', 'K2s'],
    ['AQo', 'KQo', 'QQ', 'QJs', 'QTs', 'Q9s', 'Q8s', 'Q7s', 'Q6s', 'Q5s', 'Q4s', 'Q3s', 'Q2s'],
    ['AJo', 'KJo', 'QJo', 'JJ', 'JTs', 'J9s', 'J8s', 'J7s', 'J6s', 'J5s', 'J4s', 'J3s', 'J2s'],
    ['ATo', 'KTo', 'QTo', 'JTo', 'TT', 'T9s', 'T8s', 'T7s', 'T6s', 'T5s', 'T4s', 'T3s', 'T2s'],
    ['A9o', 'K9o', 'Q9o', 'J9o', 'T9o', '99', '98s', '97s', '96s', '95s', '94s', '93s', '92s'],
    ['A8o', 'K8o', 'Q8o', 'J8o', 'T8o', '98o', '88', '87s', '86s', '85s', '84s', '83s', '82s'],
    ['A7o', 'K7o', 'Q7o', 'J7o', 'T7o', '97o', '87o', '77', '76s', '75s', '74s', '73s', '72s'],
    ['A6o', 'K6o', 'Q6o', 'J6o', 'T6o', '96o', '86o', '76o', '66', '65s', '64s', '63s', '62s'],
    ['A5o', 'K5o', 'Q5o', 'J5o', 'T5o', '95o', '85o', '75o', '65o', '55', '54s', '53s', '52s'],
    ['A4o', 'K4o', 'Q4o', 'J4o', 'T4o', '94o', '84o', '74o', '64o', '54o', '44', '43s', '42s'],
    ['A3o', 'K3o', 'Q3o', 'J3o', 'T3o', '93o', '83o', '73o', '63o', '53o', '43o', '33', '32s'],
    ['A2o', 'K2o', 'Q2o', 'J2o', 'T2o', '92o', '82o', '72o', '62o', '52o', '42o', '32o', '22']
])

trainer = TwoCardPoker()
player_1_strat, player_2_strat = trainer.train(10000000)

action_frequencies = {key:value for key, value in player_1_strat.items() if np.isin(key, hands_array)}
ranks = 'AKQJT98765432'

# Bet/check range for player 1
ranges = np.zeros((13, 13))

for i in range(13):
    for j in range(13):
        hand = hands_array[i, j]
        if hand in action_frequencies:
            actions = action_frequencies[hand]
            bet_freq = actions.get('0.5b', 0) + actions.get('1b', 0)
            ranges[i, j] = bet_freq

# Plot the range table
plt.figure(figsize=(10, 8))
sns.heatmap(ranges, annot=hands_array, fmt='', cmap='Greens')
plt.title("Player 1 betting frquency")
plt.xticks([])
plt.yticks([])
plt.savefig('Player_1_betting_frequency.png')




