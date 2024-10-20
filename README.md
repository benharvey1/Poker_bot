# Poker_bot
Attempt to make poker bots using chance sampling CFR.
<br><br> An overview of CFR theory is provided (source: http://modelai.gettysburg.edu/2013/cfr/cfr.pdf). 
<br><br> Bots are implemented for simple poker games, with increaing complexity. The aim is to develop the code for pot limit Texas Hold'em:
### Kuhn poker
Kuhn poker is a very simple form of poker developed by Harold Kuhn. It is a zero sum, two player game of imperfect information. Before dealing, both players ante 1 chip into the pot. Three different cards (e.g. 0, 1 and 2) are shuffled and one card is dealt to each player face down. The players look at their own cards (but do not know their opponent's cards) and play alternates starting with player 1. On each turn, a player may either _pass_ or _bet_. A _pass_ passes the action to the other player without putting any more chips into the pot. A player that _bets_ places another chip into the pot. After two successive passes or bets, the players show their cards and the player with the highest card wins and takes all the chips in the pot. If a player passes after the other has bet, then the chips go to the player that bet.

### Leduc poker
Leduc poker is a slight extension of Kuhn poker. There are now 6 cards in the deck, but still only three unique card values (e.g. 0, 0, 1, 1, 2, 2). Before dealing, both players ante 1 chip into the pot. The players are dealt a card and play alternates starting with player 1. When playing first or when facing a pass from the other player, a player can either pass or bet 2 chips. When facing a bet from the other player, a player can either fold, call the 2 chips or raise to 4 chips. When facing a raise a player can either call or fold. 
<br> After the first betting round is finished and if no one has folded, one card is randomly chosen from the remaining 4 cards and dealt in the flop. The flop card is then combined with each of the players' pocket card to make up for their hand. The ranks work as in Texas Hold'em, with pairs beating high cards. When the flop is dealt, there's a second betting round, in which the same rules as in the first betting round apply. If the showdown is reached, the player with the highest ranked hand wins the pot.

### Two card poker
We now deal with a full deck of 52 cards and players have stacks of 100 chips. Each player antes 1 chip into the pot and is dealt two cards. Play alternates starting with player 1. When playing first or facing a check from the other player, a player can either check or bet half or full pot. When facing a bet, a player can fold, call or raise by the size of the pot. Play continues until one player folds, a player calls the other or an all in occurs. 

