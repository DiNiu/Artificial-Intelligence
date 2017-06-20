'''
Ex5.1: policy evaluation
fixed policy: player sticks if the player's sum is 20 or 21, and otherwise hits.
              state space: {usable 'A', dealer's showing card, cards sum}
              usable 'A' = true, false
              dealer's showing card: 'A','1', ...,'9','10'('J','Q','K').
              card sum: [12,21]. if sum<12, player must hit, or he will lose. 
                        In other words, pi(a=hit|sum<12)=1. Therefore, there's no need take sum<12 as a state

'''
import Env.BlackJackEnv as env
import numpy as np


new = env.BlackJack()

# state value init



