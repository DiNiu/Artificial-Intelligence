###
import numpy as np


class BlackJack:
    '''
    Blackjack environment:
    cards: 
        Infinite deck, one deck of cards: {2,3,4,5,6,7,8,9,10,J,Q,K,A}
        face cards = 10, A = 1,11
    
        
    two players:
        dealer: fixed strategy
            sum of cards<17, hit
            sum of cards>=17, stick
        
        player:
            state space: {dealer's showing card, player's current sum, usable ace}
                         player's current sum: {12,21}
            action space: {hit,stick}
            Rewards: 
                win -> +1, lose -> -1, draw -> 0
                In the process, reward = 0 
            
            transition p(s',r|s,a): it depends on actions of dealer and player
    
    NOTE:        
        If the player/dealer holds multiple 'A', there is at most one card 'A' = 11, because the cards sum must not exceed 21, or he'll go bust and lose the game. 
        Therefore, each time player/dealer gets a 'A', it first is treated as 11, and AceFlag=true. If the cards sum > 21, then in order to prevent going bust, 'A' will be resigned as 1, and re-calculate
        cards sum.
    '''
    def __init__(self):
        '''
        Constructor
        '''
        self.action = ['hit', 'stick']
        self.cardValue = {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, 'J':10, 'Q':10, 'K':10, 'A': 1}      # A=1/11, face cards = 10
        self.card = list(self.cardValue.keys())
        self.playerCards = []
        self.dealerCards = []
        self.CardSum = {'player':0, 'dealer': 0}    
        self.Ace11Flag = {'player': False, 'dealer': False}   # flag of 'A'=11
        self.reward = 0
        self.gameEnd = False    # game over
        
        # initial 2 cards
        for _ in range(0,2) :
            # player
            idx = np.random.randint(0,13)
            self.playerCards.append(self.card[idx])
            self.card_sum(self.card[idx], 'player')
            # dealer
            idx = np.random.randint(0,13)
            self.dealerCards.append(self.card[idx])
            self.card_sum(self.card[idx], 'dealer')
        
        # dealer's showing card
        self.showCard = self.dealerCards[0]

        # natural? 
        if self.CardSum['player'] == 21:
            print()    
            print('Player is NATURAL!!')
            if self.CardSum['dealer'] != 21:
                print('Player Win!!')
                self.reward = 1
            else :
                print('Draw!!')
                self.reward = 0
            self.gameEnd = True
            
            print('Player initial cars: ', self.playerCards)  
            print('Player Ace11Flag: ', self.Ace11Flag['player'])    
            print('Player CardSum: ', self.CardSum['player'])
                  
            print('Dealer initial cars: ', self.dealerCards)
            print('Dealer Ace11Flag: ', self.Ace11Flag['dealer'])
            print('Dealer CardSum: ', self.CardSum['dealer'])      
                          
            print('Game Over!!')
        
        
    def card_sum(self, card, who):
        '''
        calculate player/dealer cards sum
        input param.:
            card: card element
                if card=='A' and AceFlag==False
                    'A' = 11  
                    sum += 11
                    AceFlag = true
                    if sum>21            # special case, 1st 'A': '9','9','A'
                       'A' = 1
                        AceFlag = False
                        sum -= 10
                if card=='A' and AceFlag==true
                    'A' = 1
                    sum += 1
                    if sum>21                # special case: 2nd 'A': 'A',5,5,'A' 
                        AceFlag = False      # stupid player makes a stupid 'hit' decision
                        sum -= 10
            who: player, or dealer    
        '''
        if card=='A' and self.Ace11Flag[who]==False :
            self.CardSum[who] += 11 
            self.Ace11Flag[who] = True
            # prevent going bust, set 'A'=1
            if self.CardSum[who]>21 :
                self.CardSum[who] -= 10
                self.Ace11Flag[who] = False
        elif card=='A' and self.Ace11Flag[who]==True :
            self.CardSum[who] += 1 
            # prevent exception: stupid player makes a stupid 'hit' decision when he got 21
            if self.CardSum[who]>21 :
                self.CardSum[who] -= 10
                self.Ace11Flag[who] = False
        elif self.Ace11Flag[who]==True :
            self.CardSum[who] += self.cardValue[card]
            if self.CardSum[who]>21 :
                self.CardSum[who] -= 10
                self.Ace11Flag[who] = False
        else: 
            self.CardSum[who] += self.cardValue[card]       
                
                
    def one_deal(self, playerAction):
        '''
        dealing one card
        if playerAction == hit
            dealing cards, player first
        if playerAction == stick
            dealer's decision
        '''
        if self.CardSum['player']>21 :
            raise ValueError('NO MORE DEAL; Player has already gone bust!!')
        print('Player action: ', action)    
        # hit action
        if playerAction==self.action[0] :
            # player first
            idx = np.random.randint(0,13)
            self.playerCards.append(self.card[idx])
            
            # cal card sum
            self.card_sum(self.card[idx], 'player')
            
            # check state
            # bust
            if self.CardSum['player']>21 :
                print('player goes bust')
                self.gameEnd = True
                self.reward = -1
            # natural
            elif self.CardSum['player']==21 :
                print('Player is NATURAL!!')
                if self.CardSum['dealer'] != 21:
                    print('Player Win!!')
                    self.reward = 1
                else :
                    print('Draw!!')
                    self.reward = 0
                self.gameEnd = True 
            # continue        
            else :
                self.gameEnd = False
        # stick action
        else :
            # dealer's turn
            print('Dealer\'s turn')
            while self.gameEnd==False :
                # hit
                if self.CardSum['dealer']<17 :
                    print('Dealer hit')
                    # deal
                    idx = np.random.randint(0,13)
                    self.dealerCards.append(self.card[idx])
                    # cal cards sum
                    self.card_sum(self.card[idx], 'dealer')
                    
                    self.gameEnd = False
                # stick, and judge who win
                else :
                    self.gameEnd = True
                    # dealer goes bust, player win
                    if self.CardSum['dealer'] > 21 :
                        print('Dealer goes bust')
                        self.reward = 1
                    # dealer > player, player lose
                    elif self.CardSum['dealer'] > self.CardSum['player'] :
                        print('Dealer cars sum > player\'s')
                        self.reward = -1
                    # dealer == player, draw
                    elif self.CardSum['dealer'] == self.CardSum['player'] :
                        print('Dealer cars sum = player\'s')
                        self.reward = 0
                    # dealer < player, player win    
                    elif self.CardSum['dealer'] < self.CardSum['player'] :
                        print('Dealer cars sum < player\'s')
                        self.reward = 1       
                       
                    
if __name__=='__main__' :
    '''
    init test
    '''
#     for _ in range(1000) :    
#         new = BlackJack()
#         if new.playerCards[0]=='A' and new.playerCards[1]=='A' :
#             print()
#             print('Player initial cars: ', new.playerCards)  
#             print('Player Ace11Flag: ', new.Ace11Flag['player'])    
#             print('Player CardSum: ', new.CardSum['player'])
#             break
#         
#         if new.dealerCards[0]=='A' and new.dealerCards[1]=='A' :
#             print()
#             print('Dealer initial cars: ', new.dealerCards)
#             print('Dealer Ace11Flag: ', new.Ace11Flag['dealer'])
#             print('Dealer CardSum: ', new.CardSum['dealer'])   
#             break
    '''
    play one round
    player's policy: if sum<20, then hit; otherwise, stick 
    '''            
    # init
    new = BlackJack()
    print('Dealer initial cars: ', new.dealerCards)
    print('Player initial cars: ', new.playerCards)
    
    ite = 0
    while new.gameEnd==False :
        ite += 1
        print()
        print('ite = ', ite)
        if new.CardSum['player'] < 20 :
            action = new.action[0]
            new.one_deal(action)
            if new.gameEnd == True :
                break
        else :
            action = new.action[1]
            new.one_deal(action)
    print()        
    print('Dealer cars: ', new.dealerCards)
    print('Dealer AceFlag: ', new.Ace11Flag['dealer'])
    print('Dealer CardSum: ', new.CardSum['dealer'])    
     
    print('Player cars: ', new.playerCards)
    print('Player Ace11Flag: ', new.Ace11Flag['player'])    
    print('Player CardSum: ', new.CardSum['player'])
                            
    print('Player get reward: ', new.reward)                
                    
                         
        
