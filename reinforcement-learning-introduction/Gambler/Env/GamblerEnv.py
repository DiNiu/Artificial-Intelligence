import numpy as np

class GamblerEnv:
    '''
    Gambler's problem: undiscounted, episodic, finite MDP
        final state: s = 0, or 100
        state sapce: s = [1,99]
        action space: a = [1, min(s,100-s)]
        reward: 0, or 1
        transition: if coil flip = head, p(s',r|s,a) = 0.4
                    else, p(s',r|s,a) = 0.6
                    
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.GOAL = 100                                     # reaching goal of $100
        self.state = [_ for _ in range(0,self.GOAL+1)]      # terminal states: 0, 100
        self.winProb = 0.4                                  # transition probability: p(winning) = p(head) = 0.4
        
    
    def transition(self, state, action, stateValue):
        '''
        transition process: (s,a) -> (s',r)
        calculate qValue: q(s,a)    
        input param.:
            @state: present state 
            @action: action 
            @stateValue: v(s) at present stage
        output param.:
            @qValue: q(s,a)
        NOTE:
            if win, s' = s + a, probability=self.winProb
            else, s' = s - a, probability=1-self.winProb
            
            if lose, reward = 0
            else, reward = 0, 1    
        '''
        if action not in range(0,min(state,self.GOAL-state)+1) :
            raise ValueError('Prohibitive stake')
        # reward
        # if win, reward = 0,1
        if (state+action)==self.GOAL :
            rewardWin = 1
        else: 
            rewardWin = 0
        
        # if lose, reward = 0,-1    
        if (state-action)==0 :
            rewardLoss = -1
        else: 
            rewardLoss = 0

        qValue = self.winProb*(rewardWin+stateValue[state+action]) + (1-self.winProb)*(rewardLoss+stateValue[state-action])
        
        return qValue
    
############################
### test        
if __name__ == '__main__' :
    # init
    new = GamblerEnv()
    stateValue = np.zeros(new.GOAL+1)   # state value v(s)
    state = 1  # capital
    qValue = []
    action = [_ for _ in range(1,min(state,new.GOAL-state)+1)]
    # gamble 
    for act in action:     # stake
        qValue.append(new.transition(state, act, stateValue))
    
    stateValue[state] = max(qValue)    
    optPolicy = action[np.argmax(qValue)]
            
    print(stateValue[state])
    print(qValue)
    print(optPolicy)   
        
        
        
        
