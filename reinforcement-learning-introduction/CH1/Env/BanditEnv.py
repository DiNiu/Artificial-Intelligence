import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Bandit:
    '''
    stationary k-armed Bandit.
    value estimation:
        Using sample average to update estimation
    Policy update:
        Using epsilon-greedy to explore and exploit
    '''
    
    def __init__(self, kArm, epsilon):
        '''
        param.:
            kArm: number of arms, k-armed bandit problem
            epsilon: epsilon-greedy
        '''
        if not(isinstance(kArm, int)and(kArm>0)) :
            raise ValueError('kArm must be a positive integer')
        else : 
            self.nArm = kArm
        
        if epsilon>=1 :  
            raise ValueError('epsilon must be in the range [0,1)')  
        else :
            self.epsilon = epsilon
        
        self.index = np.arange(self.nArm)
        
        self.reward = []        # reward of each bandit, and the distribution is N(m,1)
        self.nAction = []       # action number of each bandit    
        self.Q = []             # action value of each bandit
        # initialization 
        np.random.seed(17)
        for _ in range(0, self.nArm):
            self.reward.append(np.random.randn())   # m -> N(0,1)
            self.nAction.append(0)
            self.Q.append(0)
        
        self.bestAct = np.argmax(self.reward) 
    
    def banditShow(self):
        '''
        show k-armed bandit's reward distribution
        '''
        data1 = np.random.randn(1000, self.nArm) + self.reward   # reward distribution
        sns.violinplot(data=data1)
        plt.xlabel("Action")
        plt.ylabel("Reward distribution")
        print("Best action is: ", self.bestAct)
        
         
    def getAction(self):
        '''
        get the action At:
            using epsilon-greedy to get the action            
        '''
        tmp = random.uniform(0,1) 
        if tmp<=self.epsilon :                  # explore
            random.shuffle(self.index)
            return self.index[0]
        else :                                  # exploit
            return np.argmax(self.Q)
            
        
        
    def  Qvalue(self, action):
        '''
        estimate Q value
            using sample average to estimate Q(a) <-- Q(a) + (R-Q(a))/N(a)
        '''   
        # generate a reward 
        reward = self.reward[action] + random.gauss(0,1)   # reward: m + N(0,1)
        self.nAction[action] += 1
        
        # sample average
        self.Q[action] += (reward - self.Q[action])/self.nAction[action]
        
        return reward
        
        
                
