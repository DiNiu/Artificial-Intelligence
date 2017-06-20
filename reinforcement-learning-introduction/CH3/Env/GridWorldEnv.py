import numpy as np


class GridWorldEnv:
    '''
    GridWorld Env: stationary environment
        state space {S}: 5*5 grids
        action space {A}: left, right, up, down
        action prob. pi(a|s): equiprobability
        state tran. p(s'|s,a): 1
        special state transitiOn:
            A(0,1) -> A'(4,1)
            B(0,3) -> B'(2,3)
        Reward: normal actions, R = 0
                actions taking agent off the grid, R = -1
                actions movings agent out of special state A(0,1), R = 10
                                                           B(0,3), R = 5
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.SIZE = 5                           # 5*5 grid
        self.action = ['L', 'U', 'R', 'D']      # left, up, right, down
        self.tran = {'L':0.25, 'U':0.25, 'R':0.25, 'D':0.25}    # default policy: uniform sampling pi(a|s)= 0.25
        self.A = [0, 1]         # special state A position
        self.APrime = [4, 1]    # special state A' position
        self.B = [0, 3]         # special state B position
        self.BPrime = [2, 3]    # special state B' position

        # initialize value of each state
        self.value = np.zeros((self.SIZE,self.SIZE), dtype='float')
        
        # query s' and reward using table search: [grid x][grid y][action]
        self.next_state = []
        self.reward = []
        for i in range(0,self.SIZE) :
            self.next_state.append([])
            self.reward.append([])
            for j in range(0,self.SIZE) :
                action = {}
                reward = {}
                ## state machine switch   
                # up action
                if i==0 :
                    action['U'] = [i,j]
                    reward['U'] = -1
                else :
                    action['U'] = [i-1,j]
                    reward['U'] = 0
                    
                # down action
                if i==self.SIZE-1 :
                    action['D'] = [i,j]
                    reward['D'] = -1
                else :
                    action['D'] = [i+1,j]        
                    reward['D'] = 0
                
                # left action
                if j==0 :
                    action['L'] = [i,j]
                    reward['L'] = -1
                else :
                    action['L'] = [i,j-1]
                    reward['L'] = 0
                        
                # right action
                if j==self.SIZE-1 :
                    action['R'] = [i,j]
                    reward['R'] = -1
                else :
                    action['R'] = [i,j+1]
                    reward['R'] = 0
                
                ## special case
                # A position
                if [i,j] == self.A :
                    action['L'] = action['R'] = action['U'] = action['D'] = self.APrime
                    reward['L'] = reward['R'] = reward['U'] = reward['D'] = 10 
                
                # B position
                if [i,j] == self.B :
                    action['L'] = action['R'] = action['U'] = action['D'] = self.BPrime
                    reward['L'] = reward['R'] = reward['U'] = reward['D'] = 5 
                    
                self.next_state[i].append(action)
                self.reward[i].append(reward)
                    
                     
            
        
if __name__ == '__main__' :
    new = GridWorldEnv(0.9)
    for i in range(0,new.SIZE) :
        for j in range(0,new.SIZE) :
            for act in new.action :
                print('action: ', act)
                print("state: {}, next state: {}". format([i,j], new.next_state[i][j][act]))
                    
        
            
