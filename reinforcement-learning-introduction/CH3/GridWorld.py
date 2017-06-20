import numpy as np
import Env.GridWorldEnv as env
import matplotlib.pyplot as plt

GAMMA = 0.9
new = env.GridWorldEnv()

ERROR = 1e-3 
STEPS = 200

def valueSim():
    '''
    policy evaluation: v(s) at the default policy 
        v(s) = E{Gt|s} = sum(pi(a|s)*q(s,a), a)
        q(s,a) = r(s,a) + gamma*sum(p(s'|s,a)*v(s'),s')
        r(s,a) = sum(r*p(r|s,a), r)
        where, p(s',r|s,a) = p(s'|s,a) = p(r|s,a) = 1
    ''' 
    lastValue = np.zeros((new.SIZE,new.SIZE),dtype='float')
    for step in range(0,STEPS):
        new.value = np.zeros((new.SIZE,new.SIZE),dtype='float')
        for i in range(0,new.SIZE) :
            for j in range(0,new.SIZE) :
                    for act in new.action :
                        sPrime = new.next_state[i][j][act]
                        new.value[i,j] += new.tran[act] * (new.reward[i][j][act] + GAMMA * lastValue[sPrime[0], sPrime[1]]) 
        
        err = np.sum(np.abs(new.value-lastValue))
        
        if err<=ERROR :
            print('stable v(s) at step :', step)
            print(new.value)
            break                
        else :
            lastValue = new.value
            
        if step==STEPS-1 :
            print('Unstable stable v(s):')
            print(new.value)
    
    

def optValueSim():
    '''
    calculate optimal policy and related v(s): policy evaluation & improvement using value iteration
        optimal v(s) = max(q(s,a), a)
        q(s,a) = r(s,a) + gamma*sum(p(s'|s,a)*v(s'),s')
        r(s,a) = sum(r*p(r|s,a), r)
        where, p(s',r|s,a) = p(s'|s,a) = p(r|s,a) = 1
    '''
    lastValue = np.zeros((new.SIZE,new.SIZE),dtype='float')
    for step in range(0,STEPS):
        optPolicy = []  
        new.value = np.zeros((new.SIZE,new.SIZE),dtype='float')
        for i in range(0,new.SIZE) :
            optPolicy.append([])
            for j in range(0,new.SIZE) :
                tmp = []
                # policy evaluation
                for act in new.action :
                    sPrime = new.next_state[i][j][act]
                    tmp.append(new.reward[i][j][act] + GAMMA * lastValue[sPrime[0], sPrime[1]])     
                # policy improvement
                new.value[i,j] = max(tmp)                                                       
                optPolicy[i].append(new.action[np.argmax(tmp)])
                
        err = np.sum(np.abs(new.value-lastValue))
        if err<=ERROR :
            print('stable v(s) at step :', step)
            print(new.value)
            break                
        else :
            lastValue = new.value
            
        if step==STEPS-1 :
            print('Unstable stable v(s):')
            print(new.value)        
    
    for i in range(0,new.SIZE) :
        for j in range(0,new.SIZE) :
            print(optPolicy[i][j].rjust(4), end=' ')
        print()    
    

def main():
    print('Default policy:')
    valueSim()
     
    print('Optimal policy')
    optValueSim()
     
    

if __name__ == '__main__':
    main()
