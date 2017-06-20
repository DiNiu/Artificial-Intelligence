import numpy as np
import matplotlib.pyplot as plt
import Env.GamblerEnv as env

STEPS = 100
ERROR = 1e-3
PH = 0.4

new = env.GamblerEnv()

def value_iteration():
    stateValue = np.zeros(new.GOAL+1)   # state value v(s)
    lastValue = np.zeros(new.GOAL+1)
    optPolicy = np.zeros(new.GOAL+1, dtype='uint8')    # optimal policy at each state
    
    for step in range(0,STEPS) :
        print('step: ', step)
        for state in range(1,new.GOAL) :    # two terminal states: 0,100
            action = [_ for _ in range(1,min(state,new.GOAL-state)+1)]  # action space
            qValue = []
            lastValue[state] = stateValue[state]
            # policy evaluation
            for act in action :
                qValue.append(new.transition(state, act, stateValue))
            # policy improvement    
            stateValue[state] = max(qValue)
            optPolicy[state] = action[np.argmax(qValue)]
        
        # convergence?
        delta = np.sum(np.abs(stateValue-lastValue))
        if delta < ERROR :
            print('stable v(s): ')
            print(stateValue)
            print()
            print('stable optimal policy:')
            print(optPolicy)
            print()
            break
        # time up
        if step==STEPS-1 :
            print('unstable v(s): ')
            print(stateValue)
            print()
            print('unstable optimal policy:')
            print(optPolicy)
            print()  
                      
    plt.figure()
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.plot(range(1,new.GOAL), stateValue[1:new.GOAL])        
    
    plt.figure()
    plt.xlabel('Capital')
    plt.ylabel('optimal policy')
    plt.plot(range(1,new.GOAL), optPolicy[1:new.GOAL],'r.')
    
    return optPolicy
    
def real_Play(credit, policy):
    '''
    
    ''' 
    step = 0   
    while credit>0 and credit<new.GOAL :
        step += 1
        print('Step: ', step)
        # stake
        stake = policy[credit]  #
        print('stake: ', stake)
        # coin flip
        if np.random.binomial(1, PH)==1 :   # win 
            credit += stake
            print('WIN, present credit: ', credit)
        else :  # lose
            credit -= stake
            print('LOSS, present credit: ', credit)
        print()      
            
    if credit==0 :
        print('$0')
    elif credit==new.GOAL: 
        print('$100')   
    else:
        print('Something is wrong...')        
    
    return credit
             
def main():
    policy = value_iteration()
    
    print()
    
#     print('Now, real game begin:')
#     credit = np.random.random_integers(1,new.GOAL-1)
#     print('Your initial credit is: ', credit)
#     real_Play(credit, policy)
    credit = np.zeros(5000, dtype='uint8')
    for i in range(0,5000) :
        credit[i] = real_Play(50, policy)
    
    aveCredit50 = np.mean(credit)
    print('Average return of $50 is: ', aveCredit50)
    
    plt.show()
    
if __name__ == '__main__':
    main()
