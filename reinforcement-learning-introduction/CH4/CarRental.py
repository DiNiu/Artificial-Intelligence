###
import Env.CarRentalEnv as env
import numpy as np
#import matplotlib

# initialization
ERROR = 1e-3
STEPS = 100
new = env.CarRentalEnv()


def value_iterateion():
    '''
    Value iteration:
        policy estimation & improvement
    '''
    lastValue = np.zeros((new.MAX_CARS+1,new.MAX_CARS+1), dtype='float')       # v(s) = max(q(s,a), a)
    stateValue = np.zeros((new.MAX_CARS+1,new.MAX_CARS+1), dtype='float')       # v(s) = max(q(s,a), a)
    optPolicy = np.zeros((new.MAX_CARS+1,new.MAX_CARS+1), dtype='int8')        # a = pi(s) = argmax(q(s,a),a)
    for step in range(0,STEPS) :
        print('step ', step)
        print()
        for carA in range(0,new.MAX_CARS+1) :         # state of loc A
            for carB in range(0, new.MAX_CARS+1) :    # state of loc B
                lastValue[carA, carB] = stateValue[carA, carB]
                state= [carA, carB]
                qValue = []
                for act in new.action :
                    # action constraint
                    if act<0 : # B -> A  
                #        print('Action = {}, which means, from B to A, moving cars: {}'. format(act, abs(act))) 
                        if state[0]-act>new.MAX_CARS :
                        #    print('Oops...# of cars in loc A overflow!!!')
                        #    print()
                            continue
                        if state[1]+act<0 :
                        #    print('Oops...# of cars in loc B empty!!')
                        #    print()
                            continue
                    else : # A -> B
                #        print('Action = {}, which means, from A to B, moving cars: {}'. format(act, act))
                        if state[0]-act<0 :
                        #    print('Oops...# of cars in loc A empty!!')
                        #    print()
                            continue
                        if state[1]+act>new.MAX_CARS :
                        #    print('Oops...# of cars in loc B overflow!!')
                        #    print()
                            continue
    
                    # policy evaluation
                    qValue.append(new.transition(state, act, stateValue))
                # policy improvement
                stateValue[carA,carB] = max(qValue)
                optPolicy[carA, carB] = new.action[np.argmax(qValue)]    
        # convergence?
        err = np.sum(np.abs(stateValue-lastValue))    
        if err<=ERROR :
            print('Stable state value at step ', step)
            print(stateValue)
            break;   
             
        if step==STEPS-1 :
            print('Unstable stable v(s):')
            print(stateValue)
    
    for i in range(0,new.MAX_CARS+1) :
        for j in range(0,new.MAX_CARS+1) :
            print(repr(optPolicy[i,j]).rjust(4), end=' ')
        print()          
    
    
def main():
    value_iterateion()   

        
if __name__ == '__main__':
    main()
