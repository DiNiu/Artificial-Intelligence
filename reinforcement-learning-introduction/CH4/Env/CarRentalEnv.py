###
import numpy as np
import scipy.stats as stat
from math import *

class CarRentalEnv:
    '''
    Jack's car rental 
        two locations: A, B
        car rental fee: $10/car
        car moving fee: $2/car
        max car moving: 5 cars/night
        max car holding: 20 cars/(day,location)
        car request prob. per day: A -> poisson(3,n), B -> poisson(4,n)
        car return prob. per day: A -> poisson(3,n), B -> poisson(2,n)
        state space: # of cars/(end of day, location), [0,20]. It can be treated as a grid, s[0] = A, s[1] = B
        action space: # of cars moving from A to B/night, [-5,5]. 
                      positive # means moving car from A to B; negative # means moving car from B to A  
        Reward: car rental fee - car moving fee              
    '''
    
    def __init__(self):
        '''
        Constructor
        '''
        self.CAR_RENTAL = 10
        self.CAR_MOVE_FEE = 2
        self.MAX_MOVE_CARS = 5  
        self.MAX_CARS = 20      
        self.CAR_REQUEST = {'A':3, 'B':4}   # A -> poisson(3,n), B -> poisson(4,n)
        self.CAR_RETURN = {'A':3, 'B':2}    # A -> poisson(3,n), B -> poisson(2,n)
        
        self.action = [_ for _ in range(-self.MAX_MOVE_CARS, self.MAX_MOVE_CARS+1)]  # action space
        self.GAMMA = 0.9     # discounted coeff.
        self.poiTable = {}  # poisson table
        
        
    def poisson_pmf(self, n, lam):
        '''
        calculate poisson pmf: 
            p(n,lam) = exp(-lam)*lam^n/n!
        also store those (n,lam) that have already been calculated, 
        so that next time (n,lam) pair is queried, use table search 
        '''
        key = n*10 + lam
        if key not in self.poiTable.keys():
            self.poiTable[key] = stat.poisson.pmf(n,lam)
        return self.poiTable[key]
    
       
    def transition(self, state, action, stateValue):
        '''
        calculate the next state and reward: (s,a) -> s', r
        input param.:
            @state: # of cars at loc A and B at the end of day1
            @action: # of car move action overnight1, but make sure action won't make loc A & B empty or overflow
            @stateValue: v(s)
        output param.:    
            @qValue: q(s,a) = r(s,a) + gamma * sum(v(s')*p(s'|s,a), s') - action*slef.CAR_MOVE_FEE, r(s,a) = sum(r*p(r|s,a),r)
                     By using accumulated sum, we can treat p(s',r|s,a) = p(s'|s,a) = p(r|s,a), and then
                     q(s,a) = sum(p(s',r|s,a)*(r+gamma*v(s')), s') - action*slef.CAR_MOVE_FEE
        local important param.:
            nextState: # of cars at loc A and B  at the end of day2
            reward: reward at day2
            prob: p(s',r|s,a)
        NOTE:    
            s'[0] = ((s[0]-a)^-car_quest)'+car_return)^, s'[1] = (((s[1]+a)^-car_quest)'+car_return)^
            s.t.: 
                s,s' = [0,self.MAX_CARS]
                (x)' means max(x, 0)
                (x)^ means min(x, self.MAX_CARS) 
                
            r = (carReqA+carReqB)*self.CAR_RENTAL
            s.t.:
                carReqA = [0,(s[0]-a)^]
                carReqB = [0,(s[1]+a)^]
        '''
        nextState = [0,0]       # state at the end of day2
        numOfCarsLocA = {}      # number of cars of loc A at each time step: move, request
        numOfCarsLocB = {}      # number of cars of loc B
        reward = 0              # reward at day2
        qValue = 0              # q(s,a)
        
        # NIGHT1
        if (state[0]-action)<0 :
            raise ValueError('# of cars in loc A is less than present action')
        
        if (state[0]-action)>self.MAX_CARS :
            raise ValueError('# of cars in loc A will exceed MAX_CARS with present action')
        
        if (state[1]+action)<0 :
            raise ValueError('# of cars in loc B is less than present action')
         
        if (state[1]+action)>self.MAX_CARS :
            raise ValueError('# of cars in loc B will exceed MAX_CARS with present action')
        
        # after moving, the # of cars
        numOfCarsLocA['MOVE'] = state[0] - action     #  car # in loc A
        numOfCarsLocB['MOVE'] = state[1] + action     #  car # in loc A
#        print('After car moving, state is: {}, {}'. format(numOfCarsLocA['MOVE'], numOfCarsLocB['MOVE'])) 

        qValue -= self.CAR_MOVE_FEE*abs(action) 

        # DAY2 
        # Car request can't exceeds the present car storage        
        for carReqA in range(0, numOfCarsLocA['MOVE']+1) :
            for carReqB in range(0, numOfCarsLocB['MOVE']+1) :    
                probReqA = self.poisson_pmf(carReqA,self.CAR_REQUEST['A'])
                probReqB = self.poisson_pmf(carReqB,self.CAR_REQUEST['B'])
#                print('DAY2 CAR request: {}, {}, prob. is {}'. format(carReqA, carReqB, probReqA*probReqB))
                            
                # After renting, the # of cars            
                numOfCarsLocA['Req'] = numOfCarsLocA['MOVE'] - carReqA  # 
                numOfCarsLocB['Req'] = numOfCarsLocB['MOVE'] - carReqB
#                print('After car renting, state is: {}, {}'. format(numOfCarsLocA['Req'], numOfCarsLocB['Req']))
#                print()
                # car return can't exceeds the maximum storage            
                for carRtnA in range(0, self.MAX_CARS-numOfCarsLocA['Req']+1) :
                    for carRtnB in range(0, self.MAX_CARS-numOfCarsLocB['Req']+1) :
                        probRtnA = self.poisson_pmf(carRtnA,self.CAR_RETURN['A'])
                        probRtnB = self.poisson_pmf(carRtnB,self.CAR_RETURN['B'])
#                        print('DAY2 CAR return: {}, {}, prob. is: {}'. format(carRtnA, carRtnB, probRtnA*probRtnB))    
                        
                        # Final car number at the end of day
                        nextState[0] = numOfCarsLocA['Req'] + carRtnA
                        nextState[1] = numOfCarsLocB['Req'] + carRtnB
#                        print('After car return, final state is: {}, {}'. format(nextState[0], nextState[1]))
         
                        # reward at DAY2 
                        reward = self.CAR_RENTAL*(carReqA+carReqB)        
                        prob = probReqA * probRtnA * probReqB * probRtnB    # 
                        qValue += prob * (reward + self.GAMMA*stateValue[nextState[0],nextState[1]])
                        
        return qValue        
    

    
if __name__ == "__main__" :
    new = CarRentalEnv()
    state = [5, 5]
    print('At the end of DAY1, # of cars in loc A is: {}, # of cars in loc B is: {}'. format(state[0], state[1]))
    stateValue = np.zeros((new.MAX_CARS+1,new.MAX_CARS+1), dtype='float')
    qValue = []
    for act in new.action :
        print()
        # action constraint
        if act<0 : # B -> A  
            print('Action = {}, which means, from B to A, moving cars: {}'. format(act, abs(act))) 
            if state[0]-act>CarRentalEnv().MAX_CARS :
                print('Oops...# of cars in loc A overflow!!!')
                print()
                continue
            if state[1]+act<0 :
                print('Oops...# of cars in loc B empty!!')
                print()
                continue
        else : # A -> B
            print('Action = {}, which means, from A to B, moving cars: {}'. format(act, act))
            if state[0]-act<0 :
                print('Oops...# of cars in loc A empty!!')
                print()
                continue
            if state[1]+act>CarRentalEnv().MAX_CARS :
                print('Oops...# of cars in loc B overflow!!')
                print()
                continue
        
        qValue.append(CarRentalEnv().transition(state, act, stateValue))
                
    print(qValue)
    
