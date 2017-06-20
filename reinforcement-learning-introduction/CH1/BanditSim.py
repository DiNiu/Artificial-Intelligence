import numpy as np
import Env.BanditEnv as env
import matplotlib.pyplot as plt

kArm = 10
eps = [0, 0.01, 0.1]

def banditSim(horizon, run):
    '''
    record the reward in each run, best action times, and Q value of each action 
    '''
    aveReward = np.zeros((horizon,len(eps)), dtype='float')       # average reward
    bestActCnt = np.zeros((horizon,len(eps)), dtype='float')    # best action count
    for i in range(0, len(eps)) :
        for cycle in range(0,run) :
            Bandit = env.Bandit(kArm, eps[i])
            for step in range(0,horizon) :
                action = Bandit.getAction()
                aveReward[step][i] += Bandit.Qvalue(action)
                if action==Bandit.bestAct :
                    bestActCnt[step][i] += 1
    
    aveReward /= run            
    bestActCnt /= run

    plt.figure()
    plt.plot(range(horizon), aveReward)
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend(["eps = 0","eps = 0.01", "eps = 0.1"])
    
    plt.figure()
    plt.plot(range(horizon), bestActCnt)
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend(["eps = 0","eps = 0.01", "eps = 0.1"]) 
    
def main():
    # show the reward distribution of the k-armed bandit
    env.Bandit(kArm, eps[0]).banditShow()
    
    # simulation
    banditSim(1000, 2000)
    
    # show 
    plt.show()
    
    
    
if __name__ == '__main__':
    main()
