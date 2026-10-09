import random
import numpy as np
import matplotlib.pyplot as p

def getSamplar():
    mu=np.random.normal(0,10)
    sd=abs(np.random.normal(5,2))
    getSample=lambda: np.random.normal(mu,sd)
    return getSample

def e_greedy(Q, e):
   #random choose a number from 1:10
   #if the number < 10*e  use second random action A= random
      # all action choose randomly
   #else use the argmax  A= argmax
      #choose the largest reward from previous Q
      #if there only one action has max reward, A= argmax
      # else randomly choose one action from all actions with highest reward

    import random
    num=random.randint(1,10)
    if num < 10*e:
        action= random.randint(1,10)
    else:
        getMaxValue = max(Q.values())
        actionindex=list({key for key, value in Q.items() if value == getMaxValue})
        if len(actionindex) == 1:
            action= actionindex[0]
        else:
            import random
            action=random.choice(actionindex)

    return action
    
def upperConfidenceBound(Q, N, c):
#   if there have actions out of all actions non-taken
#       if there is 1 action is non taken
#           take that action
#       else
#           random choose one action among all of non taken actions
#           take that action

#   else all actions are taken
#       Compute all actions value by using the A(t) function
#       action= argmax form previous action values
#       if there are multiple actions have same highest values
#           randomly choose one action
    UCB={}
    for a, r in Q.items():
        UCB[a] = r + (c * sqrt(ln(t) / N[a]))
    nontakenaction=[a for a, co in N.items() if co == 0]
    if len(nontakenaction)==1:
        action=nontakenaction[0]
    else:
        if len(nontakenaction)>1:
            import random
            action=random.choice(nontakenaction)
        else:
            getMaxReward = max(UCB.values())
            actionid = list({key for key, value in Q.items() if value == getMaxReward})
            if len(actionid) == 1:
                action = actionid[0]
            else:
                import random
                action= random.choice(actionid)

    return action

def updateQN(action, reward, Q, N):

#upadated reward function = Q(a) <- Q(a)+ 1/N(A)[R-Q(A)]
#updated Count function= from the N(A) dictionary to add one
    QNew={}
    for a, r in Q.items():
        QNew[a] = r
        QNew.update({action:Q[a]+(1/N[a])*(reward-Q[a])})
    NNew={}
    for a, c in N.items():
        NNew[a] = c
        NNew.update ({action:N[a]+1})

    return QNew, NNew

def decideMultipleSteps(Q, N, policy, bandit, maxSteps):

#update maxsteps
    action=[]
    for i in range(maxSteps):
        ac= policy(Q,N)
        action.append(ac)

    reward=[]
    for i in range(maxSteps):
       re= bandit(action)
       reward.append(re)

    actionReward= tuple(zip(action,reward))

    Qn= lambda Q,N: updateQN(action,reward,Q,N)[0]
    Nn= lambda Q,N: updateQN(action,reward,Q,N)[1]

    return {'Q':Qn, 'N':Nn, 'actionReward':actionReward}

def plotMeanReward(actionReward,label):
    maxSteps=len(actionReward)
    reward=[reward for (action,reward) in actionReward]
    meanReward=[sum(reward[:(i+1)])/(i+1) for i in range(maxSteps)]
    plt.plot(range(maxSteps), meanReward, linewidth=0.9, label=label)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')

def main():
    np.random.seed(2020)
    K=10
    maxSteps=1000
    Q={k:0 for k in range(K)}
    N={k:0 for k in range(K)}
    testBed={k:getSamplar() for k in range(K)}
    bandit=lambda action: testBed[action]()
    
    policies={}
    policies["e-greedy-0.5"]=lambda Q, N: e_greedy(Q, 0.5)
    policies["e-greedy-0.1"]=lambda Q, N: e_greedy(Q, 0.1)
    policies["UCB-2"]=lambda Q, N: upperConfidenceBound(Q, N, 2)
    policies["UCB-20"]=lambda Q, N: upperConfidenceBound(Q, N, 20)
    
    allResults = {name: decideMultipleSteps(Q, N, policy, bandit, maxSteps) for (name, policy) in policies.items()}
    
    for name, result in allResults.items():
         plotMeanReward(allResults[name]['actionReward'], label=name)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    


if __name__=='__main__':
    main()
