import numpy as np
from Driver import Driver
SEED = 520
N_OBS = 20 #number of monte carlo simulation observations
ACTIONS = ['park','move right'] #0 park, or 1 move right

class QuadDriver(Driver):
    def __init__(self,pf, f, c, N):
        super().__init__(pf, f, c, N)

    #This function evaluates cost to go at every stage given current policy
    def policy_eval(self):
        MonteCarloCosts = self.Jt.copy()
        #for every state, we need to sample state-cost pairs using Monte Carlo simulation. Simulation follows current policy
        for i in range(self.N - 1, -1, -1):
            Jtilda = np.zeros(N_OBS,float)
            for o in range(N_OBS):
                np.random.seed(o)
                # Generate a random probability vector using new seed
                probs = np.random.rand(self.N)

                # Find cost of closest parking spot due to Greedy Heuristic
                p = False
                # set k+1 spot to free and get cost with greedy heuristic
                fobs = np.zeros_like(self.f)
                # Generate observations for remaining spots based on pf
                fobs[(np.where(self.pf > probs))] = 1
                iter=i
                while iter < self.N and not p:
                    if fobs[iter] == 1 and self.policy[iter] == ACTIONS[0]:
                            Jf = self.c[iter]
                            p = True
                    else:
                        iter += 1
                if not p:
                    Jf = self.c[self.N]
                Jtilda[o] = Jf

            # Obtain Monte Carlo simulation average
            MonteCarloCosts[i] = Jtilda.mean()

        #now we train the quadratic cost architecture using obtained samples
        X = np.vstack([np.arange(0,self.N,1,float)**2, np.arange(0,self.N,1,float), np.ones(self.N)]).T
        self.w = np.linalg.lstsq(X,MonteCarloCosts[:-1],rcond=None)[0]
        print('The cost approximation architecture weights are:',self.w)

        #now we evaluate the policy using the trained architecture
        for i in range(self.N - 1, -1, -1):
            Jtiplus1 = np.matmul([i**2, i , 1], self.w)
            #if policy says park at current state
            if self.policy[i] == ACTIONS[0]:
                self.Jt[i] = (self.pf * self.c[i]) + ((1 - self.pf) * (Jtiplus1))
            #else policy says move right
            else:
                self.Jt[i] = Jtiplus1
        print('Cost to go is:\n',self.Jt)
        return
