import numpy as np

SEED = 520
ACTIONS = ['park','move right'] #0 park, or 1 move right

class Driver:
    def __init__(self, pf, f, c, N):
        self.pf = pf
        self.f = f.copy()
        self.c = c.copy()
        self.N = N
        # array that tells you what action to do at what state, i.e. policy. element at index i (stage i) is the action taken by current policy
        self.policy = np.where(np.zeros(N,int)==0,ACTIONS[0],ACTIONS[1]) #0 is park, 1 is move right
        self.Jt = np.zeros(N + 1, float)
        self.Jt[N] = self.c[N]


    # The function that calculates new policy to take given previous policy evaluation. This will iterate over every possible state
    # compute the one step lookahead plus cost to go of current policy at next state, and choose action based on
    def policy_imp(self):
        newpolicy = self.policy.copy()
        policy_changed = False
        for i in range(self.N):
            # if spot is free
            if self.f[i] == 1:
                # min of (either park and evaluate cost of moving and to go (c[i]), or move right and evaluate cost of moving and to go ())
                if self.c[i] <= self.Jt[i + 1]:
                    # park if better than or equal to cost to go of current policy
                    newpolicy[i] = ACTIONS[0]
                # else cost of current spot cost is not better than optimal of right
                else:
                    # move right
                    newpolicy[i] = ACTIONS[1]
            else:
                newpolicy[i] = ACTIONS[1]  # move right
            if newpolicy[i] != self.policy[i]:
                policy_changed = True
        self.policy = newpolicy
        return policy_changed


    #This function evaluates cost to go at every stage given current policy
    def policy_eval(self):
        for i in range(self.N - 1, -1, -1):
            #if policy says park at current state
            if self.policy[i] == ACTIONS[0]:
                self.Jt[i] = (self.pf * self.c[i]) + ((1 - self.pf) * (self.Jt[i+1]))
            #else policy says move right
            else:
                self.Jt[i] = self.Jt[i+1]
        print('Cost to go is:\n',self.Jt)
        return


    def policy_iter(self):
        policy_changed = True
        i = 0
        print('\n\n')
        while policy_changed:
            print('Iteration',i,'Evaluating policy')
            print('Current policy is:\n',self.policy)
            self.policy_eval()
            print('Done evaluating. Improving policy')
            policy_changed = self.policy_imp()
            print('New policy is:\n',self.policy)
            print('Current policy:','changed' if policy_changed else 'not changed. Iterations stop.')
            i += 1
            print('\n\n')
        return self.policy

