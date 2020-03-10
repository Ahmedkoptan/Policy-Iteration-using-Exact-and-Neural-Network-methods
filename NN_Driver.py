import numpy as np
from Driver import Driver
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.metrics import mse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


SEED = 520
N_OBS = 20 #number of monte carlo simulation observations
ACTIONS = ['park','move right'] #0 park, or 1 move right

class NN_driver(Driver):
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
        print('Monte Carlo costs are',MonteCarloCosts)

        #now we train the quadratic cost architecture using obtained samples
        X = np.vstack([np.arange(0,self.N,1,float), np.ones(self.N)]).T
        #self.w = np.linalg.lstsq(X,MonteCarloCosts[:-1],rcond=None)[0]

        input_shape = X.shape[1]
        h1 = 10
        h2 = 5
        o = 1
        lr = 0.001
        epochs = 200
        reg = 1e-5
        mypatience = 10
        model = build_model(input_shape,h1,h2,o,lr,epochs,reg,mypatience)

        # The patience parameter is the amount of epochs to check for improvement
        early_stop = keras.callbacks.EarlyStopping(monitor='val_mse', patience=mypatience, min_delta=0,
                                                   restore_best_weights=True)
        early_history = model.fit(X, MonteCarloCosts[:-1],
                                  epochs=epochs, validation_split=0.1, verbose=1,
                                  callbacks=[early_stop], shuffle=True, batch_size=128)

        print('Final model is',model.summary())

        #now we evaluate the policy using the trained architecture
        for i in range(self.N - 1, -1, -1):
            Jtiplus1  = model.predict([[i,1]]).flatten()
            #if policy says park at current state
            if self.policy[i] == ACTIONS[0]:
                self.Jt[i] = (self.pf * self.c[i]) + ((1 - self.pf) * (Jtiplus1))
            #else policy says move right
            else:
                self.Jt[i] = Jtiplus1
        print('Cost to go is:\n',self.Jt)
        return





def build_model(input_shape,hidden_layer_shape_1,hidden_layer_shape_2,output_layer_shape,lr,epochs,reg,mypatience):
    model = Sequential([
        Dense(hidden_layer_shape_1,
                                      activation='relu',
                                      input_shape=([input_shape]),
                                      kernel_regularizer=l2(reg)),
        Dense(hidden_layer_shape_2,
                                      activation='relu',
                                      kernel_regularizer=l2(reg)),

        Dense(output_layer_shape)
    ])
    opti = Adam(learning_rate = lr)
    model.compile(loss='mse',
                  optimizer = opti,
                  metrics=['mae','mse'])
    return model