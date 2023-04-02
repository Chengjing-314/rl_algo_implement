import numpy as np
from random_walk import RandomWalk
from tqdm import tqdm

class TdLambda:
    
    def __init__(self, lda = 0.8, alpha = 0.01, num_states = 19):
        self.lda = lda
        self.alpha = alpha
        self.states_value = np.zeros(num_states + 2)
        
    def value(self, state):
        return self.states_value[state]
    
    def update(self, state, diff):
        self.states_value[state] += self.alpha * diff
        
        
        

def main(num_episodes = 1000):
    rd = RandomWalk()
    td = TdLambda()
    thres = 1e-3
    
    for _ in tqdm(range(num_episodes)):
        states = [rd.current_state]
        rewards = [0]
        
        while not rd.done:
            rd.step()
            states.append(rd.current_state)
            rewards.append(rd.get_current_reward())
        
        states = np.array(states)
        rewards = np.array(rewards)
    
        T = len(states) - 1
        for t in range(T):
            gt_lambda = 0 
            state = states[t]
            for n in range(1, T-t):
                gt_tn = np.sum(rewards[t+1:t+n+1]) + td.value(states[t+n]) # sum from t to t+n, r1 + r2 + ... + rn + V(Sn)
                lambda_power = np.power(td.lda, n-1) # lambda^(n-1)
                if lambda_power < thres: # early termination
                    break
                
                gt_lambda += lambda_power * gt_tn # lambda^(n-1) * (r1 + r2 + ... + rn + V(Sn))
            
            gt_lambda *=(1 - td.lda)
            if lambda_power > thres: # if not early termination
                gt_lambda += lambda_power * rewards[-1] # add the last reward term
            
            td.update(state, gt_lambda - td.value(state)) # update V(S)
                
        rd.random_reset() # reset the environment
            
    np.set_printoptions(precision=2, suppress=True)
    print(f"Final values {td.states_value}")
    

if __name__ == "__main__":
    main()
            
    
    
    
    
    
        
    