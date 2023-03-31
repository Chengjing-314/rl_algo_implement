import numpy as np
from random_walk import RandomWalk

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
    
    for i in range(num_episodes):
        rewards = [rd.get_current_reward()]
        states = [rd.current_state]
        while not rd.done or len(states) < rd.max_steps:
            rd.step()
            rewards.append(rd.get_current_reward())
            states.append(rd.current_state)
        
        rewards = np.array(rewards)
        states = np.array(states)
        
        T = len(states) - 1
        for t in range(T):
            iter_state = states[t]
            gt = 0 
            for n in range(1, T-t):
                gttn_sum = rewards[t:t+n].sum()
                gt += (td.lda ** n) * gttn_sum
            
            td.update(iter_state, gt - td.value(iter_state))
    
    np.set_printoptions(precision=2, suppress=True)
    print(f"Final values {td.states_value}")
    

if __name__ == "__main__":
    main()
            
    
    
    
    
    
        
    