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
    rd = RandomWalk() # default is 19 states
    td = TdLambda(lda = 0.8, alpha = 0.01)
    
    states = range(rd.num_states + 1)
    
    eligibility_trace = {s: 0 for s in states}
    
    for _ in tqdm(range(num_episodes)):
        cnt = 0
        # initialize S 
        cur_eg_trace = eligibility_trace.copy() # eligibility trace for current episode
        while not rd.done:
            state = rd.current_state
            rd.step()
            delta = rd.get_current_reward() + td.value(rd.current_state) - td.value(state)
            cur_eg_trace[state] += 1
            
            # update on the fly
            for s in states: # update all states
                td.update(s, delta * cur_eg_trace[s])
                cur_eg_trace[s] *= td.lda # update eligibility trace
    
        rd.random_reset() # reset the environment
        
    np.set_printoptions(precision=2, suppress=True)
    print(f"Final values {td.states_value}")
    

if __name__ == "__main__":
    main()
            