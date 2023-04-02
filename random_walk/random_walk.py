import numpy as np 

class RandomWalk:
    
    def __init__(self, num_states = 19, discrete = True, step_size = 1, max_steps = 1000):
        self.num_states = num_states
        self.start = (num_states - 1) // 2
        self.left_terminal = 0
        self.right_terminal = num_states + 1
        self.current_state = self.start
        self.step_size = step_size
        self.discrete = discrete
        self.done = False
        self.max_steps = max_steps
        
        
    def reset(self):
        self.current_state = self.start
        self.done = False
        return self.current_state
    
    def random_reset(self):
        self.current_state = np.random.choice(range(1, self.num_states))
        self.done = False
        return self.current_state
        
    
    def step(self):
        direction = np.random.choice([-1, 1])
        new_state = self.current_state
        current_step = 0 
        if self.discrete:
            current_step = self.step_size
        else:
            current_step = np.random.uniform(self.step_size[0], self.step_size[1])
            
        new_state += direction * current_step
        
        if new_state <= self.left_terminal:
            new_state = self.left_terminal
            self.done = True
        elif new_state >= self.right_terminal:
            self.done = True
            new_state = self.right_terminal
        
        self.current_state = new_state
        
        return direction, current_step
        
    
    def reward(self, state):
        if state == self.left_terminal:
            return -1
        elif state == self.right_terminal:
            return 1
        else:
            return 0
        

    def get_current_reward(self):
        return self.reward(self.current_state)