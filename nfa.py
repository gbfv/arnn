


class NFA():

    def __init__(self, sigma, init_state, final_states):
        self.sigma = sigma
        self.init_state = init_state
        self.final_states = final_states

    def predict(self, X):
        res = []
        current_states = {self.init_state}
        for x in X:
            current_states.update(self.sigma.get((self.init_state, 300), set())) # epsilon transitions depuis l'état initial
            #print(f"Current states: {current_states}, input: {x}")
            next_states = {self.init_state}
            for state in current_states:
                next_states.update(self.sigma.get((state, x), set()))
            current_states = next_states

            is_final = any(state in self.final_states for state in current_states)
            res.append(int(is_final))
            if is_final:
                current_states = {self.init_state} #reset to initial state
        return res