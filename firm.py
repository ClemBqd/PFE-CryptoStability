from mesa import Agent

class Firm(Agent):
         '''
         Create a new Firm agent.

         Args:
            unique_id: Unique identifier for the agent. put 2
            growth factor ???
        '''
    def __init__(self, unique_id, model):
        self.capital = self.random.randint(1000, 20000)
        self.production = 0
        self.wealth = 0 # = capital - debt
        self.debt = 0
    
    # to put in BtcModel
    def production(self):
        self.production = "equation 8"
    
    def give_salaries(self):

        pass
    
    def growth(self):
        pass

    def step(self):

        pass