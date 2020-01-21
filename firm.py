from mesa import Agent

class Firm(Agent):
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        super().__init__(unique_id,model)
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