from mesa import Agent

class Bank(Agent):
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        super().__init__(unique_id, model)
        self.loan = []
        self.deposits = 0
        #self.reserves = ((self.reserve_percent / 100)*self.deposits)

    def step(self):

        pass
