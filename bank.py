from mesa import Agent

class Bank(Agent):
         '''
         Create a new Bank agent.

         Args:
            unique_id: Unique identifier for the agent. put 1
            reserve_percent: percent that the bank keep in reserve -> maybe to add
        '''
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.loan = []
        self.deposits = 0
        #self.reserves = ((self.reserve_percent / 100)*self.deposits)

    def bank_balance(self):
        self.reserves = ((self.reserve_percent / 100)*self.deposits)
        #self.bank_to_loan = (self.deposits - (self.reserves + self.bank_loans))

    def step(self):

        pass
