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

class Household(Agent):
     '''
         Create a new Household agent.

         Args:
            unique_id: Unique identifier for the agent.
        Type:    
            agent_type: Indicator for the agent's type (risk_lover=1, risk_averse=0, )
        '''
    def __init__(self, unique_id, model):
        super().__init__(unique_id,model)
        self.type = "type" #Def risk profile here
        # self.capital = 0
        self.wage = 0 #salary month
        self.debt = 0
        self.savings = 0 
        self.conso = 0
        # self.bank ?

    def risk_profile(self):

        pass    

    def savings(self):

        pass #put 2 different

    def consumption(self):
        # if self.type = risklover: self.conso = "equation 4"
        self.conso = "equation 3"

    def step(self):
        
        pass

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