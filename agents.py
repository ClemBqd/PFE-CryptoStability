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
            agent_type: Indicator for the agent's type (risk_lover=1, risk_averse=-1, risk_neutral=0 )
        '''
    def __init__(self, unique_id, model, alpha=0.2, beta=0.8):
        super().__init__(unique_id,model)
        self.type = -1 #Def risk profile here
        self.capital = 0
        self.wage = 0 #salary month
        self.debt = 0
        self.savings = 0 
        self.conso = 0
        self.loan = 0
        self.rate_loan = 0.1 
        self.happiness = 0 # To define
        self.speculator_portfolio = 0 
        
        # Variable wich households doesn't impact on
        ispe = 0 # Cours du bitcoin
        P = 0 #Find what that is -> related to risk_lovers
        rs = 0.1 #To define = %of savings
        sp = 0.1 #To define = %of savings speculators
        # self.bank ? 

    def savings(self):
        if self.risk_profile == -1:
            self.savings = (1 - rs)*self.savings + self.wage - self.conso
        else:
            self.savings = (1 - sp + rs)*self.savings + self.wage - self.conso + self.loan(1 + self.rate_loan)
        return self.savings

    def speculators_portfolio(self):
        if risk_profile == 1:
            self.speculator_portfolio = self.speculators_portfolio*(1 + ispe) + P*self.savings + self.loan
        return self.speculator_portfolio

    def consumption(self):
        if self.type == -1:
            self.conso = self.capital*self.savings*(self.happiness^(1-alpha))*(1 - alpha*beta)
        else:
            self.conso = self.capital*(self.savings + self.speculator_portfolio)*(self.happiness^(1-alpha))*(1 - alpha*beta)

    def step(self, ispe):
        
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