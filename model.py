from mesa import Agent, Model
from mesa.time import RandomActivation
from agents import *

class BtcModel(Model):
    def __init__(self, n_households):
        	"""
		Basic constructor with mostly default values

		Parameters
		n_agents : dict containing the number of each
			type of agents that we want our model to include
		"""
        """
        self.n_banks = n_agent['banks']
        self.n_firms = n_agent['firms']
        self.n_households = n_agent['households']
    """
        # Chose a scheduler
        self.scheduler = RandomActivation(self)

        # Create  a bank, a firm and n household
        self.bank = Bank(1, self) 
        self.firm = Firm(2)
    
        for i in range(self.n_households):
            # def proportion of risk here or in class household
            h = Household(i+2, self.n_households)
            self.scheduler.add(h)
