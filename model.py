from mesa import Agent, Model
from mesa.time import RandomActivation
from household import *
from bank import *
from firm import * 

class BtcModel(Model):
    def __init__(self, n_households):
        	"""
		Basic constructor with mostly default values

		Parameters
		n_households : number of household in the model
    """
        # Chose a scheduler
        self.scheduler = RandomActivation(self)

        # Create  a bank, a firm and n household
        self.bank = Bank(1, self) 
        self.schedule.add(self.bank)
        self.firm = Firm(2)
        self.schedule.add(self.firm)
    
        for i in range(self.n_households):
            h = Household(i+2, self.n_households)
            self.scheduler.add(h) 

        # Create datacollector here

        # Put variable comun of all the model too

        def step(self):
            # Tell all the agents in the model to run their step function
            self.schedule.step()

            # Collect data
            # self.datacollector.collect(self)

        
