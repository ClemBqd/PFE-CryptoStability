from mesa import Agent, Model
from mesa.time import RandomActivation
from household import Household
from bank import Bank
from firm import Firm

class BtcModel(Model):
    def __init__(self, n_households):
        self.n_households = n_households
        super().__init__(n_households)
        # Chose a schedule
        self.schedule = RandomActivation(self)

        # Create  a bank, a firm and n household
        bank = Bank(1, self) 
        self.schedule.add(bank)
        firm = Firm(2, self)
        self.schedule.add(firm)
        
    
        for i in range(self.n_households):
            h = Household(i+2, self)
            self.schedule.add(h) 

        # Create datacollector here

        # Put variable comun of all the model too

    def step(self):
        # Tell all the agents in the model to run their step function
        self.schedule.step()

        # Collect data
        # self.datacollector.collect(self)

        
empty_model = BtcModel(200)

print(empty_model.n_households)