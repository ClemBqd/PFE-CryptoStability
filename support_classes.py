
"""
Define the support classes needed in this program
"""
import datetime
import calendar

class Loan():
	"""
	The class loan will hold data concerning a loan 
	given by one agent to another agent.


	"""
	

	def __init__(self, start_date, end_date,
				 debtor, interest_rate, value, loan_type="basic"):
		# Start by checking the compatibility of the parameters
		
		self.start_date = start_date
		self.end_date = end_date
		self.debtor = debtor
		self.interest_rate = interest_rate
		self.value = value
		self.type = loan_type
		self.nb_month = ((end_date.year - start_date.year)*12)+end_date.month - start_date.month 
		self.mensuality = value/self.nb_month + ((value/self.nb_month)*interest_rate)
		self.remaining_val_with_interest = self.mensuality*self.nb_month
		self.total_val = 0 #to store total_val => maybe reemplace below total_val by value

	def __getattribute__(self, name):
		# Definition of dynamic variables
		if name=="total_val":
			return self.value * (1 + self.interest_rate)

		else:
			super().__getattribute__(name)

	def set_total_val(self, val):
		self.value = val + val * self.interest_rate
	def decrease_total_val(self, val):
		"""
		value is an amount of the loan that has been repaid
		"""
		self.value = (self.value - val) / (1 + self.interest_rate)
		
	def get_payment(self, current_date):
		"""
		This function will return hoow much money is due for this month

		Return : amount of money due, debtor and end.
			- Amount of money due is the amount the debtor should pay to
			the owner of the loan
			
			- Debtor is the number of the agent who should pay this amount
			- end is a boolean which says wether or not this loan can be deleted
				from the database
		"""
		end_date=current_date # TO CHANGE end_date could be on a list 
		if self.type == "basic loan":
			"""
			In the case of a basic loan, everything is paid at once at
			the end date
			"""
			if current_date == end_date:
				return self.value + (self.interest_rate) * self.value,\
					self.debtor,\
					True

			
		if self.type == "household loan":
			if current_date < end_date:
				self.decrease_total_val(self.mensuality)
				
				return self.mensuality,\
					self.debtor,\
					False

			elif current_date == end_date:
				self.decrease_total_val(self.mensuality)

				return self.mensuality,\
					self.debtor,\
					self.total_val == 0

			# Dans le cas où le prêt est arrivé à échéance mais qu'il reste à payer
			elif self.total_val != 0:
				return self.total_val,\
					self.debtor,\
					True
					
			# Dans le cas où le prêt est entièrement payé
			else:
				return 0, self.debtor, True

		if self.type == "bank loan":
			"""in the case of bank loan from 1 to 12 months"""
			if current_date < end_date:
				pass
				
				

			


###### Helper functions ######
def addMonth(source, n_months=1):
	"""
	This function takes a datetime as input and returns a date n_months
	later than datetime
	"""
	month = source.month - 1 + n_months
	year = source.year + month // 12
	month = month % 12 + 1
	day = min(source.day, calendar.monthrange(year, month)[1])
	return datetime.date(year, month, day)
