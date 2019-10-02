#================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#									SIMPLE LINEAR REGRESSION
#----------------------------------------------------------------------------------------------------------------
#================================================================================================================

#Simple linear regression is applied to stock data
# x values are time and y values are the stock closing price.
# just an example

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import datetime
#
# #Quandl for getting stock data
# import quandl

#for plotting
plt.style.use('ggplot')

class CustomLinearRegression:

	def __init__(self):
		self.intercept = 0
		self.slope = 0

	#finding the slope in best fit line
	def best_fit(self, dimOne, dimTwo):
		self.slope = ( (np.mean(dimOne) * np.mean(dimTwo) ) - np.mean(dimOne*dimTwo) ) / ( np.mean(dimOne)**2 - np.mean(dimOne**2) ) #formula for finding slope
		return self.slope

	#finding the best fit intercept
	def y_intercept(self, dimOne ,dimTwo):
		self.intercept = np.mean( dimTwo ) - ( self.slope * np.mean(dimOne) )
		return self.intercept

	#predict for future values based on model
	def predict(self, ip):
		ip = np.array(ip)
		# create a "predicted" array where the index corresponds to the index of the input
		predicted = [(self.slope*param) + self.intercept for param in ip]
		return predicted

	#find the squared error
	def squared_error(self, original, model):
		return sum((model - original) **2)

	#find co-efficient of determination for R^2
	def cod(self, original, model):
		am_line = [np.mean(original) for y in original]
		sq_error = self.squared_error(original, model)
		sq_error_am = self.squared_error(original, am_line)
		#R^2 is 1 - of squared error for our model / squared error if the model only consisted of the mean
		return 1 - (sq_error/sq_error_am)

def main():
	# stk = quandl.get("WIKI/TSLA")
	# homedepot stock price
	stk = pd.read_csv('./data/EOD-HD.csv')
	simpl_linear_regression = CustomLinearRegression()

	#reset index to procure date - date was the initial default index
	stk = stk.reset_index()

	#Add them headers
	stk = stk[['Date','Adj_Open','Adj_High','Adj_Low','Adj_Close', 'Volume']]

	stk['Date'] = pd.to_datetime(stk['Date'])
	stk['Date'] = (stk['Date'] - stk['Date'].min())  / np.timedelta64(1,'D')


	#The column that needs to be forcasted using linear regression
	forecast_col = 'Adj_Close'

	#take care of NA's
	# stk.fillna(-999999, inplace = True)
	stk['label'] = stk[forecast_col]

	stk.dropna(inplace = True)

	x = np.array(stk['Date'])
	y = np.array(stk['label'])

	#Always in the order: first slope, then intercept
	slope = simpl_linear_regression.best_fit(x, y) #find slope
	intercept = simpl_linear_regression.y_intercept(x, y) #find the intercept
	reg = [(slope*param) + intercept for param in x]
	r_sqrd = simpl_linear_regression.cod(y, reg)
	print("R^2 Value: " ,r_sqrd)

	plt.scatter(x, y)
	plt.plot(x, reg)
	plt.show()

	# input a point to predict
	ippoint = list(map(int, input("Enter x to predict y: \n").split()))

	line = simpl_linear_regression.predict(ippoint) #predict based on model

	print("Predicted value(s) after linear regression :", line)

	plt.scatter(x, y)
	plt.scatter(ippoint, line, color = "blue")
	plt.plot(x, reg)
	plt.show()

if __name__ == "__main__":
	main()
