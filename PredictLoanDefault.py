import pandas
import numpy

loanDefaultDataset = pandas.read_csv( 'LoanDefaultData.csv' )

loanDefaultDataset.rename( { 'BAD': 'defaultResult', 'LOAN': 'loanRequested', 'MORTDUE': 'currentMortageDue',\
                             'VALUE': 'currentPropertyValue', 'REASON': 'reasonForRequest', 'JOB': 'Job',\
                             'YOJ': 'yearsAtJob', 'DEROG': 'derogatoryReports', 'DELINQ': 'delinquentLines',\
                             'CLAGE': 'ageOldestTradeLine', 'NINQ': 'recentCreditLines', 'CLNO': 'totalCreditLines',\
                             'DEBTINC': 'debtToIncome' }, inplace = True, axis = 'columns' )

#print the shape of the dataset
# print( 'shape of dataset:', loanDefaultDataset.shape, '\n' )

#replace nan values with averages and modes
loanDefaultDataset.currentMortageDue.fillna( loanDefaultDataset.currentMortageDue.mean(), inplace = True )
loanDefaultDataset.currentPropertyValue.fillna( loanDefaultDataset.currentPropertyValue.mean(), inplace = True )
loanDefaultDataset.reasonForRequest.fillna( loanDefaultDataset.reasonForRequest.mode().iloc[ 0 ] , inplace = True )
loanDefaultDataset.Job.fillna( loanDefaultDataset.Job.mode().iloc[ 0 ], inplace = True )
loanDefaultDataset.yearsAtJob.fillna( loanDefaultDataset.yearsAtJob.mean(), inplace = True )
loanDefaultDataset.derogatoryReports.fillna( loanDefaultDataset.derogatoryReports.mode().iloc[ 0 ], inplace = True )
loanDefaultDataset.delinquentLines.fillna( loanDefaultDataset.delinquentLines.mode().iloc[ 0 ], inplace = True )
loanDefaultDataset.ageOldestTradeLine.fillna( loanDefaultDataset.ageOldestTradeLine.mean(), inplace = True )
loanDefaultDataset.recentCreditLines.fillna( loanDefaultDataset.recentCreditLines.mode.iloc[ 0 ], inplace = True )

#print how many empty cells are in each column
# for column in loanDefaultDataset.columns:
#     listOfNullValues = [ element for element in loanDefaultDataset[ column ] if pandas.isnull(element) ]
#     print( column, len( listOfNullValues ) )

print( loanDefaultDataset.head() )