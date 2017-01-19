import numpy as np
import csv
from fancyimpute import NuclearNormMinimization
from common import reconstruction_error

data = np.genfromtxt('query_result.csv', delimiter=',', dtype=None) #Read entire CSV into array

unnorm= data[1:,1:] #Last row is normalization ... need to change norm.. somehow?  Max of each row?
incompl = []

for el in unnorm:
 	#incompl.append([float(x)/float(el[-1]) if float(el[-1]) !=0 else 0 for x in el[:-1]]) #Does norm even matter??
	#incompl.append([float(x) for x in el[:-1]])
	maxn = max([float(x) for x in el[:-1]])
	incompl.append([float(x)/maxn if float(maxn) !=0 else 0 for x in el[:-1]])

incompl= np.asarray(incompl)
incompl[incompl == 0] = np.nan
np.savetxt("normed.csv", incompl, delimiter=",") #Save the formatted matrix for comparison after we remove extra entries for bootstrapping
incompl[3,6]=np.nan
mask = np.isnan(incompl)
solver = NuclearNormMinimization(require_symmetric_solution=False, min_value= 0, max_value=1,fast_but_approximate=False)
solved = solver.complete(incompl)
np.savetxt("solved.csv", solved, delimiter=",")
