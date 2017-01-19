import numpy as np
from numpy.linalg import matrix_rank
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

mask = np.isnan(incompl)
solver = NuclearNormMinimization(require_symmetric_solution=False, min_value= 0, max_value=1,fast_but_approximate=False)
solved = solver.complete(incompl)
np.savetxt("solved.csv", solved, delimiter=",")

print("Nuclear norm minimization perfectly recovers most "
	  "low-rank nxn matrices of rank r, if O(n^1.2 r log n) entries sampled uniformly at random are observed")
print ("Reconstructed Matrix has rank: " + str(matrix_rank(solved,0.1)))
print ("n^1.2 r log n = " + str(np.power(len(incompl),1.2)*matrix_rank(solved,0.1)*np.log(len(incompl))))
print ("Total elements in matrix: " + str(incompl.shape[0]*incompl.shape[1]))
print ("Elements we knew in matrix: "+ str(np.sum(np.logical_not(mask))))

