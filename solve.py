import numpy as np
from fancyimpute import NuclearNormMinimization, KNN, MatrixFactorization

data = np.genfromtxt('query_result.csv', delimiter=',', dtype=None) #Read entire CSV into array

unnorm= data[1:,1:] #Last row is normalization ... need to change norm.. somehow?  Max of each row?
incompl = []

for el in unnorm:
    if float(el[-1]) != float(0):# remove rows which have no information
        #incompl.append([float(x)/float(el[-1]) if float(el[-1]) !=0 else 0 for x in el[:-1]]) #Does norm even matter??
        #incompl.append([float(x) for x in el[:-1]])
        maxn = max([float(x) for x in el[:-1]])
        #incompl.append([float(x)/maxn if float(maxn) !=0 else 0 for x in el[:-1]])
        incompl.append([round(float(x)/maxn*5)/5 if float(maxn) !=0 else 0 for x in el[:-1]]) #Round to nearest 0.2n

incompl= np.asarray(incompl)
data_rank=np.linalg.matrix_rank(incompl)
incompl[incompl == 0] = np.nan
np.savetxt("normed.csv", incompl, delimiter=",") #Save the formatted matrix for comparison after we remove extra entries for bootstrapping

mask = np.isnan(incompl)
solver = NuclearNormMinimization(require_symmetric_solution=False, min_value= 0, max_value=1,fast_but_approximate=False)
#solver = KNN(k=3, min_value=0, max_value=1) #This seems to have problems with overlaps, see errors?
#solver = MatrixFactorization(min_value = 0, max_value=1) #Similar to a netflix algo
solved = solver.complete(incompl)
#solved = [np.round(el*5)/5 for el in solved] #This just made things worse, i.e., higher rank solution
np.savetxt("solved.csv", solved, delimiter=",")
np.savetxt("mask.csv", mask, delimiter=",") #ones where we reconstructed the data, zeros where we already had enteries
# u,s,v = np.linalg.svd(solved)
# print(s)
#
# print ("Reconstructed Matrix has rank: " + str(np.linalg.matrix_rank(solved)))

# print("Nuclear norm minimization perfectly recovers most "
# 	  "low-rank nxn matrices of rank r, if O(n^1.2 r log n) entries sampled uniformly at random are observed")
# print ("n^1.2 r log n = " + str(np.power(len(incompl),1.2)*data_rank*np.log(len(incompl))))
# print ("Total elements in matrix: " + str(incompl.shape[0]*incompl.shape[1]))
# print ("Our data matrix has " + str(incompl.shape[0]) + " rows (customers) and "+str(incompl.shape[1]))+" columns (products)
# print ("Elements we knew in matrix: "+ str(np.sum(np.logical_not(mask))))

#Bootstrapping
R = np.random.rand(incompl.shape[0],incompl.shape[1])
rfraction = 0.05 #Fraction of data to discard
bstrap = np.copy(incompl) #Copy the data, don't just reference it

extraremovals = np.logical_and(R < rfraction , mask==0)

bstrap[extraremovals]= np.nan #Remove some fraction of known data

bsolved = solver.complete(bstrap)

diffs = incompl[extraremovals]-bsolved[extraremovals]
for x in diffs:
	print("{0:.3f}".format(x))

print("Average difference: "+ str(np.average(diffs)))
print("Maximum difference: " +str(np.max(diffs)))

np.savetxt("bstrap.csv", bsolved, delimiter=",")




