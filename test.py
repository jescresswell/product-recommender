from fancyimpute import NuclearNormMinimization
import numpy as np


from common import reconstruction_error
from low_rank_data import create_rank_k_dataset

def test_nuclear_norm_minimization_with_low_rank_random_matrix():
	solver = NuclearNormMinimization(require_symmetric_solution=False)
	compl, incomp, mask = create_rank_k_dataset(20,20,1,.7)
	print(incomp)
	print(compl)
	solved = solver.complete(incomp)
	print (solved-compl)
	_, missing_mae = reconstruction_error(
	    compl, solved, mask, name="NuclearNorm")
	assert missing_mae < 0.1, "Error too high!"
	

if __name__ == "__main__":
	#test_rank1_convex_solver()
	#test_rank1_symmetric_convex_solver()
	test_nuclear_norm_minimization_with_low_rank_random_matrix()
