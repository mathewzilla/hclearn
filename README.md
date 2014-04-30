hclearn
=======

Unitary coherent hippocampus model code, created by Charles Fox and Alan Saul and edited by Mathew Evans

Below are notes on how I (M.E.) understand what the code does.


rbm.py : Restricted Boltzmann Machine code, based on Hinton's TRBM work. Includes a number of functions for building the network, setting observed and hidden units, training weights and measuring the 'Energy'. 
		List Of Functions
		boltzmannProbs(W,x) : Returns the probability of a node being on (is the weight*x above a half?)

		trainPriorBias(hids) : uses addBias(hids) to concatenate the hidden node values (their energy?)

		addBias(xs) : reshapes and concatenates input vector with a column of ones (I think! 30/4/14). Also adds/removes tiny values to/from zeros/ones



cffun.py : More functions, probably from Charles Fox (hence CFfun).

