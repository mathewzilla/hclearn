hclearn
=======

Unitary coherent hippocampus model code, created by Charles Fox and Alan Saul and edited by Mathew Evans

Below are notes on how I (M.E.) understand what the code does. This document has now been superseeded by the notebook, but a detailed list of functions might stay here.


rbm.py : Restricted Boltzmann Machine code, based on Hinton's TRBM work. Includes a number of functions for building the network, setting observed and hidden units, training weights and measuring the 'Energy'. 
		List Of Functions
		boltzmannProbs(W,x) : Returns the probability of a node being on (is the weight*x above a half?)

		trainPriorBias(hids) : Normalises and concatenates the hidden values (I think! 30/4/14). Uses addBias(hids) to concatenate the hidden node values (their energy?). Also adds/removes tiny values to/from zeros/ones, and uses invsig to normalise the output.

		addBias(xs) : reshapes and concatenates input vector with a column of ones (I think! 30/4/14). 



cffun.py : More functions, probably from Charles Fox (hence CFfun).

		invsig(x) : takes neg log of 1/x - 1. Squashes x into the range 0:1, with infinitely large outputs near those values. (Try inputing -log((1/x)-1)) into google.

