import numpy as np
import itertools as it
from dhn_feed_forward import FeedForwardNetwork as dhnff
from FFNN import Substrate


def decode(cppn, sub_input_dimensions, sub_outputs, sheet_dimensions=None):
	'''
	Decodes cppn into Substrate.

	cppn -- input CPPN to be decoded
	sub_input_dimensions -- matrix dimensions of input layer for Substrate (row,col)
	sub_outputs -- number of outputs in Substrate
	sheet_dimensions -- matrix dimenstions of Substrate sheets. Defaults to input.
	'''

	# Get x and y coordinates for Substrate input layer and create layer
	# neat-python uses floats for coordinates so we need to convert int->float
	x = range(1,sub_input_dimensions[1]+1)
	x = [float(i) for i in x]
	y = range(1, sub_input_dimensions[0]+1)
	y = [float(i) for i in y]
	sub_input_layer = list(it.product(x,y))

	# We want the output nodes to have their first coord at (1.0,1.0)
	x = range(1,sub_outputs+1)
	x = [float(i) for i in x]
	y = range(1,2)
	y = [float(i) for i in y]
	sub_out_layer = list(it.product(x,y))

	# Check if sheet dimensions have been provided
	if sheet_dimensions != None:
		x = range(1,sheet_dimensions[1]+1)
		x = [float(i) for i in x]
		y = range(1, sheet_dimensions[0]+1)
		y = [float(i) for i in y]
		sheet = list(it.product(x,y))
	else:
		sheet = sub_input_layer

	# Initialize substrate with input (1,0) and output (0,0) layers
	# Should do dict comprehension
	substrate = {
		(0,0): sub_out_layer,
		(1,0): sub_input_layer
	}

	# Traverse CPPN Output Nodes (CPPNONs) and add layers and sheets to the substrate
	for node in cppn.output_nodes:
		if cppn.nodes[node].cppn_tuple[0] not in substrate:
			substrate[cppn.nodes[node].cppn_tuple[0]] = sheet
		if cppn.nodes[node].cppn_tuple[1] not in substrate:
			substrate[cppn.nodes[node].cppn_tuple[1]] = sheet

	# Build weight matrices output by CPPN by going through each CPPNON and querying
	# them for the mappings between sheets they represent
	weights = {}	
	for node in cppn.output_nodes:
		source_sheet = cppn.nodes[node].cppn_tuple[0]
		target_sheet = cppn.nodes[node].cppn_tuple[1]
		# The weight dictionary is a dictionary of matrices (or dictionary of lists of lists)
		weights[cppn.nodes[node].cppn_tuple] = []
		# Each passing through the entire for loop creates a new row in the corresponding 
		# weight matrix.
		# Each for loop traverses (1,1) to (m,n) going y/col first and x/row last
		for coord1 in substrate[source_sheet]:
			w = []
			for coord2 in substrate[target_sheet]:
				w.append(query_cppn(coord1,coord2,cppn,node))
			# Append the matrix with the new row
			weights[cppn.nodes[node].cppn_tuple].append(w)
			#print w

	return Substrate((substrate,weights))

def query_cppn(coord1, coord2, cppn, node, max_weight=5.0):
	'''
	Get the weight from one point to another using the CPPN - takes into 
	consideration which point is source/target.

	coord1 -- coordinates for neuron in source sheet
	coord2 -- coordinates for neuron in target sheet
	cppn -- CPPN being queried
	node -- index of which CPPNON to read
	max_weight -- weight threshold 
	'''
	# print (coord1[0],coord1[1],coord2[0],coord2[1])
	i = [coord1[0], coord1[1], coord2[0], coord2[1], 1.0]
	w = cppn.activate(i, node)

	return w*max_weight