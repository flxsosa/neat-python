import sys
import neat
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import population
import reproduction

sys.path.insert(0, "../../hyperneat/shared/")
from visualize import draw_net
from neat.reporting import ReporterSet
from dhngenome import DHNGenome
from dhn_feed_forward import FeedForwardNetwork
from phenome import decode
from activations import ActivationFunctionSet
from aggregations import AggregationFunctionSet

# Config file for dhn_proof
config = neat.config.Config(DHNGenome, reproduction.DefaultReproduction,
							neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
							'config_dhn_proof')
# Substrate sheet dimensions and number of substrate outputs
input_dimensions = (1,2)
sheet_dimensions = (1,1)
num_outputs = 1

xor_inputs = [(0.0,0.0),(0.0,1.0),(1.0,0.0),(1.0,1.0)]
expected_outputs = [0.0, 1.0, 1.0, 0.0]

def evaluate_xor(genomes, config):
	'''
	Evaluates candidate genomes from a population for a given number of
	generations.

	genomes -- candidate genomes to be evaluated
	config -- config file
	'''
	# Iterate through and evaluate candidate genomes
	for genome_index, genome in genomes:
		print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++"
		print "Evaluating genome {0}".format(genome_index)
		# Convert genome into usable CPPN
		cppn = FeedForwardNetwork.create(genome, config)
		# Decode CPPN into substrate
		substrate = decode(cppn, input_dimensions, num_outputs,sheet_dimensions)
		
		sum_square_error = 0.0
		for inputs, expected in zip(xor_inputs, expected_outputs):
			print("Expected Output: {0}".format(expected))
			actual_output = substrate.activate(inputs)[0]
			print("Actual Output: {0}".format(actual_output))
			sum_square_error += ((actual_output - expected)**2.0)/4.0
			print("Loss: {0}".format(sum_square_error))
		genome.fitness = 1.0 - sum_square_error - len(genome.output_keys)*0.01
		print("Genome {0} Fitness is {1}".format(genome_index, genome.fitness))
		print

def evaluate_debug(genomes, config):
	'''
	Evaluates candidate genomes from a population for a given number of
	generations.

	genomes -- candidate genomes to be evaluated
	config -- config file
	'''
	# Iterate through and evaluate candidate genomes
	for genome_index, genome in genomes:
		print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++"
		print "Evaluating genome {0}".format(genome_index)
		print "CPPN Output Nodes:"

		# Print out all CPPN output nodes in current genome
		for key in genome.output_keys:
			try:
				print "\tKey: {0} CPPNON Gene: {1}".format(key, genome.nodes[key])
			except KeyError:
				print "Attempted key: {0}, Key List: {1}".format(key, genome.output_keys)
		# Convert genome into usable CPPN
		cppn = FeedForwardNetwork.create(genome, config)
		# Decode CPPN into substrate
		substrate = decode(cppn, input_dimensions, num_outputs)
		substrate = Substrate(substrate)
		# Print resulting substrate to user
		print "Current substrate from genome {0}: {1}".format(genome_index, substrate)

		genome.fitness = 0.5 + len(genome.output_keys)*0.01
		print

def run(generations):
	'''
	Runs DHN

	generations -- number of generations to evolve for
	'''
	pop = population.Population(config)

	# Run the population under a simple evaluation scheme
	winner = pop.run(evaluate_xor, generations)
	cppn = FeedForwardNetwork.create(winner, config)
	substrate = decode(cppn,input_dimensions,num_outputs, sheet_dimensions)
	print("WINNER: Genome {0}\n".format(winner.key))
	print("Winner Fitness is {0}".format(winner.fitness))
	for node in winner.nodes:
		print("\t{0}: {1}".format(winner.nodes[node].key,winner.nodes[node].activation))
	for connection in winner.connections:
		print("\t{0},{1},{2}".format(connection,winner.connections[connection].weight,winner.connections[connection].enabled))
	sum_square_error = 0.0
	for inputs, expected in zip(xor_inputs, expected_outputs):
		print("Expected Output: {0}".format(expected))
		actual_output = substrate.activate(inputs)[0]
		print("Actual Output: {0}".format(actual_output))
		sum_square_error += ((actual_output - expected)**2.0)/4.0
		print("Loss: {0}".format(sum_square_error))
	draw_net(cppn, filename="dhn_cppn_winner")
	draw_net(substrate, filename="dhn_substrate_winner")
	for node in substrate.node_evals:
		print("Node: \t{0}:".format(node))
	for connection in substrate.values:
		print("Value:\t{0}".format(connection))
if __name__ == '__main__':
	run(1200)
	print "DONE"
