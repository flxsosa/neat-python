import sys
import neat
import random
import reproduction
import population
from substrate_visualizer import visualize_activations
from neat.reporting import ReporterSet
from dhngenome import DHNGenome
from dhn_feed_forward import FeedForwardNetwork
from phenome import decode
from activations import ActivationFunctionSet
from aggregations import AggregationFunctionSet
sys.path.insert(0, "../../hyperneat/shared/")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from visualize import draw_net
'''
Final proof of DHN 

Felix Sosa
'''

'''
CREATE CONFIG
'''
# Need to create a configuration file
config = neat.config.Config(DHNGenome, reproduction.DefaultReproduction,
							neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
							'config_dhn_proof')

config2 = neat.config.Config(DHNGenome, reproduction.DefaultReproduction,
							neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
							'config_dhn_proof')
'''
CREATE CPPN
'''

# Create reporters from neat.reporting.ReporterSet()
reporters = ReporterSet()
# Create stagnation from neat.stagnation.DefaultStagnation()
stagnation = config.stagnation_type(config.stagnation_config, reporters)
# Create reproduction from neat.reproduction.DefaultReproduction
reproduction = config.reproduction_type(config.reproduction_config, reporters, stagnation)
# Create new population
population = reproduction.create_new(config.genome_type, config.genome_config, 1)
# Retrieve cppn from population
genome = population[1]

'''
DECODE TEST FOR CPPN
'''

'''
# Inputs are from (1,1) to (x,1)
inputs = [1.0,1.23,0.1,1.2]
print "Inputs:\n", inputs
substrate_input_dimensions = (2,4)
num_substrate_outputs = 3

cppn = FeedForwardNetwork.create(genome, config)
substrate = decode(cppn, substrate_input_dimensions, num_substrate_outputs)

output = substrate.activate(inputs+inputs)
print "Output:", output
for node in substrate.node_evals:
	print("Node: \t{0}:".format(node))
'''

'''
MANUAL MUTATION TEST FOR CPPN - IncDepth - PASS
'''
'''
# Substrate parameters
substrate_input_dimensions = (1,4)
substrate_sheet_dimensions = None
num_substrate_outputs = 4

# Test parameters
num = 10
count = 0

# Generate random inputs and activate substrate an arbitraty number of times
for i in range(num):
	# Random inputs
	inputs = [random.random() for x in range(4)]
	# print "Inputs:\n", inputs

	# CPPN before mutation(s)
	cppn = FeedForwardNetwork.create(genome, config)
	# draw_net(cppn, filename="IncDepth_Study/cppn_before_mutation")
	substrate = decode(cppn, substrate_input_dimensions, num_substrate_outputs, substrate_sheet_dimensions)
	# draw_net(substrate, filename="IncDepth_Study/substrate_before_mutation")
	output_before = substrate.activate(inputs)

	# Induce IncDepth mutation(s)
	genome.mutate_increment_depth(config.genome_config)
	genome.mutate_increment_depth(config.genome_config)
	genome.mutate_increment_depth(config.genome_config)
	genome.mutate_increment_depth(config.genome_config)

	# CPPN after mutation(s)
	cppn = FeedForwardNetwork.create(genome, config)
	# draw_net(cppn, filename="IncDepth_Study/cppn_after_mutation")
	substrate = decode(cppn, substrate_input_dimensions, num_substrate_outputs, substrate_sheet_dimensions)
	# draw_net(substrate, filename="IncDepth_Study/substrate_after_mutation")
	output_after = substrate.activate(inputs)

	# Check if mutations preserved function of original CPPN
	if output_before == output_after:
		count += 1

# Output results
print "Identity reached {0} out of {1} times".format(count, num)
'''

'''
MANUAL MUTATION TEST FOR CPPN - IncBreadth - PASS
'''
'''
# Substrate parameters
substrate_input_dimensions = (1,4)
substrate_sheet_dimensions = None
num_substrate_outputs = 4

# Test parameters
num = 20
count = 0

# Generate random inputs and activate substrate an arbitraty number of times
for i in range(num):
	# Random inputs
	inputs = [random.random() for x in range(4)]
	# print "Inputs:\n", inputs

	# Induce IncDepth mutation
	genome.mutate_increment_depth(config.genome_config)
	# CPPN before mutation(s)
	cppn = FeedForwardNetwork.create(genome, config)
	# draw_net(cppn, filename="IncDepth_Study/cppn_before_mutation")
	substrate = decode(cppn, substrate_input_dimensions, num_substrate_outputs, substrate_sheet_dimensions)
	# draw_net(substrate, filename="IncDepth_Study/substrate_before_mutation")
	output_before = substrate.activate(inputs)

	# Induce IncBreadth mutation(s)
	genome.mutate_increment_breadth(config.genome_config)
	genome.mutate_increment_breadth(config.genome_config)
	genome.mutate_increment_breadth(config.genome_config)
	genome.mutate_increment_breadth(config.genome_config)

	# CPPN after mutation(s)
	cppn = FeedForwardNetwork.create(genome, config)
	# draw_net(cppn, filename="IncDepth_Study/cppn_after_mutation")
	substrate = decode(cppn, substrate_input_dimensions, num_substrate_outputs, substrate_sheet_dimensions)
	# draw_net(substrate, filename="IncDepth_Study/substrate_after_mutation")
	output_after = substrate.activate(inputs)

	# Check if mutations preserved function of original CPPN
	# print "Inputs: \t{0}".format(inputs)
	# print "Output Before: \t {0} \n Output After: \t {1}".format(output_before, output_after)
	# if output_before == output_after:
	# 	count += 1
	if sum([((k-v)**2)**(0.5) for (k,v) in zip(output_before, output_after)]) < 0.005:
		count +=1

# Output results
print "Identity reached {0} out of {1} times".format(count, num)
'''

'''
MANUAL MUTATION TEST FOR CPPN - IncBreadth & IncDepth - PASS
'''
'''
# Substrate parameters
substrate_input_dimensions = (1,4)
substrate_sheet_dimensions = None
num_substrate_outputs = 4

# Test parameters
num = 20
count = 0

# Generate random inputs and activate substrate an arbitraty number of times
for i in range(num):
	# Random inputs
	inputs = [random.random() for x in range(4)]
	# print "Inputs:\n", inputs

	# CPPN before mutation(s)
	cppn = FeedForwardNetwork.create(genome, config)
	# draw_net(cppn, filename="IncDepth_Study/cppn_before_mutation")
	substrate = decode(cppn, substrate_input_dimensions, num_substrate_outputs, substrate_sheet_dimensions)
	# draw_net(substrate, filename="IncDepth_Study/substrate_before_mutation")
	output_before = substrate.activate(inputs)

	# Induce IncDepth mutation(s)
	genome.mutate_increment_depth(config.genome_config)
	genome.mutate_increment_depth(config.genome_config)
	genome.mutate_increment_depth(config.genome_config)
	genome.mutate_increment_depth(config.genome_config)

	# Induce IncBreadth mutation(s)
	genome.mutate_increment_breadth(config.genome_config)
	genome.mutate_increment_breadth(config.genome_config)
	genome.mutate_increment_breadth(config.genome_config)
	genome.mutate_increment_breadth(config.genome_config)

	# CPPN after mutation(s)
	cppn = FeedForwardNetwork.create(genome, config)
	# draw_net(cppn, filename="IncDepth_Study/cppn_after_mutation")
	substrate = decode(cppn, substrate_input_dimensions, num_substrate_outputs, substrate_sheet_dimensions)
	# draw_net(substrate, filename="IncDepth_Study/substrate_after_mutation")
	output_after = substrate.activate(inputs)

	# Check if mutations preserved function of original CPPN
	# print "Inputs: \t{0}".format(inputs)
	# print "Output Before: \t {0} \n Output After: \t {1}".format(output_before, output_after)
	# if output_before == output_after:
	# 	count += 1
	if sum([((k-v)**2)**(0.5) for (k,v) in zip(output_before, output_after)]) < 0.005:
		count +=1

# Output results
print "Identity reached {0} out of {1} times".format(count, num)
'''

'''
VISUALIZATION TEST FOR SUBSTRATE
'''

# Substrate parameters
substrate_input_dimensions = (4,4)
substrate_sheet_dimensions = (4,4)
num_substrate_outputs = 4

# Inputs
inputs = [random.random() for x in range(16)]

# Induce IncDepth mutation(s)
genome.mutate_increment_depth(config.genome_config)

# CPPN
cppn = FeedForwardNetwork.create(genome, config)
# draw_net(cppn, filename="IncDepth_Study/cppn_after_mutation")
substrate = decode(cppn, substrate_input_dimensions, num_substrate_outputs, substrate_sheet_dimensions)
# draw_net(substrate, filename="IncDepth_Study/substrate_after_mutation")
output = substrate.activate(inputs)
print(substrate.values)
visualize_activations(substrate.values, substrate_input_dimensions, substrate_sheet_dimensions)



'''
MANUAL CROSSOVER TEST FOR CPPN
'''
'''
# Population of two parents
population = reproduction.create_new(config.genome_type, config.genome_config, 2)

# Parent 1
parent_1 = population[2]
# Parent 2 
parent_2 = population[3]

# Mutate Parent 1 and 2
parent_1.mutate_increment_depth(config.genome_config)
parent_1.mutate_increment_depth(config.genome_config)
parent_1.mutate_increment_depth(config.genome_config)
parent_1.mutate_increment_depth(config.genome_config)
parent_1.mutate_increment_depth(config.genome_config)
parent_1.mutate_increment_depth(config.genome_config)
parent_2.mutate_increment_depth(config2.genome_config)
parent_2.mutate_increment_breadth(config2.genome_config)
parent_2.mutate_increment_depth(config2.genome_config)
parent_2.mutate_increment_breadth(config2.genome_config)
parent_2.mutate_increment_breadth(config2.genome_config)
parent_2.mutate_increment_depth(config2.genome_config)
parent_2.mutate_increment_breadth(config2.genome_config)
parent_2.mutate_increment_depth(config2.genome_config)
parent_2.mutate_increment_breadth(config2.genome_config)
parent_2.mutate_increment_breadth(config2.genome_config)
# Give parent 1 higher fitness
parent_1.fitness = 1
parent_2.fitness = 0

# Show Parent 1
parent_1_genome = FeedForwardNetwork.create(parent_1,config)
draw_net(parent_1_genome, filename="parent_1")
print("\nParent 1:\n")
parent_1_sub = decode(parent_1_genome, (1,4), 1)
draw_net(parent_1_sub, filename="parent_1_substrate")

# Show Parent 2
parent_2_genome = FeedForwardNetwork.create(parent_2,config)
draw_net(parent_2_genome, filename="parent_2")
print("\nParent 2:\n")
parent_2_sub = decode(parent_2_genome, (1,4), 1)
draw_net(parent_2_sub, filename="parent_2_substrate")

# Create child
gid = next(reproduction.genome_indexer)
child = config.genome_type(gid)

# Crossover child
child.configure_crossover(parent_1,parent_2,config.genome_config)

# Show child
child_genome = FeedForwardNetwork.create(child,config)
draw_net(child_genome, filename="child")
print("\nChild:\n")
child_sub = decode(child_genome, (1,4), 1)
draw_net(child_sub, filename="child_substrate")
'''
'''
DECODE INTO SUBSTRATE AND VISUALIZE
'''
'''
SAVE AND LOAD CPPN AND DECODE
'''