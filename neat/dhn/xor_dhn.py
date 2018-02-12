import sys
import neat 
import neat.nn
import cPickle as pickle
import population
import reproduction
import config

sys.path.insert(0,"../../shared/")
sys.path.append("../../hyperneat/")
sys.path.insert(0, "../../hyperneat/shared/")

# Import new DHN genome
from dhngenome import DHNGenome

from FFNN import Substrate
# Import FeedForwardNetwork == Decode()
from dhn_feed_forward import FeedForwardNetwork

from visualize import draw_net
from dhn_decode import decode

# Network inputs and expected outputs.
xor_inputs  = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [    (0.0,),     (1.0,),     (1.0,),     (0.0,)]

# Substrate coordinates
input_coordinates  = [(-1.0, -1.0),(0.0, -1.0),(1.0, -1.0)]
hidden_coordinates = [[(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]]
output_coordinates = [(0.0, 1.0)]
activations = len(hidden_coordinates) + 2


# Config for CPPN.
config = config.Config(DHNGenome, reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn_xor')

def eval_fitness(genomes, config):
    
    for idx, g in genomes:
        print
        print "ENTERING EVAL FITNESS"
        print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print
        print "Genome Node Dict: "
        
        for key in g.output_keys:
            if g.nodes[key].cppn_tuple == ((),()):
                g.nodes[key].cppn_tuple =((1,0),(0,0))
        # for node in g.nodes:
        #     print(g.nodes[node])
        #     print(g.nodes[node].cppn_tuple)
        print "Genome Key:", g
        cppn = FeedForwardNetwork.create(g, config)
        print "CPPN:",cppn.nodes
        print "CPPN Tuple:", cppn.nodes[0].cppn_tuple
        net = decode(cppn, (1,2), 1)
        print "Decoded CPPN:", net
        print
        net = Substrate(net)
        #print net

        sum_square_error = 0.0
        for inputs, expected in zip(xor_inputs, xor_outputs):

            new_input = inputs
            # print "Inputs", new_input
            output = net.query(new_input)
            # print "Output", output
            sum_square_error += ((output[0] - expected[0])**2.0)/4.0
            # print "Error", sum_square_error
 
        g.fitness = 1 - sum_square_error

# Create the population and run the XOR task by providing the above fitness function.
def run(gens):
    # Create population
    pop = population.Population(config)
    # Gather statistics from population and add that reporter
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))

    # Run your population under some evaluation scheme for a number of generations
    winner = pop.run(eval_fitness, gens)
    print("dhn_xor done")
    return winner, stats

# If run as script.
if __name__ == '__main__':
    winner = run(300)[0]
    print('\nBest genome:\n{!s}'.format(winner))
    cppn = FeedForwardNetwork.create(winner, config)
    winner_net = decode(cppn, (1,2), 1)
    winner_net = Substrate(winner_net)
    # Verify network output against training data.
    print('\nOutput:')
    
    sum_square_error = 0.0
    for inputs, expected in zip(xor_inputs, xor_outputs):
        new_input = inputs
        # print "Inputs", new_input
        output = winner_net.query(new_input)
        # print "Output", output
        sum_square_error += ((output[0] - expected[0])**2.0)/4.0
        # print "Error", sum_square_error
        print("  input {!r}, expected output {!r}, got {!r}".format(inputs, expected, output))

    # # Save CPPN if wished reused and draw it to file along with the winner.
    # with open('hyperneat_xor_cppn.pkl', 'wb') as output:
    #     pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)
    # draw_net(cppn, filename="hyperneat_xor_cppn")
    # draw_net(winner_net, filename="hyperneat_xor_winner")