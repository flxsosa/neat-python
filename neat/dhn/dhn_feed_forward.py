from neat.graphs import feed_forward_layers
from neat.six_util import itervalues


class FeedForwardNetwork(object):
    '''
    Class for a feed forward CPPN
    '''
    def __init__(self, inputs, outputs, node_evals, nodes=None):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)
        self.nodes = nodes

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            # print("CPPN Input Node and Inputs", (k,v))
            self.values[k] = v
        # print("CPPN Values Now", self.values)
        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = []
            # print("Node:", node)
            for i, w in links:
                # print("CPPN Link",i,w)
                node_inputs.append(self.values[i] * w)
            s = agg_func(node_inputs)
            self.values[node] = act_func(bias + response * s)

        return self.values

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]

        layers = feed_forward_layers(genome.input_keys, genome.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                node_expr = [] # currently unused
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))
                        node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))


                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        # originally had output keys coming from config.genome_config.output_keys but now just coming from
        # genome.output_keys
        return FeedForwardNetwork(genome.input_keys, genome.output_keys, node_evals, 
                                    genome.nodes)