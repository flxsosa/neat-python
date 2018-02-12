from neat.genes import DefaultNodeGene
from neat.genome import DefaultGenome, create_node, DefaultGenomeConfig
from neat.attributes import BaseAttribute

class DHNNodeGene(DefaultNodeGene):
    ''' Node gene for Deep HyperNEAT. Contains Source/Target Tuple '''
	_gene_attributes = [FloatAttribute('bias'),
                        FloatAttribute('response'),
                        StringAttribute('activation', options='sigmoid'),
                        StringAttribute('aggregation', options='sum')]

    def __init__(self, key, tuple=((),())):
        self.tuple = tuple
        assert isinstance(key, int), "DefaultNodeGene key must be an int, not {!r}".format(key)
        DefaultNodeGene.__init__(self, key)

class DHNGenome(DefaultGenome):
    ''' Genome for Deep HyperNEAT. Contains DHN mutations.'''
    
    def __init__(self,key):
        DefaultGenome.__init__(self, key)

    @classmethod
    def mutate(cls, config):
        """ Mutates this genome. """

        if config.single_structural_mutation:
            div = max(1,(config.node_add_prob + config.node_delete_prob +
                         config.conn_add_prob + config.conn_delete_prob))
            r = random()
            if r < (config.node_add_prob/div):
                DefaultGenome.mutate_add_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob)/div):
                DefaultGenome.mutate_delete_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob)/div):
                DefaultGenome.mutate_add_connection(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob + config.conn_delete_prob)/div):
                DefaultGenome.mutate_delete_connection()
        else:
            if random() < config.node_add_prob:
                DefaultGenome.mutate_add_node(config)

            if random() < config.node_delete_prob:
                DefaultGenome.mutate_delete_node(config)

            if random() < config.conn_add_prob:
                DefaultGenome.mutate_add_connection(config)

            if random() < config.conn_delete_prob:
                DefaultGenome.mutate_delete_connection()

        # Mutate connection genes.
        for cg in DefaultGenome.connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.).
        for ng in DefaultGenome.nodes.values():
            ng.mutate(config)

class DHNGenomeConfig(DefaultGenomeConfig):
    pass