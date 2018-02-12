"""Handles genomes (individuals in the population)."""
from __future__ import division, print_function


from itertools import count
from random import choice, random, shuffle

import sys

from random import randint
from activations import ActivationFunctionSet
from aggregations import AggregationFunctionSet
from neat.config import ConfigParameter, write_pretty_params
from dhngenes import DefaultConnectionGene, DHNNodeGene
from neat.graphs import creates_cycle
from neat.six_util import iteritems, iterkeys
from dhngenes import DHNNodeGene

class DefaultGenomeConfig(object):
    """Sets up and holds configuration information for the DefaultGenome class."""
    allowed_connectivity = ['unconnected', 'fs_neat_nohidden', 'fs_neat', 'fs_neat_hidden',
                            'full_nodirect', 'full', 'full_direct',
                            'partial_nodirect', 'partial', 'partial_direct']

    def __init__(self, params):
        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        # ditto for aggregation functions - name difference for backward compatibility
        self.aggregation_function_defs = AggregationFunctionSet()
        self.aggregation_defs = self.aggregation_function_defs

        self._params = [ConfigParameter('num_inputs', int),
                        ConfigParameter('num_outputs', int),
                        ConfigParameter('num_hidden', int),
                        ConfigParameter('feed_forward', bool),
                        ConfigParameter('compatibility_disjoint_coefficient', float),
                        ConfigParameter('compatibility_weight_coefficient', float),
                        ConfigParameter('conn_add_prob', float),
                        ConfigParameter('conn_delete_prob', float),
                        ConfigParameter('node_add_prob', float),
                        ConfigParameter('node_delete_prob', float),
                        ConfigParameter('inc_depth_prob', float),
                        ConfigParameter('inc_breadth_prob', float),
                        ConfigParameter('single_structural_mutation', bool, 'false'),
                        ConfigParameter('structural_mutation_surer', str, 'default'),
                        ConfigParameter('initial_connection', str, 'full_direct')]

        # Gather configuration data from the gene classes.
        self.node_gene_type = params['node_gene_type']
        self._params += self.node_gene_type.get_config_params()
        self.connection_gene_type = params['connection_gene_type']
        self._params += self.connection_gene_type.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

        self.connection_fraction = None

        # Verify that initial connection type is valid.
        # pylint: disable=access-member-before-definition
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

        # Verify structural_mutation_surer is valid.
        # pylint: disable=access-member-before-definition
        if self.structural_mutation_surer.lower() in ['1','yes','true','on']:
            self.structural_mutation_surer = 'true'
        elif self.structural_mutation_surer.lower() in ['0','no','false','off']:
            self.structural_mutation_surer = 'false'
        elif self.structural_mutation_surer.lower() == 'default':
            self.structural_mutation_surer = 'default'
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)

        self.node_indexer = None

    def add_activation(self, name, func):
        self.activation_defs.add(name, func)

    def add_aggregation(self, name, func):
        self.aggregation_function_defs.add(name, func)

    def save(self, f):
        if 'partial' in self.initial_connection:
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")
            f.write('initial_connection      = {0} {1}\n'.format(self.initial_connection,
                                                                 self.connection_fraction))
        else:
            f.write('initial_connection      = {0}\n'.format(self.initial_connection))

        assert self.initial_connection in self.allowed_connectivity

        write_pretty_params(f, self, [p for p in self._params
                                      if not 'initial_connection' in p.name])

    def get_new_node_key(self, node_dict):
        if self.node_indexer is None:
            self.node_indexer = count(max(list(iterkeys(node_dict))) + 1)

        new_id = next(self.node_indexer)

        assert new_id not in node_dict

        return new_id

    def check_structural_mutation_surer(self):
        if self.structural_mutation_surer == 'true':
            return True
        elif self.structural_mutation_surer == 'false':
            return False
        elif self.structural_mutation_surer == 'default':
            return self.single_structural_mutation
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)

class DHNGenome(object):
    # Config methods

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DHNNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    # Initialization methods

    def __init__(self, key):
        # Unique identifier for a genome instance.
        self.key = key

        # (gene_key, gene) pairs for gene sets.
        self.connections = {}
        self.nodes = {}

        # Number of layers encoded by CPPN
        self.num_layers = 0

        # Fitness results.
        self.fitness = None
        self.num_inputs = 5
        self.num_outputs = 1
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""

        # Create node genes for the output pins.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key, ((1,0),(0,0)))

        # Add hidden nodes if requested.
        if config.num_hidden > 0:
            for i in range(config.num_hidden):
                node_key = config.get_new_node_key(self.nodes)
                assert node_key not in self.nodes
                node = self.create_node(config, node_key)
                self.nodes[node_key] = node

        # Add connections based on initial connectivity type.

        if 'fs_neat' in config.initial_connection:
            if config.initial_connection == 'fs_neat_nohidden':
                self.connect_fs_neat_nohidden(config)
            elif config.initial_connection == 'fs_neat_hidden':
                self.connect_fs_neat_hidden(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = fs_neat will not connect to hidden nodes;",
                        "\tif this is desired, set initial_connection = fs_neat_nohidden;",
                        "\tif not, set initial_connection = fs_neat_hidden",
                        sep='\n', file=sys.stderr);
                self.connect_fs_neat_nohidden(config)
        elif 'full' in config.initial_connection:
            if config.initial_connection == 'full_nodirect':
                self.connect_full_nodirect(config)
            elif config.initial_connection == 'full_direct':
                self.connect_full_direct(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = full with hidden nodes will not do direct input-output connections;",
                        "\tif this is desired, set initial_connection = full_nodirect;",
                        "\tif not, set initial_connection = full_direct",
                        sep='\n', file=sys.stderr);
                self.connect_full_nodirect(config)
        elif 'partial' in config.initial_connection:
            if config.initial_connection == 'partial_nodirect':
                self.connect_partial_nodirect(config)
            elif config.initial_connection == 'partial_direct':
                self.connect_partial_direct(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = partial with hidden nodes will not do direct input-output connections;",
                        "\tif this is desired, set initial_connection = partial_nodirect {0};".format(
                            config.connection_fraction),
                        "\tif not, set initial_connection = partial_direct {0}".format(
                            config.connection_fraction),
                        sep='\n', file=sys.stderr);
                self.connect_partial_nodirect(config)

    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """
        # Sanity check
        assert isinstance(genome1.fitness, (int, float))
        assert isinstance(genome2.fitness, (int, float))

        # Determine fitter parent
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent11, parent2 = genome2, genome1

        # Inherit fitter parent's output nodes
        self.output_keys = parent1.output_keys
        # Inherit connection genes
        for key, cg1 in iteritems(parent1.connections):
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2)

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes
        
        for key, ng1 in iteritems(parent1_set):
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)
        
        # Determine disjoint CPPNON genes
        disjoint_output_nodes = [parent2.nodes[n] for n in parent2.output_keys if n not in parent1.output_keys if n not in self.nodes]
        for node in disjoint_output_nodes:
            flag = True
            for key in self.output_keys:
                if self.nodes[key].cppn_tuple == node.cppn_tuple:
                    flag = False
            if flag:
                self.nodes[node.key] = node.copy()
                self.output_keys.append(node.key)
                self.recursively_add_connection(parent2, node)
    
    # Mutation methods

    def mutate(self, config):
        """ Mutates this genome. """

        # Structural/topological mutations
        if config.single_structural_mutation:
            # Normalize
            div = max(1,(config.node_add_prob + config.node_delete_prob +
                        config.conn_add_prob + config.conn_delete_prob +
                        config.inc_depth_prob + config.inc_breadth_prob))
            # Grab a random number
            r = random()
            # Traverse mutations based on RNG
            if r < (config.node_add_prob/div):
                self.mutate_add_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob)/div):
                self.mutate_delete_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob)/div):
                self.mutate_add_connection(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob + config.conn_delete_prob)/div):
                self.mutate_delete_connection()
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob + config.conn_delete_prob +
                       config.inc_depth_prob)/div):
                self.mutate_increment_depth(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob + config.conn_delete_prob +
                       config.inc_depth_prob + config.inc_breadth_prob)/div):
                self.mutate_increment_breadth(config)

        else:
            if random() < config.node_add_prob:
                self.mutate_add_node(config)

            if random() < config.node_delete_prob:
                self.mutate_delete_node(config)

            if random() < config.conn_add_prob:
                self.mutate_add_connection(config)

            if random() < config.conn_delete_prob:
                self.mutate_delete_connection()

            if random() < config.inc_depth_prob:
                self.mutate_increment_depth(config)

            if random() < config.inc_breadth_prob:
                self.mutate_increment_breadth(config)

        # Mutate connection genes.
        for cg in self.connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.nodes.values():
            ng.mutate(config)

    def mutate_add_node(self, config):
        if not self.connections:
            if config.check_structural_mutation_surer():
                self.mutate_add_connection(config)
            return

        # Choose a random connection to split
        conn_to_split = choice(list(self.connections.values()))
        new_node_id = config.get_new_node_key(self.nodes)
        ng = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = ng

        # Disable this connection and create two new connections joining its nodes via
        # the given node.  The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        conn_to_split.enabled = False

        i, o = conn_to_split.key
        del self.connections[conn_to_split.key]
        self.add_connection(config, i, new_node_id, 1.0, True)
        self.add_connection(config, new_node_id, o, conn_to_split.weight, True)

    def add_connection(self, config, input_key, output_key, weight, enabled):
        # TODO: Add further validation of this connection addition?
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0
        assert isinstance(enabled, bool)
        key = (input_key, output_key)
        connection = config.connection_gene_type(key)
        connection.init_attributes(config)
        connection.weight = weight
        connection.enabled = True
        self.connections[key] = connection

    def mutate_add_connection(self, config):
        """
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        """
        possible_outputs = list(iterkeys(self.nodes))
        out_node = choice(possible_outputs)

        possible_inputs = possible_outputs + config.input_keys
        in_node = choice(possible_inputs)

        # Don't duplicate connections.
        key = (in_node, out_node)
        if key in self.connections:
            # TODO: Should this be using mutation to/from rates? Hairy to configure...
            if config.check_structural_mutation_surer():
                self.connections[key].enabled = True
            return

        # Don't allow connections between two output nodes
        if in_node in config.output_keys and out_node in config.output_keys:
            return

        if in_node in config.output_keys:
            return
        # No need to check for connections between input nodes:
        # they cannot be the output end of a connection (see above).

        # For feed-forward networks, avoid creating cycles.
        if config.feed_forward and creates_cycle(list(iterkeys(self.connections)), key):
            return

        cg = self.create_connection(config, in_node, out_node)
        self.connections[cg.key] = cg

    def mutate_delete_node(self, config):
        '''
        Deletes a node from the CPPN. Does not delete any CPPNONs.

        TODO: Allow for the deletion of CPPNONs.
        '''
        # Do nothing if there are no non-output nodes.
        available_nodes = [k for k in iterkeys(self.nodes) if k not in self.output_keys]
        if not available_nodes:
            return -1

        del_key = choice(available_nodes)

        connections_to_delete = set()
        for k, v in iteritems(self.connections):
            if del_key in v.key:
                connections_to_delete.add(v.key)

        for key in connections_to_delete:
            del self.connections[key]

        del self.nodes[del_key]

        return del_key
        # # Available nodes are only non-output nodes
        # available_nodes = [k for k in iterkeys(self.nodes) if k != 0]
        # # If there are no available nodes, return 
        # if not available_nodes:
        #     return -1

        # # Randomly choose from available nodes to delete
        # del_key = choice(available_nodes)

        # # Gather connections to delete associated with chosen node
        # connections_to_delete = set()
        # for k, v in iteritems(self.connections):
        #     if del_key in v.key:
        #         connections_to_delete.add(v.key)
        
        # if del_key in self.output_keys:
        #     print("Deleted CPPNON", del_key, "with Tuple", self.nodes[del_key].cppn_tuple)
        #     source_sheet = 
        
        # # Delete connections
        # for key in connections_to_delete:
        #     del self.connections[key]

        # if del_key in self.output_keys:
        #     # Assign source and target sheets
        #     source_sheet = self.nodes[del_key].cppn_tuple[0]
        #     target_sheet = self.nodes[del_key].cppn_tuple[1]
            
        #     # Delete node
        #     del self.nodes[del_key]
        #     self.output_keys.remove(del_key)

        #     # Assign mappings
        #     outgoing_mappings = []
        #     incoming_mappings = []

        #     # Go through the CPPNONs
        #     for node_key in self.output_keys:
        #         # Check if there are any outgoing connections
        #         if self.nodes[node_key].cppn_tuple[0] == source_sheet:
        #             del self.nodes[node_key]
        #             self.output_keys.remove(node_key)
        #         # Chec if there are any incoming connections
        #         if self.nodes[node_key].cppn_tuple[1] == target_sheet:
        #             del self.nodes[node_key]
        #             self.output_keys.remove(node_key)
            
        #     # Adjust CPPNON tuples
        #     self.sanitize_afer_crossover()
            
        # # Delete node
        # del self.nodes[del_key]

        # # Return deleted node key if mutation succesful
        # return del_key

    def mutate_delete_connection(self):
        if self.connections:
            key = choice(list(self.connections.keys()))
            del self.connections[key]

    def mutate_increment_depth(self, config):
        '''
        Add a new layer node to the CPPN
        '''
        if not self.connections:
            if config.check_structural_mutation_surer():
                self.mutate_add_connection(config)
            return

        # Define Source and Target layers for CPPNON
        # Should be set comprehension
        seen = [0]
        for key in self.output_keys:
            num = self.nodes[key].cppn_tuple[0][0]
            if num not in seen:
                seen.append(num)

        source_layer = len(seen)
        target_layer = 0

        # Define target sheet as 0 because we are making a new layer in the substrate
        target_sheet = 0
        source_sheet = 0

        # Adjust tuples for previous CPPNONs
        for key in self.output_keys:
            try:
                tup = self.nodes[key].cppn_tuple
            except KeyError:
                print("Attempted key: {0}, Key List: {1}".format(key, self.nodes))
            if tup[1] == (0,0):
                self.nodes[key].cppn_tuple = (tup[0], (source_layer,source_sheet))
                
        # Create two new Gaussian nodes
        new_node_id_g1 = config.get_new_node_key(self.nodes)
        new_node_id_g2 = config.get_new_node_key(self.nodes)
        new_gauss1_node = self.create_node(config, new_node_id_g1)
        new_gauss2_node = self.create_node(config, new_node_id_g2)

        # Create new CPPN Output Node (CPPNON)
        cppn_tuple = ((source_layer, source_sheet),(target_layer,target_sheet))
        new_node_id = config.get_new_node_key(self.nodes)
        new_cppn_on = self.create_node(config, new_node_id, cppn_tuple)
        new_cppn_on.activation = 'linear'
        new_cppn_on.bias = 0.0
        
        # Add new CPPNON key to list of output keys in genome
        self.num_outputs += 1
        self.output_keys.append(new_cppn_on.key)

        # Add new Gauss nodes to node list in genome
        new_gauss1_node.activation = 'dhngauss'
        new_gauss2_node.activation = 'dhngauss'
        new_gauss1_node.aggregation = 'foo'
        new_gauss2_node.aggregation = 'foo'
        new_gauss1_node.bias = 0.0
        new_gauss2_node.bias = 0.0
        self.nodes[new_gauss1_node.key] = new_gauss1_node
        self.nodes[new_gauss2_node.key] = new_gauss2_node

        # Create check node where Guass connects to
        new_node_check_id = config.get_new_node_key(self.nodes)
        new_check_node = self.create_node(config, new_node_check_id)
        new_check_node.activation = 'dhngauss2'
        new_check_node.aggregation = 'sum'
        new_check_node.bias = 0.0
        self.nodes[new_node_check_id] = new_check_node
        # Add new CPPNON to node list
        self.nodes[new_cppn_on.key] = new_cppn_on

        # Add connections
        # Assuming CPPN only has four inputs: x1 in [0], y1 in [1], x2 in [2], y2 in [3]     
        # x1 to Gauss 1
        self.add_connection(config, self.input_keys[0], new_node_id_g1, 1.0, True)
        # x2 to Gauss 1
        self.add_connection(config, self.input_keys[1], new_node_id_g2, 1.0, True)
        # y1 to Gauss 2
        self.add_connection(config, self.input_keys[2], new_node_id_g1, 1.0, True)
        # y2 to Gauss 2
        self.add_connection(config, self.input_keys[3], new_node_id_g2, 1.0, True) 
        # Gauss 1 to CPPNON
        self.add_connection(config, new_node_id_g1, new_check_node.key, 1.0, True)
        # Gauss 2 to CPPNON
        self.add_connection(config, new_node_id_g2, new_check_node.key, 1.0, True)
        # Check to CPPNON
        self.add_connection(config, new_check_node.key,new_cppn_on.key,1.0,True)

        # Increment number of layers
        self.num_layers += 1

    def mutate_increment_breadth(self, config):
        '''
        Creates a CPPNON to represent a new sheet in existing layer.
        '''
        # Find out how many layers are represented by current set of CPPNONs and 
        # pick a random layer in that interval

        # seen = [0]
        # for key in self.output_keys:
        #     try:
        #         num = self.nodes[key].cppn_tuple[0][0]
        #     except KeyError:
        #         print("Attempted key: {0}, Key List: {1}".format(key, self.output_keys))
        #     if num not in seen:
        #         seen.append(num)
        seen = {self.nodes[key].cppn_tuple[0][0] for key in self.output_keys}
        seen.add(0)
        # print(seen)
        num_layers = len(seen)
       
        # Can only expand a layer with more sheets if there is a hidden layer
        if num_layers <= 2:
            self.mutate_increment_depth(config)
       
        else:
            layer = randint(2,num_layers-1)
            # Find out how many sheets are represented by current set of CPPNONs and
            # pick a random sheet in that interval
            num_sheets = 0
            for key in self.output_keys:
                if self.nodes[key].cppn_tuple[0][0] == layer:
                    num_sheets += 1

            assert num_sheets >= 1
            sheet = randint(0,num_sheets-1)

            # Initiate copied sheet
            copied_sheet = (layer, sheet)
            # List for keys to append to output_keys after loop to prevent infinite loop
            keys_to_append = []

            # Search for CPPNONs that contain the selected sheet to be copied in their tuple
            for key in self.output_keys:

                # Create CPPNONs to represent outgoing connections from new sheet
                if self.nodes[key].cppn_tuple[0] == copied_sheet:
                    # create new cppn node for newly copied sheet
                    cppn_tuple = ((layer,num_sheets),self.nodes[key].cppn_tuple[1])
                    new_cppn_id = config.get_new_node_key(self.nodes)
                    new_cppn_on = self.create_node(config, new_cppn_id, cppn_tuple)
                    new_cppn_on.activation = self.nodes[key].activation
                    new_cppn_on.bias = self.nodes[key].bias
                    self.nodes[new_cppn_on.key] = new_cppn_on
                    keys_to_append.append(new_cppn_on.key)

                    # Create connections in CPPN and halve existing connections
                    for conn in list(self.connections):
                        if conn[1] == key:
                            # print(self.connections[conn])
                            self.connections[conn].weight /= 2.0
                            # print(self.connections[conn])
                            self.add_connection(config, conn[0], new_cppn_on.key, self.connections[conn].weight, True)
                
                # Create CPPNONs to represent the incoming connections to new sheet
                if self.nodes[key].cppn_tuple[1] == copied_sheet:
                    # create new cppn node for newly copied sheet
                    cppn_tuple = (self.nodes[key].cppn_tuple[0],(layer,num_sheets))
                    new_cppn_id = config.get_new_node_key(self.nodes)
                    new_cppn_on = self.create_node(config, new_cppn_id, cppn_tuple)
                    new_cppn_on.activation = self.nodes[key].activation
                    new_cppn_on.bias = self.nodes[key].bias
                    self.nodes[new_cppn_on.key] = new_cppn_on
                    keys_to_append.append(new_cppn_on.key)

                    # Create connections in CPPN
                    for conn in list(self.connections):
                        if conn[1] == key:
                            self.add_connection(config, conn[0], new_cppn_on.key, self.connections[conn].weight, True)      
            
            # Add new CPPNONs to genome
            self.num_outputs += len(keys_to_append)
            self.output_keys.extend(keys_to_append)

    # Helper methods

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Compute node gene distance component.
        node_distance = 0.0
        if self.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in iterkeys(other.nodes):
                if k2 not in self.nodes:
                    disjoint_nodes += 1

            for k1, n1 in iteritems(self.nodes):
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance +
                             (config.compatibility_disjoint_coefficient *
                              disjoint_nodes)) / max_nodes

        # Compute connection gene differences.
        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in iterkeys(other.connections):
                if k2 not in self.connections:
                    disjoint_connections += 1

            for k1, c1 in iteritems(self.connections):
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2, config)

            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (connection_distance +
                                   (config.compatibility_disjoint_coefficient *
                                    disjoint_connections)) / max_conn

        distance = node_distance + connection_distance
        return distance

    def size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        num_enabled_connections = sum([1 for cg in self.connections.values() if cg.enabled])
        return len(self.nodes), num_enabled_connections

    def __str__(self):
        ''' String printed out for results of genome after experiment'''

        s = "Key: {0}\nFitness: {1}\nNodes:".format(self.key, self.fitness)
        for k, ng in iteritems(self.nodes):
            s += "\n\t{0} {1!s} Tuple: {2}".format(k, ng, self.nodes[k].cppn_tuple)
        s += "\nConnections:"
        connections = list(self.connections.values())
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)
        return s

    @staticmethod
    def create_node(config, node_id, cppn_tuple=((),())):
        ''' 
        Creates a node for CPPN
        '''
        node = DHNNodeGene(node_id, cppn_tuple)
        node.init_attributes(config)
        return node

    @staticmethod
    def create_connection(config, input_id, output_id):
        connection = config.connection_gene_type((input_id, output_id))
        connection.init_attributes(config)
        return connection

    def connect_fs_neat_nohidden(self, config):
        """
        Randomly connect one input to all output nodes
        (FS-NEAT without connections to hidden, if any).
        Originally connect_fs_neat.
        """
        input_id = choice(config.input_keys)
        for output_id in config.output_keys:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_fs_neat_hidden(self, config):
        """
        Randomly connect one input to all hidden and output nodes
        (FS-NEAT with connections to hidden, if any).
        """
        input_id = choice(config.input_keys)
        others = [i for i in iterkeys(self.nodes) if i not in config.input_keys]
        for output_id in others:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def compute_full_connections(self, config, direct):
        """
        Compute connections for a fully-connected feed-forward genome--each
        input connected to all hidden nodes
        (and output nodes if ``direct`` is set or there are no hidden nodes),
        each hidden node connected to all output nodes.
        (Recurrent genomes will also include node self-connections.)
        """
        hidden = [i for i in iterkeys(self.nodes) if i not in config.output_keys]
        output = [i for i in iterkeys(self.nodes) if i in config.output_keys]
        connections = []
        if hidden:
            for input_id in config.input_keys:
                for h in hidden:
                    connections.append((input_id, h))
            for h in hidden:
                for output_id in output:
                    connections.append((h, output_id))
        if direct or (not hidden):
            for input_id in config.input_keys:
                for output_id in output:
                    connections.append((input_id, output_id))

        # For recurrent genomes, include node self-connections.
        if not config.feed_forward:
            for i in iterkeys(self.nodes):
                connections.append((i, i))

        return connections

    def connect_full_nodirect(self, config):
        """
        Create a fully-connected genome
        (except without direct input-output unless no hidden nodes).
        """
        for input_id, output_id in self.compute_full_connections(config, False):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_full_direct(self, config):
        """ Create a fully-connected genome, including direct input-output connections. """
        for input_id, output_id in self.compute_full_connections(config, True):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial_nodirect(self, config):
        """
        Create a partially-connected genome,
        with (unless no hidden nodes) no direct input-output connections."""
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, False)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial_direct(self, config):
        """
        Create a partially-connected genome,
        including (possibly) direct input-output connections.
        """
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, True)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def sanitize_afer_crossover(self):
        '''
        Goes through current genome's output nodes and checks that
        they properly encode a deep substrate.
        '''

        # Count number of layers currently encoded
        seen = [0]
        for key in self.output_keys:
            try:
                num = self.nodes[key].cppn_tuple[0][0]
            except KeyError:
                print("Attempted key: {0}, Key List: {1}".format(key, self.output_keys))
            if num not in seen:
                seen.append(num)
        num_layers = len(seen)

        # Keep track of number of layers
        number_of_layers = num_layers
        halt = 0
        
        if num_layers <= 2:
            return

        # Traverse CPPNONs
        for i in range(2, number_of_layers):
            for output_key in self.output_keys:
                if self.nodes[output_key].cppn_tuple[0][0] == i:
                    if halt != 0:
                        self.nodes[output_key].cppn_tuple = ((halt, self.nodes[output_key].cppn_tuple[0][1]), 
                            (self.nodes[output_key].cppn_tuple[1][0],self.nodes[output_key].cppn_tuple[1][1]))
                        i = halt
                        halt = 0
                        number_of_layers = num_layers
              
            # Layer was not found
            if halt == 0:
                halt = i
            i += 1
            number_of_layers += 1

    def recursively_add_connection(self, parent, target):
        # Recursively adds connections and nodes to self genome
        # parent -- parent genome grabbing connections and nodes from
        # target -- target cppnon attempting to add
        # Rraverse the parent's connection genes
        for key, cg in iteritems(parent.connections):
            # If the target node in the current connection is out current node
            if key[1] == target.key:
                # Check if we havet the source of that connection
                # If we don't, add that source and recursively check if we have it's source, etc.
                if key[0] not in self.nodes and key[0] > 0:
                    self.nodes[key[0]] = parent.nodes[key[0]].copy()
                    self.recursively_add_connection(parent, parent.nodes[key[0]])
                # Add the connection
                self.connections[key] = cg.copy()