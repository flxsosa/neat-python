import numpy as np
import matplotlib.pyplot as plt
import itertools as it

def visualize_activations(activations, input_dim, sheet_dim):

    input_layer_activations = [x for x in activations]

    print(input_layer_activations)
    # # Create figure and axis
    # fig, ax = plt.subplots()

    # min_val, max_val = 24,24

    # x_coords = y_coords = np.linspace(-1.0, 1.0, 24)

    # xy_coords = np.array(list(it.product(x_coords, y_coords)), ndmin=2)
    # print(xy_coords)

    # # Weight matrix
    # weight_matrix =  np.eye(24,24)

    # # Draw the matrix on the pot
    # ax.matshow(weight_matrix, cmap=plt.cm.Blues)

    # # Traverse the matrix, going through each row at a time and 
    # # annotate the plot with its values
    # for i in xrange(24):
    #     for j in xrange(24):
    #         # Grab the value at the particular cell
    #         weight = weight_matrix[j,i]
    #         # Annotate the plot with that value at that cell
    #         ax.text(i, j, str(weight), va='center', ha='center')

    # ax.matshow(weight_matrix, cmap=plt.cm.Blues)

    # # ax.set_xlim(min_val, max_val)
    # # ax.set_ylim(min_val, max_val)
    # ax.set_xticks(np.arange(max_val))
    # ax.set_yticks(np.arange(max_val))
    # # Place x tick labels on bottom of plot
    # ax.xaxis.set_ticks_position('bottom')
    # # Set x and y tick labels
    # ax.set_xticklabels([round(x,2) for x in np.linspace(-1.0,1.0,24)], rotation=50)
    # ax.set_yticklabels([round(x,2) for x in np.linspace(1.0,-1.0,24)])
    # plt.show()
    # # ax.grid()