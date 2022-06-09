import h5py
import glob
import pprint
import numpy as np
import pandas as pd

from sonata.circuit import File

net = File(data_files=['network/biophysical_biophysical_edges.h5', 'network/biophysical_nodes.h5'],
           data_type_files=['network/biophysical_biophysical_edge_types.csv', 'network/biophysical_node_types.csv'])

print('Contains nodes: {}'.format(net.has_nodes))
print('Contains edges: {}'.format(net.has_edges))


file_edges = net.edges
print('Edge populations in file: {}'.format(file_edges.population_names))
recurrent_edges = file_edges['biophysical_to_biophysical']

conver_onto = 925

con_count = 0
for edge in recurrent_edges.get_target(conver_onto):  # we can also use get_targets([id0, id1, ...])
    assert (edge.target_node_id == conver_onto)
    print("cell %d has cell %d converging onto it" % (conver_onto, edge.source_node_id))
    con_count += 1

print('There are {} connections onto target node #{}'.format(con_count, conver_onto))

