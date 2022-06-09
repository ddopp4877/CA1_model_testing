"""Script for running the network built in build_network.py
Also saves a file called Connections.csv that consists of information about
each synapse in the simulation.
"""

import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)

from bmtk.simulator import bionet
import numpy as np
from neuron import h
import synapses
import pandas as pd



def save_connections(graph, sim):
    """Saves Connections.csv based on the given network.
    Parameters
    ----------
    graph : BioNetwork
        the network that the connections are retrieved from
    sim : BioSimulator
        the simulation about to be run (not used in this function)
    """
    cells = graph.get_local_cells()
    cell = cells[list(cells.keys())[0]]

    h.distance(sec=cell.hobj.soma[0])  # Makes the distances correct.

    sec_types = []  # soma, apic, or dend
    weights = []  # scaled conductances (initW)
    dists = []  # distance from soma
    node_ids = []  # node id within respective node population (exc, prox_inh, dist_inh)
    names = []  # full NEURON str representation of postsynaptic segment
    source_pops = []  # node population
    release_probs = []  # propability of release.
    x_angle = []
    y_angle = []
    z_angle = []
    loc = []
    gid = []
    cell_type = []

    for c in cell.connections():
        con = c._connector
        source = c.source_node
        syn = con.syn()
        seg = con.postseg()
        fullsecname = seg.sec.name()

        #source_pops.append(source.node_gid)
        #node_ids.append(source._node_id)

        #weights.append(float(syn.initW))
        #release_probs.append(float(syn.P_0))
        names.append(str(seg))
        sec_types.append(fullsecname.split(".")[1][:4])
        dists.append(float(h.distance(seg)))
    #gets cell pos
    for i in range(len(cells)):
        x_angle.append(cells[i].node.rotation_angle_xaxis)
        y_angle.append(cells[i].node.rotation_angle_yaxis)
        z_angle.append(cells[i].node.rotation_angle_zaxis)
        loc.append(np.array(cells[i].soma_position))
        gid.append(cells[i].gid)
        cell_type.append(cells[i].hobj)
    df = pd.DataFrame()
    df["gid"] = gid
    df['cell type'] = cell_type
    df["cell location"] = loc
    df["x_roation"] = x_angle
    df["y_roation"] = y_angle
    df["z_roation"] = z_angle
    df.to_csv("Cell_rotations.csv", index=False)

synapses.load()
config_file = 'simulation_config.json'
conf = bionet.Config.from_json(config_file, validate=True)
conf.build_env()
graph = bionet.BioNetwork.from_config(conf)
pop = graph._node_populations['biophysical']
for node in pop.get_nodes():
    node._node._node_type_props['morphology'] = node.model_template[1]
sim = bionet.BioSimulator.from_config(conf, network=graph)

save_connections(graph, sim)
