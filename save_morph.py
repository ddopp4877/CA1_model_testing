from bmtk.simulator import bionet
import numpy as np
from neuron import h
import synapses

pc = h.ParallelContext()  # object to access MPI methods
MPI_size = int(pc.nhost())
MPI_rank = int(pc.id())

synapses.load()
syn = synapses.syn_params_dicts()

config_file = 'simulation_config.json'

conf = bionet.Config.from_json(config_file, validate=True)
conf.build_env()

graph = bionet.BioNetwork.from_config(conf)
sim = bionet.BioSimulator.from_config(conf, network=graph)

from analyze_area import analyze_area, make_seg_df

cell_id = 1  # change this to change what cell your looking at

make_seg_df(list(graph.get_local_cells().values())[cell_id]) # change this number to pick what cell you wanna look at
#analyze_area(list(graph.get_local_cells().values())[0]._morph.seg_prop)

#sim.run()
pc.barrier()
