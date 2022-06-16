from bmtk.builder import NetworkBuilder
from bmtk.builder.auxi.node_params import positions_list, xiter_random
from bmtk.utils.sim_setup import build_env_bionet
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator
from math import exp
import numpy as np
import pandas as pd
import random
import synapses

synapses.load()
syn = synapses.syn_params_dicts()

seed = 999

rng = np.random.default_rng(seed)

net = NetworkBuilder("biophysical")
# amount of cells
numAAC = 147
numCCK =  360
numNGF = 580
numOLM =  164
numPV =  553
numPyr =  31150

# arrays for cell location csv
cell_name = []
cell_x = []
cell_y = []
cell_z = []
# amount of cells per layer
numAAC_inSO = int(round(numAAC*0.238))
numAAC_inSP = int(round(numAAC*0.7))
numAAC_inSR = int(round(numAAC*0.062))
numCCK_inSO = int(round(numCCK*0.217))
numCCK_inSP = int(round(numCCK*0.261))
numCCK_inSR = int(round(numCCK*0.325))
numCCK_inSLM = int(round(numCCK*0.197))
numNGF_inSR = int(round(numNGF*0.17))
numNGF_inSLM = int(round(numNGF*0.83))
numPV_inSO = int(round(numPV*0.238))
numPV_inSP = int(round(numPV*0.701))
numPV_inSR = int(round(numPV*0.0596))

totalCellNum = numAAC_inSO + numAAC_inSP + numAAC_inSR + numCCK_inSO + numCCK_inSP + numCCK_inSR + numCCK_inSLM + numNGF_inSR + numNGF_inSLM + numPV_inSO + numPV_inSP + numPV_inSR



def AAC_to_PYR(src, trg, a, x0, sigma, max_dist):
    if src.node_id == trg.node_id:
        return 0

    sid = src.node_id
    tid = trg.node_id

    src_pos = src['positions']
    trg_pos = trg['positions']

    dist = np.sqrt((src_pos[0] - trg_pos[0]) ** 2 + (src_pos[1] - trg_pos[1]) ** 2 + (src_pos[2] - trg_pos[2]) ** 2)
    prob = a * exp(-((dist - x0) ** 2) / (2 * sigma ** 2))
    #print(dist)
    #prob = (prob/100)
    #print(prob)

    if dist <= max_dist:
        global count
        count = count + 1
    if dist <= max_dist and np.random.uniform() < prob:
        connection = 1
        #print("creating {} synapse(s) between cell {} and {}".format(1,sid,tid))
    else:
        connection = 0
    return connection

def n_connections(src, trg, max_dist, prob=0.1):
    """Referenced by add_edges() and called by build() for every source/target pair. For every given target/source
    pair will connect the two with a probability prob (excludes self-connections)"""
    if src.node_id == trg.node_id:
        return 0

    src_pos = src['positions']
    trg_pos = trg['positions']
    dist = np.sqrt((src_pos[0] - trg_pos[0]) ** 2 + (src_pos[1] - trg_pos[1]) ** 2 + (src_pos[2] - trg_pos[2]) ** 2)
    if dist <= max_dist:
        if np.random.uniform() > prob:
            return 0
        else:
            return 1
# total 400x1000x450
# Order from top to bottom is SO,SP,SR,SLM total


def make_layer_grid(xstart,ystart,zstart,x_length,y_length,z_length,min_dist):
    x_grid = np.arange(xstart, x_length+min_dist, min_dist)
    y_grid = np.arange(ystart, y_length+min_dist, min_dist)
    z_grid = np.arange(zstart, z_length+min_dist, min_dist)
    xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
    return np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T


#wrapper function for adding nodes to net, since there are not many params to change each time
def setNodes(netObj,Number,posList,popName,mTemplate):
    inds = rng.choice(np.arange(0, posList.shape[0]), Number, replace=False)

    # Place cell
    netObj.add_nodes(N=Number, pop_name=popName,
                positions=positions_list(positions=posList[inds, :]),
                mem_potential='e',
                model_type='biophysical',
                model_template=mTemplate,
                rotation_angle_zaxis=(np.pi/2), #90 degrees
                morphology=None)

    return np.delete(posList, inds, 0)

def setEdges(netObj,src,dest,conParams,dynamics_name,dist_range,secName,secID,secx):
    netObj.add_edges(source={'pop_name': src}, target={'pop_name': dest},
                     
                     connection_rule=n_connections,
                     connection_params={'prob': conParams[0], 'max_dist': conParams[1]},  
                     syn_weight=1,
                     delay=0.1,
                     dynamics_params=dynamics_name,
                     model_template=syn[dynamics_name]['level_of_detail'],
                     distance_range=dist_range,
                     target_sections=[secName],
                     sec_id = secID,  # check and is working putting syn on right location
                     sec_x = secx)

############################## x,y,z,     xlen,ylen,zlen, space
pos_list_SO = make_layer_grid( 0,0,320,   400,1000,450,  20)
pos_list_SP = make_layer_grid( 0,0,290,   400,1000,320,   8)
pos_list_SR = make_layer_grid( 0,0, 80,   400,1000,290,  20)
pos_list_SLM = make_layer_grid(0,0,  0,   400,1000, 79,  20)

########## add cells and connections to the network

pos_list_SO = setNodes(net,numAAC_inSO,pos_list_SO,'AAC','hoc:axoaxoniccell')
pos_list_SO = setNodes(net,numOLM,pos_list_SO,'OLM','hoc:olmcell')
pos_list_SO = setNodes(net,numPV_inSO,pos_list_SO,'PV','hoc:pvbasketcell')

pos_list_SP = setNodes(net,numPyr,pos_list_SP,'Pyr','hoc:pyramidalcell')
pos_list_SP = setNodes(net,numAAC_inSP,pos_list_SP,'AAC','hoc:axoaxoniccell')
pos_list_SP = setNodes(net,numPV_inSP,pos_list_SP,'PV','hoc:pvbasketcell')

pos_list_SR = setNodes(net,numAAC_inSR,pos_list_SR,'AAC','hoc:axoaxoniccell')
pos_list_SR = setNodes(net,numPV_inSR,pos_list_SR,'PV','hoc:pvbasketcell')

#############src     dest   prob      maxd                   dist_range                 sid sec
setEdges(net,'AAC','Pyr',[ 0.072,     400],'CHN2PN.json', [-10000.0, 10000.0],'axonal', 6, 0.5)
setEdges(net,'Pyr','AAC',[ 0.009635,  400],'PN2CHN.json', [0.0,        400.0],'apical', 6, 0.5)
setEdges(net,'PV','Pyr', [ 0.05366,   400],'PV2PN.json',  [0.0,        400.0],'somatic',0, 0.5)
setEdges(net,'Pyr','PV', [ 0.0238,    400],'PV2PN.json',  [0.0,        400.0],'apical', 6, 0.5)
setEdges(net,'PV','AAC', [ 0.135,     400],'PV2CHN.json', [0.0,        400.0],'somatic',0, 0.5)
setEdges(net,'PV','PV',  [ 0.135,     400],'PV2PV.json',  [0.0,        400.0],'somatic',0, 0.5)
setEdges(net,'OLM','Pyr',[ 0.08300,   400],'OLM2PN.json', [0.0,        400.0],'apical', 4, 0.5)
setEdges(net,'OLM','AAC',[ 0.0800,    400],'OLM2CHN.json',[0.0,        400.0],'apical', 4, 0.5)
setEdges(net,'OLM','PV', [ 0.0800,    400],'OLM2PV.json', [0.0,        400.0],'apical', 4, 0.5)
setEdges(net,'OLM','OLM',[ 0.0800,    400],'OLM2OLM.json',[0.0,        400.0],'basal',  0, 0.9)
setEdges(net,'Pyr','OLM',[  0.1320,   400],'OLM2OLM.json',[0.0,        400.0],'basal',  2, 0.5)


net.build()
net.save(output_dir='network')

psg = PoissonSpikeGenerator(population='bgpn',
       seed=222)

psg.add(node_ids=range(0,totalCellNum),  # need same number as cells
        firing_rate=5,    # spikes/second
        times=(0, 300/1000))#seconds

psg.to_sonata('CA1_inputs/bg_pn_spikes.h5')


vNet = NetworkBuilder('bgpn')
vNet.add_nodes(
    N=totalCellNum,
    pop_name='bgpn',
    potential='exc',
    model_type='virtual'
)


vNet.add_edges(source=vNet.nodes(), target=net.nodes(),
              connection_rule= lambda source,target: source.node_id == target.node_id,
              syn_weight=1,
              target_sections=['somatic'],
              delay=0.1,
              distance_range=[0.0, 400],

              dynamics_params='AMPA_ExcToExc.json',
              model_template='exp2syn')
vNet.build()
vNet.save_nodes(output_dir='network')
vNet.save_edges(output_dir='network')



#use the below on the first build, then add mechanisms directory to the circuit config manually and comment out what is below
"""
build_env_bionet(base_dir='./',
                network_dir='./network',
                config_file='config.json',
                tstop=t_stim, dt=0.1,
                report_vars=['v'],
                components_dir='biophys_components',
                
                spikes_inputs=[('bgpn', 'CA1_inputs/bg_pn_spikes.h5')],

                v_init=-70,
                compile_mechanisms=False)
"""



#print('Number of background spikes to PN cells: {}'.format(psg.n_spikes()))




