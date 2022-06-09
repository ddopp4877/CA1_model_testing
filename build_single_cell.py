from bmtk.builder import NetworkBuilder
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator
import numpy as np
import sys



net = NetworkBuilder("biophysical")

#PV BASKET
#net.add_nodes(
#        mem_potential='e',
#        model_type='biophysical',
#        model_template='hoc:pvbasketcell',
#        morphology=None)

#CHN
#net.add_nodes(
#        mem_potential='e',
#        model_type='biophysical',
#        model_template='hoc:axoaxoniccell',
#        morphology=None)

#CCK basket
#net.add_nodes(
#        mem_potential='e',
#        model_type='biophysical',
#        model_template='hoc:cckcell',
#        morphology=None)

#OLM
#net.add_nodes(
#        mem_potential='e',
#        model_type='biophysical',
#        model_template='hoc:olmcell',
#        morphology=None)

#NGF
#net.add_nodes(
#        mem_potential='e',
#        model_type='biophysical',
#        model_template='hoc:ngfcell',
#        morphology=None)

#PYR
net.add_nodes(N=5, pop_name='Pyr',
        mem_potential='e',
        model_type='biophysical',
        model_template='hoc:pyramidalcell',
        morphology=None)

thalamus = NetworkBuilder('bg_pn')
thalamus.add_nodes(N=5,
                   pop_name='tON',
                   potential='exc',
                   model_type='virtual')

def one_to_one(source, target):
    sid = source.node_id
    tid = target.node_id
    if sid == tid:
        # print("connecting cell {} to {}".format(sid,tid))
        tmp_nsyn = 1
    else:
        return None

    return tmp_nsyn

net.add_edges(source=thalamus.nodes(), target=net.nodes(pop_name='Pyr'),
              connection_rule=one_to_one,
              syn_weight=1,
              target_sections=['somatic'],
              delay=0.1,
              distance_range=[0.0, 300.0],
              dynamics_params='AMPA_ExcToExc.json',
              model_template='exp2syn')

net.build()
net.save(output_dir='network')

thalamus.build()
thalamus.save(output_dir='network')


psg = PoissonSpikeGenerator(population='bg_pn')
psg.add(node_ids=range(5),  # Have 5 nodes to match mthalamus
        firing_rate=0.2,    # 2 Hz
        times=(0.0, 5))  # time is in seconds for some reason
psg.to_sonata('bg_pn_spikes.h5')

print('Number of spikes: {}'.format(psg.n_spikes()))


from bmtk.utils.sim_setup import build_env_bionet
build_env_bionet(base_dir='./',
                network_dir='./network',
                tstop=5000.0, dt=0.1,
                report_vars=['v'],
                config_file='config.json',
                spikes_inputs=[('bg_pn', 'bg_pn_spikes.h5')],
                components_dir='biophys_components',
                compile_mechanisms=False)


#current_clamp = {
#                   'amp': .300,
#                    'delay': 100,
#                    'duration': 600
#                },




