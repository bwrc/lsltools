#shebangs here

from lsltools import sim,vis

eeg_data = sim.EEGData(nch=3,stream_name="example")
eeg_data.start()


streams = vis.pylsl.resolve_byprop("name","example")
eeg_graph = vis.Grapher(streams[0],512*5,'y')


