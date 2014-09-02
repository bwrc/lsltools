#!/usr/bin/env python3

from lsltools import sim,vis


# STEP 1: Initialize a generator for simulated EEG and start it up.
eeg_data = sim.EEGData(nch=3,stream_name="example")
eeg_data.start()

# STEP 2: Find the stream started in step 1 and pass it to the vis.Grapher
streams = vis.pylsl.resolve_byprop("name","example")
eeg_graph = vis.Grapher(streams[0],512*5,'y')

# STEP 3: Enjoy the graph.
