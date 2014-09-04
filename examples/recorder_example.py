#!/usr/bin/env python3

from lsltools import sim,rec
import time

# STEP 1: Initialize and start up three data streams
data1 = sim.EEGData(nch=3)
data2 = sim.RandomData(srate=128)
data3 = sim.LinearData(srate=64)
data1.start()
data2.start()
data3.start()

# STEP 2: Initialize recorder to save three streams of step 1 to a file
streams = rec.pylsl.resolve_streams()
recorder = rec.StreamRecorder('/tmp/example_recording.xdf',streams)

# STEP 3: Start recorder and let it run for 30s 
recorder.start_recording()
time.sleep(30)
recorder.end_recording()


data1.stop()
data2.stop()
data3.stop()
