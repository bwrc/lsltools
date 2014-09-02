#!/usr/bin/env python3

# This file is part of the LSLTools package.
# Copyright 2014
# Jari Torniainen <jari.torniainen@ttl.fi>, Andreas Henelius <andreas.henelius@ttl.fi>
# Finnish Institute of Occupational Health
# 
# This code is released under the MIT license
# http://opensource.org/licenses/mit-license.php
#
# Please see the file LICENSE for details

import pyqtgraph as pg
import numpy as np
from . import pylsl_python3 as pylsl
import sys
from PyQt4 import QtGui 
import os


class Grapher():
    """ Grapher object for an LSL stream. 
        
        Grapher object visualizes the input LSL stream using pyqtgraph. 
        Graph title is derived from StreamInfo. Buffer size of the displayed
        data and color can be defined when initializing the grapher object.
    """
    def __init__(self,stream,buffer_size,col="w"):
        """ Initializes the grapher.
        
        Args:
            stream: <pylsl.StreamInfo instance> pointer to a stream
            buffer_size: <integer> visualization buffer length in samples
            col: <char> color of the line plot (b,r,g,c,m,y,k and w)
        """

        self.stream = stream
        self.inlet = pylsl.StreamInlet(stream)

        self.buffer_size = buffer_size
        self.channel_count = self.inlet.channel_count 
        self.gbuffer = np.zeros(self.buffer_size*self.channel_count)
        self.gtimes = np.zeros(self.buffer_size)+pylsl.local_clock()
        self.col = col
        self.fill_buffer()
        self.start_graph()

    def fill_buffer(self):
        """ Fill buffer before starting the grapher. """
        num_of_smp = 0
        while num_of_smp < self.buffer_size:
            c,t = self.inlet.pull_chunk(timeout=0.0)
            new_c = []
            new_t = []
            while c:
                new_c+=c
                new_t+=t
                c,t=self.inlet.pull_chunk(timeout=0.0)
        
            # add samples to buffer
            if any(new_c):
                # add samples
                num_of_smp += len(new_c)
                data_v = [item for sublist in new_c for item in sublist]
                self.gbuffer = np.roll(self.gbuffer,-len(data_v))
                self.gbuffer[-len(data_v):]=data_v
                # add timestamps
                if new_t:
                    self.gtimes = np.roll(self.gtimes,-len(new_t))
                    self.gtimes[-len(new_t):]=new_t


    def update(self):
        """ Updates the buffer and plot if there are new chunks available. """
        # pull all available chunks
        c,t = self.inlet.pull_chunk(timeout=0.0)
        new_c = []
        new_t = []
        while c:
            new_c+=c
            new_t+=t
            c,t=self.inlet.pull_chunk(timeout=0.0)
        
        # add samples to buffer
        if any(new_c):
            # add samples
            data_v = [item for sublist in new_c for item in sublist]
            self.gbuffer = np.roll(self.gbuffer,-len(data_v))
            self.gbuffer[-len(data_v):]=data_v
            # add timestamps
            if new_t:
                self.gtimes = np.roll(self.gtimes,-len(new_t))
                self.gtimes[-len(new_t):]=new_t

        # update graph handles
        if self.gbuffer.any():
            for k in range(0,self.channel_count):
                self.handles[k].setData(self.gtimes,
                                        self.gbuffer[k::self.channel_count])
    
    def start_graph(self):
        """ Starts graphing. """
        # setup plot title and initialize plots+handles
        title_str = "%s(%s)"%(self.stream.name(),self.stream.type())
        self.win = pg.GraphicsWindow(title=title_str)
        self.plots = []
        self.handles = []

        # add each channel as a (vertical) subplot
        for k in range(0,self.channel_count):
            self.plots.append(self.win.addPlot(title="ch"+str(k)))
            self.handles.append(self.plots[k].plot(pen=self.col))
            if k<self.channel_count-1:
                self.plots[k].showAxis('bottom',show=False)
                self.win.nextRow()
        
        # its go time
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(15) # heuristically derived sleepytime

        # this stuff is for insuring a clean exit
        QtGui.QApplication.instance().exec_()
        os._exit(0)

if __name__ == "__main__":
   
    if len(sys.argv)<2 or len(sys.argv)>4:
        print("Takes between 1 and 3 arguments")
    else:
    
        stream = pylsl.resolve_byprop('name',sys.argv[1],timeout=5)

        if len(sys.argv)>2:
            buffer_size = int(sys.argv[2])
        else:
            buffer_size = 512

        if len(sys.argv)>3:
            plot_color = sys.argv[3]
        else:
            plot_color = 'w'

        if stream:
            g = Grapher(stream[0],buffer_size,plot_color)
        else:
            print("Stream %s not found"%sys.argv[1])
