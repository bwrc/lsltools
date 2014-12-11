#!/usr/bin/env python3

# This file is part of the LSLTools package.
# Copyright 2014
# Jari Torniainen <jari.torniainen@ttl.fi>,
# Andreas Henelius <andreas.henelius@ttl.fi>
# Finnish Institute of Occupational Health
#
# This code is released under the MIT license
# http://opensource.org/licenses/mit-license.php
#
# Please see the file LICENSE for details

import sys
import random
import threading
import numpy
from . import pylsl_python3 as pylsl
import nitime.algorithms.autoregressive as ar
import scipy.signal
import uuid
import subprocess


class RandomData(threading.Thread):
    """ Generates a stream which inputs randomly generated values into the LSL

        Generates a multichannel stream which inputs random values into the LSL.
        Generated random values follow gaussian distribution where mean and
        std values can be specified as arguments.
    """
    def __init__(self, stream_name="RANDOM", stream_type="RND", nch=3,
                 srate=128, mean=0, std=1, fmt='float32', nsamp=0):
        """ Initializes a data generator

        Args:
            stream_name: <string> name of the stream in LSL (default="RANDOM")
            stream_type: <string> type of the stream in LSL (default="RND")
            nch: <integer> number of channels (default=3)
            srate: <integer> sampling rate (default=128)
            mean: <float> mean value for the random values (default=0)
            std: <float> standard deviation for the random values (default=1)
            fmt: <string> sample data format (default='float32')
            nsamp: <integer> number of samples in total (0=inf)
        """
        threading.Thread.__init__(self)

        # Stream stuff
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.nch = nch
        self.srate = srate
        self.fmt = fmt
        self.uuid = str(uuid.uuid1())[0:4]

        # Synthetic data stuff, makes normally distributed noise for now
        self.data_mean = mean
        self.data_std = std

        # Setup timing related variables
        self.last_push = pylsl.local_clock()
        self.interval = 1.0 / float(self.srate)
        if nsamp == 0:
            self.nsamp = numpy.inf
        else:
            self.nsamp = nsamp

        self.running = True

        # Outlet
        self.outlet = pylsl.StreamOutlet(pylsl.StreamInfo(self.stream_name,
                                                          self.stream_type,
                                                          self.nch,
                                                          self.srate,
                                                          self.fmt,
                                                          self.uuid))

    def set_srate(self, srate):
        """ Changes the sampling rate of the stream.

        Args:
            srate: <integer> new sampling rate
        """
        self.srate = srate
        self.interval = 1.0 / float(self.srate)

    def set_mean(self, mean):
        """ Changes the mean of the random values.

        Args:
            mean: <float> new mean value for random samples
        """
        self.data_mean = mean

    def set_std(self, std):
        """ Changes the standard deviation of the random samples

        Args:
            std: <float> new standard deviation for random samples
        """
        self.data_std = std

    def push_sample(self):
        """ Pushes samples to LSL. """
        new_sample = []
        for n in range(0, self.nch):
            new_sample.append(numpy.random.normal(self.data_mean,
                                                  self.data_std))
        self.outlet.push_sample(new_sample)

    def stop(self):
        """ Stops streaming. """
        self.running = False

    def run(self):
        """ Loops for a specified time or forever. """
        current_sample = 0
        while current_sample < self.nsamp and self.running:
            if pylsl.local_clock() - self.last_push >= self.interval:
                self.last_push = pylsl.local_clock()
                current_sample += 1
                self.push_sample()


class LinearData(threading.Thread):
    def __init__(self, stream_name="LINEAR", stream_type="LIN", nch=3,
                 srate=128, max_val=1000, fmt='float32', nsamp=0):
        """ Initializes the linear data generator.

        Args:
            stream_name: <string> name of the stream in LSL (default="LINEAR")
            stream_type: <string> type of the stream in LSL (default="LIN")
            nch: <integer> number of channels (default=3)
            srate: <integer> sampling rate (default=128)
            max_val: <float> maximum value of the data (default=1000)
            fmt: <string> format of the samples (default='float32')
            nsamp: <integer> number of samples to stream (0=inf)
        """
        threading.Thread.__init__(self)
        # Stream stuff
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.nch = nch
        self.srate = srate
        self.fmt = fmt
        self.uuid = str(uuid.uuid1())[0:4]

        self.value = 0
        self.MAX_VAL = max_val
        self.running = True

        # Setup update intervals and total number of samples to be streamed
        self.last_push = pylsl.local_clock()
        self.interval = 1.0 / float(self.srate)
        if nsamp == 0:
            self.nsamp = numpy.inf
        else:
            self.nsamp = nsamp

        # Create the LSL outlet
        self.outlet = pylsl.StreamOutlet(pylsl.StreamInfo(self.stream_name,
                                                          self.stream_type,
                                                          self.nch,
                                                          self.srate,
                                                          self.fmt,
                                                          self.uuid))

    def set_srate(self, srate):
        """ Changes the sampling rate of the stream.

        Args:
            srate: <integer> new sampling rate
        """
        self.srate = srate
        self.interval = 1.0 / float(self.srate)

    def set_max_val(self, max_val):
        """ Changes the maximum value of the linear data.

        Args:
            max_val: <float> new maximum value for linear data
        """
        self.MAX_VAL = max_val

    def push_sample(self):
        """ Pushes linearly increasing value into the outlet. """
        new_sample = []
        for n in range(0, self.nch):
            new_sample.append(self.value)
        self.outlet.push_sample(new_sample)

    def stop(self):
        """ Stops pushing samples. """
        self.running = False

    def close_outlet(self):
        """ Close outlet (might cause an error). """
        self.outlet.__del__()

    def run(self):
        """ Loop for pushing samples. Loops for a set amount or forever. """
        current_sample = 0
        while current_sample < self.nsamp and self.running:
            if pylsl.local_clock() - self.last_push >= self.interval:
                self.last_push = pylsl.local_clock()
                current_sample += 1
                self.push_sample()
                self.value += 1
                if self.value > self.MAX_VAL:  # Reset value
                    self.value = 0


class MarkerData(threading.Thread):
    """ Creates a (string) marker LSL stream.

        Creates a stream which sends string formatted markers to the LSL
        at a constant rate. The stream can send just one type of marker or it
        can randomly select the marker from a list of markers.
    """
    def __init__(self, stream_name="MARKER", stream_type="MRK", srate=[1, 4],
                 markers=["ping", "pong"], nsamp=0):
        """ Initializes a marker generator

        Args:
            stream_name: <string> name of the stream in LSL (default='MARKER')
            stream_type: <string> type of the stream in LSL (default='MRK')
            srate: [int, int] sampling rate of the stream (default=[1,4])
            markers: <list> list of (string) markers (default=["ping","pong"])
            nsamp: number of samples to stream (0=inf)
        """
        threading.Thread.__init__(self)
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.srate = srate
        self.markers = markers
        self.uuid = str(uuid.uuid1())[0:4]

        self.interval = 1.0 / float(random.randint(self.srate[0],
                                                   self.srate[1]))
        self.last_push = pylsl.local_clock()
        self.running = True

        if nsamp == 0:
            self.nsamp = numpy.inf
        else:
            self.nsamp = nsamp

        self.outlet = pylsl.StreamOutlet(pylsl.StreamInfo(self.stream_name,
                                                          self.stream_type,
                                                          1,
                                                          self.srate[0],
                                                          'string',
                                                          self.uuid))

    def set_srate(self, srate):
        """ Changes the sampling rate of the stream.

        Args:
            srate: [int, int] new sampling rate range
        """
        self.srate = srate
        self.interval = 1.0 / float(random.randint(self.srate[0],
                                                   self.srate[1]))
        self.last_push = pylsl.local_clock()

    def set_markers(self, markers, append=False):
        """ Changes the markers or adds new ones.

        Args:
            markers: <list> list of markers
            append: <Boolean> True to append new markers, False to replace
        """
        if append:
            self.markers + markers
        else:
            self.markers = markers

    def push_sample(self):
        """ Pushes a randomly selected marker to the outlet. """
        self.outlet.push_sample([random.choice(self.markers)])

    def stop(self):
        """ Stops looping. """
        self.running = False

    def run(self):
        """ Main loop for pushing samples. """
        current_sample = 0
        while current_sample < self.nsamp and self.running:
            if pylsl.local_clock() - self.last_push >= self.interval:
                self.last_push = pylsl.local_clock()
                self.interval = 1.0/float(random.randint(self.srate[0],
                                                         self.srate[1]))
                current_sample += 1
                self.push_sample()


class ECGData(threading.Thread):
    """ Generates and feeds synthetic ECG in to LSL.

        Uses the C implementation ECGSYN to generate simulated ECG
        (http://www.physionet.org/physiotools/ecgsyn/). Needs a path to
        ECGSYN as the first argument.
    """
    def __init__(self, ecgsynpath, O="/tmp/dump.dat", n=256, s=256, S=256, a=0,
                 h=60, H=1, f=0.1, F=0.25, v=0.01, V=0.01, q=0.5, R=1,
                 stream_name="ECGSYN"):
        """ Initializes the synthetic ECG data  generator.

        Args:
            ecgsynpath: <string> path to ecgsyn
            O: <string> path to temporary file (default "/tmp/dump.dat")
            n: <integer> total number of heart beats per block (default=256)
            s: <integer> ECG sampling rate (default=256)
            S: <integer> internal sampling rate (default=256)
            a: <float> amplitude of additive uniform noise (default=0)
            h: <float> heart rate mean (default=60)
            H: <float> heart rate std (default=1)
            f: <float> low frequency (default=0.1)
            F: <float> high frequency (default=0.25)
            v: <float> low frequency standard deviation (default=0.01)
            V: <float> high frequency standard deviation (default=0.01)
            q: <float> LF/HF ratio (default=0.5)
            R: <float> seed (default=1)
            stream_name: <string> stream name
        """
        threading.Thread.__init__(self)

        self.path = ecgsynpath
        self.uuid = str(uuid.uuid1())[0:4]
        self.srate = s
        self.buffer_warning = 10 * self.srate  # this might break

        # Timing
        self.interval = 1.0 / float(self.srate)
        self.last_push = pylsl.local_clock()

        # Same parameters as ecgsyn_c, read their documentation for more
        # detailed information

        self.O = str(O)  # Datadumping file
        self.n = str(n)  # Number of heartbeats
        self.s = str(s)  # ECG sampling rate
        self.S = str(S)  # Internal sampling rate
        self.a = str(a)  # Amplitude of additive uniform noise
        self.h = str(h)  # Heart rate mean
        self.H = str(H)  # Heart rate std
        self.f = str(f)  # Low frequency
        self.F = str(F)  # High frequency
        self.v = str(v)  # Low frequency standard deviation
        self.V = str(V)  # High frequency standard deviation
        self.q = str(q)  # LF/HF ratio
        self.R = str(R)  # Seed

        # Setup LSL
        info = pylsl.StreamInfo(stream_name, 'ECG', 1, self.srate,
                                'float32', self.uuid)
        self.outlet = pylsl.StreamOutlet(info)

        # Need a thread lock
        self.lock = threading.Lock()

        # Generate some of this to get this going
        self.data = numpy.empty(0)
        self.generate_data(False)
        self.running = True

    def generate_data(self, append_flag):
        """ Generates more data and appends (maybe) it to buffer.

        Args:
            append_flag: <Boolean> does new data append to or replace current
        """
        # generate more data to a tmpfile by calling ecgsyn_c with subprocess
        subprocess.call([self.path, "-O", self.O, "-n", self.n,
                         "-s", self.s, "-S", self.S, "-a", self.a,
                         "-h", self.h, "-H", self.H, "-f", self.f,
                         "-F", self.F, "-v", self.v, "-V", self.V,
                         "-q", self.q, "-R", self.R])
        # read data from tmpfile
        d = numpy.genfromtxt(self.O, delimiter=" ")
        # and add it to the buffer
        self.lock.acquire()
        if append_flag:
            self.data = numpy.append(self.data, d, axis=0)
        else:
            self.data = d
        self.lock.release()

    def set_n(self, n):
        """ Changes the number of heart beats generated per block.

        Args:
            n: <integer> number of heart beats per block of data
        """
        self.n = str(n)

    def set_s(self, s):
        """ Changes the ECG sampling rate.

        Args:
            s: <integer> new sampling rate
        """
        self.s = str(s)
        self.interval = 1.0 / float(s)

    def set_S(self, S):
        """ Changes the internal sampling rate.

        Args:
            S: <integer> new internal sampling rate
        """
        self.S = str(S)

    def set_a(self, a):
        """ Changes the amplitude of the additive (uniform) noise.

        Args:
            a: <float> new amplitude of the additive noise
        """
        self.a = str(a)

    def set_h(self, h):
        """ Changes the mean heart rate.

        Args:
            h: <float> new mean heart rate
        """
        self.h = str(h)

    def set_H(self, H):
        """ Changes the standard deviation of the heart rate.

        Args:
            H: <float> new heart rate standard deviation
        """
        self.H = str(H)

    def set_f(self, f):
        """ Changes low frequency.

        Args:
            f: <float> new low frequency
        """
        self.f = str(f)

    def set_F(self, F):
        """ Changes high frequency.

        Args:
            F: <float> new high frequency
        """
        self.F = str(F)

    def set_v(self, v):
        """ Changes low frequency standard deviation.

        Args:
            v: <float> new low frequency standard deviation
        """
        self.v = str(v)

    def set_V(self, V):
        """ Changes high frequency standard deviation.

        Args:
            V: <float> new high frequency standard deviation
        """
        self.V = str(V)

    def set_q(self, q):
        """ Changes LF/HF ratio

        Args:
            q: <float> new LF/HF ratio
        """
        self.q = str(q)

    def set_r(self, R):
        """ Changes seed.

        Args:
        R: <float> new seed
        """
        self.R = str(R)

    def reset(self):
        """ Resets data streaming. Replaces current buffer with new data. """
        r = threading.Thread(target=self.generate_data, args=[False])
        r.start()
        r.join()

    def stop(self):
        """ Stops streaming. """
        self.running = False

    def run(self):
        """ Streaming loops. Pushes samples and requests data when needed. """
        while self.running:
            # push sample
            if pylsl.local_clock() - self.last_push >= self.interval:
                self.lock.acquire()
                self.last_push = pylsl.local_clock()
                self.outlet.push_sample([float(self.data[0, 1])])
                self.data = numpy.delete(self.data, (0), axis=0)
                self.lock.release()
            # get more data
            if self.data.shape[0] < self.buffer_warning:
                t = threading.Thread(target=self.generate_data, args=[True])
                t.start()
                t.join()


class EEGData(threading.Thread):
    """ Generates and feeds synthetic EEG into LSL.

        Generates random EEG by driving white noise through an all-pole IIR
        filter. The filter coefficients are defined by autoregression estimate
        of the EEG signal. AR coefficients can be estimated from offline EEG
        data or coefficients can be directly passed to the object.
    """
    def __init__(self, data=False, ar_coefs=False, p=20, p2p=80, nch=1,
                 srate=512, stream_name='EEGSYN'):
        """ Initialize the EEG-streamer object.

        Args:
            data: <array_like> EEG data used for estimating AR coefficients
            ar_coefs: <array_like> AR coefficients
            p: <integer> model order for the AR estimation (default p=20)
            p2p: <float> peak-to-peak amplitude of simulated EEG (default=80)
            nch: <integer> number of channels
            srate: <float> sampling rate of the data
            stream_name: <string> name of the stream in LSL (default='EEGSYN')
        """
        threading.Thread.__init__(self)

        if data:
            # Calculate AR model from input data
            data = data - numpy.mean(data)
            self.A = ar.AR_est_YW(data, p)
            self.A *= -1
            self.A = numpy.insert(self.A, 0, 1)
            self.srate = srate
            self.p2p = numpy.std(data)

        elif ar_coefs:
            # Use the provided AR coefficients
            self.A = ar_coefs
            self.srate = srate

        else:
            # Use default AR coefficients
            self.A = numpy.array((1, -2.45, 1.7625, 0.0116, -0.3228))
            self.srate = 512

        self.nch = nch
        self.p2p = p2p  # peak-to-peak amplitude
        self.osc = dict()
        self.interval = 1.0/float(self.srate)
        self.last_push = pylsl.local_clock()
        self.lock = threading.Lock()
        self.buffer_size = self.srate*10
        self.buffer_warning = self.buffer_size/2
        self.uuid = str(uuid.uuid1())[0:4]

        self.running = True

        # Setup LSL
        info = pylsl.StreamInfo(stream_name, 'EEG',
                                self.nch, self.srate, 'float32', self.uuid)

        self.outlet = pylsl.StreamOutlet(info)
        self.data = numpy.empty(0)
        self.generate_data(False)

    def generate_data(self, append_flag):
        """ Add new data to the streaming buffer.

        Args:
            append_flag: <Boolean> does new data append to or replace current
        """
        noise = numpy.random.normal(0, 1, (self.buffer_size, self.nch))
        new_data = scipy.signal.lfilter((1, 0), self.A, noise, axis=0)
        new_data *= self.p2p / numpy.std(new_data)

        # Add oscillations (if there are any)
        if self.osc:
            for o in self.osc:
                s = self.generate_oscillation(self.osc[o])
                s *= ((self.osc[o][-1] * numpy.max(abs(new_data))) /
                      numpy.max(abs(s)))
                new_data += s

        self.lock.acquire()
        if append_flag:
            self.data = numpy.append(self.data, new_data, axis=0)
        else:
            self.data = new_data
        self.lock.release()

    def generate_oscillation(self, osc):
        """ Generate oscillations for the signal.

        Args:
            osc: <list> list containing oscillation parameters
        """
        w0, w1, wb, coef = osc
        w0 = float(w0) / (float(self.srate) / 2.0)
        w1 = float(w1) / (float(self.srate) / 2.0)
        wp = [w0, w1]  # Passband

        wb0 = float(w0-wb) / (float(self.srate) / 2.0)
        wb1 = float(w1+wb) / (float(self.srate) / 2.0)
        ws = [wb0, wb1]  # Stop band

        N, wn = scipy.signal.cheb1ord(wp, ws, 0.1, 30)
        b, a = scipy.signal.cheby1(N, 0.5, wn, btype='bandpass', output='ba')
        s = numpy.random.normal(0, 1, (self.buffer_size, self.nch))
        s = scipy.signal.lfilter(b, a, s, axis=0) * coef
        return s

    def add_oscillation(self, name, f_low, f_high, f_bound, ampl):
        """ Adds a new oscillation.

        Args:
            name: <string> name of the stream
            f_low: <float> lower boundary of the frequency band (in Hz)
            f_high: <float> upper boundary of the frequency band (in Hz)
            f_bound: <float> length of the transition band (in Hz)
            ampl: <float> relative amplitude of the oscillation
        """
        # should sanitize inputs here
        if name in self.osc:
            self.osc[name] = [f_low, f_high, f_bound, ampl]
        else:
            self.osc.update({name: [f_low, f_high, f_bound, ampl]})

    def remove_oscillation(self, name):
        """ Removes an oscillation from the list.

        Args:
            name: <string> name of the oscillation to be removed
        """
        if name in self.osc:
            del self.osc[name]
        else:
            print("Invalid oscillation name!")

    def list_oscillations(self):
        """ Lists all the current oscillations. """
        print("Oscillation list:")
        for osc in self.osc.keys():
            print(osc)

    def set_p2p(self, p2p):
        """ Change peak-to-peak amplitude of the EEG signal.

        Args:
            p2p: <float> new peak-to-peak amplitude
        """
        self.p2p = p2p

    def set_AR(self, AR, srate=0):
        """ Change autoregression coefficients of the signal.

        Args:
            AR: <array_like> array of new AR coefficients
            srate: <float> sampling rate corresponding to the new AR model
        """
        self.A = AR
        if srate:
            self.srate = srate

    def reset(self):
        """ Reset data streaming. Replaces current buffer with new data. """
        r = threading.Thread(target=self.generate_data, args=[False])
        r.start()
        r.join()

    def stop(self):
        """ Stops streaming. """
        self.running = False

    def run(self):
        """ Streaming loop. Pushes samples and generates new data as needed. """
        while self.running:
            # check if we need to push a new sample
            if pylsl.local_clock() - self.last_push >= self.interval:
                self.lock.acquire()
                self.last_push = pylsl.local_clock()
                self.outlet.push_sample(list(self.data[0, :]))
                self.data = numpy.delete(self.data, 0, axis=0)
                self.lock.release()

            # check if we need to generate more data
            if self.data.shape[0] < self.buffer_warning:
                t = threading.Thread(target=self.generate_data, args=[True])
                t.start()
                t.join()


if __name__ == '__main__':
    """ Give arguments, receive streams. Parameters not supported, will run with
        defaults.

    Args:
        'Random':   Starts a random stream
        'Linear':   Starts a linear stream
        'Marker':   Starts a marker stream
        'ECG':      Starts an ECG stream
        'EEG':      Starts an EEG stream

    """

    if len(sys.argv) > 1:

        streamers = []

        for arg in sys.argv[1:]:

            datatype = arg.lower()
            if datatype == "random":
                streamers.append(RandomData())
            elif datatype == "linear":
                streamers.append(LinearData())
            elif datatype == "marker":
                streamers.append(MarkerData())
            elif datatype == "ecg":
                path_to_ecgsyn = input("input path to ecgsyn: ")
                streamers.append(ECGData(path_to_ecgsyn))
            elif datatype == "eeg":
                streamers.append(EEGData(nch=4))
            else:
                print("unknown data type!")

        if streamers:
            print("Starting stream(s)")
            for s in streamers:
                s.start()
            while input('enter q to quit ') != 'q':
                pass
            print("Shutting down...")
            for s in streamers:
                s.stop()
            print("Finished.")

    else:
        print("Provide at least 1  argument:",
              "RANDOM,LINEAR,MARKER,ECG OR EEG!")
