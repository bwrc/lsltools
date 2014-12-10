#!/usr/bin/env python3

# This file is part of the LSLTools package.
# Copyright 2014
#   Jari Torniainen <jari.torniainen@ttl.fi>,
#   Andreas Henelius <andreas.henelius@ttl.fi>
# Finnish Institute of Occupational Health
#
# This code is released under the MIT license
# http://opensource.org/licenses/mit-license.php
#
# Please see the file LICENSE for details

import time
import uuid
import struct
import threading
import numpy as np
from . import pylsl_python3 as pylsl
import xml.etree.cElementTree as ET


# -----------------------------------------------------------------------------
# XDF TOOLS
# -----------------------------------------------------------------------------
def check_magic_code(f):
    """ Asserts that there is a magic code at the beginning of file.

    Args:
        f: <_io.TextIOWrapper> file pointer of the xdf-file
    Returns:
        True/False: depedning of if the magic word was found
    """

    if not(f.read(4).decode("utf-8")=="XDF:"):
        return False
    else:
        return True

def convert_to_byte_size(byteStr):
    """ Converts MATLAB/C types to python types.

    Args:
        byteStr: <str> "MATLABish" type definition
    Returns:
        Python-style data type and byte length
    """
    lookup = {
    'uint8':    ['B',1],
    'uint16':   ['H',2],
    'uint32':   ['I',4],
    'uint64':   ['Q',8],
    'int8':     ['b',1],
    'int16':    ['h',2],
    'int32':    ['i',4],
    'int64':    ['q',8],
    'float32':  ['f',4],
    'double64': ['d',8],
    'string':   ['s',3]}
    return lookup[byteStr]

def read_var_len(f):
    """ Reads the chunk length from file.
    Args:
        f: <_io.TextIOWrapper> file pointer of the xdf-file
    Returns:
        length of the chunk (-1 if EOF or error)
    """
    val = f.read(1)
    if len(val)==0:
        print("Reading complete")
        return -1
    else:
        var_size = ord(val)
        if var_size==1:
            return ord(f.read(var_size))
        elif var_size==4:
            return struct.unpack('I',f.read(var_size))[0]
        elif var_size==8:
            return struct.unpack('Q',f.read(var_size))[0]
        else:
            print("erroor")
            return -1

def read_tag(f):
    """ Reads chunk tag from file.

    Args:
        f: <_io.TextIOWrapper> file pointer of the xdf-file
    Returns:
        tag: <int> chunk tag
    """
    return struct.unpack('H',f.read(2))[0]

def read_stream_id(f):
    """ Reads stream ID from file.

    Args:
        f: <_io.TextIOWrapper> file pointer of the xdf-file
    Returns:
        stream_id: <int> id of the stream
    """
    return struct.unpack('I',f.read(4))[0]

def parse_stream_header(stream_header_str):
    """ Parses the xml-formatted stream header string.

    Args:
        stream_header_str: <str> xml-formatted header string
    Returns:
        stream_name: <str>: name of the stream
        channel_count: <int>: number of channels in stream
        nominal_srate: <float>: stream sampling rate
        channel_format: <str>: format of the samples
    """
    hdr_xml = ET.fromstring(stream_header_str)
    stream_name = hdr_xml.find('name').text
    channel_count = int(hdr_xml.find('channel_count').text)
    nominal_srate = int(hdr_xml.find('nominal_srate').text)
    channel_format = convert_to_byte_size(hdr_xml.find('channel_format').text)
    print ("\t" + hdr_xml.find('name').text + "(" + str(channel_count) + ")")
    return stream_name, channel_count, nominal_srate, channel_format

def load_xdf(filename):
    """ Main function for loading xdf-files.

    Args:
        filename: <string> name + path to the xdf-file
    Returns:
        streams: <dict> contents of the xdf-file.
    """
    f = open(filename, 'rb')
    if not(check_magic_code(f)):
        return 0
    reading = True
    # Initialize a structure for the streams
    streams = dict()
    while True:
        varlen = read_var_len(f)
        if varlen == -1:
            return streams
        tag = read_tag(f)
        if tag == 1:      # File header
            file_header = f.read(varlen-2).decode("utf-8")
        elif tag == 2:    # Stream header
            stream_id = read_stream_id(f)
            print("\tstreamID=" + str(stream_id))
            stream_hdr_str = f.read(varlen-6).decode("utf-8")
            snm, chc, nsr, cf = parse_stream_header(stream_hdr_str)
            streams.update({stream_id: {'name': snm,
                                        'hdr': stream_hdr_str,
                                        'channels': chc,
                                        'srate': nsr,
                                        'fmt': cf,
                                        'data': [],
                                        'timess': [],
                                        'clock_timess': [],
                                        'clock_values': []}})
        elif tag == 3:    # Sample
            stream_id = read_stream_id(f)
            is_string = streams[stream_id]['fmt'][0] == "s"
            num_samples = read_var_len(f)
            # allocate space for timestamps and values
            if is_string:
                values = []
            else:
                values = np.zeros((streams[stream_id]['channels'], num_samples))
            timess = np.zeros((1, num_samples))
            for s in range(0, num_samples):
                has_timestamp = ord(f.read(1))
                if has_timestamp:
                    timess[0, s] = struct.unpack('d', f.read(8))[0]
                else:
                    timess[0, s] = -1  # this should come from nominal_srate
                if is_string:
                    new_string = ""
                    str_n = read_var_len(f)
                    for c in range(0, str_n):  # assuming 1-channel triggers!
                        new_string += f.read(1).decode("utf-8")
                    values.append(new_string)
                else:
                    for c in range(0, streams[stream_id]['channels']):
                        values[c,s] = struct.unpack(streams[stream_id]['fmt'][0],
                                        f.read(streams[stream_id]['fmt'][1]))[0]
            streams[stream_id]['data'].append(values)
            streams[stream_id]['timess'].append(timess)
        elif tag == 4:    # ClockOffset
            stream_id = read_stream_id(f)
            streams[stream_id]['clock_timess'].append(
                                                struct.unpack('d',f.read(8))[0])
            streams[stream_id]['clock_values'].append(
                                                struct.unpack('d',f.read(8))[0])
        elif tag == 5:
            pass
        elif tag == 6:    # StreamFooter
            stream_id = read_stream_id(f)
            footer_content = f.read(varlen-6).decode("utf-8")
            streams[stream_id]['ftr'] = footer_content
        else:           # Unknown tag
            print("Unkown tag encountered! " + str(tag))
            f.read(varlen-2)

def clock_offsets2string(x):
    """ Parses clock-offset vector into a human readable string.

    Args:
        x: <dict> stream contents
    Returns:
        clock_str: <str> string containing clock timetamps and offset values
    """
    clock_str = "clock_time,clock_offset\n"
    for a,b in zip(x['clock_timess'],x['clock_values']):
        clock_str+="%f,%f\n"%(a,b)
    return clock_str


def stream2csv(filename,x):
    """ Writes xdf-formatted stream into a csv-file.

    Args:
        filename: <str> name of the output file
        x: <dict> contents of a single stream
    """
    data = np.concatenate(x['data'],axis=1).T
    times = np.concatenate(x['timess'],axis=1).T
    hdr = x['hdr']
    if x['clock_timess']:
            hdr+=clock_offsets2string(x)

    np.savetxt(filename,np.concatenate([times,data],axis=1),
                delimiter=',',
                header=hdr)

def everything2csv(prefix,x):
    """ Write all streams in dict x into separate csv-files.

    Args:
        prefix: <str> prefix for all files (each file is named as prefix +
                      the name of the stream)
        x: <dict> contents of the xdf file
    """
    for stream_id in x.keys():
        stream2csv((prefix+x[stream_id]['name']+".csv"),x[stream_id])

# -----------------------------------------------------------------------------
# STREAMWRITER
# -----------------------------------------------------------------------------
class StreamWriter(threading.Thread):
    """ Object for writing a single stream into a specified file. """

    def __init__(self,f,lock,stream,stream_id):
        """ Initializes the streamWriter object.

        Args:
            f: <_io.TextIOWrapper> file pointer to the output file
            lock: <threading.Lock> lock for thread-safe file writing
            stream: <lsl_stream> pointer to the LSL-stream
            stream_id: <int> id of this stream in the output file
        """
        threading.Thread.__init__(self)

        # Open output file for writing (this will be changed)
        self.stream = stream
        self.f = f
        self.lock = lock
        # Prepare stream and inlet
        self.inlet = pylsl.StreamInlet(stream)
        self.stream_id = stream_id
        self.is_recording = False

        self.byte_table = {
                            'uint8':    ['B',1],
                            'uint16':   ['H',2],
                            'uint32':   ['I',4],
                            'uint64':   ['Q',8],
                            'int8':     ['b',1],
                            'int16':    ['h',2],
                            'int32':    ['i',4],
                            'int64':    ['q',8],
                            'float32':  ['f',4],
                            'double64': ['d',8],
                            'string':   ['s',3]}

        chunk_fmt = pylsl.fmt2string[self.stream.channel_format()]

        # Need to know if we are recording strings or numerics
        if chunk_fmt is 'string':
            self.is_strings = True
        else:
            self.is_strings = False

        # Assign some byte-sizes for easy access
        chunk_pack_info = self.byte_table[chunk_fmt]
        self.sample_pack_key = chunk_pack_info[0]
        self.timess_pack_key = 'd'
        self.sample_byte_size = chunk_pack_info[1]
        self.timess_byte_size = 8

        self.boundary_uuid = uuid.uuid4()

        # Initialize timing variables for pulling, clock offsets and boundaries.
        self.last_ofs = time.time()
        self.last_uuid = time.time()
        self.last_pull = time.time()

        self.rec_started = time.time()

        self.ofs_interval = 6
        self.uuid_interval = 10

        # Calculating pull interval from nominal_srate
        # it appears that the lsl stream has an internal buffer of 1024
        # which is the MAX AMOUNT OF SAMPLES PASSED AS A SINGLE CHUNK.
        # the internal buffer of a stream could potentially be a lot larger.
        # What we do instead is pull chunks at full-buffers.
        # If sampling rate is irregular we assume 500 Hz for pull

        self.srate = self.stream.nominal_srate()
        if self.srate:
            self.pull_interval = 1024.0/self.srate
        else:
            self.pull_interval = 1024.0/500.0

        # Min pull interval = 4s
        if self.pull_interval > 4.0:
            self.pull_interval = 4.0

    def timestamp(self):
        """ Utility function for printing timestamps to console. """
        return str("[%03d]{S%02d} "%
                                  (time.time()-self.rec_started,self.stream_id))

    def get_num_of_bytes(self,chunk_length):
        """This is a duplicate function for returning the amount of bytes
           required to encode number-of-samples i.e. the number indicating the
            length of the chunk contents.

        Args:
            chunk_length: <int> lenght of the chunk
        Returns:
            Chunk size in bytes and the packing "type"
        """
        size_in_bytes = float(chunk_length.bit_length())/8
        if size_in_bytes <=1:
            return 1,'B'
        elif size_in_bytes <=4:
            return 4,'I'
        else:
            return 8,'Q'

    def calculate_chunksize_string(self,chunk,timess):
        """ Calculates the size of string contents of a chunk and the
            corresponding timestamps.

        Args:
            chunk: <list> chunk contets
            timess: <list> vector of timestamps
        Returns:
            num_of_samples: <int> number of samples in chunk
            chunk_size: <int> chunk size in bytes
        """
        num_of_samples = len(chunk)
        nsb,nsbkey = self.get_num_of_bytes(num_of_samples)
        chunk_size = 2+4+1+nsb # tag,stream_id,nsb,nsbvalue
        # Find sizes of all strings in this chunk (channels*samples)
        str_sizes = []
        for k in chunk:
            str_sizes.extend([len(n) for n in k])
        # Calculate how many bytes are required to encode these string lengths
        enc_sizes = [self.get_num_of_bytes(i)[0] for i in str_sizes]
        chunk_size+= sum(str_sizes) + sum(enc_sizes) + len(str_sizes)

        if timess:
            chunk_size+=num_of_samples*9
        else:
            chunk_size+=num_of_samples

        return num_of_samples,chunk_size

    def calculate_chunksize_numeric(self,chunk,timess):
        """ Calculate and return the number of samples in a numeric chunk and
            chunk size in bytes.

        Args:
            chunk: <list> chunk contents
            timess: <list> vector of timestamps
        Returns:
            num_of_samples: <int> number of samples in chunk
            chunk_size: <int> chunk size in bytes
        """
        num_of_samples = len(chunk)
        chunk_size = (self.stream.channel_count()*num_of_samples*
                        self.sample_byte_size)
        nsb,nsbkey = self.get_num_of_bytes(num_of_samples)
        chunk_size+=nsb+6+1 # 6=tag+stream_id,1=nsb_size
        if timess:
            # Need extra space per sample for timestamp-flags and timestamps
            chunk_size+=num_of_samples*(self.timess_byte_size+1)
        else:
            # Need one byte of extra space per sample for timestamp-flags
            chunk_size+num_of_samples
        return num_of_samples,chunk_size

    def write_chunk_length(self,content_length):
        """ Calculates how many bytes are needed to encode content_length
            then writes both the size_in_bytes and content length to
            the file.

        Args:
            content_length: <int> lenght of the chunk content
        """
        size_in_bytes = float(content_length.bit_length())/8
        if size_in_bytes <=1:
            self.f.write(struct.pack('B',1))
            self.f.write(struct.pack('B',content_length))
        elif size_in_bytes <=4:
            self.f.write(struct.pack('B',4))
            self.f.write(struct.pack('I',content_length))
        else:
            self.f.write(struct.pack('B',8))
            self.f.write(struct.pack('Q',content_length))

    def write_tag(self,tag):
        """ Writes a specified tag to file.

        Args:
            tag: <int> chunk id tag
        """
        self.f.write(struct.pack('H',tag))

    def write_stream_id(self):
        """ Writes stream_id to file. """
        self.f.write(struct.pack('I',self.stream_id))

    def write_stream_header(self):
        """ Writes stream header (in xml) to file. """
        header_str = self.stream.as_xml()
        self.write_chunk_length(len(header_str)+6)
        self.write_tag(2)
        self.write_stream_id()
        #self.f.write(header_str.encode("utf-8"))
        self.f.write(header_str)

    def write_samples_string(self,chunk,timess):
        """ Writes a sample chunk of strings into the output file.

        Args:
            chunk: <list> list of samples
            timess: <list> vector of timestamps
        """
        # Calculate how many bytes we need"
        num_of_samples,chunk_size=self.calculate_chunksize_string(chunk,timess)
        self.write_chunk_length(chunk_size)
        self.write_tag(3)
        self.write_stream_id()
        self.write_chunk_length(num_of_samples)
        if timess:
            for s,t in zip(chunk,timess):
                self.f.write(struct.pack('B',8))
                self.f.write(struct.pack(self.timess_pack_key,t))
                # Write strings from all "channels"
                for v in s:
                    self.write_chunk_length(len(v))
                    self.f.write(v.encode("utf-8"))
        else:
            for s in chunk:
                self.f.write(struct.pack('B',0))
                for v in s:
                    self.write_chunk_length(len(v))
                    self.f.write(v.encode("utf-8"))

    def write_samples_numeric(self,chunk,timess):
        """ Writes a sample chunk of numeric data into the output file.

        Args:
            chunk: <list> list of samples
            timess: <list> vector of timestamps
        """
        num_of_samples,chunk_size=self.calculate_chunksize_numeric(chunk,timess)
        self.write_chunk_length(chunk_size)
        self.write_tag(3)
        self.write_stream_id()
        # Also need special case for strings
        self.write_chunk_length(num_of_samples)
        if timess:
            for s,t in zip(chunk,timess):
                self.f.write(struct.pack('B',8))
                self.f.write(struct.pack(self.timess_pack_key,t))
                self.f.write(struct.pack(
                         self.sample_pack_key*self.stream.channel_count(),*s))
        else:
            for s in chunk:
                self.f.write(struct.pack('B',0))
                self.f.write(struct.pack(
                         self.sample_pack_key*self.streams.channel_count(),*s))

    def write_clock_offset(self):
        """ Writes the current clock offset of the stream. """
        self.write_chunk_length(22)
        self.write_tag(4)
        self.write_stream_id()
        # current time
        time_now = pylsl.local_clock()
        # offset
        time_ofs = self.inlet.time_correction(timeout=1.0)
        self.f.write(struct.pack('d',time_now))
        self.f.write(struct.pack('d',time_ofs))

    def write_boundary(self):
        """ Writes an UUID boundary chunk. """
        self.write_chunk_length(18)
        self.write_tag(5)
        self.f.write(self.boundary_uuid.bytes_le)

    def write_stream_footer(self):
        """ Writes footer, placeholder for now. """
        footer_str = "<?xml version=\"1.0\"?><info></info>"
        self.write_chunk_length(len(footer_str)+6)
        self.write_tag(6)
        self.write_stream_id()
        self.f.write(footer_str.encode("utf-8"))

    def clear_inlet(self):
        """ Empties all the samples from this stream. """
        while self.inlet.pull_sample(timeout=0.0)[0]:()

    def terminate_stream(self):
        """ This might be a bad idea. """
        # BAD IDEA CONFIRMED, DO-NOT-USE-THIS
        # I think this causes segfaults by messing up some threading locks in
        # lslboost
        #self.inlet.__del__()

    def stop(self):
        """ Terminate the recording. Pull all remaining chunks from the buffer
            and shutdown the writer.
        """
        self.is_recording = False
        print(self.timestamp() + "!SHUTDOWN!")
        # We need to pull all the remaining samples here
        chunk,t = self.inlet.pull_chunk(timeout=0.0)
        chunks = []
        times  = []
        while chunk:
            chunks.extend(chunk)
            times.extend(t)
            chunk,t=self.inlet.pull_chunk(timeout=0.0)
        # Write the remaining samples
        self.lock.acquire()
        if self.is_strings:
            self.write_samples_string(chunks,times)
        else:
            self.write_samples_numeric(chunks,times)
        self.write_clock_offset()
        self.write_stream_footer()
        self.lock.release()
        self.terminate_stream()

    def run(self):
        """ Start recording. """
        self.is_recording = True
        self.rec_started = time.time()

        print(self.timestamp() + "Recording started")
        self.clear_inlet()

        # Need to navigate the lock here
        self.lock.acquire()
        self.write_stream_header()
        self.lock.release()

        # Main recording loop
        while self.is_recording:
            # Pull and write a chunk of samples
            if (time.time()-self.last_pull>self.pull_interval
                                       and self.is_recording):
                self.last_pull = time.time()
                chunk,t = self.inlet.pull_chunk(timeout=0.0)
                chunks = []
                times  = []
                while chunk:
                    chunks.extend(chunk)
                    times.extend(t)
                    chunk,t=self.inlet.pull_chunk(timeout=0.0)
                if chunks:
                    print(self.timestamp() + "Found %d samples"%len(chunks))
                    self.lock.acquire()
                    if self.is_strings:
                        self.write_samples_string(chunks,times)
                    else:
                        self.write_samples_numeric(chunks,times)
                    self.lock.release()
                else:
                    print(self.timestamp() + "No samples found")
            # Write clock OFS
            if (time.time()-self.last_ofs>self.ofs_interval
                                     and self.is_recording):
                print(self.timestamp() + "Writing clock offset")
                self.last_ofs = time.time()
                self.lock.acquire()
                self.write_clock_offset()
                self.lock.release()
            # Write a boundary block
            if (time.time()-self.last_uuid>self.uuid_interval
                                       and self.is_recording):
                print(self.timestamp() + "Writing UUID")
                self.last_uuid = time.time()
                self.lock.acquire()
                self.write_boundary()
                self.lock.release()

            time.sleep(0.1)
        print(self.timestamp() + "Recording finished, exit OK")


# -----------------------------------------------------------------------------
# STREAMRECORDER
# -----------------------------------------------------------------------------
class StreamRecorder():
    """ Object for writing multiple streams into a single file. """

    def __init__(self,filename,streams):
        """ Initializes the StreamRecorder-object for writing mutliple LSL
            streams into a file.

        Args:
            filename: <str> name of the output file
            streams: <list> list of streams to record
        """
        self.f = open(filename,"wb+")
        self.lock = threading.Lock()
        self.writers = dict()
        # Make a dict for the streams
        for n,s in enumerate(streams):
            self.writers.update({n:StreamWriter(self.f,self.lock,s,n)})
        # Lets take a copy of all the stream_ids used at this point
        # if we need to assign new one
        self.all_ids = self.writers.keys()

    def add_stream(self,stream,running=True):
        """ Add a new stream to the recording.

        Args:
            stream: <lsl_stream> pointer to the new stream
            running: <Boolean> flag indicating if recording of the new stream
                               is started immediately
        """
        new_id = max(self.all_ids)+1
        self.writers.update({new_id:StreamWriter(self.f,self.lock,
                                                 stream,new_id)})
        self.all_ids.append(new_id)
        if running:
            self.writers[new_id].start()

    def kill_stream(self,stream_id):
        """ Stop writing a stream and remove it from the writers list.

        Args:
            stream_id: <int> id of the stream to terminate
        """
        self.writers[stream_id].stop()
        del self.writers[stream_id]

    def end_recording(self):
        """ Kill all writers and close the file. """
        # Sends stop signal to all writers
        for sw in self.writers:
            self.writers[sw].stop()
        # wait for them all to finish writing
        writers_alive = [True]*len(self.writers)
        while any(writers_alive):
            for n,sw in enumerate(self.writers):
                writers_alive[n]=self.writers[sw].is_alive()
        for sw in self.writers:
            self.writers[sw].join()
        self.f.close()

    def write_magic(self):
        """ Writes the magic word. """
        self.f.write("XDF:".encode("utf-8"))

    def write_chunk_length(self,content_length):
        """ Calculates how many bytes are needed to encode content_length
            then writes both the size_in_bytes and content length to
            the file.

        Args:
            content_length: <int> lenght of the chunk content
        """
        size_in_bytes = float(content_length.bit_length())/8
        if size_in_bytes <=1:
            self.f.write(struct.pack('B',1))
            self.f.write(struct.pack('B',content_length))
        elif size_in_bytes <=4:
            self.f.write(struct.pack('B',4))
            self.f.write(struct.pack('I',content_length))
        else:
            self.f.write(struct.pack('B',8))
            self.f.write(struct.pack('Q',content_length))

    def write_tag(self,tag):
        """ Writes a specified tag to file.

        Args:
            tag: <int> chunk id tag
        """
        self.f.write(struct.pack('H',tag))

    def write_file_header(self):
        """ Writes file header in xml. Only version tag is required. """
        header_str = ("<?xml version=\"1.0\"?>"
                            "<info>"
                                "<version>1.0</version>"
                            "</info>")
        self.write_chunk_length(len(header_str)+2)
        self.write_tag(1)
        self.f.write(header_str.encode("utf-8"))

    def start_recording(self):
        """ Starts recording from all specified streams. """
        self.write_magic()
        self.write_file_header()
        for sw in self.writers:
            self.writers[sw].start()
