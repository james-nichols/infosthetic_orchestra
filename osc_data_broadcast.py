#!/usr/bin/env python

"""
    Stream the 2nd column of a 2-column CSV file as OSCAre

    This uses the pyOSC package. On a Mac try running:

        sudo easy_install pip
        pip install pyOSC
"""

import argparse, time
import csv
import OSC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pdb
import datetime

#import pyaudio
import rtmidi


def open_file(file='some.csv'):
    out = []
    with open(file, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            out.append(row)
    return out

def midi_transform(data_1, data_2):
    """ A definable transform from data to midi notes """
  
    # Do the calc here
    midi_offset = 40.
    vel_off = 90.
    note = int(data_1 * (127. - midi_offset) + midi_offset + data_2 * 10.) 
    velocity = int(data_1 * (127. - vel_off) + vel_off)
    
    # Define the on and off note
    note_on = [0x90, note, velocity] # channel 1, middle C, velocity 112
    note_off = [0x80, note, 0] 
    
    return [note_on, note_off]

def send_osc_msgs(address, data, clients):
    
    # Single OSC message
    msg = OSC.OSCMessage()
    msg.setAddress(address) # set OSC address
    msg.append(data) # int
    for c in clients:
        c.send(msg) # send it!


def main():
    inputfile = ''
    parser = argparse.ArgumentParser(description='Stream two column CSV as OSC.')
    parser.add_argument('file_name',
                       help='CSV file to read and stream')
    parser.add_argument('-n', '--normalise', action='store_false',
                       help='The values will be normalised to a range between 0 and 1. Default=True')
    parser.add_argument('-log', '--logarithmic', action='store_true',
                       help='The stream_1 will be taken on a logarithmic scale. Default=False')
    parser.add_argument('-l', '--loop', action='store_false',
                       help='Loop the data if the end of the CSV file is reached. Default=True')
    parser.add_argument("-t", "--time", dest='time', type=float, default=240.,
                        help="Total time to run OSC broadcast, in seconds. Default=240")
    parser.add_argument("-p", "--path", default='data',
                       help="OSC message path e.g. 'jamdata' for a path of /jamdata. Default='data1'")
    parser.add_argument("--ip", default="127.0.0.1",
                       help="The ip of the OSC server. Default=127.0.0.1")
    parser.add_argument("--port", type=int, default=57120,
                       help="The port the OSC server is listening on. Default=57120 (SuperCollider's OSC port).")
    args = vars(parser.parse_args())
        
    # Parse the file
    csv = open_file(args['file_name'])

    times = map(float, zip(*csv)[0])
    stream_1 = map(float, zip(*csv)[1])
    stream_2 = map(float, zip(*csv)[2])

    if args['logarithmic']:
        stream_1 = [np.log(v+1.) for v in stream_1]
        stream_2 = [np.log(v+1.) for v in stream_2]

    if args['normalise']:
        high = max(stream_1)
        low = min(stream_1)
        stream_1 = [(v - low) / (high - low) for v in stream_1]
        high = max(stream_2)
        low = min(stream_2)
        stream_2 = [(v - low) / (high - low) for v in stream_2]

    interval = args['time'] / float(len(stream_1))
    print "Interval is ", interval, " seconds."
    
    # Parse the IPs
    #mc = OSC.OSCMultiClient()
    
    ips = []
    ip_file = open('ips.txt', 'r')
    ip_line = ip_file.readline()
    while ip_line:
        ips.append(ip_line[:-1])
        #send_address = (ip_line[:-1], args['port'])
        #mc.setOSCTarget(send_address)
        ip_line = ip_file.readline()

    # Initialise MIDI
    midiout = rtmidi.MidiOut()
    available_ports = midiout.get_ports()

    if available_ports:
        print available_ports
        midiout.open_port(0)
    else:
        midiout.open_virtual_port("My virtual output")
    
    note_on = [0x90, 60, 127] # channel 1, middle C, velocity 112
    note_off = [0x80, 60, 0]

    # Send the OSC message
    #send_address = (args['ip'], args['port'])
    #c = OSC.OSCClient()
    #c.connect( send_address ) # set the address for all following messages
    clients = []
    for ip in ips:
        send_address = (ip, args['port'])
        clients.append(OSC.OSCClient())
        print "connecting to", send_address
        clients[-1].connect(send_address)


    fig = plt.figure(figsize=(8,8))
    plt.xlim(0,len(stream_1))
    plt.ylim(0,1.0)
    line1, = plt.plot([],[],'r-')
    line2, = plt.plot([],[],'g-')

    #plt.show()

    while args['loop']:
        #for x in stream_1:
        for i in range(len(stream_1)):
            data_1 = stream_1[i]
            data_2 = stream_2[i]
            
            line1.set_data(range(i), stream_1[:i])
            line2.set_data(range(i), stream_2[:i])

            avg_1 = sum(stream_1[:i]) / float(i+1)
            avg_2 = sum(stream_2[:i]) / float(i+1)

            #line1.set_data(np.linspace(0., i * interval, interval), stream_1[:i])
            
            send_osc_msgs("/"+args['path']+"1", data_1, clients)
            send_osc_msgs("/"+args['path']+"2", data_1, clients)
            send_osc_msgs("/avg2", avg_1, clients)
            send_osc_msgs("/avg1", avg_2, clients)

            d = datetime.datetime.fromtimestamp(times[i])
            send_osc_msgs("/year", d.year, clients)
            send_osc_msgs("/month", d.month, clients)
            send_osc_msgs("/day", d.day, clients)
            

            # Single MIDI note
            if i%100 == 0:
                
                print "sending...\t", data_1, "\t", data_2, "\t avg:", avg_1, "\t", avg_2, "\t", d.day, d.month, d.year
                
                midiout.send_message(note_off)            
                [note_on, note_off] = midi_transform(data_1, data_2)
                midiout.send_message(note_on)

                #cv.set_data([data_1, data_2]) 

                # Print diagnostics
                #print "sent MIDI note %s with velocity %s"%(str(note), str(velocity))
            
            # Send CV via PyAudio

            time.sleep(interval)

    del midiout

if __name__ == "__main__":
    main()


# Initialise audio (which is in turn CV)
"""
cv = CVBuffer()


pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paFloat32,
        channels=2,
        rate=cv.sample_rate,
        output=True,
        frames_per_buffer=cv.buffer_length,
        stream_callback=cv.poll_buffer)
stream.start_stream()
"""

"""
class CVBuffer:

    def __init__(self, buffer_length=4096, interp_length=256, sample_rate=44100):

        self.buffer_length = buffer_length
        self.interp_length = interp_length
        self.sample_rate = sample_rate

        self.buf = np.zeros([self.buffer_length, 2])#, dtype='int16')
        self.int_buf = np.zeros([self.buffer_length, 2], dtype='int16')

        self.old_data = [0., 0.]
        self.new_data = [0., 0.]
        self.is_new_data = False

    def set_data(self, data):
        # It is assumed that the data comes in between 0.0 and 1.0

        self.old_data = self.new_data
        self.new_data = data
        self.is_new_data = True



    def poll_buffer(self, in_data, frame_count, time_info, status):
        
        if self.is_new_data:
            # Then do an interpolation
            for i in range(self.interp_length):
                self.buf[i,:] = (self.new_data - self.old_data) * float(i) / float(self.interp_length) + self.old_data
            self.buf[self.interp_length:, :] = self.new_data[:]
            self.int_buf = (65536. * self.buf - 32768.).astype(np.int16)
            self.is_new_data = False
        else:
            self.buf[:,:] = self.new_data

        return (self.buf.flatten().astype(np.float32), pyaudio.paContinue)
"""
