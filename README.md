# infosthetic_orchestra

A collection of scripts to facilitiate the infosthetic orchestra.

The main script is osc_data_broadcast.py. It has a few library dependencies:
 - OSC (https://trac.v2.nl/wiki/pyOSC)
 - pyaudio (https://people.csail.mit.edu/hubert/pyaudio/)
 - rtmidi (https://pypi.python.org/pypi/python-rtmidi/#downloads)

I can't remember how I installed them, but I think all but the OSC package are in pip.

To execute this you need to supply a data set. mtgoxAUD.csv is included - it's the history of bitcoin transactions on the mtgox exchange in Australia.
