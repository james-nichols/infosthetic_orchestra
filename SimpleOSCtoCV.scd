
s = Server.default;
s.boot;
thisProcess.openPorts; // list all open ports

n = NetAddr.localAddr

SynthDef("cv", {|value=0.0, out=0|
	Out.ar(out, SinOsc.ar(55*value+880));
},[0.01,0]).add;

SynthDef("cv", {|value=0.0, out=0|
	Out.ar(out, Line.ar(0.0, 0.0, 0.0, 1.0, value));
},[0.01,0]).add;

x = Synth("cv", [\out, 0]);
y = Synth("cv", [\out, 1]);

x.free;
y.free;
s.scope
(
l =	OSCFunc(
	{ |msg, time, addr, recvPort|
		x.set(\value,2.0*(msg[1].asFloat)-1.0);
		msg.postln;
	}
	, '/data1'); // path

m =	OSCFunc(
	{ |msg, time, addr, recvPort|
		y.set(\value,2.0*(msg[1].asFloat)-1.0);
		msg.postln;
	}
	, '/data2'); // path

)
m.disable;
l.disable;