/*
	Playign around with the original Dark Sea Horns code. I figure since it was
	Using feedback i might try and replace this with a chaotic function

	Here I try different Chaos generators
*/

s = Server.default;
s.boot;
thisProcess.openPorts; // list all open ports

n = NetAddr.localAddr

m =	OSCFunc(
	{ |msg, time, addr, recvPort|
		var main_freq;
		//main_freq = 80;
		main_freq = 100 * msg[1].asFloat + 50;
		main_freq.postln;
		Ndef(\x).set(\freq, main_freq);
	}
	, '/data2'); // path

l =	OSCFunc(
	{ |msg, time, addr, recvPort|
		var main_freq = 80 * msg[1].asFloat + 40;
		main_freq.postln;
		2.0 * msg[1].asFloat.postln;
		Ndef(\x).set(\chaosAmp, 2.0 * msg[1].asFloat);
		Ndef(\x).set(\freq, main_freq);
	}
	, '/data1'); // path

j =	OSCFunc(
	{ |msg, time, addr, recvPort|
		msg.postln;
	}, '*'); // path

x = OSCFunc.trace(true);
x = OSCFunc.trace(false);

m.disable;
l.disable;
j.disable;
(
Ndef(\x).clear;
Ndef(\chaos, {
	// "...useful values from just below 1.0 to just above 2.0.
	// Towards 2.0 the sound crackles."
	Crackle.ar(SinOsc.ar(0.5, 0, 0.5, 1.5), 0.1)
});
Ndef(\x).ar(2);
Ndef(\x,{ | amp=1.0, freq=70, chaosAmp=1.0 |
	n = LFNoise1;
	a = SinOsc.ar(freq + n.ar(0.1!2,5),(Ndef(\x) +  Ndef(\chaos).ar) * chaosAmp * n.ar(0.1!2,3), n.ar(3,6)).tanh;
	9.do{
		a = AllpassL.ar(a,0.3,{0.2.rand+0.1}!2, 5)
	};
	a.tanh * amp;
}).play;


Ndef(\chaos).fadeTime = 1;
Ndef(\x).fadeTime = 1;
Ndef(\x).set(\freq, 70);
Ndef(\x).set(\chaosAmp, 1.0);

)

// By defining different chaos functions we get different effects as they feed into
// the playing Ndef(\x)
(
	Ndef(\chaos, {
		HenonC.ar(
			SampleRate.ir/8,
			LFNoise2.kr(1, 0.2, 1.2),
			LFNoise2.kr(1, 0.15, 0.15)
		) * 0.05
	});
)
(
	Ndef(\chaos, {
		LinCongC.ar(
			LFNoise2.kr(1, 1e4, 1e4),
			LFNoise2.kr(0.1, 0.5, 1.4),
			LFNoise2.kr(0.1, 0.1, 0.1),
			LFNoise2.kr(0.1)
		) * 0.5
	}).play;
)
(
	Ndef(\chaos, {
		LatoocarfianC.ar(
			SampleRate.ir/4,
			LFNoise2.kr(2,1.5,1.5),
			LFNoise2.kr(2,1.5,1.5),
			LFNoise2.kr(2,0.5,1.5),
			LFNoise2.kr(2,0.5,1.5)
		) * 0.5
	})
)
(
	Ndef(\chaos, { QuadC.ar(SampleRate.ir/4) * 0.2 })
)
