"use strict";
import { buildNetwork } from "./nnv.js";

var gNNConf = null;
var theNN = null;

globalThis.onmessage = function(e) {
	if (e && e.data) {
		const edata = e.data;
		if (edata.conf) {
			console.log("[Worker] Received NN configuration");
			gNNConf = edata.conf;
			theNN = buildNetwork(gNNConf);
		} else if (edata.learn) {
			const chunkSize = edata.learn;
			advanceLearning(chunkSize);
		}
	}
};

function advanceLearning(chunkSize) {
	for (let i = 0;i < chunkSize;++i) {
		
	} 
}

// ------------------------------------
console.log("[Worker] ready.");
globalThis.postMessage({
	workerReady: true
});