"use strict";
import { buildNetwork, setTrainDataset } from "./nnv.js";

var gNNConf = null;
var theNN = null;
var gLearning = false;

globalThis.onmessage = function(e) {
	if (e && e.data) {
		const edata = e.data;
		if (edata.conf) {
			console.log("[Worker] Received NN configuration.");
			gNNConf = edata.conf;
			theNN = buildNetwork(gNNConf);
			setTrainDataset(theNN, edata.trainDataset);
			globalThis.postMessage({
				nnBuilt: true
			});
		} else if (edata.learn) {
			console.log("[Worker] Learning has started.");
			gLearning = true;
			advanceLearning();
		} else if (edata.stop) {
			gLearning = false;
			if (edata.sendBackParam) {
				globalThis.postMessage({
					stopped: edata.sendBackParam
				});
			}
		}
	}
};


function advanceLearning() {
	if (!gLearning) {
		console.log("[Worker] Stopped.");
		return false;
	}

	const nIters = 30;
	const st = performance.now();
	for (let i = 0;i < nIters;++i) {
		theNN.advanceLearning();		
	}
	const et = performance.now() - st;

	const lossList = makeLossList(theNN);
	globalThis.postMessage({
		learned: {
			params: theNN.exportInternalParams(),
			et: et,
			lastTotalError: theNN.lastTotalError,
			iterationCount: theNN.iterationCount,
			lossList: lossList,
			chunkIterations: nIters
		}
	});

	globalThis.setTimeout(advanceLearning, 1);
}

function makeLossList(nn) {
	const n = nn.numOfSamples;

	const resList = new Float32Array(n + 1); // each sample + total
	nn.forEachTrainDataSample( (sample, index) => {
		resList[index] = sample.lastErrorAmount;
	} );

	resList[n] = nn.lastTotalError;
	return resList;
}

// ------------------------------------
console.log("[Worker] ready.");
globalThis.postMessage({
	workerReady: true
});