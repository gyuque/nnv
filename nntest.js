"use strict";
import { NNErrorLogChart } from "./nnchart.js";
import { buildNetwork, renderNetwork } from "./nnv.js";

const TestNN = {
	LayersConfig: [
		{n: [16, 16], one: false},
		{n: 10, one: false},
		{n: 10}
	],

	Initialization: {
		// ======= Number of training data =======
		numOfSamples: 10,

		// ======= Weight initialization =======
		// M:  V = 2/M
		// MM: V = 2/(M1+M2)
		denominator: "M",
		randomFunc: "uniform",

		// ======= Learning rate =======
		learningRate: 0.003
	}
};

var theNN = null;
var theLogChart = null;
var restFrameCount = 318;

window.launch = function() {
	const cv = document.getElementById("cv");
	const nn = buildNetwork(TestNN);

	for (let i = 0;i < TestNN.Initialization.numOfSamples;++i) {
		nn.setClassificationTrainData(i, TRAIN_DATA_1[i].data, i);
	}

	theNN = nn.ready();
	setupCharts(document.getElementById("elog-container"), TestNN.Initialization.numOfSamples);

	renderNetwork(cv, theNN, window.devicePixelRatio);
	setTimeout(enterFrame, 300);
};

function setupCharts(containerElement, n) {
	theLogChart = new NNErrorLogChart(100, 192, window.devicePixelRatio, n+1);
	containerElement.appendChild(theLogChart.canvas);
	theLogChart.render();
}

function enterFrame() {
	const cv = document.getElementById("cv");
	for (let i = 0;i < 29;++i) {
		theNN.advanceLearning();
	}
	const showSampleIndex = 9-((restFrameCount >> 3) % 10);
	theNN.advanceLearning( (nn, sampleIndex) => {
		if (sampleIndex === showSampleIndex) {
			nn.doForwardPropagation();
			renderNetwork(cv, nn, window.devicePixelRatio);
		}
	} );

	theNN.forEachTrainDataSample( (sample, s_index) => {
		theLogChart.pushValue(s_index, sample.lastErrorAmount);
	} );
	theLogChart.pushValue(theNN.numOfSamples, theNN.lastTotalError);
	theLogChart.render();

	if (--restFrameCount > 0) {
		requestAnimationFrame(enterFrame);
	}
}
