"use strict";
import { ButtonManager, LearningThrobber, NNErrorLogChart } from "./nnchart.js";
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
		learningRate: 0.003,

		// ======= Completion threshold =======
		completionThreshold: 0.001
	}
};

var theNN = null;
var theLogChart = null;
var theControlButtons = null;
var theThrobber = null;
var currentFrameCount = 0;
var gIterationCountDisplay = null;
var gAnimationActive = false;

window.launch = function() {
	resetNN();
	theControlButtons = new ButtonManager("control-button-container", onCommandButtonClick);
	theControlButtons.setDisabled("p", true);
};

function resetNN() {
	const cv = document.getElementById("cv");
	const nn = buildNetwork(TestNN);

	for (let i = 0;i < TestNN.Initialization.numOfSamples;++i) {
		nn.setClassificationTrainData(i, TRAIN_DATA_1[i].data, i);
	}

	theNN = nn.ready();
	setupCharts(document.getElementById("elog-container"), TestNN.Initialization.numOfSamples);
	renderNetwork(cv, theNN, window.devicePixelRatio);

	currentFrameCount = 0;
}

function setupCharts(containerElement, n) {
	containerElement.innerHTML = "";

	gIterationCountDisplay = document.createElement("p");
	gIterationCountDisplay.className = "iteration-counter";
	containerElement.appendChild(gIterationCountDisplay);
	updateIterationCounter(0);

	theLogChart = new NNErrorLogChart(100, 192, window.devicePixelRatio, n+1);
	containerElement.appendChild(theLogChart.canvas);
	theLogChart.render();

	theThrobber = new LearningThrobber();
	containerElement.appendChild(theThrobber.element);
}

function updateIterationCounter(i) {
	if (gIterationCountDisplay) {
		gIterationCountDisplay.innerHTML = "";
		gIterationCountDisplay.appendChild( document.createTextNode(`Iteration: ${i}`) );
	}
}

function enterFrame() {
	const cv = document.getElementById("cv");
	for (let i = 0;i < 29;++i) {
		theNN.advanceLearning();
	}
	const showSampleIndex = (currentFrameCount >> 3) % 10;
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

	updateIterationCounter(theNN.iterationCount);

	if (theNN.lastTotalError <= TestNN.Initialization.completionThreshold) {
		stopLearning(true);
	}

	++currentFrameCount;
	if (gAnimationActive) {
		requestAnimationFrame(enterFrame);
	}
}

function onCommandButtonClick(_manager, _button, command) {
	switch(command) {
		case 'r':
			resetNN();
			break;

		case 'x':
			if (!gAnimationActive) {
				theThrobber.showThinkingFrame();
				gAnimationActive = true;
				theControlButtons.setDisabled("x", true);
				theControlButtons.setDisabled("p", false);
				enterFrame();
			}
			break;
		case 'p':
			stopLearning();
			break;
	}
}

function stopLearning(complete) {
	gAnimationActive = false;
	theControlButtons.setDisabled("x", false);
	theControlButtons.setDisabled("p", true);

	if (complete) {
		theThrobber.nowComplete();
	} else {
		theThrobber.nowReady();
	}
}
