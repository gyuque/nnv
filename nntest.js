"use strict";
import { ButtonManager, LearningThrobber, NNErrorLogChart, TabPager } from "./nnchart.js";
import { buildNetwork, renderNetwork } from "./nnv.js";

const TestNN = {
	LayersConfig: [
		{n: [16, 16], one: false},
		{n: 10, one: true},
		{n: 10}
	],

	Initialization: {
		// ======= Number of training data =======
		numOfSamples: 50,

		// ======= Weight initialization =======
		// M:  V = 2/M
		// MM: V = 2/(M1+M2)
		denominator: "M",
		randomFunc: "uniform",

		// ======= Learning rate =======
		learningRate: 0.02,

		// ======= Completion threshold =======
		completionThreshold: 0.001
	}
};

var theNN = null;
var theLogChart = null;
var theControlButtons = null;
var theControlTabPager = null;
var theThrobber = null;
var currentFrameCount = 0;
var gIterationCountDisplay = null;
var gAnimationActive = false;

window.launch = function() {
	resetNN();
	theControlButtons = new ButtonManager("control-button-container", onCommandButtonClick);
	theControlButtons.setDisabled("p", true);

	theControlTabPager = new TabPager("pages-container");
	theControlTabPager.selectByIndex(0);

	showTrainDataPreview("train-data-preview", TRAIN_DATA_1);
};

function resetNN() {
	pickParams();

	const cv = document.getElementById("cv");
	const nn = buildNetwork(TestNN);

	for (let i = 0;i < TestNN.Initialization.numOfSamples;++i) {
		nn.setClassificationTrainData(i, TRAIN_DATA_1[i].data, i % 10);
	}

	// Set output label
	nn.forEachNodeAtLayer(-1, (node, nodeIndex) => {  node.label = `${nodeIndex}`;  });

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
	for (let i = 0;i < 49;++i) {
		theNN.advanceLearning();
	}
	const showSampleIndex = Math.floor(currentFrameCount / 6) % TestNN.Initialization.numOfSamples;
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

	if (currentFrameCount === 200) {
		theThrobber.setPanic(1);
	}

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

function pickParams() {
	TestNN.Initialization.learningRate = pickNumericInput("param-lr");
	TestNN.Initialization.completionThreshold = pickNumericInput("param-cth");
	console.log("Set learningRate to ", TestNN.Initialization.learningRate);
	console.log("Set completionThreshold to ", TestNN.Initialization.completionThreshold);
}

function pickNumericInput(id) {
	const el = document.getElementById(id);
	return parseFloat(el.value);
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

function showTrainDataPreview(container_id, dataList) {
	const containerElement = document.getElementById(container_id);
	let count = 0;
	for (const name in dataList) if(dataList.hasOwnProperty(name)) {
		const smp = dataList[name];
		const cv = document.createElement("canvas");

		const w = smp.width;
		const h = smp.height; 

		cv.width = w;
		cv.height = h;

		const g = cv.getContext("2d");
		const idat = g.getImageData(0, 0, w, h);
		const pixels = idat.data;
		let pos = 0;
		for (let y = 0;y < h;++y) {
			for (let x = 0;x < w;++x) {
				const val = Math.floor(smp.data[pos] * 255);
				pixels[(pos<<2)  ] = val;
				pixels[(pos<<2)+1] = val;
				pixels[(pos<<2)+2] = val;
				pixels[(pos<<2)+3] = 255;

				++pos;
			}
		}

		g.putImageData(idat, 0, 0);

		containerElement.appendChild(cv);
		if ((count % 5) === 4) {
			containerElement.appendChild(document.createElement("br"));
		}

		++count;
	}
}