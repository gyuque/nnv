"use strict";
import { setupInputArea } from "./datainput.js";
import { ButtonManager, LearningThrobber, NNErrorLogChart, TabPager } from "./nnchart.js";
import { buildNetwork, NNAUG_2DTRANS, NNAUG_NONE, NNMODE_LEARN, NNMODE_USE, renderNetwork } from "./nnv.js";

const TestNN = {
	LayersConfig: [
		{n: [16, 16], one: false},
		{n: 10, one: true},
		{n: 10}
	],

	Initialization: {
		// ======= Information of training data =======
		numOfSamples: 50,
		augmentation: NNAUG_NONE,

		// ======= Weight initialization =======
		// M:  V = 2/M
		// MM: V = 2/(M1+M2)
		denominator: "M",
		randomFunc: "uniform",

		// ======= Learning rate =======
		learningRate: 0.04,
		momentum: 0,

		// ======= Completion threshold =======
		completionThreshold: 0.003
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

const canvasSet = [null, null];

window.launch = function() {
	canvasSet[0] = document.getElementById("back-cv");
	canvasSet[1] = document.getElementById("cv");

	resetNN();
	theControlButtons = new ButtonManager("control-button-container", onCommandButtonClick);
	theControlButtons.setDisabled("p", true);

	theControlTabPager = new TabPager("pages-container");
	theControlTabPager.selectByIndex(0);

	showTrainDataPreview("data-preview-items", TRAIN_DATA_1);

	setupInputArea("data-input-pane", onExecuteInferenceClick);
};

function resetNN() {
	pickParams();

	const nn = buildNetwork(TestNN);

	for (let i = 0;i < TestNN.Initialization.numOfSamples;++i) {
		const src = TRAIN_DATA_1[i];
		const samp = nn.setClassificationTrainData(i, src.data, i % 10);
		samp.set2DMetrics(src.width, src.height);
		samp.augMode = TestNN.Initialization.augmentation;
	}

	// Set output label
	nn.forEachNodeAtLayer(-1, (node, nodeIndex) => {  node.label = `${nodeIndex}`;  });

	theNN = nn.ready();
	setupCharts(document.getElementById("elog-container"), TestNN.Initialization.numOfSamples);
	renderNetwork(canvasSet, theNN, window.devicePixelRatio);

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
	for (let i = 0;i < 24;++i) {
		theNN.advanceLearning();
	}
	const showSampleIndex = Math.floor(currentFrameCount / 6) % TestNN.Initialization.numOfSamples;
	theNN.advanceLearning( (nn, sampleIndex) => {
		if (sampleIndex === showSampleIndex) {
			nn.doForwardPropagation();
			renderNetwork(canvasSet, nn, window.devicePixelRatio);
		}
	} );

	theNN.forEachTrainDataSample( (sample, s_index) => {
		theLogChart.pushValue(s_index, sample.lastErrorAmount);
	} );
	theLogChart.pushValue(theNN.numOfSamples, theNN.lastTotalError);
	theLogChart.render();

	updateIterationCounter(theNN.iterationCount);

	const cycle = (currentFrameCount % 800);
	if (cycle === 395) {
		theThrobber.setPanic(1, 0);
	} else if (cycle === 795) {
		theThrobber.setPanic(1, 1);
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
	theNN.setMode(NNMODE_LEARN);

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
	TestNN.Initialization.momentum = pickNumericInput("param-mom");
	TestNN.Initialization.completionThreshold = pickNumericInput("param-cth");
	const aug = augmentModeFromInput();
	TestNN.Initialization.augmentation = aug;
	console.log("Set learningRate to ", TestNN.Initialization.learningRate);
	console.log("Set momentum to ", TestNN.Initialization.momentum);
	console.log("Set completionThreshold to ", TestNN.Initialization.completionThreshold);
	console.log("Set augmentation mode to ", (aug === NNAUG_2DTRANS) ? "[2D translation]" : "[none]");
	updateDirtyFlag();
}

function augmentModeFromInput() {
	return pickCheckboxInput("data-aug-checkbox") ? NNAUG_2DTRANS : NNAUG_NONE;
}

function pickNumericInput(id) {
	const el = document.getElementById(id);
	return parseFloat(el.value);
}

function pickCheckboxInput(id) {
	const el = document.getElementById(id);
	return el.checked;
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
		if ((count % 10) === 9) {
			containerElement.appendChild(document.createElement("br"));
		}

		++count;
	}
}

function onExecuteInferenceClick(inputData) {
	const inLayerNodes = theNN.getNodesAtLayer(0);
	const n = inLayerNodes.length;
	for (let i = 0;i < n;++i) {
		inLayerNodes[i].outValue = inputData.data[i] || 0;
	}

	theNN.setMode(NNMODE_USE);
	theNN.doForwardPropagation();

	renderNetwork(canvasSet, theNN, window.devicePixelRatio);

	const report = theNN.reportInferenceResult();
	const ambiguous = report.secondNode.outValue / report.firstNode.outValue;
	theThrobber.nowInferred(report.firstNode.label, ambiguous > 0.6);
}

window.onAnyInputChange = function() {
	updateDirtyFlag();
};

function updateDirtyFlag() {
	const lr = pickNumericInput("param-lr");
	const cth = pickNumericInput("param-cth");
	const mom = pickNumericInput("param-mom");
	const aug = augmentModeFromInput();

	const same = ( nearly_equals(lr, TestNN.Initialization.learningRate) &&
					nearly_equals(cth, TestNN.Initialization.completionThreshold) &&
					nearly_equals(mom, TestNN.Initialization.momentum) &&
					aug === TestNN.Initialization.augmentation);
	
	const el = document.getElementById("controller-container");
	el.dataset.dirty = same ? 0 : 1;
}

function nearly_equals(a, b) {
	return Math.abs(a - b) < 0.000001;
}