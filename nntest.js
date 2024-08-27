"use strict";
import { setupInputArea } from "./datainput.js";
import { ButtonManager, LearningThrobber, NNErrorLogChart, TabPager } from "./nnchart.js";
import { buildNetwork, NNAUG_2DTRANS, NNAUG_NONE, NNMODE_LEARN, NNMODE_USE, renderNetwork, setTrainDataset, TransferFunctions } from "./nnv.js";
import { PresetPopup } from "./preset.js";

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

		// ======= Default transfer function for hidden layers =======
		transferFunc: "Identity",

		// ======= Learning rate =======
		learningRate: 0.04,
		momentum: 0,

		// ======= Completion threshold =======
		completionThreshold: 0.01
	}
};

var gWorker = null;

var theNN = null;
var theLogChart = null;
var theControlButtons = null;
var theControlTabPager = null;
var theThrobber = null;
var currentFrameCount = 0;
var gIterationCountDisplay = null;
var gPerformanceDisplay = null;
var gAnimationActive = false;
var gAnimationStartTime = 0;
var gLastLossList = null;
var gLastThrobberChange = -1;
var gSoundShouldBePlayed = -1;
var gPresetPopup = null;

const canvasSet = [null, null];
const soundSet = [null, null, null]

window.launch = function() {
	setupSounds();
	canvasSet[0] = document.getElementById("back-cv");
	canvasSet[1] = document.getElementById("cv");

	gWorker = new Worker("./nn-worker.js", {type: "module", credentials: "omit"});
	gWorker.onmessage = onWorkerMessage;
};

function setupSounds() {
	const a1 = document.createElement("audio");
	a1.src = "./sound/poku.mp3";
	soundSet[0] = a1;

	const a2 = document.createElement("audio");
	a2.src = "./sound/chin.mp3";
	soundSet[1] = a2;

	const a3 = document.createElement("audio");
	a3.src = "./sound/syupo.mp3";
	soundSet[2] = a3;

	setInterval(soundInterval, 500);
}

function soundInterval() {
	if (gSoundShouldBePlayed >= 0) {
		const a = soundSet[gSoundShouldBePlayed];
		if (a) {
			a.currentTime = 0;
			a.play();
		}

		if (gSoundShouldBePlayed === 1) {
			gSoundShouldBePlayed = -1;
		}
	}
}

function setupControlUIs() {
	theControlButtons = new ButtonManager("control-button-container", onCommandButtonClick);
	theControlButtons.setDisabled("p", true);

	theControlTabPager = new TabPager("pages-container");
	theControlTabPager.selectByIndex(0);

	showTrainDataPreview("data-preview-items", TRAIN_DATA_1);
	setupInputArea("data-input-pane", onExecuteInferenceClick);

	gPresetPopup = new PresetPopup("preset-button", "preset-popup", onPresetSelected);
}

function onWorkerReady() {
	setupControlUIs();
	resetNN();
}

function resetNN() {
	gLastLossList = null;
	gLastThrobberChange = -1;
	gSoundShouldBePlayed = -1;
	pickParams();

	const b_report = {};
	const nn = buildNetwork(TestNN, b_report);

	const p_ct = document.getElementById("intl-params-count");
	p_ct.innerHTML = "";
	p_ct.appendChild( document.createTextNode(b_report.nEdges+" edge weights") );

	const trainDataset = makeTrainDataset(
		TRAIN_DATA_1, 
		TestNN.Initialization.augmentation,
		TestNN.Initialization.numOfSamples
	);
	setTrainDataset(nn, trainDataset);
	sendBuildCommand(TestNN, trainDataset);


	// Set output label
	nn.forEachNodeAtLayer(-1, (node, nodeIndex) => {  node.label = `${nodeIndex}`;  });

	theNN = nn.ready();
	setupCharts(document.getElementById("elog-container"), TestNN.Initialization.numOfSamples);
	renderNetwork(canvasSet, theNN, window.devicePixelRatio);

	currentFrameCount = 0;
}

function makeTrainDataset(source, augMode, n) {
	const sampleList = [];

	for (let i = 0;i < n;++i) {
		const inset = source[i];
		sampleList.push({
			data: inset.data,
			classIndex: i % 10,
			width: inset.width,
			height: inset.height
		});
	}

	const res = {
		sampleList: sampleList,
		augMode: augMode
	};

	return res;
}

function addCounterP(parent, className) {
	const el = document.createElement("p");
	el.className = className;
	parent.appendChild(el);

	return el;
}

function setupCharts(containerElement, n) {
	containerElement.innerHTML = "";

	gIterationCountDisplay = addCounterP(containerElement, "iteration-counter");
	updateIterationCounter(0);

	gPerformanceDisplay = addCounterP(containerElement, "iteration-counter");
	updatePerfoemanceCounter(0);

	theLogChart = new NNErrorLogChart(100, 192, window.devicePixelRatio, n+1);
	containerElement.appendChild(theLogChart.canvas);
	theLogChart.render();

	theThrobber = new LearningThrobber();
	containerElement.appendChild(theThrobber.element);
}

function updateIterationCounter(i) {
	renewTextNodeValue(gIterationCountDisplay, `Iteration: ${i}`);
}

function updatePerfoemanceCounter(t) {
	renewTextNodeValue(gPerformanceDisplay, `${t.toFixed(1)}ms/iter`);
}

function renewTextNodeValue(el, t) {
	if (el) {
		el.innerHTML = "";
		el.appendChild( document.createTextNode(t) );
	}
}

function enterFrame() {
	if (currentFrameCount === 0) {
		gAnimationStartTime = performance.now();
	}

	const etime = performance.now() - gAnimationStartTime;

	const showSampleIndex = Math.floor(etime / 500) % TestNN.Initialization.numOfSamples;
	const frontNN = theNN;

	if (0 === (currentFrameCount & 7)) {
		frontNN.selectSample(showSampleIndex);
	}
	frontNN.doForwardPropagation();
	renderNetwork(canvasSet, frontNN, window.devicePixelRatio);

	if (gAnimationActive && gLastThrobberChange !== currentFrameCount) {
		const cycle = Math.floor(etime / 1000) % 20;
		if (cycle === 9) {
			theThrobber.setPanic(1, 0);
			gLastThrobberChange = currentFrameCount;
		} else if (cycle === 19) {
			theThrobber.setPanic(1, 1);
			gLastThrobberChange = currentFrameCount;
		}	
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
			invokeRebuild();
			break;

		case 'x':
			if (!gAnimationActive) {
				theThrobber.showThinkingFrame();
				gAnimationActive = true;
				gSoundShouldBePlayed = 0;
				theControlButtons.setDisabled("x", true);
				theControlButtons.setDisabled("p", false);
				sendExecCommand();
				enterFrame();
			}
			
			break;
		case 'p':
			sendStopCommand();
			stopLearning();
			break;
	}
}

function invokeRebuild() {
	sendStopCommand(1);
}

function pickParams() {
	TestNN.Initialization.transferFunc = pickTransferFuncSelection();
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

function pickTransferFuncSelection() {
	return document.getElementById("txfunc-sel").value;
}

function stopLearning(complete) {
	sendStopCommand();
	gAnimationActive = false;
	theControlButtons.setDisabled("x", false);
	theControlButtons.setDisabled("p", true);
	gSoundShouldBePlayed = complete ? 1 : -1;

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

	soundSet[2].play();
}

window.onAnyInputChange = function() {
	updateDirtyFlag();
};

function updateDirtyFlag() {
	const txf = pickTransferFuncSelection();
	const lr = pickNumericInput("param-lr");
	const cth = pickNumericInput("param-cth");
	const mom = pickNumericInput("param-mom");
	const aug = augmentModeFromInput();

	const same = ( nearly_equals(lr, TestNN.Initialization.learningRate) &&
					nearly_equals(cth, TestNN.Initialization.completionThreshold) &&
					nearly_equals(mom, TestNN.Initialization.momentum) &&
					aug === TestNN.Initialization.augmentation) &&
					txf === TestNN.Initialization.transferFunc;
	
	const el = document.getElementById("controller-container");
	el.dataset.dirty = same ? 0 : 1;
}

function nearly_equals(a, b) {
	return Math.abs(a - b) < 0.000001;
}

// Worker communication -------------------------

function onWorkerMessage(e) {
	if (e && e.data) {
		const edat = e.data;
		if (edat.workerReady) {
			onWorkerReady();
		} else if (edat.nnBuilt) {
			theControlButtons.setDisabled("x", false);			
		} else if (edat.learned) {
			const learn_status = edat.learned;
			gLastLossList = learn_status.lossList;
			recordLossList(gLastLossList);
			updateIterationCounter(learn_status.iterationCount);
			updatePerfoemanceCounter(learn_status.et / learn_status.chunkIterations);
			theNN.importParams(learn_status.params);
			if (learn_status.lastTotalError <= TestNN.Initialization.completionThreshold) {
				stopLearning(true);
			}
		} else if (edat.stopped === 1) {
			stopLearning();
			resetNN();
		}
	}
};

function recordLossList(lossList) {
	const nLoss = lossList.length;
	for (let il = 0;il < nLoss;++il) {
		theLogChart.pushValue(il, lossList[il]);
	}
	theLogChart.render();
}

function sendBuildCommand(cf, trainDataset) {
	theControlButtons.setDisabled("x", true);
	gWorker.postMessage({conf: cf, trainDataset: trainDataset});
}

function sendExecCommand() {
	gWorker.postMessage({learn: true});
}

function sendStopCommand(sendBackParam) {
	gSoundShouldBePlayed = -1;
	gWorker.postMessage({stop: true, sendBackParam: sendBackParam});
}

function onPresetSelected(presetItem) {
	set_checkbox_value("data-aug-checkbox", presetItem.aug);
	set_direct_value("txfunc-sel", presetItem.tx);
	set_direct_value("param-lr", presetItem.lr);
	set_direct_value("param-mom", presetItem.momentum);
	set_direct_value("param-cth", presetItem.cth);
	invokeRebuild();
}

function set_checkbox_value(dest_id, val) {
	document.getElementById(dest_id).checked = val;
}

function set_direct_value(dest_id, val) {
	document.getElementById(dest_id).value = val;
}
