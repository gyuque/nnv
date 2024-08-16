"use strict";
const kSkewVector = {x: 0.949, y: 0.316};

const NT_Generic  = 0x00;
const NT_Output   = 0x01;
const NT_Constant = 0x10;

const VARINIT_M  = 1;
const VARINIT_MM = 2;

class NeuralNetwork {
	constructor(nAllLayers, learningRate, numOfSamples) {
		if ( !Number.isSafeInteger(nAllLayers) ) {
			throw new Error("Bad nAllLayers:", nAllLayers);
		}

		if ( !Number.isFinite(learningRate) ) {
			throw new Error("Bad learningRate:", learningRate);
		}

		if ( !Number.isSafeInteger(numOfSamples) ) {
			throw new Error("Bad numOfSamples:", numOfSamples);
		}

		this.learningRate = learningRate;
		this.layerList = new Array(nAllLayers);
		for (let i = 0;i < nAllLayers;++i) {
			this.layerList[i] = createLayerMetadata();
		}
		
		this.iterationCount = 0;
		this.lastTotalError = 1;
		this.currentSample = null;
		this.numOfSamples = numOfSamples;
		this.trainDataList = new Array(numOfSamples);
		for (let i = 0;i < numOfSamples;++i) {
			this.trainDataList[i] = new NNSampleData();
		}
	}

	forEachTrainDataSample(fn) {
		return this.trainDataList.forEach(fn);
	}

	setClassificationTrainData(sampleIndex, inArray, outClassIndex) {
		this.trainDataList[sampleIndex].inValues = inArray;
		this.trainDataList[sampleIndex].expectedResult = outClassIndex;
	}

	ready() {
		this.doForwardPropagation();
		return this;
	}

	doForwardPropagation() {
		const nLayers = this.layerList.length;
		for (let li = 0;li < nLayers;++li) {
			const nodeList = this.layerList[li].nodes;

			// Phase 1 receive input and apply tx-func
			if (li > 0) {
				for (const nd of nodeList) {
					nd.receiveFromEdges();
					nd.updateOutValue();
				}
			}

			// Phase 2 propagate to next edges
			for (const nd of nodeList) {
				nd.propagateToEdges();
			}

			this.updateStats(this.layerList[li]);
		}

		this.updateResultLayerError();
	}

	selectSample(trainDataIndex) {
		const samp = this.trainDataList[trainDataIndex];
		this.currentSample = samp;
		const inLayerNodes = this.getNodesAtLayer(0);

		const inA = samp.inValues;
		const n = inA.length;
		for (let i = 0;i < n;++i) {
			inLayerNodes[i].outValue = inA[i];
		}

		this.setExpectedClass(samp.expectedResult);
		return samp;
	}

	advanceLearning(onSampleLearned) {
		const n = this.numOfSamples;
		this.resetDErrors();

		let sumE = 0;
		for (let i = 0;i < n;++i) {
			const selectedSample = this.selectSample(i);
			this.doForwardPropagation();
			this.calcLastTrainingError();
			sumE += selectedSample.lastErrorAmount;

			this.doBackPropagation();
			this.updateDErrors();

			if (onSampleLearned) {
				onSampleLearned(this, i);
			}
		}

		this.updateWeights();
		this.doForwardPropagation();
		this.lastTotalError = sumE / n;

		++this.iterationCount;
	}

	doBackPropagation() {
		const startLayer = this.layerList.length - 2;
		for (let li = startLayer;li > 0;--li) {
			const nodeList = this.layerList[li].nodes;
			for (const nd of nodeList) {
				nd.updateDelta();
			}
		}
	}

	resetDErrors() {
		this.forAllBackEdge( endNode => endNode.resetBackEdgeDErrors() );
	}

	updateDErrors() {
		this.forAllBackEdge( endNode => endNode.updateBackEdgeDErrors() );
	}

	updateWeights() {
		this.forAllBackEdge( endNode => endNode.updateBackEdgeWeights(this.learningRate, this.numOfSamples) );
	}

	forAllBackEdge(fn) {
		const startLayer = this.layerList.length - 1;
		for (let li = startLayer;li > 0;--li) {
			const nodeList = this.layerList[li].nodes;
			for (const endNode of nodeList) {
				fn(endNode);
			}
		}
	}

	updateStats(layer) {
		let o_max = Number.NEGATIVE_INFINITY;
		let w_max = Number.NEGATIVE_INFINITY;
		for (const nd of layer.nodes) {
			o_max = Math.max( Math.abs(nd.outValue), o_max);
			for (const e of nd.backwards) {
				w_max = Math.max( Math.abs(e.weight), w_max);
			}
		}

		layer.stats.outMax = o_max;
		layer.stats.weightMax = w_max;
	}

	addNode(node, layerIndex) {
		if (layerIndex >= this.countLayers()) {
			console.log("**WARNING**");
			console.trace();
		}

		this.layerList[layerIndex].nodes.push(node);
	}

	addOneNode(layerIndex) {
		const nd = new NNNode(NT_Constant);
		this.layerList[layerIndex].nodes.push(nd);
	}

	countLayers() {
		return this.layerList.length;
	}

	getLastLayer() {
		const i = this.layerList.length - 1;
		return this.layerList[i];
	}

	getNodesAtLayer(index) {
		return this.layerList[index].nodes;
	}

	getStatsAtLayer(index) {
		return this.layerList[index].stats;
	}

	setLayerDimension(index, d) {
		this.layerList[index].dim = d;
	}

	setLayerNumColumns(index, c) {
		this.layerList[index].nColumns = c;
	}

	getLayerDimension(index) {
		return this.layerList[index].dim;
	}

	getLayerNumColumns(index) {
		return this.layerList[index].nColumns;
	}

	getLayerNumRows(index) {
		const cols = this.getLayerNumColumns(index);
		return Math.floor(this.layerList[index].nodes.length / cols);
	}

	connectAll() {
		const n = this.countLayers() - 1;
		for (let i = 0;i < n;++i) {
			this.connectAllAtLayer(i);
		}
	}

	connectAllAtLayer(firstLayerIndex) {
		const selfLayer = this.layerList[firstLayerIndex];
		const nextLayer = this.layerList[firstLayerIndex+1];
		for (const selfNode of selfLayer.nodes) {
			for (const nextNode of nextLayer.nodes) {
				if (nextNode.type !== NT_Constant) {
					selfNode.connectToForward(nextNode);
				}
			}
		}
	}

	setExpectedClass(raiseIndex) {
		const lastLayer = this.getLastLayer();
		const nodes = lastLayer.nodes;
		const n = nodes.length;

		for (let i = 0;i < n;++i) {
			nodes[i].expectedValue = (i === raiseIndex) ? 1 : 0;
		}
	}

	initWeights(denominatorType) {
		const lastLayer = this.getLastLayer();
		const nodes = lastLayer.nodes;
		for (const nd of nodes) {
			nd.initBackWeights(denominatorType);
		}
	}

	updateResultLayerError() {
		const lastLayer = this.getLastLayer();
		for (const nd of lastLayer.nodes) {
			nd.updateError();
		}
	}

	calcLastTrainingError() {
		const lastLayer = this.getLastLayer();
		const samp = this.currentSample;

		samp.lastErrorAmount = 0;
		for (const nd of lastLayer.nodes) {
			samp.lastErrorAmount += Math.abs( nd.calcError() );
		}

		samp.lastErrorAmount /= lastLayer.nodes.length;
	}
}

class NNSampleData {
	constructor() {
		this.inValues = null;
		this.expectedResult = null;
		this.lastErrorAmount = 1;
	}
}

function createLayerMetadata(options) {
	let dim = 1;
	if (options && Number.isSafeInteger(options.dim)) {
		dim = options.dim;
	}

	return {
		dim: dim,
		nColumns: 1,
		nodes: [],
		stats: {
			outMax: Number.NEGATIVE_INFINITY,
			weightMax: Number.NEGATIVE_INFINITY
		}
	};
}

class NNNode {
	constructor(nType) {
		this.type = nType || NT_Generic;
		this.forwards = [];
		this.backwards = [];
		this.inputBias = 0;
		this.constantValue = 1;

		this.outValue = 0;
		this.txFunc = txf_identity;
		this.txFunc_d = txd_identity;
		this.inValue = 0;

		this.expectedValue = 0;
		this.error = 0;
		this.delta = 0;

		this.viz = {};
	}

	updateError() {
		this.error = this.calcError();
		this.delta = this.error;
	}

	calcError() {
		return this.outValue - this.expectedValue;
	}

	updateDelta() {
		this.delta = 0;
		for (const e of this.forwards) {
			const counterNode = e.destNode;
			this.delta += counterNode.delta * e.weight * this.txFunc_d(this.inValue);
		}
	}

	resetBackEdgeDErrors() {
		for (const e of this.backwards) {
			e.dE = 0;
		}
	}

	updateBackEdgeDErrors() {
		for (const e of this.backwards) {
			e.dE += this.delta * e.originNode.outValue;
		}
	}

	updateBackEdgeWeights(learningRate, numOfSamples) {
		for (const e of this.backwards) {
			e.weight -= learningRate * (e.dE / numOfSamples);
		}
	}

	receiveFromEdges() {
		this.inValue = 0;
		for (const e of this.backwards) {
			this.inValue += e.calcOutValue();
		}
	}

	updateOutValue() {
		this.outValue = this.txFunc(this.inValue);
	}

	propagateToEdges() {
		for (const e of this.forwards) {
			e.inValue = this.outValue;
		}
	}

	connectToForward(fwdNode) {
		const edge = new NNConnection();
		edge.originNode = this;
		edge.destNode = fwdNode;
		this.forwards.push(edge);

		fwdNode.connectBack(edge);
	}

	connectBack(edge) {
		this.backwards.push(edge);
	}

	initBackWeights(denominatorType) {
		const M = this.backwards.length;
		for (const e of this.backwards) {
			const a = Math.sqrt(6.0 / M);

			if (e.originNode && e.originNode.type === NT_Constant) {
				e.zeroWeight();
			} else {
				e.uniformRand(a);
			}

			if (e.originNode) {
				e.originNode.initBackWeights(denominatorType);
			}
		}
	}
}

// f(u)=u
// f'(u)=1
function txf_identity(inVal) { return inVal; }
function txd_identity(_inVal) { return 1; }

class NNConnection {
	constructor() {
		this.originNode = null;
		this.destNode = null;
		this.weight = 1;
		this.dE = 0;

		this.inValue = 0;
	}

	calcOutValue() {
		return this.weight * this.inValue;
	}

	zeroWeight() {
		this.weight = 0;
	}

	uniformRand(a) {
		this.weight = (Math.random() * 2*a) - a;
	}
}

function calcNumOfNodes(input_n) {
	if (Array.isArray(input_n)) {
		return input_n.reduce(
			(new_val, cur) => new_val * cur,
			1
		);
	}

	return input_n;
}

function calcDim(input_n) {
	if (Array.isArray(input_n)) {
		return input_n.length;
	}

	return 1;
}

function calcNumColumns(input_n) {
	if (Array.isArray(input_n)) {
		return input_n[0];
	}

	return 1;
}

function addNewNodes(targetNN, layerIndex, input_n, isResult) {
	const n = calcNumOfNodes(input_n);
	for (let i = 0;i < n;++i) {
		const node = new NNNode(isResult ? NT_Output : NT_Generic);
		targetNN.addNode(node, layerIndex);
	}
}

function buildNetwork(src) {
	const ini = src.Initialization;
	const lyrConfigs = src.LayersConfig;
	const nAllLayers = lyrConfigs.length;
	const iLast = nAllLayers - 1;

	const aNN = new NeuralNetwork(nAllLayers, ini.learningRate, ini.numOfSamples);

	for (let li = 0;li < nAllLayers;++li) {
		const conf = lyrConfigs[li];
		addNewNodes(aNN, li, conf.n, li === iLast);
		if (conf.one) {
			aNN.addOneNode(li);
		}

		aNN.setLayerDimension(li, calcDim(conf.n));
		aNN.setLayerNumColumns(li, calcNumColumns(conf.n));
	}

	aNN.connectAll();

	let vd = VARINIT_M;
	if (ini.denominator.length === 2) {
		vd = VARINIT_MM;
	}

	aNN.initWeights(vd);

	return aNN;
}


// Visualization

function renderNetwork(canvas, nn, pixelRatio) {
	const w = canvas.width - 0;
	const h = canvas.height - 0;

	const g = canvas.getContext("2d");
	g.clearRect(0, 0, w, h);
	
	const nLayers = nn.countLayers();
	const segWidth = Math.floor(w / (nLayers+1));

	let x = segWidth;
	for (let li = 0;li < nLayers;++li) {
		const l_nodes = nn.getNodesAtLayer(li);
		const nNodes = l_nodes.length;

		const is2d = nn.getLayerDimension(li) > 1;
		let nCols = nn.getLayerNumRows(li);

		if (is2d) {
			layoutNodes2D(l_nodes, x, h >> 1, nCols, nn.getLayerNumRows(li), pixelRatio);
		} else {
			const vseg_height = Math.floor(h / (nNodes+1));
			let y = vseg_height;
			// Layout
			for (let i = 0;i < nNodes;++i) {
				const nd = l_nodes[i];
				nd.viz.sx = x;
				nd.viz.sy = y;
				nd.viz.skew = null;

				y += vseg_height;
			}
		}

		x += segWidth;
	}

	renderEdges(g, w, h, pixelRatio, nn);
	renderNodes(g, w, h, pixelRatio, nn);
}

function layoutNodes2D(nodeList, cx, cy, cols, rows, pixelRatio) {
	const nNodes = nodeList.length;

	const cellWidth = 8 * pixelRatio;
	const cellHeight = 8 * pixelRatio;

	const ox = (cols/2) * -cellWidth;
	const oy = (rows/2) * -cellHeight;

	for (let i = 0;i < nNodes;++i) {
		const nd = nodeList[i];

		const ci = i % cols;
		const ri = Math.floor(i / cols);

		if (nd.type === NT_Constant) {
			nd.viz.sx = cx;
			nd.viz.sy = cy - oy*2;
			nd.viz.width = cellWidth;
			nd.viz.height = cellHeight;
			nd.viz.skew = null;
		} else {
			nd.viz.sx = cx + ox + ci*cellWidth * kSkewVector.x;
			nd.viz.sy = cy + oy + ri*cellHeight + ci*cellWidth * kSkewVector.y;
			nd.viz.width = cellWidth;
			nd.viz.height = cellHeight;
			nd.viz.skew = kSkewVector;
		}
	}
}

function renderNodes(g, w, h, pixelRatio, nn) {
	const nLayers = nn.countLayers();
	for (let li = 0;li < nLayers;++li) {
		const l_nodes = nn.getNodesAtLayer(li);
		const l_stats = nn.getStatsAtLayer(li);
		const nNodes = l_nodes.length;

		const is2d = nn.getLayerDimension(li) > 1;

		for (let i = 0;i < nNodes;++i) {
			const nd = l_nodes[i];

			g.strokeStyle = "#345";
			g.fillStyle = "#e0e3fe";
			g.lineWidth = pixelRatio;

			const radius = 9 * pixelRatio;
			if (is2d) {
				render2DNode(g, nd, pixelRatio);
			} else {
				if (nd.type === NT_Constant) {
					renderConstantNode(g, nd.viz.sx, nd.viz.sy, radius, pixelRatio, nd.constantValue);
				} else {
					g.beginPath();
					g.arc(nd.viz.sx, nd.viz.sy, radius, 0, Math.PI*2, false);
					g.fill();
					g.stroke();

					if (nd.type === NT_Output) {
						renderOutputMeter(g, nd, radius, pixelRatio, l_stats);
					}
				}
			}
		}
	}	
}

function renderOutputMeter(g, node, nodeRadius, pixelRatio, layerStats) {
	const v = node.viz;

	g.save();
	const h = pixelRatio * 3;
	const w = pixelRatio * 50;

	const ox = v.sx + nodeRadius*2;
	const oy = v.sy;

	const xvMax = Math.max(layerStats.outMax, node.expectedValue);
	const ratio = node.outValue / xvMax;
	const exRatio = node.expectedValue / xvMax;
	const mcolor = "#09B";

	const absError = Math.abs(node.error);
	const ex = (w/2) * Math.min(ratio, exRatio);

	renderMeter(g, ox, oy-h*2 - pixelRatio*2, w, h*2, ratio, mcolor, node.outValue, pixelRatio);
	renderMeter(g, ox, oy     + pixelRatio*2, w, h*2, exRatio, "#EB0", node.expectedValue, pixelRatio);
	renderMeter(g, ox, oy     - pixelRatio  , w, pixelRatio*2, absError/xvMax, "#C00", "", pixelRatio, ex);

	g.restore();
}

function renderMeter(g, x, y, w, h, ratio, color, label, pixelRatio, offsetX) {
	const w2 = w/2;
	g.fillStyle = "#BBB";
	g.fillRect(x, y, w2, h);
	g.fillStyle = "#CCC";
	g.fillRect(x+w2, y, w2, h);

	g.fillStyle = color;
	const len = w2*ratio;
	g.fillRect(x+w2 + (offsetX || 0), y, len, h);

	g.fillStyle = "#444";
	g.font = "normal 12px monospace";
	g.textBaseline = "top";
	g.textAlign = "left";
	g.fillText(label, x+w+ pixelRatio*3, y);
}

function render2DNode(g, node, pixelRatio) {
	if (node.viz.skew) {
		const v = node.viz.skew;
		const ox = node.viz.sx;
		const oy = node.viz.sy;
		const hw = node.viz.width/2;
		const hh = node.viz.height/2;

		const vx = v.x * hw;
		const vy = v.y * hw;

		const b = Math.floor(node.outValue * 255);

		g.save();
		g.fillStyle = `rgb(${b},${b},${b})`;

		g.beginPath();
		g.moveTo(ox - vx + pixelRatio, oy - vy - hh + pixelRatio);
		g.lineTo(ox + vx - pixelRatio, oy + vy - hh + pixelRatio);
		g.lineTo(ox + vx - pixelRatio, oy + vy + hh - pixelRatio);
		g.lineTo(ox - vx + pixelRatio, oy - vy + hh - pixelRatio);
		g.closePath();
		g.fill();
		g.stroke();

		g.restore();
	} else {
		renderConstantNode(g, node.viz.sx, node.viz.sy, node.viz.width, pixelRatio, node.constantValue);
	}
}

function renderEdges(g, w, h, pixelRatio, nn) {
	const nLayers = nn.countLayers();
	for (let li = 1;li < nLayers;++li) {
		const l_nodes = nn.getNodesAtLayer(li);
		const l_stats = nn.getStatsAtLayer(li);
		const nNodes = l_nodes.length;
		for (let i = 0;i < nNodes;++i) {
			const nd = l_nodes[i];
			nd.backwards.forEach(edge => {
				const destNode = edge.originNode;
				const wAlpha = (Math.abs(edge.weight) / l_stats.weightMax) * 0.4;

				g.lineWidth = pixelRatio;
				g.strokeStyle = `rgba(80,80,88,${0.1+wAlpha})`;
				g.beginPath();
				g.moveTo(nd.viz.sx, nd.viz.sy);
				g.lineTo(destNode.viz.sx, destNode.viz.sy);
				g.stroke();
			});
		}
	}
}

function renderConstantNode(g, cx, cy, size, pixelRatio, label) {
	g.save();
	g.beginPath();
	g.fillStyle = "#35d";

	const x0 = cx-size;
	const x1 = cx+size;
	const y0 = cy-size;
	const y1 = cy+size;

	g.moveTo(cx     , y0);
	g.bezierCurveTo(x1,y0, x1,y0, x1, cy);
	g.bezierCurveTo(x1,y1, x1,y1, cx, y1);
	g.bezierCurveTo(x0,y1, x0,y1, x0, cy);
	g.bezierCurveTo(x0,y0, x0,y0, cx, y0);
	g.closePath();

	g.fill();
	g.stroke();

	g.fillStyle = "#fff";
	g.font = `normal ${size}px sans-serif`;
	g.textAlign = "center";
	g.textBaseline = "middle";
	g.fillText(label, cx, cy);

	g.restore();
}

export { NeuralNetwork, NNNode, NNConnection, buildNetwork, renderNetwork };