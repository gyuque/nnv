"use strict";

class DrawingInput {
	constructor(width, height, nin_width, nin_height, inferButtonCallback) {
		this.outer = document.createElement("div");
		this.outer.className = "handwrite-input-outer";
		this.cv = document.createElement("canvas");
		this.cv.width = width;
		this.cv.height = height;
		this.outer.appendChild(this.cv);

		this.neuralInputCanvas = document.createElement("canvas");
		this.neuralInputCanvas.width = nin_width;
		this.neuralInputCanvas.height = nin_height;
		this.neuralInputCanvas.className = "input-preview-canvas";
		this.ng = this.neuralInputCanvas.getContext("2d", {willReadFrequently: true});

		this.cv.addEventListener("mousedown", this.onCanvasMouseDown.bind(this), false);
		this.cv.addEventListener("mousemove", this.onCanvasMouseMove.bind(this), false)
		window.addEventListener("mouseup", this.onGlobalMouseUp.bind(this), false);

		this.outer.appendChild( document.createElement("br") );
		this.outer.appendChild( this.neuralInputCanvas );

		this.clearButton = make_button("↻", this.outer, this.onClearClick.bind(this), "btn-type-destructive");
		this.inferButton = make_button("▶", this.outer, this.onInferClick.bind(this), "btn-type-execution");

		this.drawState = {
			active: false,
			prevX: 0,
			prevY: 0
		};

		this.g = this.cv.getContext("2d");
		this.clear();

		this.inferButtonCallback = inferButtonCallback || null;
	}

	getElement() {
		return this.outer;
	}

	drawLine(x1, y1, x2, y2) {
		const g = this.g;

		g.strokeStyle = "#FFF";
		g.lineWidth = 8;
		g.lineCap = "round";

		g.beginPath();
		g.moveTo(x1, y1);
		g.lineTo(x2, y2);
		g.stroke();

		this.copyToScaledCanvas();
	}

	onCanvasMouseDown(e) {
		const s = this.drawState;
		s.active = true;
		s.prevX = e.offsetX;
		s.prevY = e.offsetY;
	}

	onCanvasMouseMove(e) {
		const s = this.drawState;
		const x = e.offsetX;
		const y = e.offsetY;

		if (s.active) {
			this.drawLine(s.prevX, s.prevY, x, y);
		}

		s.prevX = x;
		s.prevY = y;
	}

	onGlobalMouseUp(_e) {
		this.drawState.active = false;
	}

	clear() {
		const w = this.cv.width - 0;
		const h = this.cv.height - 0;
		this.g.clearRect(0, 0, w, h);
		this.g.fillStyle = "#000";
		this.g.fillRect(0, 0, w, h);
		this.copyToScaledCanvas();
	}

	onClearClick() {
		this.clear();
	}

	onInferClick() {
		if (this.inferButtonCallback) {
			const inputData = this.makeInputData();
			this.inferButtonCallback(inputData);
		}
	}

	copyToScaledCanvas() {
		const sw = this.cv.width - 0;
		const sh = this.cv.height - 0;

		const dw = this.neuralInputCanvas.width - 0;
		const dh = this.neuralInputCanvas.height - 0;

		const g = this.ng;
		g.drawImage(this.cv, 0, 0, sw, sh, 0, 0, dw, dh);
	}

	makeInputData() {
		const w = this.neuralInputCanvas.width - 0;
		const h = this.neuralInputCanvas.height - 0;
		const g = this.ng;

		const idat = g.getImageData(0, 0, w, h);
		const dataArray = [];
		const pixels = idat.data;

		let pos = 0;
		for (let y = 0;y < h;++y) {
			for (let x = 0;x < w;++x) {
				const red = pixels[pos];
				dataArray.push(red / 255.0);
				
				pos += 4;
			}
		}

		return {
			width: w,
			height: h,
			data: dataArray
		};
	}
}

function make_button(label, parent, handler, className) {
	const btn = document.createElement("button");
	btn.appendChild( document.createTextNode(label) );
	if (parent) {
		parent.appendChild(btn);
	}

	if (handler) {
		btn.addEventListener("click", handler, false);
	}

	if (className) {
		btn.className = className;
	}

	return btn;
}

function setupInputArea(container_id, inferButtonCallback) {
	const el = document.getElementById(container_id);
	const di = new DrawingInput(128, 128, 16, 16, inferButtonCallback);
	el.appendChild(di.getElement());

	return di;
}

export { setupInputArea };