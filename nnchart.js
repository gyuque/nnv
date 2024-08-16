"use strict";

class NNErrorLogChart {
	constructor(width, height, pixelRatio, nSeries) {
		const LPAD = 20;
		this.capacity = width - LPAD;
		this.yScale = height*pixelRatio / 0.8;
		this.visibleWidth = width;
		this.visibleHeight = height;
		this.pixelRatio = pixelRatio;

		const cv = create_canvas_with_pixel_ratio(width, height, pixelRatio);
		this.canvas = cv;
		this.g = cv.getContext("2d");

		this.leftPadding = pixelRatio * LPAD;
		this.bottomPadding = pixelRatio * 12;

		this.seriesList = (new Array(nSeries)).fill(null).map(_i => [] );
	}

	prescan() {
		let max = 0;
		for (const sr of this.seriesList) {
			sr.forEach( v => { max = Math.max(v, max); } );
		}
		return max;	
	}

	pushValue(seriesIndex, value) {
		const sr = this.seriesList[seriesIndex];
		sr.push(value);
		if (sr.length > this.capacity) {
			sr.shift();
		}
	}

	render() {
		const pr = this.pixelRatio;
		const g = this.g;
		const w = this.canvas.width - 0;
		const h = this.canvas.height - 0;

		// const globalMax = this.prescan();

		g.clearRect(0, 0, w, h);
		g.fillStyle = "#aaa";
		this.drawMidScale(g, h, 0.2, w, pr);
		this.drawMidScale(g, h, 0.4, w, pr);
		this.drawMidScale(g, h, 0.6, w, pr);

		g.fillStyle = "#888";
		this.drawXaxis(g, h-this.bottomPadding, w, pr);
		this.drawYaxis(g, this.leftPadding, h, pr);
		this.drawTitle(g, w, h, pr);

		this.plotAll(g, w, h, pr);
	}

	plotAll(g, canvasWidth, canvasHeight, pixelRatio) {
		const lastIndex = this.seriesList.length-1;
		let i = 0;
		g.fillStyle = "#C00";
		for (const sr of this.seriesList) {
			const isLast = (i === lastIndex);
			g.strokeStyle = isLast ? "#C00" : "#DBB";
			g.lineWidth = isLast ? pixelRatio*2 : pixelRatio;
			this.plotSeries(g, canvasWidth, canvasHeight, pixelRatio, sr, isLast);
			++i;
		}
	}

	plotSeries(g, canvasWidth, canvasHeight, pixelRatio, seriesData, drawLabel) {
		const dataLen = seriesData.length;
		if (dataLen < 1) { return; }

		g.beginPath();
		const n = this.capacity;
		for (let i = 0;i < n;++i) {
			const dataIndex = dataLen-1-i;
			if (dataIndex >= 0) {
				const y = this.calcYPos(seriesData[dataIndex], canvasHeight);
				const x = canvasWidth - i*pixelRatio;
				if (i === 0) {
					g.moveTo(x, y);
				} else {
					g.lineTo(x, y);
				}
			}

			if (dataIndex <= 0) {
				if (i > 0) {
					g.stroke();
				}
				break;
			}
		}

		if (drawLabel) {
			const lastVal = seriesData[ dataLen-1 ];
			g.textAlign = "right";
			g.textBaseline = "alphabetic";
			const labelY = this.calcYPos(lastVal, canvasHeight) - pixelRatio*6;
			g.fillText(lastVal.toFixed(4), canvasWidth-pixelRatio*4, labelY);
		}
	}

	calcYPos(val, canvasHeight) {
		return canvasHeight - this.bottomPadding - val * this.yScale;
	}

	drawMidScale(g, canvasHeight, val, w, pr) {
		const y = this.calcYPos(val, canvasHeight);
		this.drawXaxis(g, y, w, pr);
		g.font = `normal ${9*pr}px monospace`;
		g.textAlign = "right";
		g.textBaseline = "middle";
		g.fillText(val.toFixed(1), this.leftPadding-pr*3, y);
	}

	drawXaxis(g, y, w, pr) {
		g.fillRect(this.leftPadding, y, w-this.leftPadding, pr);
	}

	drawYaxis(g, x, h, pr) {
		g.fillRect(x, 0, pr, h - this.bottomPadding);
	}

	drawTitle(g, w, h, pr) {
		g.font = `normal ${9*pr}px sans-serif`;
		g.textAlign = "center";
		g.textBaseline = "alphabetic";
		g.fillText("Loss", (w + this.leftPadding)/2, h-pr);
	}
}

function create_canvas_with_pixel_ratio(w, h, pixelRatio) {
	const cv = document.createElement("canvas");
	cv.width = w * pixelRatio;
	cv.height = h * pixelRatio;

	cv.style.width = `${w}px`;
	cv.style.height = `${h}px`;
	return cv;
}

class LearningThrobber {
	constructor() {
		this.element = document.createElement("div");
		this.element.className = "throbber-container";
		this.imageList = [];

		this.addImage("./images/bb1.png");
		this.addImage("./images/bb2.png");
		this.addImage("./images/bb0.png");

		this.caption = document.createElement("span");
		this.caption.className = "throbber-caption";
		this.element.appendChild(this.caption);

		this.nowReady();
	}

	nowReady() {
		this.showDefaultFrame();
		this.caption.innerHTML = "Ready";
	}

	addImage(url) {
		const index = this.imageList.length;
		const img = document.createElement("img");
		img.src = url;
		img.className = "learning-throbber";
		img.style.zIndex = index;

		this.imageList.push(img);
		this.element.appendChild(img);
	}

	showDefaultFrame() {
		this.toggleVisibility(0, true);
		this.toggleVisibility(1, false);

		this.toggleAnimation(1, 1);
	}

	showThinkingFrame() {
		this.toggleVisibility(1, true);
		this.toggleVisibility(0, false);

		this.toggleAnimation(1, 0);
	}

	toggleVisibility(index, visible) {
		this.imageList[index].style.visibility = visible ? "" : "hidden";
	}

	toggleAnimation(index, animationSelector) {
		this.imageList[index].dataset.animation = animationSelector || 0;
	}
}

export { NNErrorLogChart, LearningThrobber };