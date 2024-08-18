"use strict";

const kEColor = "#F55";

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

		g.fillStyle = "#eee";
		this.drawXaxis(g, h-this.bottomPadding, w, pr);
		this.drawYaxis(g, this.leftPadding, h, pr);
		this.drawTitle(g, w, h, pr);

		this.plotAll(g, w, h, pr);
	}

	plotAll(g, canvasWidth, canvasHeight, pixelRatio) {
		const lastIndex = this.seriesList.length-1;
		let i = 0;
		g.fillStyle = kEColor;
		for (const sr of this.seriesList) {
			const isLast = (i === lastIndex);
			g.strokeStyle = isLast ? kEColor : "#955";
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
		this.addImage("./images/bb3.png", "throbber-sweat-3");

		this.caption = document.createElement("span");
		this.caption.className = "throbber-caption";
		this.element.appendChild(this.caption);

		this.popout = document.createElement("span");
		this.popout.className = "result-popout-outer";
		this.element.appendChild(this.popout);

		this.popoutLabel = document.createElement("span");
		this.popout.appendChild(this.popoutLabel);

		this.nowReady();
	}

	setPanic(level) {
		this.element.dataset.panic = level;
		if (level > 0) {
			this.caption.innerHTML = "Oh! busy!";
		}
	}

	setGlitter(type) {
		this.caption.dataset.glitter = type;
	}

	setPopout(level) {
		this.element.dataset.popout = level;
	}

	nowReady() {
		this.showDefaultFrame();
		this.caption.innerHTML = "Ready";
		this.setGlitter(0);
		this.setPanic(0);
		this.setPopout(0);
	}

	nowComplete() {
		this.showDefaultFrame();
		this.caption.innerHTML = "Complete!";
		this.setGlitter(1);
		this.setPanic(0);
		this.setPopout(0);
	}

	nowInferred(label, ambiguous) {
		this.showDefaultFrame();
		this.caption.innerHTML = "Inferred.";
		this.setGlitter(0);
		this.setPanic(0);

		if (ambiguous) {
			label += "…?";
		}

		this.setPopout(0);
		this.popoutLabel.innerHTML = "";
		this.popoutLabel.appendChild( document.createTextNode(label) );
		setTimeout( () => { this.setPopout(ambiguous ? 2 : 1); } , 20);
	}

	addImage(url, additionalClassName) {
		let cls = "learning-throbber";
		if (additionalClassName) {
			cls += " " + additionalClassName;
		}

		const index = this.imageList.length;
		const img = document.createElement("img");
		img.src = url;
		img.className = cls;
		img.style.zIndex = index;

		this.imageList.push(img);
		this.element.appendChild(img);
	}

	showDefaultFrame() {
		this.toggleVisibility(0, true);
		this.toggleVisibility(1, false);

		this.toggleAnimation(1, 0);
	}

	showThinkingFrame() {
		this.toggleVisibility(1, true);
		this.toggleVisibility(0, false);

		this.toggleAnimation(1, 1);
		this.caption.innerHTML = "Learning...";
		this.setGlitter(2);
		this.setPopout(0);
	}

	toggleVisibility(index, visible) {
		this.imageList[index].style.visibility = visible ? "" : "hidden";
	}

	toggleAnimation(index, animationSelector) {
		this.imageList[index].dataset.animation = animationSelector || 0;
	}
}

class ButtonManager {
	constructor(container_id, callback) {
		this.commandMap = {};

		const containerElement = document.getElementById(container_id);
		this.scan(containerElement);
		this.clickCallback = callback;
	}

	scan(containerElement) {
		const ls = containerElement.getElementsByTagName("button");
		const n = ls.length;
		for (let i = 0;i < n;++i) {
			const btn = ls[i];
			btn.addEventListener("click", this.onClick.bind(this, btn), false);

			this.commandMap[ get_button_command(btn) ] = btn;
		}
	}

	onClick(button) {
		const cmd = get_button_command(button);

		if (this.clickCallback) {
			this.clickCallback(this, button, cmd);
		}
	}

	setDisabled(cmd, disabled) {
		const b = this.commandMap[cmd];
		if (!b) { return; }

		b.disabled = disabled;
	}
}

class TabPager {
	constructor(container_id) {
		const containerElement = document.getElementById(container_id);
		this.tabElements = [];
		this.pageElements = [];

		this.scan(containerElement);
	}

	scan(containerElement) {
		const tabstrip = containerElement.getElementsByClassName("pager-tab");

		this.pushChildElements(this.tabElements, tabstrip[0], "li").forEach( (li, li_index) => {
			li.addEventListener("click", this.onTabClick.bind(this, li, li_index));
		} );

		this.pushChildElements(this.pageElements, containerElement, "div", "paged-content");
	}

	pushChildElements(outArray, parent, tagName, withClass) {
		const ls = parent.getElementsByTagName(tagName);
		const n = ls.length;
		for (let i = 0;i < n;++i) {
			if (withClass) {
				if (ls[i].className !== withClass) { continue; }
			}

			outArray.push(ls[i]);
		}

		return outArray;
	}

	selectByIndex(selIndex) {
		const n = this.tabElements.length;
		for (let i = 0;i < n;++i) {
			const sel_flag = (i === selIndex) ? 1 : 0;
			this.tabElements[i].dataset.selected = sel_flag;

			const pg = this.pageElements[i];
			if (pg) {
				pg.style.display = sel_flag ? "block" : "none";
			}
		}
	}

	onTabClick(_element, index, _e) {
		this.selectByIndex(index);
	}
}

function get_button_command(el) { return el.dataset.command; }

export { NNErrorLogChart, LearningThrobber, ButtonManager, TabPager };