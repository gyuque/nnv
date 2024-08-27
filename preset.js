"use strict";

const Presets = {
	basic: {
		aug: false,
		tx: "Identity",
		lr: 0.045,
		momentum: 0,
		cth: 0.01
	},

	advanced: {
		aug: true,
		tx: "LeakyReLU",
		lr: 0.053,
		momentum: 0.7,
		cth: 0.045
	}
};

class PresetPopup {
	constructor(trigger_id, popup_id, callback) {
		this.body = document.body;
		this.triggerElement = document.getElementById(trigger_id);
		this.triggerElement.addEventListener("click", this.onTriggerClick.bind(this), false);

		this.popupElement = document.getElementById(popup_id);
		this.popupElement.addEventListener("click", this.onPopupClick.bind(this), false);

		this.body.addEventListener("click", this.onBodyClick.bind(this), false);
		this.scan(this.popupElement);

		this.onSelectCallback = callback || null;
	}

	scan(parent) {
		const ls = parent.getElementsByTagName("a");
		const n = ls.length;

		for (let i = 0;i < n;++i) {
			this.registerLink(ls[i]);
		}
	}

	registerLink(a) {
		a.addEventListener("click", this.onItemClick.bind(this, a), false);
	}

	onTriggerClick(e) {
		this.toggle();
		e.stopPropagation();
	}

	onBodyClick(_e) {
		this.toggle(true);
	}

	onPopupClick(e) {
		e.stopPropagation();
	}

	toggle(closeOnly) {
		const newVal = 1 - get_data_opened(this.body)
		if (closeOnly && newVal) { return; }

		this.body.dataset.popen = newVal;
		setTimeout( () => {
			this.popupElement.style.opacity = newVal;
		}, 10);
	}

	onItemClick(sender, _e) {
		if (sender.dataset && sender.dataset.name) {
			const presetItem = Presets[sender.dataset.name];
			if (presetItem && this.onSelectCallback) {
				this.onSelectCallback(presetItem);
			}
		}

		this.toggle(true);
	}
}

function get_data_opened(el) {
	const v = el.dataset.popen - 0;
	return (v === 1) ? 1 : 0;
}

export { PresetPopup };