function Bg(data) {
	this.x = data.x;
	this.y = data.y;
	this.res = data.res;
	this.useLudibriumSky = !!data.useLudibriumSky;

	this.width = this.res.width;
	this.height = this.res.height;

	// Pre-generate a starfield once so it doesn't twinkle randomly each frame.
	this._stars = null;
	this._buildStars = function() {
		var count = 140;
		var arr = [];
		// Seeded pseudo-random so the field is stable across frames.
		var seed = 0xDEADBEEF;
		var rand = function() { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; };
		for (var i = 0; i < count; i++) {
			arr.push({ rx: rand(), ry: rand(), r: rand() * 1.4 + 0.4, a: rand() * 0.7 + 0.25 });
		}
		this._stars = arr;
	};

	this.draw = function(ctx) {
		if (this.useLudibriumSky) {
			// Procedural Ludibrium-style night sky: deep navy → magenta gradient
			// from top to bottom, with a soft moon glow on the upper-left and
			// a stable starfield. Drawn in the canvas viewport so it doesn't
			// scroll with the map (parallax-static, like the real game's back
			// layer scrolling at speed 0).
			var W = window.WIDTH, H = window.HEIGHT;
			var grad = ctx.createLinearGradient(0, 0, 0, H);
			grad.addColorStop(0,    '#0d0738');
			grad.addColorStop(0.45, '#231464');
			grad.addColorStop(0.85, '#3b1a72');
			grad.addColorStop(1,    '#1a0f4d');
			ctx.fillStyle = grad;
			ctx.fillRect(0, 0, W, H);

			// Soft moon halo
			var moonX = W * 0.78, moonY = H * 0.22, moonR = Math.min(W, H) * 0.18;
			var halo = ctx.createRadialGradient(moonX, moonY, 0, moonX, moonY, moonR);
			halo.addColorStop(0,   'rgba(255, 240, 220, 0.55)');
			halo.addColorStop(0.4, 'rgba(255, 220, 180, 0.18)');
			halo.addColorStop(1,   'rgba(255, 220, 180, 0)');
			ctx.fillStyle = halo;
			ctx.fillRect(0, 0, W, H);
			// Moon disc
			ctx.beginPath();
			ctx.arc(moonX, moonY, moonR * 0.45, 0, Math.PI * 2);
			ctx.fillStyle = '#fff5dc';
			ctx.fill();

			// Starfield
			if (!this._stars) this._buildStars();
			for (var i = 0; i < this._stars.length; i++) {
				var s = this._stars[i];
				var x = s.rx * W;
				var y = s.ry * H * 0.7; // stars only in the upper 70%
				ctx.beginPath();
				ctx.arc(x, y, s.r, 0, Math.PI * 2);
				ctx.fillStyle = 'rgba(255, 245, 220, ' + s.a + ')';
				ctx.fill();
			}
			return;
		}
		ctx.drawImage(this.res, this.x, this.y, window.WIDTH, window.HEIGHT);
	}
}

function UI() {
	// main_bar PNG is 800×62. We anchor the whole HUD to the bottom-right corner.
	this.res = window.resource.ui["main_bar"][0];
	this.tip = {color: "white", tip: ""};

	this.update = function() {
		// Author/help tips removed.
	}

	this.draw = function(ctx) {
		var BAR_W = 800, BAR_H = 62;
		// Bottom-right anchor. All internal x offsets are relative to barX.
		var barX = window.WIDTH - BAR_W;
		var barY = window.HEIGHT - BAR_H;

		ctx.save();
		ctx.drawImage(this.res, barX, barY);

		// HP bar
		var hpGrad = ctx.createLinearGradient(barX + 255, window.HEIGHT - 32, barX + 255, window.HEIGHT - 12);
		hpGrad.addColorStop(0, 'rgba(255,0,0, 0.8)');
		hpGrad.addColorStop(1, 'rgba(255,99,71, 0.8)');
		ctx.fillStyle = hpGrad;
		ctx.roundRect(barX + 254, window.HEIGHT - 32, 137 * window.player_attr.curr_hp / window.player_attr.max_hp, 12, 3, true, false);

		// MP bar
		var mpGrad = ctx.createLinearGradient(barX + 423, window.HEIGHT - 32, barX + 423, window.HEIGHT - 12);
		mpGrad.addColorStop(0, 'rgba(65,105,225, 0.8)');
		mpGrad.addColorStop(1, 'rgba(100,149,237, 0.8)');
		ctx.fillStyle = mpGrad;
		ctx.roundRect(barX + 423, window.HEIGHT - 32, 137 * window.player_attr.curr_mp / window.player_attr.max_mp, 12, 3, true, false);

		// EXP bar
		var expGrad = ctx.createLinearGradient(barX + 254, window.HEIGHT - 15, barX + 254, window.HEIGHT - 3);
		expGrad.addColorStop(0, 'rgba(124,252,0, 0.8)');
		expGrad.addColorStop(1, 'rgba(255,215,0, 0.8)');
		ctx.fillStyle = expGrad;
		ctx.roundRect(barX + 254, window.HEIGHT - 15, 305 * window.player_attr.curr_exp / window.player_attr.max_exp, 12, 3, true, false);

		// Level number
		ctx.fillStyle = "yellow";
		ctx.textAlign = "left";
		ctx.textBaseline = "top";
		ctx.font = "24px liwen";
		ctx.fillText(window.player_attr.level, barX + 35, window.HEIGHT - 31);

		// Numeric HP/MP/EXP labels
		ctx.fillStyle = "white";
		ctx.font = "14px liwen";
		ctx.textAlign = "right";
		ctx.fillText("[" + window.player_attr.curr_hp + "/" + window.player_attr.max_hp + "]", barX + 390, window.HEIGHT - 33);
		ctx.fillText("[" + window.player_attr.curr_mp + "/" + window.player_attr.max_mp + "]", barX + 560, window.HEIGHT - 33);
		ctx.fillText(window.player_attr.curr_exp + "[" + (window.player_attr.curr_exp / window.player_attr.max_exp * 100).toFixed(2) + "%]", barX + 560, window.HEIGHT - 16);

		ctx.restore();
	}
}
