function GameScene(scene_obj, ctx) {
	if (typeof window !== 'undefined') window.__scene = this;
	window.music_manager.playMusic(scene_obj.music_src);

	this.equipment = scene_obj.equipment;
	this.backpack = scene_obj.backpack;
	this.ability = scene_obj.ability;
	this.backpack.equipment = this.equipment;
	this.ui = scene_obj.ui;

	this.player = scene_obj.player;
	this.map = scene_obj.map;
	this.bg = scene_obj.bg;

	this.doors = scene_obj.doors;
	this.npcs = scene_obj.npcs || [];
	this.ctx = ctx;

	this.is_finish = false;
	this.next_map;

	this.normal_monsters_stack = scene_obj.normal_monsters_stack;
	this.normal_monsters = [];
	this.curr_normal_stack = 0;

	this.skill_attack_monsters_stack = scene_obj.skill_attack_monsters_stack;
	this.skill_attack_monsters = [];
	this.curr_skill_effect_attack_stack = 0;

	this.skill_attack_monster_effects = [];
	this.skill_hit_monster = [];

	this.nos = [];

	this.ajust_speed_x = 0;
	this.ajust_speed_y = 0;

	this.skill_manager = new SkillManager(this.player, this.normal_monsters, this.skill_attack_monsters, this.map, this);
	this.skill_hit = [];

	this.things = [];
	this.tips = [];

	this.is_open_thing_window = scene_obj.is_open_thing_window;
	this.is_open_ability_window = scene_obj.is_open_ability_window;
	this.is_open_equipment_window = scene_obj.is_open_equipment_window;
	this.check_player = new CheckPlayer();

	this.level_up = null;

	this.keyDownEvent = function(event) {
		// Prevent browsers from focusing the menu bar (Alt), scrolling (Space/arrows),
		// or opening the find bar (slash).
		var pd = [18, 32, 37, 38, 39, 40, 68];
		if (pd.indexOf(event.keyCode) !== -1 && event.preventDefault) event.preventDefault();
		switch (event.keyCode) {
			case 16: // shift
				this.backpack.addMp();
				break;
			case 17: // ctrl
				this.backpack.addHp();
				break;
			case 50:
				this.is_open_thing_window = !this.is_open_thing_window;
				this.backpack.mouse_point.index = -1;
				this.backpack.select_point.index = -1;
				this.backpack.count = 0;
				break;
			case 51:
				this.is_open_ability_window = !this.is_open_ability_window;
				break;
			case 52:
				this.is_open_equipment_window = !this.is_open_equipment_window;
				this.equipment.mouse_point.index = -1;
				break;
			case 83:
				if (this.player.is_skill) return;
				this.player.is_thing = true;
				break;
			case 18: // jump (Alt — often eaten by the browser menu on Windows)
			case 32: // jump (Space — often eaten by page scroll / YouTube iframe)
			case 68: // jump (D — README's documented jump key, the reliable one)
				if (this.player.is_rope) {
					this.player.ropeJump();
				} else if (this.player.is_jump || this.player.is_fall) {
					return;
				} else {
					this.player.is_jump = true;
				}
				break;
			case 37: // left
				this.player.is_key_left_up = false;
				if (this.player.is_rope || this.player.is_skill) return;
				this.player.is_right = false;
				this.player.is_walk = true;
				break;
			case 38:
				if (this.player.is_skill) return;
				this.player.is_up = true;
				break;
			case 39: // right
				this.player.is_key_right_up = false;
				if (this.player.is_rope || this.player.is_skill) return;
				this.player.is_right = true;
				this.player.is_walk = true;
				break;
			case 40:
				if (this.player.is_skill) return;
				this.player.is_down = true;
				break;
			case 70: // g
				if (!this.player.is_skill && !this.player.is_rope) {
					this.skill_manager.preSkill("二连击", this.tips);
				}
				break;
			case 81: // q
				if (!this.player.is_skill && !this.player.is_rope) {
					this.skill_manager.preSkill("勇士的意志", this.tips);
				} 
				break;
			case 87: // w
				if (!this.player.is_skill && !this.player.is_rope) {
					this.skill_manager.preSkill("影子分身0", this.tips);
				}
				break;
			case 71:
				if (!this.player.is_skill && !this.player.is_rope) {
					this.skill_manager.preSkill("影舞瞬杀", this.tips);
				}
				break;
			case 82:
				if (!this.player.is_skill && !this.player.is_rope) {
					this.skill_manager.preSkill("回旋斩", this.tips);
				} 
				break;
			case 84:
				if (!this.player.is_skill && !this.player.is_rope ) { //&& this.checkLuoYeZhan()) {
					this.skill_manager.preSkill("落叶斩1", this.tips);
				} 
				break;
		}
	}

	this.keyUpEvent = function(event) {
		switch (event.keyCode) {
			case 83:
				this.player.is_thing = false;
				break;
			case 37: // left
				this.player.is_walk = false;
				this.player.is_key_left_up = true;
				break;
			case 38:
				this.player.is_up = false;
				break;
			case 39: // right
				this.player.is_walk = false;
				this.player.is_key_right_up = true;
				break;
			case 40:
				this.player.is_down = false;
				break;
		}
	}

	// Convert a pointer event to canvas-internal pixel coords, compensating for CSS scaling.
	function _canvasCoords(event) {
		var rect = window.canvas.getBoundingClientRect();
		var sx = window.canvas.width / rect.width;
		var sy = window.canvas.height / rect.height;
		return {
			x: (event.clientX - rect.left) * sx,
			y: (event.clientY - rect.top) * sy
		};
	}

	this.mouseMoveEvent = function(event) {
		var p = _canvasCoords(event);
		if (this.is_open_thing_window) {
			this.backpack.checkItemCollision(p.x, p.y);
		}
		if (this.is_open_equipment_window) {
			this.equipment.mouseHover(p.x, p.y);
		}
	}

	this.mouseDown = function(event) {
		// While an NPC dialog is open, suppress all canvas-level clicks so the
		// underlying NPC isn't re-triggered.
		if (window.npcDialogOpen) return;
		var p = _canvasCoords(event);
		if (this.is_open_thing_window) {
			this.backpack.changeType(p.x, p.y, true);
			this.backpack.checkItemSelect(p.x, p.y, true);
			this.backpack.checkMenuSelect(p.x, p.y, true);
		}

		if (this.is_open_ability_window) {
			this.ability.addPoint(p.x, p.y, true);
		}

		// Click on an interactive NPC → show dialog overlay
		for (var i in this.npcs) {
			var n = this.npcs[i];
			var r = n.rect;
			if (p.x >= r.x && p.x <= r.x + r.width &&
			    p.y >= r.y && p.y <= r.y + r.height) {
				if (typeof window.showNpcDialog === 'function') {
					window.showNpcDialog(n.name, n.role, n.dialog);
				}
				break;
			}
		}
	}

	this.getNextMap = function() {
		return next_map;
	}
	
	this.update = function() {
		this.checkMapCollision();
		this.checkDoorCollision();
		this.checkSkillAttackCollision();
		this.checkPlayerCollision();
		this.checkMonsterCollision();

		this.checkMonsterIsDie();

		this.checkPlayerSkill();

		this.genThings();

		this.removeItem();

		this.genMonsters();

		this.adjustXY();
		this.draw();
	}

	this.adjustXY = function() {
		if (this.player.x > window.WIDTH / 2  + this.player.width && this.map.isRightMovable()) {
			this.ajust_speed_x = (window.WIDTH / 2 - this.player.x) / 20;
			if (this.player.is_walk) {
				this.ajust_speed_x = -this.player.walk_speed;
			}
		} else if (this.player.x < window.WIDTH / 2 - this.player.width && this.map.isLeftMovable()) {
			this.ajust_speed_x = (window.WIDTH / 2 - this.player.x) / 20;
			if (this.player.is_walk) {
				this.ajust_speed_x = this.player.walk_speed;
			}
		} else {
			this.ajust_speed_x = 0;
		}

		if (this.player.y < window.HEIGHT / 3 && this.map.isUpMovable() ) {
			this.ajust_speed_y = (window.HEIGHT / 3 - this.player.y) / 15
			if (this.player.is_rope && this.player.is_up) {
				this.ajust_speed_y = this.player.rope_speed;
			}
		} else if (this.player.y > window.HEIGHT / 2 && this.map.isDownMovable()) {
			this.ajust_speed_y = (window.HEIGHT / 2 - this.player.y) / 15;
			if (this.player.is_rope && this.player.is_down) {
				this.ajust_speed_y = -this.player.rope_speed;
			}
		} else {
			this.ajust_speed_y = 0;
		}

		this.player.update(this.ajust_speed_x, this.ajust_speed_y);
		this.map.update(this.ajust_speed_x, this.ajust_speed_y);

		for (var i in this.doors) {
			this.doors[i].update(this.ajust_speed_x, this.ajust_speed_y);
		}

		// NPC hitboxes scroll with the map (their rect.x/y are world-relative)
		for (var i in this.npcs) {
			this.npcs[i].rect.x += this.ajust_speed_x;
			this.npcs[i].rect.y += this.ajust_speed_y;
		}

		// Position the YouTube iframe inside the Ludibrium Maple TV's bottom screen,
		// and the playback controls inside the TV's top screen.
		// Sprite is 426x414, anchored at NPC feet. Inner-dark rects measured by
		// walking outward from each screen's center to the pink border:
		//   bottom screen inner: sprite x=170..409, y=150..329  (240×180, 4:3)
		//   top    screen inner: sprite x=159..398, y= 25..114  (240×90,  ~21:8)
		// (The screens are offset right of sprite center because of the cables on
		//  the left side of the TV cabinet.)
		var tvIframe = window.document.getElementById('tv-screen');
		var tvCtrls  = window.document.getElementById('tv-controls');
		if (tvIframe) {
			var tvNpc = null;
			for (var j in this.npcs) {
				if (this.npcs[j].id === 9250026) { tvNpc = this.npcs[j]; break; }
			}
			if (tvNpc) {
				var feetX = tvNpc.rect.x + tvNpc.rect.width / 2;
				var feetY = tvNpc.rect.y + tvNpc.rect.height;
				var spriteLeft = feetX - 213;
				var spriteTop  = feetY - 414;
				// Bottom screen — 5 px inset on each side
				var bLeft = spriteLeft + 175;
				var bTop  = spriteTop  + 155;
				var bW    = 230;
				var bH    = 170;
				var onScreenB = (bLeft + bW > 0) && (bLeft < window.WIDTH) && (bTop + bH > 0) && (bTop < window.HEIGHT);
				if (onScreenB) {
					// Let the YouTube IFrame API initialise the iframe (it loads the
					// playlist and wires up postMessage so the controls work). Don't
					// set src manually — that breaks the API binding.
					if (!window.tvPlayer && typeof window.tvInitPlayer === 'function') {
						window.tvInitPlayer();
					}
					tvIframe.style.display = 'block';
					tvIframe.style.left   = bLeft + 'px';
					tvIframe.style.top    = bTop + 'px';
					tvIframe.style.width  = bW + 'px';
					tvIframe.style.height = bH + 'px';
				} else {
					tvIframe.style.display = 'none';
				}
				// Top screen — playback controls bar (Prev / Play / Next)
				if (tvCtrls) {
					var tLeft = spriteLeft + 164;
					var tTop  = spriteTop  + 30;
					var tW    = 229;
					var tH    = 79;
					if (onScreenB) {
						tvCtrls.classList.add('visible');
						tvCtrls.style.left   = tLeft + 'px';
						tvCtrls.style.top    = tTop + 'px';
						tvCtrls.style.width  = tW + 'px';
						tvCtrls.style.height = tH + 'px';
					} else {
						tvCtrls.classList.remove('visible');
					}
				}
			}
		}

		for (var i in this.normal_monsters) {
			this.normal_monsters[i].update(this.ajust_speed_x, this.ajust_speed_y);
		}

		for (var i in this.skill_attack_monsters) {
			this.skill_attack_monsters[i].update(this.ajust_speed_x, this.ajust_speed_y);
		}


		for (var i in this.normal_monsters_stack) {
			this.normal_monsters_stack[i].update(this.ajust_speed_x, this.ajust_speed_y);
		}

		for (var i in this.skill_attack_monsters_stack) {
			this.skill_attack_monsters_stack[i].update(this.ajust_speed_x, this.ajust_speed_y);
		}

		for (var i in this.skill_attack_monster_effects) {
			this.skill_attack_monster_effects[i].update(this.ajust_speed_x, this.ajust_speed_y);
		}

		for (var i in this.skill_hit_monster) {
			this.skill_hit_monster[i].update(this.player.x + this.player.width / 2, this.player.y + this.player.height / 2);
		}

		for (var i in this.nos) {
			this.nos[i].update(this.ajust_speed_x, this.ajust_speed_y);
		}

		if (!this.skill_manager.is_finish) {
			this.skill_manager.update();
		}

		for (var i in this.skill_hit) {
			this.skill_hit[i].update(this.ajust_speed_x, this.ajust_speed_y);
		}

		for (var i in this.things) {
			if (this.things[i].is_get) {
				this.things[i].tracePlayer(this.player.x + this.player.width / 2, this.player.y + 2 * this.player.height / 3, this.ajust_speed_x, this.ajust_speed_y);
			} else {
				this.things[i].update(this.ajust_speed_x, this.ajust_speed_y);
			}
		}

		for (var i in this.tips) {
			this.tips[i].update();
		}

		if (this.level_up) {
			this.level_up.update(this.player);
		}

		this.ui.update();
	}

	this.checkMapCollision = function() {
		var result = this.map.checkCollision(this.player.getMapRect(), this.player.is_down);
		// If the player is pressing UP/DOWN over a ladder/rope, prefer the
		// climb action over standing on a co-located foothold. Without this,
		// a ladder whose base overlaps the floor is unreachable from below
		// because is_floor short-circuits the dispatch.
		var grab_intent = this.player.is_up || this.player.is_down;
		if (grab_intent && result.is_ladder) {
			this.player.rope(result.ladder_item.rect.width / 2 + result.ladder_item.rect.x, true);
		} else if (grab_intent && result.is_rope) {
			this.player.rope(result.rope_item.rect.width / 2 + result.rope_item.rect.x, false);
		} else if (result.is_floor) {
			this.player.floor(result.foor_item.rect.y);
		} else if (result.is_rope) {
			this.player.rope(result.rope_item.rect.width / 2 + result.rope_item.rect.x, false);
		} else if (result.is_ladder) {
			this.player.rope(result.ladder_item.rect.width / 2 + result.ladder_item.rect.x, true);
		}

		if (result.is_stop && (this.player.is_right == result.stop_item.stop_right)) {
			this.player.is_stop = true;
		} else {
			this.player.is_stop = false;
		}

		if (!result.is_floor && !result.is_rope && !result.is_stop && !result.is_ladder) {
			this.player.fall();
		}

		for (var i in this.things) {
			if (this.things[i].checkMapCollision(this.map.map_items)) {
				this.things[i].float();
			}
		}
	}

	this.checkDoorCollision = function() {
		// Don't trigger doors while climbing — pressing Up next to a ladder
		// that happens to be near a portal would otherwise set is_finish and
		// snuff out the climb. The grab animation would flash for one frame
		// and the scene would start transitioning instead of letting you climb.
		if (this.player.is_rope) return;
		for(var i in this.doors) {
			if (this.doors[i].checkCollision(this.player.getMapRect(), this.player.is_up)) {
				this.is_finish = true;
				this.next_map = {position: this.doors[i].next_map, orientation: this.doors[i].orientation, is_open_thing_window: this.is_open_thing_window,
					is_open_ability_window: this.is_open_ability_window, is_open_equipment_window: this.is_open_equipment_window};
				this.player.is_up = false;
			}
		}
	}

	this.checkSkillAttackCollision = function() {
		for (var i in this.skill_attack_monsters) {
			var temp = this.skill_attack_monsters[i];
			if (temp.checkSkillAttackCollision(this.player.getRect(), this.player.x > temp.x) && !temp.is_attack) {
				temp.attack();
			}
			if (temp.is_attack && temp.attack_animation.getIsFlagFrame()) {
				this.skill_attack_monster_effects.push(window.monster_skill_effect_factory.getSkillEffect(temp, this.player.x + this.player.width / 2, this.player.y + this.player.height / 2));  
			}
		}
	}

	this.checkPlayerCollision = function() {
		for (var i in this.normal_monsters) {
			if (this.normal_monsters[i].checkCollision(this.player.getRect(), this.player.can_hit && !this.player.is_skill)) {
				this.player.hit();
				var attack = this.normal_monsters[i].physicsAttack(window.player_attr.defense);
				window.player_attr.curr_hp -= attack;
				if (window.player_attr.curr_hp < 0) window.player_attr.curr_hp = 0;
				this.nos.push(window.number_factory.getNumber(window.VIOLET, attack, this.player.x + this.player.width / 2, this.player.y, 1));
			}
		}

		for (var i in this.skill_attack_monsters) {
			if (this.skill_attack_monsters[i].checkCollision(this.player.getRect(), this.player.can_hit && !this.player.is_skill)) {
				this.player.hit();
				var attack = this.skill_attack_monsters[i].physicsAttack(window.player_attr.defense);
				window.player_attr.curr_hp -= attack;
				if (window.player_attr.curr_hp < 0) window.player_attr.curr_hp = 0;
				this.nos.push(window.number_factory.getNumber(window.VIOLET, attack, this.player.x + this.player.width / 2, this.player.y, 1));
			}
		}

		for (var i in this.skill_attack_monster_effects) {
			if (this.skill_attack_monster_effects[i].checkCollision(this.player.getRect(), this.player.can_hit && !this.player.is_skill)) {
				this.player.hit();
				var magic_attack = this.skill_attack_monster_effects[i].magicAttack(window.player_attr.magic_defense);
				window.player_attr.curr_hp -= magic_attack;
				if (window.player_attr.curr_hp < 0) window.player_attr.curr_hp = 0;

				this.skill_hit_monster.push(window.monster_skill_hit_factory.getSkillHit(this.skill_attack_monster_effects[i].name, this.player.x + this.player.width / 2, this.player.y + this.player.height / 2));
				this.nos.push(window.number_factory.getNumber(window.VIOLET, magic_attack, this.player.x + this.player.width / 2, this.player.y, 1));
				this.skill_attack_monster_effects.splice(i, 1);
			}
		}

		var is_has_add = false;
		for (var i in this.things) {
			if (!this.player.is_get && !this.things[i].is_get && this.things[i].checkPlayerCollision(this.player.getRect(), this.player.is_thing)) {
				if (this.things[i].type == 0) {
					this.things[i].is_get = true;
					this.player.is_get = true;
					break;
				} else {
					if (this.backpack.checkCanAdd(this.things[i])) {
						this.things[i].is_get = true;
						this.player.is_get = true;
						break;
					} else {
						if (is_has_add == false) {
							window.tips_factory.getTip(this.things[i], 2, this.tips);
							is_has_add = true;
						}
						//this.player.is_thing = false;
					}
				}
			}
		}
		if (is_has_add) {
			this.player.is_thing = false;
		}
	}

	this.checkPlayer = function() {
		if(this.check_player.checkLevelUp()) {
			this.check_player.levelUp();
			this.level_up = new LevelUpEffect(this.player);
		}
	}

	this.checkMonsterCollision = function() {
		if (this.skill_manager.is_finish) return;
		var num_items = this.skill_manager.checkMonsterCollision();

		for (var i in num_items) {
			var skill_hurt = this.skill_manager.effect[0].skillAttack(window.monsters_attr[num_items[i].monster.name].defense); 
			this.nos.push(window.number_factory.getNumber(skill_hurt.power_hit ? window.CRI : window.RED, skill_hurt.attack, num_items[i].monster.x + num_items[i].monster.width / 2, num_items[i].monster.ay - num_items[i].monster.height, num_items[i].level));
			num_items[i].monster.hit(skill_hurt.attack);
			this.skill_hit.push(window.skill_hit_factory.getSkillHit(this.skill_manager.name, this.player.is_right, num_items[i].monster));
		}
	}

	this.checkMonsterIsDie = function() {
		for (var i in this.normal_monsters) {
			this.normal_monsters[i].checkCanDie();
		}
		for (var i in this.skill_attack_monsters) {
			this.skill_attack_monsters[i].checkCanDie();
		}
	}

	this.checkPlayerSkill = function() {
		if (this.skill_manager.is_start) {
			this.skill_manager.startSkill();
		}
	}

	this.removeItem = function() {
		for (var i in this.skill_attack_monster_effects) {
			if(this.skill_attack_monster_effects[i].is_finish) {
				this.skill_attack_monster_effects.splice(i, 1);
			}
		}

		for (var i in this.skill_hit_monster) {
			if (this.skill_hit_monster[i].getIsFinish()) {
				this.skill_hit_monster.splice(i, 1);
			}
		}

		for (var i in this.nos) {
			if (this.nos[i].is_finish) {
				this.nos.splice(i, 1);
			}
		}

		for (var i in this.skill_attack_monsters) {
			var monster = this.skill_attack_monsters[i];
			if (monster.is_finish) {
				window.tips_factory.getTip(this.skill_attack_monsters[i], 0, this.tips);
				window.player_attr.curr_exp += window.monsters_attr[this.skill_attack_monsters[i].name].exp;
				this.skill_attack_monsters_stack.push(new SkillAttackMonstersStackItem(monster.ax, monster.ay, monster.awidth, monster.count, monster.name));
				this.skill_attack_monsters.splice(i, 1);
			}
		}

		for (var i in this.normal_monsters) {
			var monster = this.normal_monsters[i];
			if (monster.is_finish) {
				window.tips_factory.getTip(this.normal_monsters[i], 0, this.tips);
				window.player_attr.curr_exp += window.monsters_attr[this.normal_monsters[i].name].exp;
				var temp = new NormalMonstersStackItem(monster.ax, monster.ay, monster.awidth, monster.count, monster.name);
				this.normal_monsters_stack.push(temp);
				this.normal_monsters.splice(i, 1);
			}
		}
		this.checkPlayer();

		if (!this.skill_manager.is_finish && this.skill_manager.getIsEnd()) {
			this.skill_manager.endSkill();
		}

		for (var i in this.skill_hit) {
			if (this.skill_hit[i].getIsFinish()) {
				this.skill_hit.splice(i, 1);
			}
		}

		for (var i in this.things) {
			if (this.things[i].is_finish) {
				window.tips_factory.getTip(this.things[i], 1, this.tips);
				if (this.things[i].type == 0) {
					window.player_attr.money += this.things[i].money;
				} else {
					this.backpack.add(this.things[i]);
				}
				this.things.splice(i, 1);
				this.player.is_get = false;
			} else if (this.things[i].is_time_finish) {
				this.things.splice(i, 1);
			}
		}

		for (var i in this.tips) {
			if (this.tips[i].is_finish) {
				this.tips.splice(i, 1);
			}
		}

		if (this.level_up != null && this.level_up.is_finish) {
			this.level_up = null;
		}
	}

	this.genMonsters = function() {
		if (this.normal_monsters_stack.length > 0) {
			this.normal_monsters_stack[this.curr_normal_stack].count++;
			if (this.normal_monsters_stack[this.curr_normal_stack].count > window.WAIT_FRAME) {
				this.normal_monsters.push(window.monster_factory.getNormalMonster(this.normal_monsters_stack[this.curr_normal_stack]));
				this.normal_monsters_stack.splice(this.curr_normal_stack, 1);
				this.curr_normal_stack = parseInt(Math.random() * this.normal_monsters_stack.length);
			}
		}
		if (this.skill_attack_monsters_stack.length > 0) {
			this.skill_attack_monsters_stack[this.curr_skill_effect_attack_stack].count++;
			if (this.skill_attack_monsters_stack[this.curr_skill_effect_attack_stack].count > window.WAIT_FRAME) {
				this.skill_attack_monsters.push(window.monster_factory.getSkillAttackMonster(this.skill_attack_monsters_stack[this.curr_skill_effect_attack_stack]));
				this.skill_attack_monsters_stack.splice(this.curr_skill_effect_attack_stack, 1);
				this.curr_skill_effect_attack_stack = parseInt(Math.random() * this.skill_attack_monsters_stack.length);
			}
		}
	}

	this.genThings = function() {
		for (var i in this.normal_monsters) {
			if (this.normal_monsters[i].is_die && this.normal_monsters[i].die_animation.getIsFlagFrame()) {
				window.things_factory.getThings(this.normal_monsters[i], this.things);
			}
		}

		for (var i in this.skill_attack_monsters) {
			if (this.skill_attack_monsters[i].is_die && this.skill_attack_monsters[i].die_animation.getIsFlagFrame()) {
				window.things_factory.getThings(this.skill_attack_monsters[i], this.things);
			}
		}

	}

	this.draw = function() {
		this.ctx.clearRect(0, 0, window.WIDTH, window.HEIGHT);
		this.bg.draw(this.ctx);
		this.map.draw(this.ctx);

		for (var i in this.normal_monsters) {
			this.normal_monsters[i].draw(ctx);
		}

		for (var i in this.skill_attack_monsters) {
			this.skill_attack_monsters[i].draw(ctx);
		}

		for (var i in this.skill_attack_monster_effects) {
			this.skill_attack_monster_effects[i].draw(ctx);
		}

		// === Husky pet — trails the player like a real MapleStory pet ===
		if (!window._huskyImgs) {
			window._huskyImgs = {};
			['stand0', 'stand1', 'move', 'jump'].forEach(function(a){
				var img = new Image(); img.src = 'pet/husky_' + a + '.png';
				window._huskyImgs[a] = img;
			});
			window._petPos = { x: this.player.x - 50, y: this.player.y + this.player.height, faceRight: false };
		}
		var pet = window._petPos;
		var playerFootY = this.player.y + this.player.height;
		var dx = this.player.x - pet.x;
		var absDx = Math.abs(dx);
		var FOLLOW_GAP = 90; // start running once farther than this
		var MAX_SPEED = 7;
		// Always face toward the player — even when idle — so the husky turns when
		// the player runs around it. ("not turning anymore" was caused by only
		// updating facing inside the moving branch.)
		if (absDx > 5) pet.faceRight = dx > 0;
		var moving = absDx > FOLLOW_GAP;
		if (moving) {
			var step = Math.min(MAX_SPEED, absDx - FOLLOW_GAP + 2);
			pet.x += Math.sign(dx) * step;
		}
		// Vertical: pet sticks to the floor (player's feet line). Smooth easing for jumps.
		pet.y += (playerFootY - pet.y) * 0.25;
		// Pick frame: cycle move↔stand0 quickly while running for a paw-stride
		// animation, alternate stand0↔stand1 slowly while idle, and add a small
		// vertical bob so the run looks alive.
		var petAction;
		var petBob = 0;
		if (moving) {
			petAction = (Math.floor(Date.now() / 130) % 2 === 0) ? 'move' : 'stand0';
			petBob = Math.abs(Math.sin(Date.now() / 80)) * 3; // little hop
		} else {
			petAction = (Math.floor(Date.now() / 600) % 2 === 0) ? 'stand0' : 'stand1';
		}
		var petImg = window._huskyImgs[petAction];
		if (petImg && petImg.complete && petImg.naturalWidth > 0) {
			var px = pet.x - petImg.naturalWidth / 2;
			var py = pet.y - petImg.naturalHeight - petBob;
			if (pet.faceRight && ctx.drawRightImage) {
				ctx.drawRightImage(petImg, px, py);
			} else {
				ctx.drawImage(petImg, px, py);
			}
		}

		this.player.draw(this.ctx);

		// === Player name tag — "Zizhao" above the head ===
		var pnX = this.player.x + this.player.width / 2;
		var pnY = this.player.y - 6;
		ctx.save();
		ctx.font = 'bold 12px Arial, sans-serif';
		ctx.textAlign = 'center';
		ctx.textBaseline = 'bottom';
		ctx.lineWidth = 3;
		ctx.strokeStyle = 'rgba(0,0,0,0.85)';
		ctx.strokeText('Zizhao', pnX, pnY);
		ctx.fillStyle = '#ffffff';
		ctx.fillText('Zizhao', pnX, pnY);
		ctx.restore();

		for (var i in this.things) {
			this.things[i].draw(ctx);
		}

		for (var i in this.skill_hit_monster) {
			this.skill_hit_monster[i].draw(ctx);
		}

		for (var i in this.skill_hit) {
			this.skill_hit[i].draw(ctx);
		}

		if (!this.skill_manager.is_finish) {
			for (var i in this.skill_manager.effect) {
				this.skill_manager.effect[i].draw(ctx);
			}
		}

		if (this.level_up) {
			this.level_up.draw(ctx);
		}

		for (var i in this.nos) {
			this.nos[i].draw(ctx);
		}

		for (var i in this.doors) {
			this.doors[i].draw(ctx);
		}

		// Interactive NPCs — draw sprite + name label + blinking "!" marker
		var npcBlink = (Math.floor(Date.now() / 250) % 2 === 0) ? 1 : 0.55;
		ctx.save();
		for (var i in this.npcs) {
			var n = this.npcs[i];
			var cx = n.rect.x + n.rect.width / 2;
			var feetY = n.rect.y + n.rect.height;
			// Skip if obviously off-screen
			if (cx < -80 || cx > window.WIDTH + 80) continue;

			// 1. NPC sprite (anchored at feet, centered horizontally)
			var sprite = n.sprite;
			var spriteH = 0;
			if (sprite && sprite.complete && sprite.naturalWidth > 0) {
				ctx.drawImage(sprite, cx - sprite.naturalWidth / 2, feetY - sprite.naturalHeight);
				spriteH = sprite.naturalHeight;
			}
			// 2. Name label — black box with white name above the sprite head
			var nameY = feetY - Math.max(spriteH, 60) - 8;
			ctx.font = "bold 12px sans-serif";
			ctx.textAlign = "center";
			ctx.textBaseline = "middle";
			var nameW = ctx.measureText(n.name).width + 10;
			ctx.fillStyle = "rgba(0,0,0,0.78)";
			ctx.fillRect(cx - nameW / 2, nameY - 9, nameW, 18);
			ctx.strokeStyle = "rgba(255,235,80,0.7)";
			ctx.lineWidth = 1;
			ctx.strokeRect(cx - nameW / 2, nameY - 9, nameW, 18);
			ctx.fillStyle = "white";
			ctx.fillText(n.name, cx, nameY + 1);

			// 3. Blinking "!" speech marker just above the name plate
			var markY = nameY - 22;
			ctx.fillStyle = "rgba(255,235,80," + npcBlink + ")";
			ctx.strokeStyle = "rgba(0,0,0,0.8)";
			ctx.lineWidth = 1.5;
			ctx.beginPath();
			ctx.arc(cx, markY, 9, 0, Math.PI * 2);
			ctx.fill();
			ctx.stroke();
			ctx.fillStyle = "black";
			ctx.font = "bold 13px sans-serif";
			ctx.fillText("!", cx, markY + 1);
		}
		ctx.restore();

		for (var i in this.tips) {
			this.tips[i].draw(ctx);
		}

		if (this.is_open_ability_window) {
			this.ability.draw(ctx);
		}

		if (this.is_open_thing_window) {
			this.backpack.draw(ctx);
		}

		if (this.is_open_equipment_window) {
			this.equipment.draw(ctx);
		}

		this.ui.draw(ctx);
	}
}
