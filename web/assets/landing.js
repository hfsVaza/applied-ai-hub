/* ============================================================================
   AI Projects Hub — landing page behaviour
   ========================================================================= */
(() => {
  'use strict';

  // Opt in to the JS-only hidden states only now that we know JS runs.
  document.documentElement.classList.add('js');

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const fine    = window.matchMedia('(pointer: fine)').matches;
  const $  = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

  /* ======================================================= boot sequence */

  const BOOT = [
    'initialising hub',
    'probing compute device',
    'four modules registered',
    'awaiting operator',
  ];

  function runBoot() {
    const el = $('#boot');
    if (!el) return;
    const cursor = '<span class="boot__cursor"></span>';

    // Returning within the session skips straight to the end: a boot
    // animation is charming once and tedious on every navigation back.
    if (reduced || sessionStorage.getItem('booted')) {
      el.innerHTML = `> ${BOOT[BOOT.length - 1]} ${cursor}`;
      return;
    }
    sessionStorage.setItem('booted', '1');

    let line = 0;
    const nextLine = () => {
      if (line >= BOOT.length) { el.innerHTML = `> ${BOOT[BOOT.length - 1]} ${cursor}`; return; }
      const text = BOOT[line++];
      let i = 0;
      const type = () => {
        el.innerHTML = `> ${text.slice(0, ++i)} ${cursor}`;
        if (i < text.length) setTimeout(type, 16 + Math.random() * 22);
        else setTimeout(nextLine, 430);
      };
      type();
    };
    nextLine();
  }

  /* ================================================== hero point cloud */

  /* A rotating, perspective-projected point cloud — the same operation the
     depth project performs when it back-projects a depth map into geometry,
     running live behind the title. Points are shaded by distance, so the
     structure reads as depth rather than as scattered dots. */
  function heroField() {
    const canvas = $('#field');
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { alpha: true });

    // Rows of a height field, stroked as polylines and drawn back to front.
    // This is the same operation the depth project performs when it turns a
    // depth map into geometry; as contour lines the relief is legible at a
    // glance, where a cloud of equally-bright points is not.
    const ROWS = 46;
    const COLS = 150;

    let w = 0, h = 0, dpr = 1;
    const pointer = { x: 0, y: 0, tx: 0, ty: 0 };

    function size() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.clientWidth; h = canvas.clientHeight;
      if (!w || !h) return;
      canvas.width = w * dpr; canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function height(u, v, t) {
      return (
        Math.sin(u * 3.0 + t * 0.7) * 0.34 +
        Math.cos(v * 2.4 - t * 0.5) * 0.28 +
        Math.sin((u + v) * 1.9 + t * 0.35) * 0.22 +
        Math.sin(Math.hypot(u, v) * 4.2 - t * 1.1) * 0.20
      );
    }

    let t = 0, raf = 0;

    function frame() {
      t += 0.0055;
      ctx.clearRect(0, 0, w, h);

      pointer.x += (pointer.tx - pointer.x) * 0.05;
      pointer.y += (pointer.ty - pointer.y) * 0.05;

      const yaw   = Math.sin(t * 0.55) * 0.34 + pointer.x * 0.26;
      const pitch = -0.52 + pointer.y * 0.12;
      const cy = Math.cos(yaw),   sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);

      const cx = w * 0.52, cyc = h * 0.42;
      const scale = Math.max(w, h) * 0.52;

      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';

      // Far rows first, so nearer ridges overlap and hide what is behind them.
      for (let iy = 0; iy < ROWS; iy++) {
        const v = (iy / (ROWS - 1) - 0.5) * 2.6;

        let started = false;
        let nearest = 0;
        ctx.beginPath();

        for (let ix = 0; ix < COLS; ix++) {
          const u = (ix / (COLS - 1) - 0.5) * 4.2;
          const z = height(u, v, t);

          const x1 =  u * cy + z * sy;
          const z1 = -u * sy + z * cy;
          const y2 =  v * cp - z1 * sp;
          const z2 =  v * sp + z1 * cp;

          const depth = z2 + 3.0;
          if (depth <= 0.35) { started = false; continue; }

          const k = 2.4 / depth;
          const sx = cx + x1 * k * scale;
          const syy = cyc + y2 * k * scale;

          nearest = Math.max(nearest, clamp((3.5 - depth) / 1.5, 0, 1));
          if (!started) { ctx.moveTo(sx, syy); started = true; }
          else ctx.lineTo(sx, syy);
        }

        // Rows closer to the camera are brighter and pick up the signal
        // colour, so the crest of the surface reads as the live scan edge.
        const eased = nearest * nearest;
        const alpha = 0.06 + eased * 0.74;
        ctx.strokeStyle = nearest > 0.60
          ? `rgba(55,226,240,${alpha})`
          : `rgba(150,170,190,${alpha * 0.72})`;
        ctx.lineWidth = 0.7 + eased * 0.9;
        ctx.stroke();
      }

      raf = requestAnimationFrame(frame);
    }

    size();
    requestAnimationFrame(size);

    if (reduced) { size(); frame(); cancelAnimationFrame(raf); raf = 0; }
    else raf = requestAnimationFrame(frame);

    window.addEventListener('resize', size, { passive: true });

    if (fine && !reduced) {
      window.addEventListener('pointermove', (e) => {
        pointer.tx = (e.clientX / window.innerWidth) * 2 - 1;
        pointer.ty = (e.clientY / window.innerHeight) * 2 - 1;
      }, { passive: true });
    }

    // Stop drawing entirely when the hero scrolls away or the tab is hidden.
    const io = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !raf && !reduced) raf = requestAnimationFrame(frame);
      else if (!entry.isIntersecting && raf) { cancelAnimationFrame(raf); raf = 0; }
    }, { threshold: 0 });
    io.observe(canvas);

    document.addEventListener('visibilitychange', () => {
      if (document.hidden && raf) { cancelAnimationFrame(raf); raf = 0; }
      else if (!document.hidden && !raf && !reduced) raf = requestAnimationFrame(frame);
    });
  }

  /* ================================================ card visualisations */

  /* Each card carries a small live diagram of what its project actually does.
     They animate only while the pointer is over the card, so four extra
     canvases cost nothing at rest. */
  const VIZ = {
    faces(ctx, w, h, t, hot) {
      const cols = 11, rows = 5;
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const px = (x + 0.5) * (w / cols);
          const py = (y + 0.5) * (h / rows);
          // One cell is the "face": it blurs out as the effect sweeps past.
          const isFace = x === 4 && y === 2;
          const sweep = (Math.sin(t * 1.6) * 0.5 + 0.5) * cols;
          const blurred = isFace && sweep > x;
          ctx.fillStyle = blurred
            ? 'rgba(55,226,240,0.16)'
            : isFace ? 'rgba(55,226,240,0.62)' : 'rgba(150,170,186,0.14)';
          const s = blurred ? 9 : 3;
          ctx.fillRect(px - s / 2, py - s / 2, s, s);
        }
      }
    },

    depth(ctx, w, h, t, hot) {
      // A depth ramp with contour bands sliding through it.
      const bands = 26;
      for (let i = 0; i < bands; i++) {
        const f = i / bands;
        const y = f * h;
        const shift = Math.sin(t * 0.9 + f * 5) * 6;
        const a = 0.05 + (1 - f) * 0.3;
        ctx.fillStyle = `rgba(167,139,250,${a})`;
        ctx.fillRect(0, y + shift, w, h / bands * 0.62);
      }
    },

    pose(ctx, w, h, t, hot) {
      // A stick figure performing a curl, driven by the same angle logic.
      const cx = w / 2, cy = h / 2;
      const s = h / 3.1;
      const curl = (Math.sin(t * 1.5) * 0.5 + 0.5);
      const elbow = -0.35 + curl * 1.5;

      const joints = {
        head:  [cx, cy - s * 1.15],
        neck:  [cx, cy - s * 0.62],
        hip:   [cx, cy + s * 0.42],
        sh:    [cx - s * 0.42, cy - s * 0.52],
        el:    [cx - s * 0.62, cy + s * 0.04],
        wr:    [cx - s * 0.62 + Math.cos(elbow) * s * 0.58,
                cy + s * 0.04 - Math.sin(elbow) * s * 0.58],
        knee:  [cx - s * 0.2, cy + s * 1.0],
        foot:  [cx - s * 0.24, cy + s * 1.5],
      };
      const bones = [['neck','hip'],['neck','sh'],['sh','el'],['el','wr'],['hip','knee'],['knee','foot']];

      ctx.strokeStyle = 'rgba(61,220,151,0.55)';
      ctx.lineWidth = 1.4; ctx.lineCap = 'round';
      for (const [a, b] of bones) {
        ctx.beginPath();
        ctx.moveTo(...joints[a]); ctx.lineTo(...joints[b]); ctx.stroke();
      }
      ctx.strokeStyle = 'rgba(61,220,151,0.75)';
      ctx.beginPath();
      ctx.arc(joints.head[0], joints.head[1], s * 0.2, 0, 6.283); ctx.stroke();

      ctx.fillStyle = 'rgba(61,220,151,0.95)';
      for (const k of ['sh','el','wr','hip','knee']) {
        ctx.beginPath(); ctx.arc(joints[k][0], joints[k][1], 1.9, 0, 6.283); ctx.fill();
      }
    },

    search(ctx, w, h, t, hot) {
      // Scattered documents drawn toward their cluster centres.
      if (!VIZ._docs) {
        VIZ._docs = Array.from({ length: 30 }, (_, i) => ({
          g: i % 3,
          x: Math.random(), y: Math.random(),
          s: Math.random() * 6.28,
        }));
      }
      const centres = [[0.24, 0.42], [0.55, 0.62], [0.82, 0.36]];
      const pull = hot ? (Math.sin(t * 0.8) * 0.5 + 0.5) : 0.18;

      for (const d of VIZ._docs) {
        const [gx, gy] = centres[d.g];
        const x = (d.x + (gx - d.x) * pull) * w + Math.sin(t + d.s) * 1.6;
        const y = (d.y + (gy - d.y) * pull) * h + Math.cos(t + d.s) * 1.6;
        ctx.fillStyle = ['rgba(242,177,52,0.66)','rgba(242,177,52,0.4)','rgba(242,177,52,0.52)'][d.g];
        ctx.fillRect(x - 1.6, y - 2.1, 3.2, 4.2);
      }
    },
  };

  function cardVisuals() {
    $$('[data-viz]').forEach((card) => {
      const canvas = $('canvas', card);
      const draw = VIZ[card.dataset.viz];
      if (!canvas || !draw) return;
      const ctx = canvas.getContext('2d');
      let w = 0, h = 0, raf = 0, t = Math.random() * 10, hot = false;

      const size = () => {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        w = canvas.clientWidth; h = canvas.clientHeight;
        if (!w || !h) return;
        canvas.width = w * dpr; canvas.height = h * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      };

      const render = () => {
        ctx.clearRect(0, 0, w, h);
        draw(ctx, w, h, t, hot);
      };

      const loop = () => { t += 0.016; render(); raf = requestAnimationFrame(loop); };
      const start = () => { if (!raf && !reduced) { hot = true; raf = requestAnimationFrame(loop); } };
      const stop  = () => { if (raf) { cancelAnimationFrame(raf); raf = 0; } hot = false; render(); };

      size(); render();
      window.addEventListener('resize', () => { size(); render(); }, { passive: true });
      card.addEventListener('pointerenter', start);
      card.addEventListener('pointerleave', stop);
      card.addEventListener('focus', start);
      card.addEventListener('blur', stop);
    });
  }

  /* ============================================== pointer-driven effects */

  function reticle() {
    if (!fine || reduced) return;
    const el = $('#reticle');
    if (!el) return;
    let x = 0, y = 0, tx = 0, ty = 0, raf = 0;

    window.addEventListener('pointermove', (e) => {
      tx = e.clientX; ty = e.clientY;
      el.classList.add('is-visible');
      const over = e.target.closest('a, button, [data-tilt]');
      el.classList.toggle('is-hot', !!over);
      if (!raf) raf = requestAnimationFrame(tick);
    }, { passive: true });

    document.addEventListener('pointerleave', () => el.classList.remove('is-visible'));

    function tick() {
      x += (tx - x) * 0.22; y += (ty - y) * 0.22;
      el.style.transform = `translate(${x}px, ${y}px)`;
      raf = (Math.abs(tx - x) > 0.3 || Math.abs(ty - y) > 0.3)
        ? requestAnimationFrame(tick) : 0;
    }
  }

  function magnetic() {
    if (!fine || reduced) return;
    $$('[data-magnetic]').forEach((el) => {
      el.addEventListener('pointermove', (e) => {
        const r = el.getBoundingClientRect();
        const dx = (e.clientX - (r.left + r.width / 2)) * 0.22;
        const dy = (e.clientY - (r.top + r.height / 2)) * 0.32;
        el.style.transform = `translate(${dx}px, ${dy}px)`;
      });
      el.addEventListener('pointerleave', () => { el.style.transform = ''; });
    });
  }

  function cardGlow() {
    $$('[data-tilt]').forEach((card) => {
      card.addEventListener('pointermove', (e) => {
        const r = card.getBoundingClientRect();
        card.style.setProperty('--mx', `${e.clientX - r.left}px`);
        card.style.setProperty('--my', `${e.clientY - r.top}px`);
        if (fine && !reduced) {
          const rx = ((e.clientY - r.top) / r.height - 0.5) * -3.2;
          const ry = ((e.clientX - r.left) / r.width - 0.5) * 3.2;
          card.style.transform = `perspective(900px) rotateX(${rx}deg) rotateY(${ry}deg) translateY(-3px)`;
        }
      });
      card.addEventListener('pointerleave', () => { card.style.transform = ''; });
    });
  }

  /* ====================================================== scroll reveals */

  function reveals() {
    const items = $$('.reveal');
    if (reduced) { items.forEach((el) => el.classList.add('is-in')); return; }

    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry, i) => {
        if (!entry.isIntersecting) return;
        // A short stagger so a row of cards arrives in sequence.
        setTimeout(() => entry.target.classList.add('is-in'), i * 70);
        io.unobserve(entry.target);
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -8% 0px' });

    items.forEach((el) => io.observe(el));

    // Failsafe: whatever happens, nothing stays invisible. An element that is
    // still hidden after a few seconds is shown regardless, so a missed
    // observer callback can never leave a section blank.
    setTimeout(() => items.forEach((el) => el.classList.add('is-in')), 3000);
  }

  function heroWords() {
    if (reduced) {
      $$('#hero-title .word').forEach((w) => { w.style.opacity = 1; w.style.transform = 'none'; });
      return;
    }
    $$('#hero-title .word').forEach((w, i) => {
      w.style.animationDelay = `${180 + i * 85}ms`;
    });
  }

  /* Count a number up once it scrolls into view. */
  function counters() {
    const items = $$('[data-count]');
    if (reduced) {
      items.forEach((el) => {
        el.textContent = `${el.dataset.prefix || ''}${el.dataset.count}${el.dataset.suffix || ''}`;
      });
      return;
    }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const el = entry.target;
        io.unobserve(el);
        const target = parseFloat(el.dataset.count);
        const decimals = (el.dataset.count.split('.')[1] || '').length;
        const pre = el.dataset.prefix || '', suf = el.dataset.suffix || '';
        const started = performance.now();
        const dur = 1150;
        const step = (now) => {
          const p = clamp((now - started) / dur, 0, 1);
          const eased = 1 - Math.pow(1 - p, 3);
          el.textContent = `${pre}${(target * eased).toFixed(decimals)}${suf}`;
          if (p < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
      });
    }, { threshold: 0.5 });
    items.forEach((el) => io.observe(el));
  }

  /* ========================================================== telemetry */

  function clock() {
    const el = $('#sb-clock');
    if (!el) return;
    const tick = () => {
      el.textContent = new Date().toTimeString().slice(0, 8);
    };
    tick(); setInterval(tick, 1000);
  }

  const LOG_LEVEL = { info: 'info', ok: 'ok', warn: 'warn', error: 'err' };

  function renderLog(events) {
    const el = $('#log');
    if (!el) return;
    if (!events || !events.length) {
      el.innerHTML = `<div class="log__line"><span class="log__msg">no events yet</span></div>`;
      return;
    }
    el.innerHTML = events.slice(-40).map((e) => `
      <div class="log__line" data-level="${LOG_LEVEL[e.level] || 'info'}">
        <span class="log__time">${e.time}</span>
        <span class="log__msg">${escapeHtml(e.message)}</span>
      </div>`).join('');
    el.scrollTop = el.scrollHeight;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  function setText(id, value) { const el = $(id); if (el) el.textContent = value; }

  async function poll() {
    try {
      const res = await fetch('/api/status', { cache: 'no-store' });
      if (!res.ok) throw new Error(res.status);
      const s = await res.json();

      setText('#sb-device', `DEVICE ${s.device.toUpperCase()}`);
      setText('#sb-models', `MODELS ${s.models.loaded}`);

      const dev = $('#r-device');
      if (dev) dev.textContent = s.device.toUpperCase();
      setText('#r-device-sub', s.device_detail);

      const m = $('#r-models');
      if (m) m.innerHTML = `${s.models.loaded}<span class="unit">loaded</span>`;
      setText('#r-models-sub', s.models.names.length
        ? s.models.names.join(' · ')
        : 'lazy — loaded on first use');

      const c = $('#r-chunks');
      if (c) c.innerHTML = `${s.search.chunks}<span class="unit">chunks</span>`;
      setText('#r-chunks-sub', s.search.built
        ? `${s.search.documents} documents indexed`
        : 'index builds on first search');

      const lib = $('#r-library');
      if (lib) lib.innerHTML = `${s.library.files}<span class="unit">files</span>`;
      setText('#r-library-sub', s.library.files
        ? `${s.library.categories} type(s) · ${s.library.readable} readable`
        : 'drop anything in');

      const led = $('#led-server');
      if (led) led.dataset.state = 'live';
      renderLog(s.events);
    } catch (err) {
      const led = $('#led-server');
      if (led) led.dataset.state = 'down';
    }
  }

  /* =============================================================== init */

  runBoot();
  heroWords();
  heroField();
  cardVisuals();
  reticle();
  magnetic();
  cardGlow();
  reveals();
  counters();
  clock();
  poll();
  setInterval(poll, 4000);
})();
