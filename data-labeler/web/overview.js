// Time-bar canvas used twice: as the whole-song overview and as the zoom bar.
// Each bar displays a time `range` of the song's peaks plus an optional
// highlighted `viewport` sub-window. The zoom bar's range is kept equal to the
// main waveform's visible range, so both stay column-aligned with it:
// the same x position always means the same moment in the song.
// Click = onPan(time); horizontal drag = onZoomRange(start, end).

export function createOverview(canvas, { onPan, onZoomRange }) {
  const ctx = canvas.getContext("2d");
  let peaks = [];
  let duration = 0;
  let range = { start: 0, end: 0 };  // displayed time window
  let view = null;                   // highlighted sub-window {start,end}
  let drag = null;                   // {x0, x1} in css px
  let cursor = -1;                   // playhead time

  const span = () => Math.max(0.001, range.end - range.start);

  function cssWidth() { return canvas.getBoundingClientRect().width; }

  function resize() {
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(r.width * devicePixelRatio));
    canvas.height = Math.max(1, Math.round(r.height * devicePixelRatio));
    draw();
  }

  function draw() {
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (!peaks.length || duration <= 0) return;
    ctx.fillStyle = "#566071";
    const mid = H / 2;
    for (let x = 0; x < W; x++) {
      const t = range.start + (x / W) * span();
      const v = Math.abs(peaks[Math.floor((t / duration) * peaks.length)] || 0);
      const h = Math.max(1, v * H);
      ctx.fillRect(x, mid - h / 2, 1, h);
    }
    if (view && view.end > view.start) {
      const x0 = ((view.start - range.start) / span()) * W;
      const x1 = ((view.end - range.start) / span()) * W;
      ctx.fillStyle = "rgba(110, 168, 254, 0.16)";
      ctx.fillRect(x0, 0, x1 - x0, H);
      ctx.strokeStyle = "rgba(110, 168, 254, 0.9)";
      ctx.lineWidth = devicePixelRatio;
      ctx.strokeRect(x0, 1, x1 - x0, H - 2);
    }
    if (cursor >= range.start && cursor <= range.end) {
      const x = ((cursor - range.start) / span()) * W;
      ctx.fillStyle = "#e6e8ec";
      ctx.fillRect(x, 0, devicePixelRatio, H);
    }
    if (drag) {
      const s = devicePixelRatio;
      ctx.fillStyle = "rgba(65, 194, 129, 0.25)";
      ctx.fillRect(Math.min(drag.x0, drag.x1) * s, 0, Math.abs(drag.x1 - drag.x0) * s, H);
    }
  }

  function timeAt(cssX) {
    const t = range.start + (cssX / cssWidth()) * span();
    return Math.max(range.start, Math.min(range.end, t));
  }

  canvas.addEventListener("mousedown", (e) => {
    if (duration <= 0) return;
    const left = canvas.getBoundingClientRect().left;
    drag = { x0: e.clientX - left, x1: e.clientX - left };
    e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!drag) return;
    drag.x1 = e.clientX - canvas.getBoundingClientRect().left;
    draw();
  });
  window.addEventListener("mouseup", () => {
    if (!drag) return;
    const { x0, x1 } = drag;
    drag = null;
    draw();
    if (Math.abs(x1 - x0) < 5) onPan(timeAt(x0));
    else onZoomRange(timeAt(Math.min(x0, x1)), timeAt(Math.max(x0, x1)));
  });
  window.addEventListener("resize", resize);

  return {
    setPeaks(p, dur) {
      peaks = p || [];
      duration = dur || 0;
      if (range.end <= range.start) range = { start: 0, end: duration };
      resize();
    },
    setRange(r) {
      range = r && r.end > r.start ? r : { start: 0, end: duration };
      draw();
    },
    setViewport(v) {
      view = v;
      draw();
    },
    setCursor(t) {
      cursor = t;
      draw();
    },
  };
}
