// Waveform + region editing, wrapping wavesurfer.js v7.
//
// Two kinds of regions live on the waveform: read-only VAD *suggestions*
// (clicking one adopts it) and a single editable *active* region that becomes
// the segment you save. The rest of the app only talks to this controller.

import WaveSurfer from "https://unpkg.com/wavesurfer.js@7.8.6/dist/wavesurfer.esm.js";
import RegionsPlugin from "https://unpkg.com/wavesurfer.js@7.8.6/dist/plugins/regions.esm.js";

const SUGGESTION_COLOR = "rgba(110, 168, 254, 0.18)";
const CONSUMED_COLOR = "rgba(120, 128, 140, 0.10)";
const ACTIVE_COLOR = "rgba(65, 194, 129, 0.28)";

export function createWaveform(container, { onActiveChange, onSuggestionClick }) {
  const ws = WaveSurfer.create({
    container,
    height: 128,
    waveColor: "#5b6473",
    progressColor: "#6ea8fe",
    cursorColor: "#e6e8ec",
    normalize: true,
    autoScroll: true,
  });
  const regions = ws.registerPlugin(RegionsPlugin.create());
  regions.enableDragSelection({ color: ACTIVE_COLOR });

  let active = null;
  let creatingActive = false;
  let loop = false;

  const isSuggestion = (region) => region.id.startsWith("vad-");

  function adopt(region) {
    if (active && active !== region) active.remove();
    active = region;
    active.setOptions({ color: ACTIVE_COLOR, drag: true, resize: true });
    onActiveChange(active.start, active.end);
  }

  regions.on("region-created", (region) => {
    if (isSuggestion(region)) return;
    if (creatingActive) {
      creatingActive = false;
      active = region;
      onActiveChange(active.start, active.end);
      return;
    }
    adopt(region); // drag-created selection
  });

  regions.on("region-updated", (region) => {
    if (region === active) onActiveChange(active.start, active.end);
  });

  regions.on("region-clicked", (region, e) => {
    e.stopPropagation();
    if (isSuggestion(region)) {
      onSuggestionClick(Number(region.id.slice(4)));
      return;
    }
    // Click inside the active region seeks to that exact spot (regions would
    // otherwise swallow the click and the bare-waveform seek never fires).
    const rect = ws.getWrapper().getBoundingClientRect();
    ws.setTime(((e.clientX - rect.left) / rect.width) * ws.getDuration());
  });

  regions.on("region-out", (region) => {
    if (loop && region === active) region.play();
  });

  return {
    load(url) {
      return new Promise((resolve) => {
        const once = () => { ws.un("ready", once); resolve(); };
        ws.on("ready", once);
        ws.load(url);
      });
    },
    reset() {
      regions.clearRegions();
      active = null;
    },
    setSuggestions(list) {
      regions.getRegions().filter(isSuggestion).forEach((r) => r.remove());
      list.forEach((r, i) =>
        regions.addRegion({
          id: `vad-${i}`, start: r.start, end: r.end,
          color: r.consumed ? CONSUMED_COLOR : SUGGESTION_COLOR,
          drag: false, resize: false,
        }),
      );
    },
    setActive(start, end) {
      if (active) {
        active.setOptions({ start, end });
        onActiveChange(start, end);
      } else {
        creatingActive = true;
        regions.addRegion({ start, end, color: ACTIVE_COLOR, drag: true, resize: true });
      }
    },
    clearActive() {
      if (active) { active.remove(); active = null; }
    },
    getActive() {
      return active ? { start: active.start, end: active.end } : null;
    },
    nudge(side, delta) {
      if (!active) return;
      const start = side === "start" ? Math.max(0, active.start + delta) : active.start;
      const end = side === "end" ? Math.min(ws.getDuration(), active.end + delta) : active.end;
      if (end > start) active.setOptions({ start, end });
    },
    // Space semantics: pause if playing; resume if paused inside the active
    // region; otherwise play the region from its start.
    playActive() {
      if (ws.isPlaying()) { ws.pause(); return; }
      const t = ws.getCurrentTime();
      if (active && t >= active.start && t < active.end) ws.play();
      else if (active) active.play();
      else ws.play();
    },
    restartActive() {
      if (active) active.play();
      else { ws.setTime(0); ws.play(); }
    },
    playPause() { ws.playPause(); },
    toggleLoop() { loop = !loop; return loop; },
    duration() { return ws.getDuration(); },
    currentTime() { return ws.getCurrentTime(); },
    setTime(t) { ws.setTime(Math.max(0, Math.min(ws.getDuration(), t))); },
    isPlaying() { return ws.isPlaying(); },
    onTime(cb) { ws.on("timeupdate", cb); },
    onPlayState(cb) {
      ws.on("play", () => cb(true));
      ws.on("pause", () => cb(false));
    },
    onFinish(cb) { ws.on("finish", cb); },
    // Zoom so [start-margin, end+margin] fills the container width.
    zoomTo(start, end, margin = 10) {
      const dur = ws.getDuration();
      if (!dur) return;
      const a = Math.max(0, start - margin);
      const b = Math.min(dur, end + margin);
      const pxPerSec = container.clientWidth / Math.max(1, b - a);
      try {
        ws.zoom(pxPerSec);
        ws.setScroll(a * pxPerSec);
      } catch { /* audio not ready yet */ }
    },
    zoomFit() {
      const dur = ws.getDuration();
      if (!dur) return;
      try { ws.zoom(container.clientWidth / dur); } catch { /* not ready */ }
    },
    // Scroll so `t` is centered, keeping the current zoom level.
    scrollToTime(t) {
      const dur = ws.getDuration();
      if (!dur) return;
      const total = ws.getWrapper().getBoundingClientRect().width;
      ws.setScroll(Math.max(0, (t / dur) * total - container.clientWidth / 2));
    },
    // Time range currently visible in the scroll window.
    getVisibleRange() {
      const dur = ws.getDuration();
      if (!dur) return { start: 0, end: 0 };
      const total = ws.getWrapper().getBoundingClientRect().width || 1;
      const left = ws.getScroll();
      return {
        start: (left / total) * dur,
        end: Math.min(dur, ((left + container.clientWidth) / total) * dur),
      };
    },
    onViewChange(cb) {
      ws.on("scroll", cb);
      ws.on("zoom", cb);
      ws.on("redraw", cb);
    },
    exportPeaks() {
      try { return ws.exportPeaks({ channels: 1, maxLength: 1500 })[0] || []; }
      catch { return []; }
    },
    setRate(rate) { ws.setPlaybackRate(rate, true); },
  };
}
