// Orchestrates the labeler: wires the waveform, lyrics panel, segment list and
// keyboard shortcuts to the backend API. Keeps all UI state in one place.
//
// Review flow: LRC lines are merged into chunk proposals (as large as fits in
// max s, split only on real VAD-detected silence). Each proposal is adopted as
// the active region with its text pre-filled; Enter saves and advances, S
// skips (persisted). Without LRC the proposals are plain VAD regions.

import { api } from "./api.js";
import { createWaveform } from "./waveform.js";
import { createLyrics } from "./lyrics.js";
import { createOverview } from "./overview.js";
import { bindShortcuts } from "./shortcuts.js";

const $ = (id) => document.getElementById(id);

// A VAD-silent stretch this long between LRC lines splits a chunk; timestamp
// gaps alone don't (long screams produce huge gaps with no silence).
const SPLIT_SILENCE_SEC = 1.5;
const EDGE_PAD_SEC = 0.15;   // kept around VAD speech when trimming chunk edges
const ZOOM_MARGIN_SEC = 10;  // context shown around the adopted proposal

const state = {
  songs: [],
  index: 0,
  song: null,
  segments: [],
  vadRegions: [],
  lrcLines: [],         // post-offset {start,end,text,span} from the backend
  suggestions: [],      // chunk proposals (with text) or raw VAD regions
  suggestionPtr: -1,
  rejected: new Set(),  // persisted skipped proposal keys
  editing: null,        // audio_filepath being edited, or null for a new segment
  selectedSpan: null,   // {start,end} lyric offsets for the pending segment
  cursor: -1,
  maxSec: 30,
  lyricsIndex: { text: "", map: [] },
  karaokeLine: -1,
  textDirty: false,     // user typed/selected text manually; stop auto-syncing
};

let wf, lyrics, overview, zoombar;

// Keep both bars in lockstep with the main view: the overview highlights the
// visible window, the zoom bar displays exactly that window (column-aligned
// with the waveform below) and highlights the active region inside it.
function syncBars() {
  const vis = wf.getVisibleRange();
  overview.setViewport(vis);
  zoombar.setRange(vis);
  zoombar.setViewport(wf.getActive());
}

// --- formatting ------------------------------------------------------------

function fmt(t) {
  const m = Math.floor(t / 60);
  const s = (t - m * 60).toFixed(2).padStart(5, "0");
  return `${m}:${s}`;
}

function toast(msg, isError = false) {
  const el = $("toast");
  el.textContent = msg;
  el.className = `toast show${isError ? " err" : ""}`;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (el.className = "toast"), 2200);
}

// --- rendering -------------------------------------------------------------

function renderSplits(splits) {
  $("splits").innerHTML =
    `train <b>${splits.train}</b> · test <b>${splits.test}</b> · valid <b>${splits.valid}</b>`;
}

function renderSelection() {
  const a = wf.getActive();
  const el = $("selection");
  if (!a) { el.textContent = "no selection"; el.classList.remove("over"); return; }
  const dur = a.end - a.start;
  el.innerHTML = `<b>${fmt(a.start)}</b> → <b>${fmt(a.end)}</b> &nbsp; (${dur.toFixed(2)}s)`;
  el.classList.toggle("over", dur > state.maxSec);
}

function renderLyrics() {
  const editingSpan = state.selectedSpan;
  const consumed = state.segments
    .filter((s) => s.lyric_end > s.lyric_start && s.audio_filepath !== state.editing)
    .map((s) => ({ lyric_start: s.lyric_start, lyric_end: s.lyric_end }));
  const karaoke = state.lrcLines[state.karaokeLine]?.span || null;
  // Re-rendering rebuilds the panel's DOM; skip when nothing actually changed,
  // otherwise rapid repeat calls make the panel flicker.
  const sig = JSON.stringify([state.index, consumed, editingSpan, state.cursor, karaoke]);
  if (sig === renderLyrics._sig) return;
  renderLyrics._sig = sig;
  lyrics.render(state.song.lyrics, consumed, editingSpan, state.cursor, karaoke);
}

function renderSegments() {
  const list = $("segList");
  list.innerHTML = "";
  state.segments.forEach((s) => {
    const li = document.createElement("li");
    li.className = "seg" + (s.audio_filepath === state.editing ? " editing" : "");
    li.innerHTML =
      `<span class="meta">${fmt(s.source_start)}–${fmt(s.source_end)}</span>` +
      `<span class="txt">${s.text || "<em>—</em>"}</span>`;
    const edit = document.createElement("button");
    edit.textContent = "✎";
    edit.title = "Edit";
    edit.onclick = () => editSegment(s.audio_filepath);
    const del = document.createElement("button");
    del.textContent = "🗑";
    del.className = "danger";
    del.title = "Delete";
    del.onclick = () => deleteSegment(s.audio_filepath);
    li.append(edit, del);
    list.append(li);
  });
}

function renderProgress() {
  const total = state.suggestions.length;
  if (!total) { $("progress").textContent = ""; return; }
  const done = state.suggestions.filter((s) => isConsumed(s) || isRejected(s)).length;
  $("progress").innerHTML = `proposal <b>${done}</b>/<b>${total}</b>`;
}

// --- lyric text matching ----------------------------------------------------

// Normalized lyrics with a map back to original offsets, so LRC line text can
// be located in the lyrics panel even when punctuation/casing differ.
function buildNormIndex(s) {
  let text = "";
  const map = [];
  let prevSpace = true;
  for (let i = 0; i < s.length; i++) {
    let c = s[i].toLowerCase();
    if (c === "’") c = "'";
    if (/[a-z0-9']/.test(c)) { text += c; map.push(i); prevSpace = false; }
    else if (!prevSpace) { text += " "; map.push(i); prevSpace = true; }
  }
  return { text, map };
}

function matchLyricSpan(lineText, fromChar = 0) {
  const { text: hay, map } = state.lyricsIndex;
  const needle = lineText.toLowerCase().replace(/’/g, "'")
    .replace(/[^a-z0-9' ]+/g, " ").replace(/\s+/g, " ").trim();
  if (!needle || !hay) return null;
  let from = 0;
  while (from < map.length && map[from] < fromChar) from++;
  let idx = hay.indexOf(needle, from);
  if (idx < 0) idx = hay.indexOf(needle);
  if (idx < 0) return null;
  return { start: map[idx], end: map[idx + needle.length - 1] + 1 };
}

// Sequential matching: each line searches after the previous match, so
// repeated chorus lines map to successive occurrences when possible.
function computeLineSpans() {
  let from = 0;
  for (const line of state.lrcLines) {
    line.span = matchLyricSpan(line.text, from);
    if (line.span) from = line.span.end;
  }
}

// --- chunk proposals ---------------------------------------------------------

function maxSilenceBetween(a, b) {
  if (b <= a) return 0;
  let t = a;
  let maxGap = 0;
  for (const r of state.vadRegions) {
    if (r.end <= a || r.start >= b) continue;
    if (r.start > t) maxGap = Math.max(maxGap, r.start - t);
    t = Math.max(t, r.end);
  }
  return Math.max(maxGap, b - t);
}

// Pull chunk edges in to the VAD speech envelope (LRC line "ends" are
// synthetic next-line starts, so trailing silence is common).
function trimToVad(start, end) {
  let first = null;
  let last = null;
  for (const r of state.vadRegions) {
    if (r.end <= start || r.start >= end) continue;
    if (first === null) first = r.start;
    last = r.end;
  }
  if (first === null) return { start, end };
  return {
    start: Math.max(start, first - EDGE_PAD_SEC),
    end: Math.min(end, last + EDGE_PAD_SEC),
  };
}

function buildChunks() {
  const lines = state.lrcLines;
  if (!lines.length) return [];
  const offset = Number(state.song.lrc_offset) || 0;
  const chunks = [];
  let cur = null;
  for (const line of lines) {
    const prev = cur && cur.lines[cur.lines.length - 1];
    if (cur && line.end - cur.start <= state.maxSec
        && maxSilenceBetween(prev.end, line.start) < SPLIT_SILENCE_SEC) {
      cur.end = line.end;
      cur.lines.push(line);
    } else {
      cur = { start: line.start, end: line.end, lines: [line] };
      chunks.push(cur);
    }
  }
  return chunks.map((c) => ({
    ...trimToVad(c.start, c.end),
    lines: c.lines,
    text: c.lines.map((l) => l.text).join(" "),
    // Keyed by the raw (offset-independent) first line time, so skipped
    // proposals stay skipped when the offset changes.
    key: `c${(c.lines[0].start - offset).toFixed(2)}`,
  }));
}

function rebuildSuggestions() {
  state.suggestions = state.lrcLines.length
    ? buildChunks()
    : state.vadRegions.map((r) => ({ ...r, key: `v${r.start.toFixed(2)}` }));
  renderSuggestions();
  renderProgress();
}

function isConsumed(s) {
  const mid = (s.start + s.end) / 2;
  return state.segments.some((seg) => seg.source_start <= mid && mid <= seg.source_end);
}

function isRejected(s) {
  return state.rejected.has(s.key);
}

function renderSuggestions() {
  const list = state.suggestions.map((s) => ({
    ...s, consumed: isConsumed(s) || isRejected(s),
  }));
  // Redrawing removes and re-creates every region (visible blink) — only do
  // it when the regions actually changed.
  const sig = JSON.stringify(list.map((s) => [s.start, s.end, s.consumed]));
  if (sig === renderSuggestions._sig) return;
  renderSuggestions._sig = sig;
  wf.setSuggestions(list);
}

// --- review queue -------------------------------------------------------------

function adoptSuggestion(i, { play = true } = {}) {
  const s = state.suggestions[i];
  if (!s) return;
  state.suggestionPtr = i;
  state.textDirty = false;
  wf.setActive(s.start, s.end);
  wf.zoomTo(s.start, s.end, ZOOM_MARGIN_SEC);
  syncTextToRegion();
  renderSelection();
  if (play) wf.restartActive();
}

function nextSuggestion(dir) {
  if (!state.suggestions.length) return;
  const n = state.suggestions.length;
  adoptSuggestion((state.suggestionPtr + dir + n) % n);
}

// After a save or skip, jump to the next pending proposal.
function autoAdvance() {
  const n = state.suggestions.length;
  for (let k = 1; k <= n; k++) {
    const i = (state.suggestionPtr + k) % n;
    const s = state.suggestions[i];
    if (!isConsumed(s) && !isRejected(s)) { adoptSuggestion(i); return; }
  }
  toast("All proposals done ✔");
  wf.zoomFit();
}

async function skipSuggestion() {
  const s = state.suggestions[state.suggestionPtr];
  if (!s || !s.key) return;
  state.rejected.add(s.key);
  try {
    await api.addRejected(state.index, s.key);
  } catch (err) {
    toast(String(err), true);
  }
  renderSuggestions();
  renderProgress();
  autoAdvance();
}

// Keep segment text in sync with the lines whose start falls inside the
// active region, until the user edits the text manually.
function syncTextToRegion() {
  if (state.editing || state.textDirty || !state.lrcLines.length) return;
  const a = wf.getActive();
  if (!a) return;
  const included = state.lrcLines.filter(
    (l) => l.start >= a.start - 0.5 && l.start <= a.end - 0.2,
  );
  $("segText").value = included.map((l) => l.text).join(" ");
  const spans = included.map((l) => l.span).filter(Boolean);
  state.selectedSpan = spans.length
    ? { start: Math.min(...spans.map((sp) => sp.start)),
        end: Math.max(...spans.map((sp) => sp.end)) }
    : null;
  renderLyrics();
}

// --- LRC offset ---------------------------------------------------------------

async function applyLrcOffset(offset) {
  const res = await api.setLrcOffset(state.index, offset);
  state.song.lrc_offset = res.offset;
  state.lrcLines = res.lrc_lines;
  state.karaokeLine = -1;
  computeLineSpans();
  rebuildSuggestions();
  $("lrcOffset").value = String(res.offset);
}

// Press T exactly when the first lyric line starts being sung.
async function tapSync() {
  if (!state.lrcLines.length) { toast("No LRC for this song", true); return; }
  const offset = Number(state.song.lrc_offset) || 0;
  const rawFirst = state.lrcLines[0].start - offset;
  const newOffset = Math.round((wf.currentTime() - rawFirst) * 10) / 10;
  try {
    await applyLrcOffset(newOffset);
    toast(`LRC offset ${newOffset >= 0 ? "+" : ""}${newOffset}s (tap-sync)`);
  } catch (err) {
    toast(String(err), true);
  }
}

// --- data flow -------------------------------------------------------------

async function loadSong(i) {
  state.index = i;
  state.suggestionPtr = -1;
  state.karaokeLine = -1;
  clearEditing();
  $("songSelect").value = String(i);

  state.song = await api.song(i);
  renderSplits(state.song.splits);
  state.cursor = -1;
  state.lyricsIndex = buildNormIndex(state.song.lyrics || "");
  state.lrcLines = state.song.lrc_lines || [];
  state.rejected = new Set(state.song.rejected || []);
  computeLineSpans();
  $("lrcOffset").value = String(state.song.lrc_offset || 0);
  $("lrcOffset").disabled = !state.lrcLines.length;
  wf.reset();
  // wf.reset() dropped all regions, so the render memos are stale.
  renderSuggestions._sig = null;
  renderLyrics._sig = null;

  if (!state.song.has_vocal) {
    toast("No vocal stem for this song", true);
    state.segments = [];
    state.suggestions = [];
    renderProgress();
    renderSegments();
    renderLyrics();
    return;
  }

  await wf.load(api.audioUrl(i));
  wf.zoomFit();
  wf.setRate(Number($("speedInput").value) || 1);
  const peaks = wf.exportPeaks();
  overview.setPeaks(peaks, wf.duration());
  overview.setRange({ start: 0, end: wf.duration() });
  zoombar.setPeaks(peaks, wf.duration());
  syncBars();
  $("clock").innerHTML = `<b>${fmt(0)}</b> / ${fmt(wf.duration())}`;
  const [{ regions }, segments] = await Promise.all([api.vad(i), api.segments(i)]);
  state.vadRegions = regions;
  state.segments = segments;
  rebuildSuggestions();
  renderSegments();
  renderLyrics();
  if (state.lrcLines.length) {
    toast(`${state.lrcLines.length} synced lines → ${state.suggestions.length} proposals`);
  }
  // Pre-adopt the first pending proposal so Space starts the review right away.
  const firstPending = state.suggestions.findIndex((s) => !isConsumed(s) && !isRejected(s));
  if (firstPending >= 0) adoptSuggestion(firstPending, { play: false });
}

async function refreshSegments() {
  state.segments = await api.segments(state.index);
  const summary = await api.song(state.index);
  renderSplits(summary.splits);
  renderSuggestions();
  renderProgress();
  renderSegments();
  renderLyrics();
}

function buildBody() {
  const a = wf.getActive();
  if (!a) { toast("Select a region first", true); return null; }
  const text = $("segText").value.trim();
  if (!text) { toast("Segment text is empty", true); return null; }
  const span = state.selectedSpan || {};
  return {
    start: a.start, end: a.end, text,
    lyric_start: span.start ?? -1, lyric_end: span.end ?? -1,
  };
}

async function save() {
  const body = buildBody();
  if (!body) return;
  const wasEditing = Boolean(state.editing);
  try {
    if (state.editing) {
      await api.updateSegment({ ...body, audio_filepath: state.editing });
      toast("Segment updated");
    } else {
      await api.createSegment({ ...body, song_index: state.index });
      toast("Segment saved");
    }
    if (state.selectedSpan) state.cursor = state.selectedSpan.end;
    clearEditing();
    wf.clearActive();
    await refreshSegments();
    renderSelection();
    if (!wasEditing) autoAdvance();
  } catch (err) {
    toast(String(err), true);
  }
}

function editSegment(audio_filepath) {
  const seg = state.segments.find((s) => s.audio_filepath === audio_filepath);
  if (!seg) return;
  state.editing = audio_filepath;
  state.selectedSpan = seg.lyric_end > seg.lyric_start
    ? { start: seg.lyric_start, end: seg.lyric_end } : null;
  $("segText").value = seg.text || "";
  wf.setActive(seg.source_start, seg.source_end);
  wf.zoomTo(seg.source_start, seg.source_end, ZOOM_MARGIN_SEC);
  renderSegments();
  renderLyrics();
  renderSelection();
}

async function deleteSegment(audio_filepath) {
  try {
    await api.deleteSegment(audio_filepath);
    if (state.editing === audio_filepath) clearEditing();
    await refreshSegments();
    toast("Segment deleted");
  } catch (err) {
    toast(String(err), true);
  }
}

function deleteEditing() {
  if (state.editing) deleteSegment(state.editing);
}

function undoLast() {
  if (state.segments.length) deleteSegment(state.segments[state.segments.length - 1].audio_filepath);
}

function clearEditing() {
  state.editing = null;
  state.selectedSpan = null;
  state.textDirty = false;
  $("segText").value = "";
}

// --- init ------------------------------------------------------------------

async function init() {
  const cfg = await api.config();
  state.maxSec = cfg.max_segment_sec;
  $("maxInput").value = state.maxSec;

  wf = createWaveform($("waveform"), {
    onActiveChange: () => { renderSelection(); syncTextToRegion(); zoombar.setViewport(wf.getActive()); },
    onSuggestionClick: (i) => adoptSuggestion(i),
  });
  overview = createOverview($("overview"), {
    onPan: (t) => wf.scrollToTime(t),
    onZoomRange: (a, b) => wf.zoomTo(a, b, 0),
  });
  zoombar = createOverview($("zoombar"), {
    onPan: (t) => wf.setTime(t),
    onZoomRange: (a, b) => wf.zoomTo(a, b, 0),
  });
  wf.onViewChange(() => requestAnimationFrame(syncBars));
  lyrics = createLyrics($("lyrics"), {
    onSelect: (text, start, end) => {
      state.selectedSpan = { start, end };
      $("segText").value = text;
      state.textDirty = true;
      renderLyrics();
    },
    // Plain click on a lyric line seeks the waveform to that line.
    onClickAt: (pos) => {
      const line = state.lrcLines.find(
        (l) => l.span && pos >= l.span.start && pos < l.span.end,
      );
      if (line) wf.setTime(line.start);
    },
  });

  wf.onTime((t) => {
    $("clock").innerHTML = `<b>${fmt(t)}</b> / ${fmt(wf.duration())}`;
    overview.setCursor(t);
    zoombar.setCursor(t);
    if (!wf.isPlaying()) return;
    const i = state.lrcLines.findIndex((l) => t >= l.start && t < l.end);
    if (i !== state.karaokeLine) { state.karaokeLine = i; renderLyrics(); }
  });
  wf.onPlayState((playing) => {
    $("playBtn").textContent = playing ? "⏸ Pause" : "▶ Play";
  });
  // Playback ran off the end of the song: move on to the next one instead of
  // rewinding to the start of the same song.
  wf.onFinish(() => {
    if (state.index < state.songs.length - 1) {
      toast("End of song → next");
      loadSong(state.index + 1);
    } else {
      toast("End of the last song");
    }
  });

  state.songs = await api.songs();
  const select = $("songSelect");
  state.songs.forEach((s) => {
    const opt = document.createElement("option");
    opt.value = String(s.index);
    opt.textContent = `${String(s.index + 1).padStart(2, "0")}. ${s.artist} – ${s.title}` +
      (s.has_vocal ? "" : " (no stem)");
    select.append(opt);
  });

  select.onchange = () => loadSong(Number(select.value));
  $("prevBtn").onclick = () => loadSong(Math.max(0, state.index - 1));
  $("nextBtn").onclick = () => loadSong(Math.min(state.songs.length - 1, state.index + 1));
  $("maxInput").onchange = () => {
    state.maxSec = Number($("maxInput").value) || 30;
    renderSelection();
    if (state.lrcLines.length) rebuildSuggestions();
  };
  $("lrcOffset").onchange = async () => {
    const offset = Number($("lrcOffset").value) || 0;
    try {
      await applyLrcOffset(offset);
      toast(`LRC offset ${offset >= 0 ? "+" : ""}${offset}s`);
    } catch (err) {
      toast(String(err), true);
    }
  };
  $("segText").addEventListener("input", () => { state.textDirty = true; });
  $("speedInput").oninput = () => {
    const v = Number($("speedInput").value) || 1;
    $("speedVal").textContent = `${v}×`;
    wf.setRate(v);
  };
  $("playBtn").onclick = () => wf.playActive();
  $("restartBtn").onclick = () => wf.restartActive();
  $("loopBtn").onclick = () => $("loopBtn").classList.toggle("primary", wf.toggleLoop());
  $("skipBtn").onclick = skipSuggestion;
  $("resetRejBtn").onclick = async () => {
    try {
      await api.clearRejected(state.index);
      state.rejected = new Set();
      renderSuggestions();
      renderProgress();
      toast("Skipped proposals restored");
    } catch (err) {
      toast(String(err), true);
    }
  };
  $("clearBtn").onclick = () => { clearEditing(); wf.clearActive(); renderSegments(); renderLyrics(); renderSelection(); };
  $("saveBtn").onclick = save;

  bindShortcuts({
    play: () => wf.playActive(),
    restart: () => wf.restartActive(),
    save,
    skip: skipSuggestion,
    tapSync,
    zoomFit: () => wf.zoomFit(),
    nudge: (side, delta) => { wf.nudge(side, delta); renderSelection(); },
    nextSuggestion,
    deleteEditing,
    undo: undoLast,
    toggleLoop: () => $("loopBtn").classList.toggle("primary", wf.toggleLoop()),
  });

  await loadSong(0);
}

init().catch((err) => toast(String(err), true));
