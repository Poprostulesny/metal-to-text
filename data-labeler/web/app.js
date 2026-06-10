// Orchestrates the labeler: wires the waveform, lyrics panel, segment list and
// keyboard shortcuts to the backend API. Keeps all UI state in one place.

import { api } from "./api.js";
import { createWaveform } from "./waveform.js";
import { createLyrics } from "./lyrics.js";
import { bindShortcuts } from "./shortcuts.js";

const $ = (id) => document.getElementById(id);

const state = {
  songs: [],
  index: 0,
  song: null,
  segments: [],
  vadRegions: [],
  suggestionPtr: -1,
  editing: null,        // audio_filepath being edited, or null for a new segment
  selectedSpan: null,   // {start,end} lyric offsets for the pending segment
  cursor: -1,
  maxSec: 30,
};

let wf, lyrics;

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
  lyrics.render(state.song.lyrics, consumed, editingSpan, state.cursor);
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

// --- data flow -------------------------------------------------------------

async function loadSong(i) {
  state.index = i;
  state.suggestionPtr = -1;
  clearEditing();
  $("songSelect").value = String(i);

  state.song = await api.song(i);
  renderSplits(state.song.splits);
  state.cursor = -1;
  wf.reset();

  if (!state.song.has_vocal) {
    toast("No vocal stem for this song", true);
    state.segments = [];
    renderSegments();
    renderLyrics();
    return;
  }

  await wf.load(api.audioUrl(i));
  const [{ regions }, segments] = await Promise.all([api.vad(i), api.segments(i)]);
  state.vadRegions = regions;
  state.segments = segments;
  wf.setSuggestions(regions);
  renderSegments();
  renderLyrics();
}

async function refreshSegments() {
  state.segments = await api.segments(state.index);
  const summary = await api.song(state.index);
  renderSplits(summary.splits);
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
  $("segText").value = "";
}

function nextSuggestion(dir) {
  if (!state.vadRegions.length) return;
  state.suggestionPtr =
    (state.suggestionPtr + dir + state.vadRegions.length) % state.vadRegions.length;
  const r = state.vadRegions[state.suggestionPtr];
  wf.setActive(r.start, r.end);
}

// --- init ------------------------------------------------------------------

async function init() {
  const cfg = await api.config();
  state.maxSec = cfg.max_segment_sec;
  $("maxInput").value = state.maxSec;

  wf = createWaveform($("waveform"), {
    onActiveChange: renderSelection,
    onSuggestionClick: (start, end) => wf.setActive(start, end),
  });
  lyrics = createLyrics($("lyrics"), {
    onSelect: (text, start, end) => {
      state.selectedSpan = { start, end };
      $("segText").value = text;
      renderLyrics();
    },
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
  $("maxInput").onchange = () => { state.maxSec = Number($("maxInput").value) || 30; renderSelection(); };
  $("playBtn").onclick = () => wf.playActive();
  $("loopBtn").onclick = () => $("loopBtn").classList.toggle("primary", wf.toggleLoop());
  $("clearBtn").onclick = () => { clearEditing(); wf.clearActive(); renderSegments(); renderLyrics(); renderSelection(); };
  $("saveBtn").onclick = save;

  bindShortcuts({
    play: () => wf.playActive(),
    save,
    nudge: (side, delta) => { wf.nudge(side, delta); renderSelection(); },
    nextSuggestion,
    deleteEditing,
    undo: undoLast,
    toggleLoop: () => $("loopBtn").classList.toggle("primary", wf.toggleLoop()),
  });

  await loadSong(0);
}

init().catch((err) => toast(String(err), true));
