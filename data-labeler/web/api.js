// Thin wrappers around the backend JSON API.

async function json(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status} ${detail}`);
  }
  return res.status === 204 ? null : res.json();
}

function post(url, body, method = "POST") {
  return json(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export const api = {
  config: () => json("/api/config"),
  songs: () => json("/api/songs"),
  song: (i) => json(`/api/songs/${i}`),
  vad: (i) => json(`/api/songs/${i}/vad`),
  segments: (i) => json(`/api/songs/${i}/segments`),
  audioUrl: (i) => `/api/songs/${i}/audio`,
  previewUrl: (i, start, end) => `/api/preview?index=${i}&start=${start}&end=${end}`,
  setLrcOffset: (i, offset) => post(`/api/songs/${i}/lrc_offset`, { offset }, "PUT"),
  addRejected: (i, key) => post(`/api/songs/${i}/rejected`, { key }),
  clearRejected: (i) => post(`/api/songs/${i}/rejected`, {}, "DELETE"),
  createSegment: (body) => post("/api/segments", body),
  updateSegment: (body) => post("/api/segments", body, "PUT"),
  deleteSegment: (audio_filepath) => post("/api/segments", { audio_filepath }, "DELETE"),
};
