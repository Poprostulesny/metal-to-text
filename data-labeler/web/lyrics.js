// Lyrics panel: free text selection -> character offsets, with consumed
// ranges greyed out, the currently edited range highlighted, a karaoke
// highlight following playback, and a cursor marker showing where you last
// left off. A plain click (no selection) reports its offset for seek-to-line.

export function createLyrics(container, { onSelect, onClickAt }) {
  let fullText = "";
  let selection = null;

  function offsetOf(node, offset) {
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    let total = 0;
    let current;
    while ((current = walker.nextNode())) {
      if (current === node) return total + offset;
      total += current.textContent.length;
    }
    return total;
  }

  container.addEventListener("mouseup", (e) => {
    const sel = window.getSelection();
    if (sel && sel.rangeCount > 0 && !sel.isCollapsed) {
      const range = sel.getRangeAt(0);
      if (!container.contains(range.startContainer) || !container.contains(range.endContainer)) return;
      let start = offsetOf(range.startContainer, range.startOffset);
      let end = offsetOf(range.endContainer, range.endOffset);
      if (start > end) [start, end] = [end, start];
      if (end > start) {
        selection = { start, end };
        onSelect(fullText.slice(start, end).replace(/\s+/g, " ").trim(), start, end);
      }
      return;
    }
    if (!onClickAt) return;
    // Collapsed selection = plain click: report the character offset.
    let pos = null;
    if (document.caretPositionFromPoint) {
      const cp = document.caretPositionFromPoint(e.clientX, e.clientY);
      if (cp && container.contains(cp.offsetNode)) pos = offsetOf(cp.offsetNode, cp.offset);
    } else if (document.caretRangeFromPoint) {
      const r = document.caretRangeFromPoint(e.clientX, e.clientY);
      if (r && container.contains(r.startContainer)) pos = offsetOf(r.startContainer, r.startOffset);
    }
    if (pos != null) onClickAt(pos);
  });

  function classFor(consumed, editing, karaoke) {
    const cls = new Array(fullText.length).fill("");
    for (const span of consumed) {
      for (let i = span.lyric_start; i < span.lyric_end && i < cls.length; i++) cls[i] = "consumed";
    }
    if (karaoke) {
      for (let i = karaoke.start; i < karaoke.end && i < cls.length; i++) {
        cls[i] = cls[i] ? `${cls[i]} karaoke` : "karaoke";
      }
    }
    if (editing) {
      for (let i = editing.start; i < editing.end && i < cls.length; i++) cls[i] = "editing";
    }
    return cls;
  }

  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  return {
    render(text, consumed = [], editing = null, cursor = -1, karaoke = null) {
      const scrollTop = container.scrollTop;
      fullText = text || "";
      selection = null;
      const cls = classFor(consumed, editing, karaoke);
      let html = "";
      let i = 0;
      // The caret is an empty span (text drawn via CSS ::after) so it never
      // adds characters that would skew lyric selection offsets.
      while (i < fullText.length) {
        if (i === cursor) html += '<span class="caret"></span>';
        const c = cls[i];
        let j = i;
        while (j < fullText.length && cls[j] === c) j++;
        const chunk = esc(fullText.slice(i, j));
        html += c ? `<span class="${c}">${chunk}</span>` : chunk;
        i = j;
      }
      if (cursor >= fullText.length) html += '<span class="caret"></span>';
      container.innerHTML = html || '<span class="hint">No lyrics.</span>';
      container.scrollTop = scrollTop;
    },
    getSelection() { return selection; },
    clearSelection() { selection = null; },
  };
}
