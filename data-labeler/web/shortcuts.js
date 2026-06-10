// Keyboard shortcuts. Typing in a text field is never hijacked, except for
// Ctrl+Enter (save) which works everywhere.

const STEP = 0.1;
const BIG_STEP = 0.5;

export function bindShortcuts(handlers) {
  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);

    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handlers.save();
      return;
    }
    if (typing) return;

    switch (e.key) {
      case " ":
        e.preventDefault();
        handlers.play();
        break;
      case "Enter":
        e.preventDefault();
        handlers.save();
        break;
      case "l":
      case "L":
        handlers.toggleLoop();
        break;
      case "Tab":
        e.preventDefault();
        handlers.nextSuggestion(e.shiftKey ? -1 : 1);
        break;
      case "ArrowLeft":
        e.preventDefault();
        handlers.nudge(e.altKey ? "start" : "end", -(e.shiftKey ? BIG_STEP : STEP));
        break;
      case "ArrowRight":
        e.preventDefault();
        handlers.nudge(e.altKey ? "start" : "end", e.shiftKey ? BIG_STEP : STEP);
        break;
      case "Delete":
      case "Backspace":
        e.preventDefault();
        handlers.deleteEditing();
        break;
      case "z":
      case "Z":
        if (e.ctrlKey || e.metaKey) { e.preventDefault(); handlers.undo(); }
        break;
      default:
        break;
    }
  });
}
