/* P0: health lamp driven by Rust-forwarded sidecar events. WS wiring lands in stage 4. */
const lamp = document.getElementById("lamp");
const statusText = document.getElementById("status-text");

function set(state, text) {
  lamp.className = "lamp " + state;
  statusText.textContent = text;
}

set("booting", "booting");

const { listen } = window.__TAURI__.event;

listen("sidecar-ready", (e) => {
  set("ready", "ready - ws://127.0.0.1:" + e.payload.port);
});

listen("sidecar-error", (e) => {
  set("error", "error - " + e.payload);
});
