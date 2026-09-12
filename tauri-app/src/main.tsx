import React from "react";
import ReactDOM from "react-dom/client";
import { getCurrentWindow } from "@tauri-apps/api/window";
import App from "./App";
import { Dashboard } from "./Dashboard";
import { sidecarSocket } from "./services/SidecarSocket";

// Forward uncaught frontend errors to the sidecar so they land in sidecar.log
// (the webview console isn't otherwise visible). Best-effort — never throws.
function forwardError(message: string) {
  try {
    sidecarSocket.send("log", { level: "error", message: `[window] ${message}`.slice(0, 1000) });
  } catch {
    /* socket not ready — ignore */
  }
  // eslint-disable-next-line no-console
  console.error(message);
}

window.addEventListener("error", (e) => {
  forwardError(`${e.message} @ ${e.filename}:${e.lineno}:${e.colno}`);
});
window.addEventListener("unhandledrejection", (e) => {
  const reason = (e && (e.reason?.stack || e.reason?.message || String(e.reason))) || "unknown";
  forwardError(`unhandledrejection: ${reason}`);
});

// This same bundle drives two Tauri windows: the always-on "main" avatar
// overlay and a normal "dashboard" window (chat + agent controls). Route by
// the window label so each renders the right root.
function currentWindowLabel(): string {
  try {
    return getCurrentWindow().label;
  } catch {
    return "main";
  }
}

const isDashboard = currentWindowLabel() === "dashboard";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>{isDashboard ? <Dashboard /> : <App />}</React.StrictMode>,
);
