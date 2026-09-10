import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
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

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
