import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { HashRouter } from "react-router-dom";
import App from "./App";
import "./styles.css";
import { applyCachedUiFontSize } from "./lib/ui-font";

// Apply the cached UI font size before the first paint, otherwise the window
// would briefly render at the default size before the stored value loads.
applyCachedUiFontSize();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </StrictMode>,
);
