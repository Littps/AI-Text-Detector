import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { applySpringVars } from './spring'
import { installReloadGuard } from './reloadGuard'

// one physical definition for the spring curves, shared by CSS keyframes and
// the JS-driven dismissal (values land on :root as --spring-* variables)
applySpringVars()

// Round-19: inside the shell a page reload only costs the running detect (the
// reply's socket dies) and once left the WebView2 compositor white. The escape
// hatch is the tray menu, which reloads on purpose. A plain browser dev server
// keeps F5 - that is where reloading is actually useful.
if ('__TAURI_INTERNALS__' in window) installReloadGuard()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
