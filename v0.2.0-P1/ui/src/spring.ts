// Spring easings for the mode popover (2026-11-21 user request).
//
// SwiftUI-style semantics, because they are the clearest way to reason about
// this: `response` is the perceived duration of the motion (roughly the time
// to the first settle), `dampingFraction` is the damping ratio — 1.0 is
// critically damped (no overshoot), < 1 gives a little bounce, and small
// values feel rubbery.
//
// We cannot call a native spring API from the webview, so each preset is
// simulated once at module load and emitted as a CSS `linear()` easing curve.
// Sampling the analytic solution (not an integrator) keeps it exact and
// deterministic, and `linear()` is supported by the bundled WebView2.
//
// The animation duration MUST match the simulated window, otherwise the curve
// gets time-scaled and the bounce turns into a slow drift: that is why each
// preset carries both `easing` and `duration`.

export interface SpringPreset {
  /** CSS `linear(...)` easing sampled from the damped-spring solution */
  easing: string;
  /** matching animation duration in ms */
  duration: number;
}

/**
 * Under-damped spring displacement, normalised to 0..1 with the target at 1.
 *   x(t) = 1 - e^(-ζω₀t) · (cos(ω_d t) + (ζω₀/ω_d) · sin(ω_d t))
 * where ω₀ = 2π / response and ω_d = ω₀√(1-ζ²).
 */
export function springEasing(
  response = 0.34,
  dampingFraction = 0.72,
  samples = 30,
): SpringPreset {
  const zeta = Math.min(Math.max(dampingFraction, 0.05), 1);
  const omega0 = (2 * Math.PI) / response;
  const span = response * 1.6; // tail long enough for the bounce to die out
  const out: string[] = ["0"];
  if (zeta >= 0.999) {
    // critically damped: x = 1 - e^(-ω₀t)(1 + ω₀t)
    for (let i = 1; i <= samples; i++) {
      const t = (i / samples) * span;
      const x = 1 - Math.exp(-omega0 * t) * (1 + omega0 * t);
      out.push(x.toFixed(4));
    }
  } else {
    const omegaD = omega0 * Math.sqrt(1 - zeta * zeta);
    const k = (zeta * omega0) / omegaD;
    for (let i = 1; i <= samples; i++) {
      const t = (i / samples) * span;
      const decay = Math.exp(-zeta * omega0 * t);
      const x = 1 - decay * (Math.cos(omegaD * t) + k * Math.sin(omegaD * t));
      out.push(x.toFixed(4));
    }
  }
  out.push("1");
  return {
    easing: `linear(${out.join(", ")})`,
    duration: Math.round(span * 1000),
  };
}

/** popover entrance: fast response, visible but restrained overshoot */
export const SPRING_POP = springEasing(0.32, 0.72);
/** dismissal: quicker and almost bounce-free, so the close never feels stuck */
export const SPRING_CLOSE = springEasing(0.26, 0.92);
/** list rows / sub-sheet swap: lighter mass, a touch of inertia */
export const SPRING_ROW = springEasing(0.3, 0.78);
/** pointer-anchored context menu: crisp, overshoot nearly invisible */
export const SPRING_MENU = springEasing(0.24, 0.86);
/** card / list expand: heavier mass, the height overshoots a hair and settles */
export const SPRING_EXPAND = springEasing(0.42, 0.76);

/** Publish the curves as CSS variables so plain CSS keyframes can use them. */
export function applySpringVars(root: HTMLElement = document.documentElement): void {
  root.style.setProperty("--spring-pop", SPRING_POP.easing);
  root.style.setProperty("--spring-pop-dur", `${SPRING_POP.duration}ms`);
  root.style.setProperty("--spring-close", SPRING_CLOSE.easing);
  root.style.setProperty("--spring-close-dur", `${SPRING_CLOSE.duration}ms`);
  root.style.setProperty("--spring-row", SPRING_ROW.easing);
  root.style.setProperty("--spring-row-dur", `${SPRING_ROW.duration}ms`);
  root.style.setProperty("--spring-menu", SPRING_MENU.easing);
  root.style.setProperty("--spring-menu-dur", `${SPRING_MENU.duration}ms`);
  root.style.setProperty("--spring-expand", SPRING_EXPAND.easing);
  root.style.setProperty("--spring-expand-dur", `${SPRING_EXPAND.duration}ms`);
}
