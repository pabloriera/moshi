import { memo, useEffect, useRef } from "react";
import { PhaseClock } from "./PhaseClock";

/**
 * Trajectory cycle phase as a ring with a dot travelling around it, plus degrees.
 * Painted from the PhaseClock on every animation frame (smooth, at most one model
 * step behind; it freezes at the last received phase when the stream stops).
 */
export const PhaseIndicator = memo(function PhaseIndicator({ clock }: { clock: PhaseClock }) {
  const textRef = useRef<HTMLSpanElement>(null);
  const ringRef = useRef<HTMLSpanElement>(null);
  const dotRef = useRef<SVGCircleElement>(null);
  const arcRef = useRef<SVGCircleElement>(null);

  useEffect(() => {
    let animation = 0;
    let lastPhase = -1;
    const paint = (now: number) => {
      const { phase } = clock.sample(now);
      if (phase !== lastPhase) {
        if (textRef.current) textRef.current.textContent = `${Math.round(phase * 360)}°`;
        ringRef.current?.setAttribute("aria-valuenow", (phase * 100).toFixed(2));
        const angle = phase * Math.PI * 2;
        dotRef.current?.setAttribute("cx", (12 + 8 * Math.sin(angle)).toFixed(2));
        dotRef.current?.setAttribute("cy", (12 - 8 * Math.cos(angle)).toFixed(2));
        arcRef.current?.setAttribute("stroke-dasharray", `${(phase * 50.27).toFixed(2)} 60`);
        lastPhase = phase;
      }
      animation = requestAnimationFrame(paint);
    };
    animation = requestAnimationFrame(paint);
    return () => cancelAnimationFrame(animation);
  }, [clock]);

  return (
    <span
      ref={ringRef}
      className="phase-ring"
      role="progressbar"
      aria-label="Fase del ciclo de la trayectoria"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={0}
      title="Fase del ciclo: avanza con cada paso del modelo, con hasta un paso de demora; se congela si se detiene el stream"
    >
      <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
        <circle className="phase-ring-track" cx="12" cy="12" r="8" />
        <circle
          ref={arcRef}
          className="phase-ring-arc"
          cx="12"
          cy="12"
          r="8"
          strokeDasharray="0 60"
          transform="rotate(-90 12 12)"
        />
        <circle ref={dotRef} className="phase-ring-dot" cx="12" cy="4" r="2.6" />
      </svg>
      <span ref={textRef} className="phase-ring-text">0°</span>
    </span>
  );
});
