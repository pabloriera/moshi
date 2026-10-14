import { FC, PointerEvent, useEffect, useRef, useState } from "react";
import { TelemetryFeed, TelemetryStatus } from "./RuntimeTelemetry";
import { LayerOscilloscope, ScopeMetric } from "./LayerOscilloscope";

type Props = {
  feed: TelemetryFeed;
  status: TelemetryStatus | null;
  live: boolean;
  onChange: (params: { layer?: number; window?: number; enabled?: boolean; components?: 2 | 3 }) => void;
};

/** Ink-only 2D/3D view. Drag changes only the 3D camera. */
export const LatentScope: FC<Props> = ({ feed, status, live, onChange }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const statsRef = useRef<HTMLSpanElement>(null);
  const camera = useRef({ yaw: -0.55, pitch: 0.35 });
  const drag = useRef<{ x: number; y: number } | null>(null);
  const components = status?.components ?? 3;
  const [metric, setMetric] = useState<ScopeMetric>("all");

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    let animation = 0;
    const resize = () => {
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(canvas.clientWidth * ratio));
      canvas.height = Math.max(1, Math.round(canvas.clientHeight * ratio));
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    resize();
    const draw = () => {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);
      const { yaw, pitch } = camera.current;
      const radius = Math.min(w, h) * 0.35;
      const project = ([x, y, z = 0]: number[]) => {
        if (components === 2) return [w / 2 + x * radius, h / 2 - y * radius];
        const rx = x * Math.cos(yaw) + z * Math.sin(yaw);
        const rz = -x * Math.sin(yaw) + z * Math.cos(yaw);
        const ry = y * Math.cos(pitch) - rz * Math.sin(pitch);
        return [w / 2 + rx * radius, h / 2 - ry * radius];
      };
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 5]);
      ctx.strokeStyle = "rgba(35,34,31,0.18)";
      for (const axis of [[1.15, 0, 0], [0, 1.15, 0], [0, 0, 1.15]].slice(0, components)) {
        const from = project(axis.map((v) => -v)), to = project(axis);
        ctx.beginPath(); ctx.moveTo(from[0], from[1]); ctx.lineTo(to[0], to[1]); ctx.stroke();
      }
      ctx.setLineDash([]);
      const trajectory = feed.trajectory;
      if (trajectory && trajectory.points.length > 1) {
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        trajectory.points.slice(1).forEach((point, index) => {
          const from = project(trajectory.points[index]), to = project(point);
          const age = (index + 1) / trajectory.points.length;
          ctx.strokeStyle = `rgba(35,34,31,${0.15 + age * 0.8})`;
          ctx.lineWidth = 0.8 + age * 1.3;
          ctx.beginPath(); ctx.moveTo(from[0], from[1]); ctx.lineTo(to[0], to[1]); ctx.stroke();
          ctx.fillStyle = `rgba(35,34,31,${0.15 + age * 0.55})`;
          ctx.beginPath(); ctx.arc(to[0], to[1], 1.5, 0, Math.PI * 2); ctx.fill();
        });
        const head = project(trajectory.points[trajectory.points.length - 1]);
        ctx.fillStyle = "#23221f";
        ctx.beginPath(); ctx.arc(head[0], head[1], 3.5, 0, Math.PI * 2); ctx.fill();
      } else {
        ctx.fillStyle = "#6e6b64";
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText(!live ? "sin conexión" : status?.error ? "proyección no disponible"
          : status?.enabled === false ? "en pausa" : `reuniendo ${status?.window ?? 50}f…`, w / 2, h / 2 + 24, Math.max(1, w - 16));
      }
      if (statsRef.current) {
        const label = trajectory ? `${trajectory.points.length}f · ${trajectory.compute_ms.toFixed(1)} ms`
          : status?.benchmark ? `${status.window}f · ${status.update_hz} Hz` : "calibrando…";
        if (statsRef.current.textContent !== label) statsRef.current.textContent = label;
        statsRef.current.title = trajectory
          ? `Proyección ${trajectory.compute_ms.toFixed(1)} ms · proceso + transferencia ${trajectory.worker_ms.toFixed(1)} ms · envío ${trajectory.latency_ms.toFixed(1)} ms`
          : "Calibración y tasa de actualización del proceso de proyección";
      }
      animation = requestAnimationFrame(draw);
    };
    animation = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(animation); observer.disconnect(); };
  }, [feed, status, live, components]);

  const move = (event: PointerEvent<HTMLCanvasElement>) => {
    if (components !== 3 || !drag.current) return;
    camera.current.yaw += (event.clientX - drag.current.x) * 0.008;
    camera.current.pitch = Math.max(-1.4, Math.min(1.4, camera.current.pitch + (event.clientY - drag.current.y) * 0.008));
    drag.current = { x: event.clientX, y: event.clientY };
  };
  const benchmark = status?.benchmark;
  const cameraHelp = components === 3 ? "Arrastrá para girar." : "Dos coordenadas de difusión, sin perspectiva.";
  const timing = benchmark ? `Calibración 3D: 20f p95 ${benchmark["20"].p95_ms.toFixed(1)} ms; 50f p95 ${benchmark["50"].p95_ms.toFixed(1)} ms. ${cameraHelp} La proyección se recalcula sobre una ventana móvil; sus ejes no tienen significado fijo.`
    : `Proyección no lineal de estados del modelo. ${cameraHelp}`;

  return (
    <div className="latent-view latent-split">
      <section className="latent-pane latent-scopes" aria-label="Osciloscopios en tres columnas">
      <div className="latent-tools">
        <span>ESTADO · OSC</span>
        <select aria-label="Métricas del osciloscopio" value={metric}
          title="Normas L2 de estados internos; entropía solo en las salidas probabilísticas"
          onChange={(e) => setMetric(e.target.value as ScopeMetric)}>
          <option value="all">‖h‖ + H</option><option value="norm">‖h‖</option><option value="entropy">H · bits</option>
        </select>
        <label title="Ventana de estados: 20 frames = 1.6 s; 50 frames = 4 s a 12.5 Hz">
          <select aria-label="Frames de la trayectoria" value={status?.window ?? 50} disabled={!status || !live}
            onChange={(e) => onChange({ window: Number(e.target.value) })}>
            <option value={20}>20f</option><option value={50}>50f</option>
          </select>
        </label>
      </div>
      <LayerOscilloscope feed={feed} status={status} metric={metric} />
      <div className="latent-foot">
        <span title="Norma euclídea del estado residual en cada bloque temporal. TXT y A1…A8: entropías de muestreo en bits. Cada canal se centra y escala para mostrar su variación; los números mantienen sus unidades.">L: NORMA L2 · TXT/A: H · AUTO</span>
        <span>{((status?.window ?? 50) / (status?.frame_rate ?? 12.5)).toFixed(1)} s</span>
      </div>
      </section>
      <section className="latent-pane latent-map" aria-label="Mapa de difusión en la cuarta columna">
      <div className="latent-tools">
        <span title={timing}>MAP</span>
        <select aria-label="Dimensiones de la proyección" value={components} disabled={!status || !live}
          title="Reducción no lineal a dos o tres dimensiones"
          onChange={(e) => { drag.current = null; onChange({ components: Number(e.target.value) as 2 | 3 }); }}>
          <option value={2}>2D</option><option value={3}>3D</option>
        </select>
        <label title="Bloque temporal observado, después de la perturbación">
          L <select aria-label="Capa observada" value={status?.layer ?? 16} disabled={!status || !live}
            onChange={(e) => onChange({ layer: Number(e.target.value) })}>
            {Array.from({ length: status?.layer_count ?? 32 }, (_, i) => <option key={i} value={i + 1}>{i + 1}</option>)}
          </select>
        </label>
        <button type="button" className="mini-key" aria-pressed={status?.enabled ?? true}
          aria-label={status?.enabled === false ? "Activar trayectoria" : "Pausar trayectoria"}
          disabled={!status || !live} onClick={() => onChange({ enabled: !status?.enabled })}>
          {status?.enabled === false ? "▷" : "Ⅱ"}
        </button>
      </div>
      <div className="latent-map-display">
      <canvas ref={canvasRef} className={components === 2 ? "latent-flat" : undefined}
        aria-label={`Trayectoria ${components}D de los estados internos de Moshi`} title={status?.error || timing}
        onPointerDown={(event) => { if (components !== 3) return; drag.current = { x: event.clientX, y: event.clientY }; event.currentTarget.setPointerCapture(event.pointerId); }}
        onPointerMove={move} onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }} />
      </div>
      <div className="latent-foot"><span>DIFFUSION</span><span ref={statsRef} title={timing} /></div>
      </section>
    </div>
  );
};
