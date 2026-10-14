import { FC, useEffect, useRef } from "react";
import { RuntimeMeasurement, TelemetryFeed, TelemetryStatus } from "./RuntimeTelemetry";
import { ScopeScale, scopePosition, scopeReadout, scopeScale } from "./ScopeScale";

export type ScopeMetric = "norm" | "entropy" | "all";
type Props = { feed: TelemetryFeed; status: TelemetryStatus | null; metric: ScopeMetric };
type Channel = { label: string; layer?: number; values: number[]; unit: string; scale?: ScopeScale };

/** Three compact columns share time; each signal occupies one row. */
export const LayerOscilloscope: FC<Props> = ({ feed, status, metric }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const windowSize = status?.window ?? 50;
  const layers = status?.layer_count ?? 32;

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    let animation = 0;
    let width = 1;
    let channels: Channel[] = [];
    let lastSample: RuntimeMeasurement | undefined;
    let lastCount = -1;
    let cellHeight = 18;
    let columns = 3;
    let rows = 1;
    const pixelRatio = window.devicePixelRatio || 1;
    const resize = () => {
      width = Math.max(1, canvas.clientWidth);
      columns = window.innerWidth > 600 ? 3 : width >= 260 ? 2 : 1;
      rows = Math.max(1, Math.ceil(channels.length / columns));
      cellHeight = Math.max(16, Math.min(18, canvas.parentElement!.clientHeight / rows));
      const height = rows * cellHeight;
      canvas.style.height = `${height}px`;
      canvas.width = Math.round(width * pixelRatio);
      canvas.height = Math.round(height * pixelRatio);
      ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(canvas.parentElement!);

    const collect = () => {
      const history = feed.history.slice(-windowSize);
      const latest = history[history.length - 1];
      if (latest === lastSample && history.length === lastCount) return;
      lastSample = latest;
      lastCount = history.length;
      const previous = new Map(channels.map((channel) => [channel.label, channel.scale]));
      channels = [];
      if (metric !== "entropy") {
        for (let index = 0; index < layers; index++) channels.push({
          label: `L${String(index + 1).padStart(2, "0")}`, layer: index + 1,
          values: history.map((sample) => sample.layer_norms[index]), unit: "L2",
        });
      }
      if (metric !== "norm") {
        channels.push({ label: "TXT", values: history.map((sample) => sample.text_bits), unit: "bits" });
        const count = latest?.audio_bits.length ?? 8;
        for (let index = 0; index < count; index++) channels.push({
          label: `A${index + 1}`, values: history.map((sample) => sample.audio_bits[index]), unit: "bits",
        });
      }
      for (const channel of channels) channel.scale = scopeScale(channel.values, previous.get(channel.label));
      resize();
    };
    const draw = () => {
      collect();
      ctx.clearRect(0, 0, width, rows * cellHeight);
      const cellWidth = width / columns;
      channels.forEach((channel, index) => {
        const left = Math.floor(index / rows) * cellWidth;
        const top = (index % rows) * cellHeight;
        const selected = channel.layer === status?.layer;
        const values = channel.values;
        const scale = channel.scale!;
        const baseline = top + cellHeight / 2;
        const plotLeft = left + 26;
        const plotWidth = Math.max(1, cellWidth - 80);
        ctx.strokeStyle = selected ? "rgba(35,34,31,0.3)" : "rgba(35,34,31,0.1)";
        ctx.lineWidth = 0.7;
        ctx.beginPath(); ctx.moveTo(left + 3, top + cellHeight - 1); ctx.lineTo(left + cellWidth - 4, top + cellHeight - 1); ctx.stroke();
        ctx.strokeStyle = "rgba(35,34,31,0.16)";
        ctx.setLineDash([1, 4]);
        ctx.beginPath(); ctx.moveTo(plotLeft, baseline); ctx.lineTo(plotLeft + plotWidth, baseline); ctx.stroke();
        ctx.setLineDash([]);
        ctx.font = `${selected ? "700" : "400"} 9px 'JetBrains Mono', monospace`;
        ctx.textAlign = "left";
        ctx.fillStyle = "#23221f";
        ctx.fillText(channel.label, left + 4, baseline + 3);
        const latest = values[values.length - 1];
        ctx.textAlign = "right";
        ctx.fillStyle = "#6e6b64";
        const text = scopeReadout(latest, channel.unit === "bits");
        ctx.fillText(text, left + cellWidth - 5, baseline + 3);
        ctx.strokeStyle = "#23221f";
        ctx.lineWidth = selected ? 1.2 : 0.8;
        ctx.lineJoin = "round";
        ctx.beginPath();
        let connected = false;
        values.forEach((value, point) => {
          if (!Number.isFinite(value)) { connected = false; return; }
          const x = plotLeft + ((windowSize - values.length + point) / (windowSize - 1)) * plotWidth;
          const y = baseline - scopePosition(value, scale) * (cellHeight * 0.38);
          if (connected) ctx.lineTo(x, y); else ctx.moveTo(x, y);
          connected = true;
        });
        ctx.stroke();
        if (Number.isFinite(latest)) {
          ctx.fillStyle = "#23221f";
          ctx.beginPath(); ctx.arc(plotLeft + plotWidth, baseline - scopePosition(latest, scale) * cellHeight * 0.38, 1.3, 0, Math.PI * 2); ctx.fill();
        }
      });
      animation = requestAnimationFrame(draw);
    };
    const hover = (event: MouseEvent) => {
      const bounds = canvas.getBoundingClientRect();
      const column = Math.min(columns - 1, Math.max(0, Math.floor((event.clientX - bounds.left) / (width / columns))));
      const index = column * rows + Math.floor((event.clientY - bounds.top) / cellHeight);
      const channel = channels[index];
      if (!channel) { canvas.title = ""; return; }
      const values = channel.values.filter(Number.isFinite);
      const latest = values[values.length - 1];
      const min = values.length ? Math.min(...values) : 0, max = values.length ? Math.max(...values) : 0;
      canvas.title = `${channel.layer ? `Capa temporal ${channel.layer} · norma L2 del estado después del bloque` : channel.label === "TXT" ? "Salida de texto · entropía de muestreo" : `Salida de audio ${channel.label} · entropía de muestreo`}: ${scopeReadout(latest, channel.unit === "bits")} ${channel.unit}. Ventana ${windowSize}f · rango ${min.toPrecision(3)}–${max.toPrecision(3)}. AUTO reciente: centro ${channel.scale!.center.toPrecision(3)} ± ${channel.scale!.radius.toPrecision(3)}. Picos fuera de escala se comprimen; valores originales intactos.`;
    };
    canvas.addEventListener("mousemove", hover);
    animation = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(animation); observer.disconnect(); canvas.removeEventListener("mousemove", hover); };
  }, [feed, layers, metric, status?.layer, windowSize]);

  return (
    <div className="layer-oscilloscope" aria-label="Osciloscopios por capa; todos los canales visibles">
      <canvas ref={canvasRef} aria-label={metric === "norm" ? "Norma L2 de cada capa temporal a lo largo del tiempo"
        : metric === "entropy" ? "Entropía de texto y de cada salida de audio a lo largo del tiempo"
          : "Norma L2 de todas las capas temporales y entropía de las salidas de texto y audio"} />
    </div>
  );
};
