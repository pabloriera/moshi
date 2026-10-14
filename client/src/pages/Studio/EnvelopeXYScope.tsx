import { FC, useEffect, useRef } from "react";

const INK = "rgba(35, 34, 31, ALPHA)";

type EnvelopeXYScopeProps = {
  inputAnalyser: AnalyserNode | null;
  outputAnalyser: AnalyserNode | null;
};

type Point = {
  x: number;
  y: number;
};

function analyserRms(analyser: AnalyserNode | null, samples: Float32Array) {
  if (!analyser) return 0;
  analyser.getFloatTimeDomainData(samples);
  let energy = 0;
  for (const sample of samples) energy += sample * sample;
  return Math.sqrt(energy / samples.length);
}

function normalizeEnvelope(rms: number) {
  return Math.min(1, Math.log1p(rms * 40) / Math.log(5));
}

/** Input/output envelopes as an ink-only XY trace. */
export const EnvelopeXYScope: FC<EnvelopeXYScopeProps> = ({
  inputAnalyser,
  outputAnalyser,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const inputSamples = new Float32Array(inputAnalyser?.fftSize ?? 2048);
    const outputSamples = new Float32Array(outputAnalyser?.fftSize ?? 2048);
    const points: Point[] = [];
    let inputEnvelope = 0;
    let outputEnvelope = 0;
    let lastSample = 0;
    let animationFrame = 0;

    const resize = () => {
      const bounds = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(bounds.width * scale));
      canvas.height = Math.max(1, Math.round(bounds.height * scale));
      context.setTransform(scale, 0, 0, scale, 0, 0);
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    resize();

    const draw = (timestamp: number) => {
      const size = canvas.clientWidth;
      context.clearRect(0, 0, size, size);

      if (timestamp - lastSample >= 32) {
        const input = normalizeEnvelope(analyserRms(inputAnalyser, inputSamples));
        const output = normalizeEnvelope(analyserRms(outputAnalyser, outputSamples));
        inputEnvelope += (input - inputEnvelope) * (input > inputEnvelope ? 0.42 : 0.08);
        outputEnvelope += (output - outputEnvelope) * (output > outputEnvelope ? 0.42 : 0.08);
        points.push({ x: inputEnvelope, y: outputEnvelope });
        if (points.length > 70) points.shift();
        lastSample = timestamp;
      }

      const centre = size / 2;
      const radius = size / 2 - 10;
      const toCanvas = (point: Point) => ({
        x: centre + (point.x * 2 - 1) * radius * 0.82,
        y: centre - (point.y * 2 - 1) * radius * 0.82,
      });

      if (points.length > 1) {
        context.lineCap = "round";
        context.lineJoin = "round";
        points.slice(1).forEach((point, index) => {
          const from = toCanvas(points[index]);
          const to = toCanvas(point);
          const strength = (index + 1) / points.length;
          context.strokeStyle = INK.replace("ALPHA", (0.1 + strength * 0.9).toFixed(2));
          context.lineWidth = 1 + strength * 2;
          context.beginPath();
          context.moveTo(from.x, from.y);
          context.lineTo(to.x, to.y);
          context.stroke();
        });
        const head = toCanvas(points[points.length - 1]);
        context.fillStyle = INK.replace("ALPHA", "1");
        context.beginPath();
        context.arc(head.x, head.y, 5, 0, Math.PI * 2);
        context.fill();
      }

      animationFrame = requestAnimationFrame(draw);
    };

    animationFrame = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
    };
  }, [inputAnalyser, outputAnalyser]);

  return (
    <div className="xy-scope">
      <canvas ref={canvasRef} aria-label="Envolvente de entrada contra envolvente de Moshi" />
      <span className="xy-tag xy-tag-out">OUT</span>
      <span className="xy-tag xy-tag-in">IN</span>
    </div>
  );
};
