import { FC, useCallback, useEffect, useRef } from "react";

const INK = "#23221f";

type AudioScopeProps = {
  analyser: AnalyserNode | null;
  /** Input is drawn dotted, Moshi solid: the two voices differ by texture, not colour. */
  variant: "input" | "output";
  label: string;
};

export const AudioScope: FC<AudioScopeProps> = ({ analyser, variant, label }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
      canvas.width = width * ratio;
      canvas.height = height * ratio;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
    }

    context.clearRect(0, 0, width, height);
    context.strokeStyle = INK;
    context.lineJoin = "round";
    if (variant === "input") {
      context.setLineDash([0, 7]);
      context.lineCap = "round";
      context.lineWidth = 4;
    } else {
      context.setLineDash([]);
      context.lineCap = "butt";
      context.lineWidth = 3;
    }

    context.beginPath();
    if (!analyser) {
      context.moveTo(0, height / 2);
      context.lineTo(width, height / 2);
    } else {
      const samples = new Uint8Array(analyser.fftSize);
      analyser.getByteTimeDomainData(samples);
      const step = Math.max(1, Math.floor(samples.length / Math.max(1, width / 2)));
      for (let index = 0; index < samples.length; index += step) {
        const x = (index / (samples.length - 1)) * width;
        const y = height / 2 + ((samples[index] - 128) / 128) * (height / 2 - 4);
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
    }
    context.stroke();
    context.setLineDash([]);
    animationRef.current = requestAnimationFrame(draw);
  }, [analyser, variant]);

  useEffect(() => {
    animationRef.current = requestAnimationFrame(draw);
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [draw]);

  return <canvas className="audio-scope" ref={canvasRef} aria-label={label} />;
};
