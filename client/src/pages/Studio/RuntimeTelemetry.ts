export type RuntimeMeasurement = {
  frame: number;
  text_bits: number;
  audio_bits: number[];
  audio_mean_bits: number;
  layer_norms: number[];
};

export type LatentTrajectory = {
  frame_ids: number[];
  points: ([number, number] | [number, number, number])[];
  components: 2 | 3;
  layer: number;
  window: number;
  method: string;
  compute_ms: number;
  worker_ms: number;
  latency_ms: number;
};

export type TelemetryStatus = {
  layer: number;
  layer_count: number;
  dimension: number;
  components: 2 | 3;
  window: number;
  frame_rate: number;
  enabled: boolean;
  method: string;
  benchmark: Record<string, { median_ms: number; p95_ms: number }> | null;
  update_hz: number;
  error: string | null;
};

/** Updated synchronously per packet, independent of React render batching. */
export type TelemetryFeed = {
  history: RuntimeMeasurement[];
  trajectory: LatentTrajectory | null;
};
