export type ScopeScale = { center: number; radius: number };

function quantile(sorted: number[], fraction: number) {
  const position = (sorted.length - 1) * fraction;
  const low = Math.floor(position), high = Math.ceil(position);
  return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}

/** Recent robust range keeps an old spike from hiding small current changes. */
export function scopeScale(values: number[], previous?: ScopeScale): ScopeScale {
  const recent = values.slice(-20).filter(Number.isFinite).sort((a, b) => a - b);
  if (!recent.length) return { center: 0, radius: 1 };
  const center = quantile(recent, 0.5);
  const resolution = Math.max(1e-12, Math.abs(center) * 1e-7);
  const radius = Math.max(resolution, center - quantile(recent, 0.1), quantile(recent, 0.9) - center);
  if (!previous || recent[0] === recent[recent.length - 1]) return { center, radius };
  // Adapt once per incoming frame, not per animation frame. The scale freezes
  // with inference; historical outliers are bounded by the display transform.
  return { center: previous.center + 0.35 * (center - previous.center),
    radius: previous.radius + 0.35 * (radius - previous.radius) };
}

export function scopePosition(value: number, scale: ScopeScale) {
  return Math.tanh((value - scale.center) / scale.radius);
}

export function scopeReadout(value: number, entropy: boolean) {
  if (!Number.isFinite(value)) return "—";
  if (value !== 0 && (Math.abs(value) < 0.01 || Math.abs(value) >= 10000)) return value.toExponential(2);
  return value.toFixed(entropy ? 3 : 2);
}
