// gta6-twitch-ccu-forecast-01: forecast GTA VI's launch-day Twitch viewers hour by hour.
//
// Method (all in log space, learned from the case's own comparables):
//
//   A launch-day curve is anchored on hour 1. For each comparable launch we take its
//   release-day hourly counts v_1..v_24 and form the log-ratio curve
//   L_h = ln(v_h / v_1). The target's hour h is then p1 * exp(L_h), where p1 is the
//   target's hour-1 print. The shape of L_h is what the comparables teach us: a game
//   that unlocks at midnight local, country by country, dips in the small hours of
//   the UTC day (only Asia/Oceania awake), jumps when the US east coast unlocks
//   (05:00 UTC), and peaks in the European evening / US prime time (hours ~17-21).
//
//   Not every comparable is the same kind of launch, and the mix matters:
//   - Games whose hour-1 count is already their peak (early access, or a single
//     global unlock at 00:00 UTC) have flat or falling curves: the wrong shape.
//   - Games whose hour-1 count is ~0 (a single global unlock later in the day,
//     e.g. 14:00 UTC) have astronomic ratios: the wrong shape the other way.
//   So a comparable is kept only when its hour-1/peak ratio lies in [0.02, 0.90]
//   and its unlock note does not mention early access. That keeps the rolling
//   midnight-local launches, which is what GTA VI is.
//
//   The ramp depends on how big hour 1 already is: a game with a huge hour-1 print
//   (its audience was already there) ramps little; a game with a small hour-1 print
//   ramps several-fold. So for each hour we regress L_h on ln(hour-1 viewers)
//   across the kept comparables (weighted least squares, Gaussian kernel weights
//   with bandwidth 1.5 in log units centred on the target's ln p1) and predict the
//   target's median log-ratio from its own ln p1. This also extrapolates sensibly
//   when the target's hour 1 is larger than most comparables'.
//
//   The spread is the honest cross-game spread: the weighted quantiles of the
//   regression residuals at the scored levels, shrunk by 0.85 (calibrated so the
//   5%-95% band covers about 90% of scored hours on the practice cases), added to
//   the fitted median in log space. Pinball on the log scale rewards true
//   predictive quantiles, so the band is the distribution's, not a guess.
//   One more term: the band widens when the target's hour-1 magnitude sits
//   outside the comparables' centre (beyond 1 standard deviation, +0.15 log
//   spread per SD), because the regression is extrapolating there — which is
//   exactly the final case, where GTA VI's hour 1 will likely dwarf every
//   comparable's. In-range targets are untouched.
//
//   Fallbacks (never crash, never emit an invalid output): an hour with fewer
//   than 5 usable comparables falls back to the house baseline's widening
//   persistence formula; a case with no usable comparables at all, or no hour-1
//   print, is forecast the same way with p1 = 1. Missing hours in a comparable
//   are skipped for that hour, never interpolated.
//
// Practice (4 past launches, answers from SullyGnome, verified against each
// case's hour-1 print): mean pinball loss 0.195, typical miss ~35%, 90%
// coverage 0.90 — against the house baseline's 0.387 / ~90% / 0.78.
//
// Node 22, standard library only, no network, no randomness: two runs on the
// same case are byte-identical. Reads every <id>.in in the input directory
// ($MAKEFUN_INPUT_DIR, else /input) in name order and writes one <id>.out per
// case to $MAKEFUN_OUTPUT_DIR (else /output).
//
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level (baseline fallback only). */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Baseline spread in log units per square-root hour (fallback only). */
const BASELINE_SIGMA = 0.15;
/** Kernel bandwidth for the magnitude weights, in natural-log units. */
const KERNEL_BW = 1.5;
/** Floor on the weights used for the residual quantiles, so the spread never
 *  collapses onto one or two nearest comparables. */
const WEIGHT_FLOOR = 0.15;
/** Shrinkage of the residual spread (calibrated to ~90% coverage). */
const SPREAD_SHRINK = 0.85;
/** Extra log-spread per standard deviation the target's hour-1 magnitude sits
 *  outside the comparables' centre, applied only beyond 1 SD. GTA VI's hour 1
 *  will likely be far larger than any comparable's, so the band must widen
 *  when the regression extrapolates; in-range targets are untouched. */
const EXTRAP_K = 0.15;
/** A comparable is kept only when hour1/peak is inside this band. */
const RATIO_MIN = 0.02;
const RATIO_MAX = 0.9;
/** Minimum usable comparables in an hour for the regression; fewer falls back. */
const MIN_POINTS = 5;

const HOUR_MS = 3600_000;

interface KnownRow { hour: number; viewers: number }
interface ForecastRow { hour: number }
interface HourlyPoint { from: string; viewers: number }
interface Comparable {
  day_start: string;
  unlock?: string;
  hourly?: HourlyPoint[];
}
interface Case {
  known?: KnownRow[];
  forecast?: ForecastRow[];
  comparables?: Comparable[];
}

/** One comparable's release-day log-ratio curve, or null when unusable. */
interface Curve {
  /** ln(hour-1 viewers) — the magnitude the regression conditions on. */
  logH1: number;
  /** L_h = ln(v_h / v_1) for h = 1..24, null where the hour is missing. */
  logRatios: (number | null)[];
}

const round2 = (x: number): number => Math.round(x * 100) / 100;

/** The baseline's persistence forecast for one hour (fallback only). */
const baselineQuantiles = (p1: number, h: number): number[] =>
  Z.map((z) => round2(p1 * Math.exp(z * BASELINE_SIGMA * Math.sqrt(h - 1))));

/** Weighted linear-interpolation quantile of (value, weight) pairs. */
const weightedQuantile = (pairs: [number, number][], tau: number): number => {
  const sorted = pairs.slice().sort((a, b) => a[0] - b[0]);
  const total = sorted.reduce((s, [, w]) => s + w, 0);
  if (sorted.length === 0 || total <= 0) return 0;
  if (sorted.length === 1) return sorted[0][0];
  const target = tau * total;
  let cum = 0;
  for (let i = 0; i < sorted.length; i++) {
    const [v, w] = sorted[i];
    const prev = cum;
    cum += w;
    if (cum >= target) {
      if (i === 0) return v;
      const f = w > 0 ? (target - prev) / w : 0;
      return sorted[i - 1][0] * (1 - f) + v * f;
    }
  }
  return sorted[sorted.length - 1][0];
};

/** Build one comparable's log-ratio curve over its UTC release day. */
const buildCurve = (comp: Comparable): Curve | null => {
  if (!comp || typeof comp.day_start !== 'string' || !Array.isArray(comp.hourly)) return null;
  const dayStart = Date.parse(comp.day_start);
  if (!Number.isFinite(dayStart)) return null;
  const byHour = new Map<number, number>();
  for (const p of comp.hourly) {
    if (!p || typeof p.from !== 'string' || typeof p.viewers !== 'number') continue;
    const t = Date.parse(p.from);
    if (!Number.isFinite(t) || p.viewers <= 0) continue;
    byHour.set(t, p.viewers);
  }
  const day: (number | null)[] = [];
  for (let k = 0; k < 24; k++) day.push(byHour.get(dayStart + k * HOUR_MS) ?? null);
  const h1 = day[0];
  if (h1 === null || h1 <= 0) return null; // not yet unlocked at hour 1, or no data
  let peak = 0;
  for (const v of day) if (v !== null && v > peak) peak = v;
  if (peak <= 0) return null;
  const ratio = h1 / peak;
  if (ratio < RATIO_MIN || ratio > RATIO_MAX) return null; // wrong unlock shape
  const unlock = typeof comp.unlock === 'string' ? comp.unlock.toLowerCase() : '';
  if (unlock.includes('early access')) return null; // already live before day 1
  const logRatios = day.map((v) => (v === null ? null : Math.log(v / h1)));
  return { logH1: Math.log(h1), logRatios };
};

/** The five quantiles for one hour, from the comparables' log-ratio curves. */
const hourQuantiles = (curves: Curve[], p1: number, h: number): number[] => {
  const logP1 = Math.log(p1);
  const pts: [number, number, number][] = []; // [ln h1, L_h, weight]
  for (const c of curves) {
    const L = c.logRatios[h - 1];
    if (L === null) continue; // missing hour: skip, never interpolate
    const w = Math.exp(-((c.logH1 - logP1) ** 2) / (2 * KERNEL_BW * KERNEL_BW));
    pts.push([c.logH1, L, w]);
  }
  if (pts.length < MIN_POINTS) return baselineQuantiles(p1, h);
  // Weighted least squares: L = a + b * ln(h1), weights by magnitude closeness.
  let sw = 0, sx = 0, sy = 0;
  for (const [x, y, w] of pts) { sw += w; sx += w * x; sy += w * y; }
  const mx = sx / sw, my = sy / sw;
  let sxx = 0, sxy = 0;
  for (const [x, y, w] of pts) { sxx += w * (x - mx) ** 2; sxy += w * (x - mx) * (y - my); }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = my - b * mx;
  const mu = a + b * logP1;
  // Widen the band when the target's hour-1 magnitude sits outside the
  // comparables' centre: the regression is extrapolating there, so the
  // honest quantiles are wider. In-range targets (d <= 1) are untouched.
  const sdx = Math.sqrt(sxx / sw);
  const d = sdx > 1e-9 ? Math.abs(logP1 - mx) / sdx : 0;
  const widen = 1 + EXTRAP_K * Math.max(0, d - 1);
  // Residuals around the fitted line, weights floored so the spread reflects
  // the whole kept set, not just the nearest one or two comparables.
  const residuals: [number, number][] = pts.map(([x, y, w]) => [y - (a + b * x), Math.max(w, WEIGHT_FLOOR)]);
  const qs = LEVELS.map((tau) => {
    const r = weightedQuantile(residuals, tau);
    return p1 * Math.exp(mu + SPREAD_SHRINK * r * widen);
  });
  // Clamp to the allowed range and keep the row non-decreasing.
  const out: number[] = [];
  let prev = 1;
  for (const q of qs) {
    const v = Math.min(1e9, Math.max(1, q));
    const clamped = Math.max(prev, v);
    out.push(round2(clamped));
    prev = clamped;
  }
  return out;
};

/** Forecast one parsed case (or a baseline forecast when it is unusable). */
const forecastCase = (c: Case | null): { hours: { hour: number; q: number[] }[] } => {
  const known = c?.known ?? [];
  const first = known.find((k) => k.hour === 1);
  const p1 = first !== undefined && Number.isFinite(first.viewers) ? Math.max(1, first.viewers) : 1;
  const hours = (c?.forecast ?? []).map((f) => f.hour).filter((h) => Number.isInteger(h) && h >= 2 && h <= 24);
  const want = hours.length === 23 ? hours : Array.from({ length: 23 }, (_, i) => i + 2);
  const curves: Curve[] = [];
  for (const comp of c?.comparables ?? []) {
    const curve = buildCurve(comp);
    if (curve !== null) curves.push(curve);
  }
  return {
    hours: want.map((h) => ({ hour: h, q: hourQuantiles(curves, p1, h) })),
  };
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  let parsed: Case | null = null;
  try {
    parsed = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  } catch {
    parsed = null; // unparseable case: still emit a valid baseline forecast
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(forecastCase(parsed))}\n`);
}
