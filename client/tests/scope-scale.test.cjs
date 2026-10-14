const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const ts = require("typescript");
const filename = path.join(__dirname, "../src/pages/Studio/ScopeScale.ts");
const compiled = ts.transpileModule(fs.readFileSync(filename, "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
});
const scope = { exports: {} };
vm.runInNewContext(compiled.outputText, scope);
const { scopeScale, scopePosition, scopeReadout } = scope.exports;

test("tiny entropy changes occupy visible space and retain meaningful readouts", () => {
  const values = Array.from({ length: 20 }, (_, i) => 0.000001 + i * 1e-9);
  const scale = scopeScale(values);
  const positions = values.map((v) => scopePosition(v, scale));
  assert(Math.max(...positions) - Math.min(...positions) > 1);
  assert.notEqual(scopeReadout(values[0], true), "0.000");
});

test("old spikes do not flatten recent detail and remain bounded on screen", () => {
  const recent = Array.from({ length: 20 }, (_, i) => 0.1 + 0.0001 * Math.sin(i));
  const values = [10, ...Array(29).fill(0.1), ...recent];
  const scale = scopeScale(values);
  assert(scale.radius < 0.001);
  const positions = recent.map((v) => scopePosition(v, scale));
  assert(Math.max(...positions) - Math.min(...positions) > 1);
  assert(scopePosition(10, scale) <= 1);
});

test("constant and missing signals remain finite and do not invent motion", () => {
  for (const value of [0, 1e-15, 20]) {
    const scale = scopeScale(Array(20).fill(value), { center: 100, radius: 10 });
    assert.equal(scopePosition(value, scale), 0);
    assert(scale.radius > 0);
  }
  assert(Number.isFinite(scopeScale([NaN, Infinity]).radius));
});

test("scale adapts gradually to a new range without modifying measurements", () => {
  const values = Array.from({ length: 20 }, (_, i) => i);
  const original = values.slice();
  const target = scopeScale(values);
  const adapted = scopeScale(values, { center: 0, radius: 1 });
  assert(adapted.center > 0 && adapted.center < target.center);
  assert(adapted.radius > 1 && adapted.radius < target.radius);
  assert.deepEqual(values, original);
});
