#!/usr/bin/env node
// ─────────────────────────────────────────────────────────────
// Tests for tax-engine.js — runs with plain Node.js (no deps)
// Usage: node test_tax_engine.js
// ─────────────────────────────────────────────────────────────

'use strict';

var assert = require('assert');
var T = require('./tax-engine.js');

var passed = 0;
var failed = 0;

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log('  PASS  ' + name);
  } catch (e) {
    failed++;
    console.log('  FAIL  ' + name);
    console.log('        ' + e.message);
  }
}

function approxEqual(actual, expected, tolerance, msg) {
  var diff = Math.abs(actual - expected);
  assert.ok(diff <= tolerance,
    (msg || '') + ' expected ~' + expected + ' got ' + actual + ' (diff=' + diff + ')');
}

// ─────────────────────────────────────────────────────────────
console.log('\n=== Formatting Utilities ===');
// ─────────────────────────────────────────────────────────────

test('fmt: positive number', function () {
  assert.strictEqual(T.fmt(1234), '$1,234');
});

test('fmt: negative number', function () {
  assert.strictEqual(T.fmt(-5678), '-$5,678');
});

test('fmt: zero', function () {
  assert.strictEqual(T.fmt(0), '$0');
});

test('fmt: rounds to integer', function () {
  assert.strictEqual(T.fmt(1234.7), '$1,235');
});

test('fmtk: thousands', function () {
  assert.strictEqual(T.fmtk(150000), '$150K');
});

test('fmtk: millions', function () {
  assert.strictEqual(T.fmtk(1200000), '$1.2M');
});

test('fmtk: negative millions', function () {
  assert.strictEqual(T.fmtk(-2500000), '-$2.5M');
});

test('fmtk: small number falls back to fmt', function () {
  assert.strictEqual(T.fmtk(500), '$500');
});

test('pct: formats percentage', function () {
  assert.strictEqual(T.pct(0.315), '31.5%');
});

test('pct: zero', function () {
  assert.strictEqual(T.pct(0), '0.0%');
});

test('parseInputValue: plain number', function () {
  assert.strictEqual(T.parseInputValue('1234'), 1234);
});

test('parseInputValue: formatted with commas', function () {
  assert.strictEqual(T.parseInputValue('1,234,567'), 1234567);
});

test('parseInputValue: with dollar sign', function () {
  assert.strictEqual(T.parseInputValue('$50,000'), 50000);
});

test('parseInputValue: empty string returns 0', function () {
  assert.strictEqual(T.parseInputValue(''), 0);
});

test('parseInputValue: non-numeric returns 0', function () {
  assert.strictEqual(T.parseInputValue('abc'), 0);
});

// ─────────────────────────────────────────────────────────────
console.log('\n=== calcBrackets ===');
// ─────────────────────────────────────────────────────────────

test('calcBrackets: zero income', function () {
  var r = T.calcBrackets(0, T.FEDERAL_SINGLE);
  assert.strictEqual(r.tax, 0);
  assert.strictEqual(r.topRate, 0);
});

test('calcBrackets: income in first bracket only', function () {
  var r = T.calcBrackets(10000, T.FEDERAL_SINGLE);
  assert.strictEqual(r.tax, 10000 * 0.10);
  assert.strictEqual(r.topRate, 0.10);
});

test('calcBrackets: income spanning two brackets', function () {
  // 11600 at 10% + 8400 at 12% = 1160 + 1008 = 2168
  var r = T.calcBrackets(20000, T.FEDERAL_SINGLE);
  approxEqual(r.tax, 2168, 1, 'tax');
  assert.strictEqual(r.topRate, 0.12);
});

test('calcBrackets: high income hits 37% bracket', function () {
  var r = T.calcBrackets(1000000, T.FEDERAL_SINGLE);
  assert.strictEqual(r.topRate, 0.37);
  assert.ok(r.tax > 300000, 'tax should be > 300k for 1M income');
});

// ─────────────────────────────────────────────────────────────
console.log('\n=== computeTaxes ===');
// ─────────────────────────────────────────────────────────────

test('computeTaxes: zero income', function () {
  var r = T.computeTaxes({ salary: 0, filing: 'single' });
  assert.strictEqual(r.gross, 0);
  assert.strictEqual(r.totalTax, 0);
  assert.strictEqual(r.takeHome, 0);
});

test('computeTaxes: basic 200K salary single', function () {
  var r = T.computeTaxes({ salary: 200000, filing: 'single' });
  assert.strictEqual(r.gross, 200000);
  assert.ok(r.federal.tax > 0, 'federal tax > 0');
  assert.ok(r.state.tax > 0, 'state tax > 0');
  assert.ok(r.city.tax > 0, 'city tax > 0');
  assert.ok(r.ssTax > 0, 'ss tax > 0');
  assert.ok(r.medicareTax > 0, 'medicare tax > 0');
  assert.ok(r.takeHome > 0, 'take-home > 0');
  assert.ok(r.takeHome < 200000, 'take-home < gross');
});

test('computeTaxes: 240K scenario (verified against manual calc)', function () {
  // 200K salary + 40K bonus, 23500 401k, 3600 insurance
  var r = T.computeTaxes({
    salary: 200000,
    bonus: 40000,
    retirement: 23500,
    insurance: 3600,
    filing: 'single',
  });

  assert.strictEqual(r.gross, 240000);

  // Federal taxable = 240000 - 27100 - 14600 = 198300
  approxEqual(r.taxableIncome, 198300, 1, 'taxable income');

  // Federal tax ~41142
  approxEqual(r.federal.tax, 41142, 50, 'federal tax');

  // SS: min(240000, 168600) * 0.062 = 10453.20
  approxEqual(r.ssTax, 10453, 1, 'SS tax');

  // Medicare: 240000 * 0.0145 + (240000-200000) * 0.009 = 3480 + 360 = 3840
  approxEqual(r.medicareTax, 3840, 1, 'medicare tax');

  // Effective rate ~31.2%
  approxEqual(r.effectiveRate, 0.312, 0.01, 'effective rate');

  // Take-home should be roughly 137K-138K
  approxEqual(r.takeHome, 137921, 200, 'take-home');
});

test('computeTaxes: SS caps at wage base', function () {
  var r = T.computeTaxes({ salary: 500000, filing: 'single' });
  approxEqual(r.ssTax, T.CONSTANTS.ssWageBase * T.CONSTANTS.ssRate, 1, 'SS capped');
});

test('computeTaxes: additional Medicare for high earner', function () {
  var r = T.computeTaxes({ salary: 300000, filing: 'single' });
  var baseMedicare = 300000 * 0.0145;
  var additionalMedicare = (300000 - 200000) * 0.009;
  approxEqual(r.medicareTax, baseMedicare + additionalMedicare, 1, 'medicare with surcharge');
});

test('computeTaxes: married filing has different thresholds', function () {
  var single = T.computeTaxes({ salary: 300000, filing: 'single' });
  var married = T.computeTaxes({ salary: 300000, filing: 'married' });
  // Married should pay less federal tax due to wider brackets
  assert.ok(married.federal.tax < single.federal.tax, 'married federal < single federal');
  // Married Medicare threshold is 250K vs 200K, so less additional medicare
  assert.ok(married.medicareTax < single.medicareTax, 'married medicare < single medicare');
});

test('computeTaxes: pre-tax deductions reduce taxable income', function () {
  var noDeductions = T.computeTaxes({ salary: 200000, filing: 'single' });
  var withDeductions = T.computeTaxes({
    salary: 200000, retirement: 23500, insurance: 3600, filing: 'single',
  });
  assert.ok(withDeductions.federal.tax < noDeductions.federal.tax,
    'deductions reduce federal tax');
  assert.ok(withDeductions.taxableIncome < noDeductions.taxableIncome,
    'deductions reduce taxable income');
});

test('computeTaxes: otherIncome adds to gross', function () {
  var base = T.computeTaxes({ salary: 200000, filing: 'single' });
  var withOther = T.computeTaxes({ salary: 200000, otherIncome: 50000, filing: 'single' });
  assert.strictEqual(withOther.gross, 250000);
  assert.ok(withOther.totalTax > base.totalTax, 'more income = more tax');
});

// ─────────────────────────────────────────────────────────────
console.log('\n=== computeSpending ===');
// ─────────────────────────────────────────────────────────────

test('computeSpending: empty spending', function () {
  var s = T.computeSpending({});
  assert.strictEqual(s.totalAnnual, 0);
});

test('computeSpending: annual only', function () {
  var s = T.computeSpending({ annual: { vacations: 5000, flights: 2000 } });
  assert.strictEqual(s.annual, 7000);
  assert.strictEqual(s.totalAnnual, 7000);
});

test('computeSpending: mixed frequencies', function () {
  var s = T.computeSpending({
    annual: { vacations: 1000 },
    monthly: { rent: 3000 },
    weekly: { dining: 100 },
    daily: { coffee: 5 },
  });
  assert.strictEqual(s.annual, 1000);
  assert.strictEqual(s.monthly, 3000);
  assert.strictEqual(s.weekly, 100);
  assert.strictEqual(s.daily, 5);
  assert.strictEqual(s.monthlyAnnual, 36000);
  assert.strictEqual(s.weeklyAnnual, 5200);
  assert.strictEqual(s.dailyAnnual, 1825);
  assert.strictEqual(s.totalAnnual, 1000 + 36000 + 5200 + 1825);
});

test('computeSpending: senior preset totals are reasonable', function () {
  var s = T.computeSpending(T.PRESETS.senior.spending);
  // Senior preset should have total annual spending roughly 80-120K
  assert.ok(s.totalAnnual > 60000, 'senior spending > 60K: ' + s.totalAnnual);
  assert.ok(s.totalAnnual < 150000, 'senior spending < 150K: ' + s.totalAnnual);
});

// ─────────────────────────────────────────────────────────────
console.log('\n=== computeBudget ===');
// ─────────────────────────────────────────────────────────────

test('computeBudget: combines taxes and spending', function () {
  var budget = T.computeBudget(
    { salary: 200000, filing: 'single' },
    { monthly: { rent: 3000 } }
  );
  assert.ok(budget.taxes.takeHome > 0, 'has take-home');
  assert.strictEqual(budget.spending.monthlyAnnual, 36000);
  assert.strictEqual(budget.remainder, budget.taxes.takeHome - 36000);
});

test('computeBudget: savings rate is correct', function () {
  var budget = T.computeBudget(
    { salary: 200000, filing: 'single' },
    { annual: { trips: 10000 } }
  );
  var expected = (budget.taxes.takeHome - 10000) / budget.taxes.takeHome;
  approxEqual(budget.savingsRate, expected, 0.001, 'savings rate');
});

test('computeBudget: deficit when spending exceeds income', function () {
  var budget = T.computeBudget(
    { salary: 50000, filing: 'single' },
    { monthly: { rent: 5000 } }
  );
  assert.ok(budget.remainder < 0, 'should be in deficit');
  assert.ok(budget.savingsRate < 0, 'savings rate should be negative');
});

// ─────────────────────────────────────────────────────────────
console.log('\n=== Slider Helpers ===');
// ─────────────────────────────────────────────────────────────

test('sliderToMultiplier: center (3) = 1.0x', function () {
  approxEqual(T.sliderToMultiplier(3), 1.0, 0.001, 'center');
});

test('sliderToMultiplier: min (1) < 1.0x', function () {
  var m = T.sliderToMultiplier(1);
  assert.ok(m > 0 && m < 0.5, 'min should be 0.3-0.4x, got ' + m);
});

test('sliderToMultiplier: max (5) > 2.5x', function () {
  var m = T.sliderToMultiplier(5);
  assert.ok(m > 2.5 && m < 4, 'max should be 2.5-4x, got ' + m);
});

test('sliderToMultiplier: monotonically increasing', function () {
  var prev = 0;
  for (var v = 1; v <= 5; v += 0.5) {
    var m = T.sliderToMultiplier(v);
    assert.ok(m > prev, v + ' -> ' + m + ' should be > ' + prev);
    prev = m;
  }
});

test('getSliderForItem: finds correct slider', function () {
  assert.strictEqual(T.getSliderForItem('monthly', 'rent'), 'housing');
  assert.strictEqual(T.getSliderForItem('weekly', 'dining'), 'food');
  assert.strictEqual(T.getSliderForItem('weekly', 'bars'), 'nightlife');
  assert.strictEqual(T.getSliderForItem('annual', 'vacations'), 'travel');
  assert.strictEqual(T.getSliderForItem('monthly', 'gym'), 'health');
  assert.strictEqual(T.getSliderForItem('annual', 'clothing'), 'shopping');
});

test('getSliderForItem: returns null for unmapped items', function () {
  assert.strictEqual(T.getSliderForItem('monthly', 'streaming'), null);
  assert.strictEqual(T.getSliderForItem('annual', 'taxpro'), null);
  assert.strictEqual(T.getSliderForItem('weekly', 'transit'), null);
});

test('applySliderMultiplier: at 1.0x returns base values', function () {
  var base = { 'annual:vacations': 5000, 'annual:flights': 2000 };
  var result = T.applySliderMultiplier('travel', 1.0, base);
  assert.strictEqual(result.scaledValues['annual:vacations'], 5000);
  assert.strictEqual(result.scaledValues['annual:flights'], 2000);
  assert.strictEqual(result.annualImpact, 7000);
});

test('applySliderMultiplier: at 2.0x doubles values', function () {
  var base = { 'monthly:rent': 3000, 'monthly:utilities': 150 };
  var result = T.applySliderMultiplier('housing', 2.0, base);
  assert.strictEqual(result.scaledValues['monthly:rent'], 6000);
  assert.strictEqual(result.scaledValues['monthly:utilities'], 300);
  // Annual impact: (6000 + 300) * 12 = 75600
  assert.strictEqual(result.annualImpact, 75600);
});

test('applySliderMultiplier: skips missing base values', function () {
  var base = { 'annual:vacations': 5000 }; // flights not in base
  var result = T.applySliderMultiplier('travel', 1.5, base);
  assert.strictEqual(Object.keys(result.scaledValues).length, 1);
  assert.strictEqual(result.scaledValues['annual:flights'], undefined);
});

// ─────────────────────────────────────────────────────────────
console.log('\n=== FREQ_TO_ANNUAL ===');
// ─────────────────────────────────────────────────────────────

test('FREQ_TO_ANNUAL: correct multipliers', function () {
  assert.strictEqual(T.FREQ_TO_ANNUAL.annual, 1);
  assert.strictEqual(T.FREQ_TO_ANNUAL.monthly, 12);
  assert.strictEqual(T.FREQ_TO_ANNUAL.weekly, 52);
  assert.strictEqual(T.FREQ_TO_ANNUAL.daily, 365);
});

// ─────────────────────────────────────────────────────────────
console.log('\n=== Preset Data Integrity ===');
// ─────────────────────────────────────────────────────────────

test('all presets have required fields', function () {
  var names = Object.keys(T.PRESETS);
  assert.ok(names.length >= 6, 'at least 6 presets');
  names.forEach(function (name) {
    var p = T.PRESETS[name];
    assert.ok(typeof p.salary === 'number', name + ' has salary');
    assert.ok(typeof p.bonus === 'number', name + ' has bonus');
    assert.ok(p.spending && p.spending.annual, name + ' has annual spending');
    assert.ok(p.spending && p.spending.monthly, name + ' has monthly spending');
    assert.ok(p.spending && p.spending.weekly, name + ' has weekly spending');
    assert.ok(p.spending && p.spending.daily, name + ' has daily spending');
  });
});

test('presets are ordered by increasing total comp', function () {
  var order = ['junior', 'mid', 'senior', 'staff', 'director', 'exec'];
  for (var i = 1; i < order.length; i++) {
    var prev = T.PRESETS[order[i - 1]];
    var curr = T.PRESETS[order[i]];
    var prevComp = prev.salary + prev.bonus;
    var currComp = curr.salary + curr.bonus;
    assert.ok(currComp > prevComp,
      order[i] + ' (' + currComp + ') > ' + order[i - 1] + ' (' + prevComp + ')');
  }
});

test('all preset budgets produce positive take-home', function () {
  Object.keys(T.PRESETS).forEach(function (name) {
    var p = T.PRESETS[name];
    var r = T.computeTaxes({
      salary: p.salary, bonus: p.bonus, otherIncome: p.otherIncome,
      retirement: p.retirement, insurance: p.insurance, hsa: p.hsa,
      otherDeductions: p.otherDeductions, filing: 'single',
    });
    assert.ok(r.takeHome > 0, name + ' take-home > 0: ' + r.takeHome);
  });
});

// ─────────────────────────────────────────────────────────────
console.log('\n=== SLIDER_MAPPING Integrity ===');
// ─────────────────────────────────────────────────────────────

test('all slider items reference valid frequencies', function () {
  var validFreqs = ['annual', 'monthly', 'weekly', 'daily'];
  Object.keys(T.SLIDER_MAPPING).forEach(function (name) {
    T.SLIDER_MAPPING[name].forEach(function (item) {
      assert.ok(validFreqs.indexOf(item.freq) >= 0,
        name + ': invalid freq ' + item.freq);
      assert.ok(typeof item.key === 'string' && item.key.length > 0,
        name + ': empty key');
    });
  });
});

test('no spending item is mapped to two sliders', function () {
  var seen = {};
  Object.keys(T.SLIDER_MAPPING).forEach(function (name) {
    T.SLIDER_MAPPING[name].forEach(function (item) {
      var k = item.freq + ':' + item.key;
      assert.ok(!seen[k], k + ' is mapped to both ' + seen[k] + ' and ' + name);
      seen[k] = name;
    });
  });
});

// ─────────────────────────────────────────────────────────────
console.log('\n=== Edge Cases ===');
// ─────────────────────────────────────────────────────────────

test('computeTaxes: very high income (1.5M)', function () {
  var r = T.computeTaxes({ salary: 1000000, bonus: 500000, filing: 'single' });
  assert.strictEqual(r.gross, 1500000);
  assert.ok(r.effectiveRate > 0.35, 'effective rate > 35% for 1.5M');
  assert.ok(r.effectiveRate < 0.50, 'effective rate < 50% for 1.5M');
  assert.ok(r.takeHome > 0, 'still has positive take-home');
});

test('computeTaxes: income below standard deduction', function () {
  var r = T.computeTaxes({ salary: 10000, filing: 'single' });
  assert.strictEqual(r.taxableIncome, 0);
  assert.strictEqual(r.federal.tax, 0);
  // Still pays FICA
  assert.ok(r.ssTax > 0, 'still pays SS');
  assert.ok(r.medicareTax > 0, 'still pays Medicare');
});

test('computeTaxes: all deductions fields used together', function () {
  var r = T.computeTaxes({
    salary: 300000, bonus: 50000, otherIncome: 20000,
    retirement: 23500, insurance: 6000, hsa: 4150, otherDeductions: 3000,
    filing: 'married',
  });
  assert.strictEqual(r.gross, 370000);
  assert.strictEqual(r.totalPreTax, 36650);
  assert.ok(r.takeHome > 200000, 'reasonable take-home for 370K married');
});

// ─────────────────────────────────────────────────────────────
// Summary
// ─────────────────────────────────────────────────────────────

console.log('\n' + '─'.repeat(50));
console.log('Results: ' + passed + ' passed, ' + failed + ' failed');
console.log('─'.repeat(50));

process.exit(failed > 0 ? 1 : 0);
