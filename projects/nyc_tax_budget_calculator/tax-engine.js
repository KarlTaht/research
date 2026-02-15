// ─────────────────────────────────────────────────────────────
// NYC Tax & Budget Calculator — Pure Computation Engine
//
// Zero DOM dependencies. Works in both browser and Node.js.
// Browser: exposes window.TaxEngine
// Node:    module.exports = TaxEngine
// ─────────────────────────────────────────────────────────────

const TaxEngine = (function () {
  'use strict';

  // ── Tax Brackets (2024) ───────────────────────────────────

  const FEDERAL_SINGLE = [
    [11600, 0.10],
    [47150 - 11600, 0.12],
    [100525 - 47150, 0.22],
    [191950 - 100525, 0.24],
    [243725 - 191950, 0.32],
    [609350 - 243725, 0.35],
    [Infinity, 0.37],
  ];

  const FEDERAL_MARRIED = [
    [23200, 0.10],
    [94300 - 23200, 0.12],
    [201050 - 94300, 0.22],
    [383900 - 201050, 0.24],
    [487450 - 383900, 0.32],
    [731200 - 487450, 0.35],
    [Infinity, 0.37],
  ];

  const NY_STATE = [
    [8500, 0.04],
    [11700 - 8500, 0.045],
    [13900 - 11700, 0.0525],
    [80650 - 13900, 0.055],
    [215400 - 80650, 0.06],
    [1077550 - 215400, 0.0685],
    [5000000 - 1077550, 0.0965],
    [25000000 - 5000000, 0.103],
    [Infinity, 0.109],
  ];

  const NYC_TAX = [
    [12000, 0.03078],
    [25000 - 12000, 0.03762],
    [50000 - 25000, 0.03819],
    [Infinity, 0.03876],
  ];

  const CONSTANTS = {
    standardDeductionSingle: 14600,
    standardDeductionMarried: 29200,
    nyStandardDeductionSingle: 8000,
    nyStandardDeductionMarried: 16050,
    ssRate: 0.062,
    ssWageBase: 168600,
    medicareRate: 0.0145,
    medicareAdditionalRate: 0.009,
    medicareThresholdSingle: 200000,
    medicareThresholdMarried: 250000,
  };

  // ── Frequency Multipliers ─────────────────────────────────

  const FREQ_TO_ANNUAL = { annual: 1, monthly: 12, weekly: 52, daily: 365 };

  // ── Slider Configuration ──────────────────────────────────

  const SLIDER_MAPPING = {
    housing: [
      { freq: 'monthly', key: 'rent' },
      { freq: 'monthly', key: 'utilities' },
      { freq: 'monthly', key: 'rentersins' },
      { freq: 'monthly', key: 'laundry' },
      { freq: 'monthly', key: 'internet' },
      { freq: 'monthly', key: 'phone' },
    ],
    food: [
      { freq: 'monthly', key: 'groceries' },
      { freq: 'weekly', key: 'dining' },
      { freq: 'weekly', key: 'takeout' },
      { freq: 'daily', key: 'lunch' },
      { freq: 'daily', key: 'coffee' },
      { freq: 'daily', key: 'snacks' },
      { freq: 'daily', key: 'tips' },
    ],
    nightlife: [
      { freq: 'weekly', key: 'bars' },
      { freq: 'weekly', key: 'entertainment' },
      { freq: 'weekly', key: 'rideshare' },
    ],
    travel: [
      { freq: 'annual', key: 'vacations' },
      { freq: 'annual', key: 'flights' },
    ],
    health: [
      { freq: 'monthly', key: 'gym' },
      { freq: 'monthly', key: 'therapy' },
      { freq: 'annual', key: 'medical' },
    ],
    shopping: [
      { freq: 'annual', key: 'clothing' },
      { freq: 'annual', key: 'electronics' },
      { freq: 'annual', key: 'furniture' },
      { freq: 'annual', key: 'gifts' },
    ],
  };

  // ── Presets ────────────────────────────────────────────────

  const PRESETS = {
    junior: {
      salary: 140000, bonus: 10000, otherIncome: 0,
      retirement: 10000, insurance: 2400, hsa: 0, otherDeductions: 0,
      spending: {
        annual: { vacations: 3000, flights: 1000, furniture: 500, clothing: 1500, electronics: 800, gifts: 500, medical: 500, taxpro: 0 },
        monthly: { rent: 2200, utilities: 100, internet: 60, phone: 85, streaming: 35, gym: 50, groceries: 450, laundry: 60, subscriptions: 20, pet: 0, rentersins: 25, therapy: 0 },
        weekly: { transit: 33, dining: 80, bars: 40, entertainment: 30, rideshare: 20, takeout: 40 },
        daily: { coffee: 5, lunch: 14, snacks: 3, tips: 2 },
      },
    },
    mid: {
      salary: 190000, bonus: 35000, otherIncome: 0,
      retirement: 23500, insurance: 3000, hsa: 0, otherDeductions: 0,
      spending: {
        annual: { vacations: 5000, flights: 2000, furniture: 1500, clothing: 2500, electronics: 1200, gifts: 1000, medical: 800, taxpro: 300 },
        monthly: { rent: 2800, utilities: 130, internet: 70, phone: 85, streaming: 50, gym: 100, groceries: 550, laundry: 70, subscriptions: 40, pet: 0, rentersins: 30, therapy: 0 },
        weekly: { transit: 33, dining: 120, bars: 60, entertainment: 40, rideshare: 35, takeout: 50 },
        daily: { coffee: 6, lunch: 16, snacks: 4, tips: 3 },
      },
    },
    senior: {
      salary: 280000, bonus: 70000, otherIncome: 0,
      retirement: 23500, insurance: 3600, hsa: 4150, otherDeductions: 0,
      spending: {
        annual: { vacations: 8000, flights: 3000, furniture: 2500, clothing: 4000, electronics: 2000, gifts: 2000, medical: 1000, taxpro: 500 },
        monthly: { rent: 3500, utilities: 150, internet: 70, phone: 85, streaming: 60, gym: 150, groceries: 650, laundry: 80, subscriptions: 60, pet: 100, rentersins: 30, therapy: 200 },
        weekly: { transit: 33, dining: 180, bars: 80, entertainment: 60, rideshare: 50, takeout: 70 },
        daily: { coffee: 7, lunch: 18, snacks: 5, tips: 4 },
      },
    },
    staff: {
      salary: 350000, bonus: 150000, otherIncome: 0,
      retirement: 23500, insurance: 3600, hsa: 4150, otherDeductions: 1200,
      spending: {
        annual: { vacations: 15000, flights: 5000, furniture: 4000, clothing: 6000, electronics: 3000, gifts: 3000, medical: 1500, taxpro: 1000 },
        monthly: { rent: 4500, utilities: 180, internet: 80, phone: 100, streaming: 70, gym: 200, groceries: 800, laundry: 100, subscriptions: 80, pet: 150, rentersins: 35, therapy: 300 },
        weekly: { transit: 33, dining: 250, bars: 100, entertainment: 80, rideshare: 80, takeout: 100 },
        daily: { coffee: 8, lunch: 20, snacks: 6, tips: 5 },
      },
    },
    director: {
      salary: 500000, bonus: 250000, otherIncome: 0,
      retirement: 23500, insurance: 4800, hsa: 4150, otherDeductions: 3600,
      spending: {
        annual: { vacations: 25000, flights: 8000, furniture: 5000, clothing: 8000, electronics: 4000, gifts: 5000, medical: 2000, taxpro: 2000 },
        monthly: { rent: 6000, utilities: 200, internet: 100, phone: 120, streaming: 80, gym: 250, groceries: 1000, laundry: 120, subscriptions: 100, pet: 200, rentersins: 40, therapy: 400 },
        weekly: { transit: 0, dining: 350, bars: 150, entertainment: 120, rideshare: 150, takeout: 120 },
        daily: { coffee: 10, lunch: 25, snacks: 8, tips: 7 },
      },
    },
    exec: {
      salary: 700000, bonus: 500000, otherIncome: 0,
      retirement: 23500, insurance: 6000, hsa: 4150, otherDeductions: 6000,
      spending: {
        annual: { vacations: 40000, flights: 15000, furniture: 8000, clothing: 12000, electronics: 5000, gifts: 8000, medical: 3000, taxpro: 5000 },
        monthly: { rent: 8500, utilities: 250, internet: 120, phone: 150, streaming: 100, gym: 350, groceries: 1400, laundry: 150, subscriptions: 150, pet: 300, rentersins: 50, therapy: 500 },
        weekly: { transit: 0, dining: 500, bars: 200, entertainment: 200, rideshare: 250, takeout: 150 },
        daily: { coffee: 12, lunch: 30, snacks: 10, tips: 10 },
      },
    },
  };

  // ── Formatting Utilities ──────────────────────────────────

  function fmt(n) {
    if (n < 0) return '-$' + Math.abs(Math.round(n)).toLocaleString('en-US');
    return '$' + Math.round(n).toLocaleString('en-US');
  }

  function fmtk(n) {
    var abs = Math.abs(n);
    if (abs >= 1000000) return (n < 0 ? '-' : '') + '$' + (abs / 1000000).toFixed(1) + 'M';
    if (abs >= 1000) return (n < 0 ? '-' : '') + '$' + (abs / 1000).toFixed(0) + 'K';
    return fmt(n);
  }

  function pct(n) {
    return (n * 100).toFixed(1) + '%';
  }

  function parseInputValue(str) {
    return parseFloat(String(str).replace(/[^0-9.]/g, '')) || 0;
  }

  // ── Core Tax Computation ──────────────────────────────────

  function calcBrackets(income, brackets) {
    var tax = 0;
    var remaining = income;
    var topRate = 0;
    for (var i = 0; i < brackets.length; i++) {
      if (remaining <= 0) break;
      var size = brackets[i][0];
      var rate = brackets[i][1];
      var taxable = Math.min(remaining, size);
      tax += taxable * rate;
      topRate = rate;
      remaining -= taxable;
    }
    return { tax: tax, topRate: topRate };
  }

  /**
   * Compute all taxes from inputs. Pure function — no DOM access.
   *
   * @param {Object} inputs
   * @param {number} inputs.salary
   * @param {number} inputs.bonus
   * @param {number} inputs.otherIncome
   * @param {number} inputs.retirement
   * @param {number} inputs.insurance
   * @param {number} inputs.hsa
   * @param {number} inputs.otherDeductions
   * @param {string} inputs.filing - 'single' or 'married'
   * @returns {Object} Full tax computation results
   */
  function computeTaxes(inputs) {
    var salary = inputs.salary || 0;
    var bonus = inputs.bonus || 0;
    var otherIncome = inputs.otherIncome || 0;
    var retirement = inputs.retirement || 0;
    var insurance = inputs.insurance || 0;
    var hsa = inputs.hsa || 0;
    var otherDed = inputs.otherDeductions || 0;
    var filing = inputs.filing || 'single';

    var gross = salary + bonus + otherIncome;
    var totalPreTax = retirement + insurance + hsa + otherDed;

    // FICA is computed on gross (before 401k deduction)
    var ssIncome = Math.min(gross, CONSTANTS.ssWageBase);
    var ssTax = ssIncome * CONSTANTS.ssRate;

    var medicareThreshold = filing === 'married'
      ? CONSTANTS.medicareThresholdMarried
      : CONSTANTS.medicareThresholdSingle;
    var medicareTax = gross * CONSTANTS.medicareRate;
    if (gross > medicareThreshold) {
      medicareTax += (gross - medicareThreshold) * CONSTANTS.medicareAdditionalRate;
    }

    // Federal taxable income
    var standardDed = filing === 'married'
      ? CONSTANTS.standardDeductionMarried
      : CONSTANTS.standardDeductionSingle;
    var federalBrackets = filing === 'married' ? FEDERAL_MARRIED : FEDERAL_SINGLE;
    var taxableIncome = Math.max(0, gross - totalPreTax - standardDed);

    // NY State + NYC taxable income (different standard deduction)
    var nyStandardDed = filing === 'married'
      ? CONSTANTS.nyStandardDeductionMarried
      : CONSTANTS.nyStandardDeductionSingle;
    var stateTaxableIncome = Math.max(0, gross - totalPreTax - nyStandardDed);

    var federal = calcBrackets(taxableIncome, federalBrackets);
    var state = calcBrackets(stateTaxableIncome, NY_STATE);
    var city = calcBrackets(stateTaxableIncome, NYC_TAX);

    var totalTax = federal.tax + state.tax + city.tax + ssTax + medicareTax;
    var totalDeductions = totalTax + totalPreTax;
    var takeHome = gross - totalDeductions;
    var effectiveRate = gross > 0 ? totalTax / gross : 0;

    return {
      // Income
      salary: salary,
      bonus: bonus,
      otherIncome: otherIncome,
      gross: gross,

      // Pre-tax
      retirement: retirement,
      insurance: insurance,
      hsa: hsa,
      otherDed: otherDed,
      totalPreTax: totalPreTax,
      taxableIncome: taxableIncome,
      stateTaxableIncome: stateTaxableIncome,

      // Taxes
      federal: federal,
      state: state,
      city: city,
      ssTax: ssTax,
      medicareTax: medicareTax,
      medicareThreshold: medicareThreshold,
      totalTax: totalTax,

      // Summary
      totalDeductions: totalDeductions,
      takeHome: takeHome,
      effectiveRate: effectiveRate,
    };
  }

  /**
   * Compute annual spending from a spending map.
   *
   * @param {Object} spending - Map of freq -> { key: amount }
   * @returns {Object} Breakdown and total
   */
  function computeSpending(spending) {
    var annualRaw = 0;
    var monthlyRaw = 0;
    var weeklyRaw = 0;
    var dailyRaw = 0;

    for (var key in (spending.annual || {})) {
      annualRaw += spending.annual[key] || 0;
    }
    for (var key in (spending.monthly || {})) {
      monthlyRaw += spending.monthly[key] || 0;
    }
    for (var key in (spending.weekly || {})) {
      weeklyRaw += spending.weekly[key] || 0;
    }
    for (var key in (spending.daily || {})) {
      dailyRaw += spending.daily[key] || 0;
    }

    var annualTotal = annualRaw;
    var monthlyAnnual = monthlyRaw * 12;
    var weeklyAnnual = weeklyRaw * 52;
    var dailyAnnual = dailyRaw * 365;
    var totalAnnual = annualTotal + monthlyAnnual + weeklyAnnual + dailyAnnual;

    return {
      annual: annualRaw,
      monthly: monthlyRaw,
      weekly: weeklyRaw,
      daily: dailyRaw,
      monthlyAnnual: monthlyAnnual,
      weeklyAnnual: weeklyAnnual,
      dailyAnnual: dailyAnnual,
      totalAnnual: totalAnnual,
    };
  }

  /**
   * Compute full budget: taxes + spending + savings.
   *
   * @param {Object} inputs - Same as computeTaxes inputs
   * @param {Object} spending - Spending map (freq -> { key: amount })
   * @returns {Object} Combined results
   */
  function computeBudget(inputs, spending) {
    var taxes = computeTaxes(inputs);
    var spend = computeSpending(spending);

    var remainder = taxes.takeHome - spend.totalAnnual;
    var savingsRate = taxes.takeHome > 0 ? remainder / taxes.takeHome : 0;

    return {
      taxes: taxes,
      spending: spend,
      remainder: remainder,
      savingsRate: savingsRate,
    };
  }

  // ── Slider Helpers ────────────────────────────────────────

  function sliderToMultiplier(val) {
    // val 1-5, center at 3 = 1.0x
    // Exponential curve: pow(1.8, val-3)
    // 1 → 0.31x, 2 → 0.56x, 3 → 1.0x, 4 → 1.8x, 5 → 3.24x
    return Math.pow(1.8, val - 3);
  }

  function getSliderForItem(freq, key) {
    for (var name in SLIDER_MAPPING) {
      var items = SLIDER_MAPPING[name];
      for (var i = 0; i < items.length; i++) {
        if (items[i].freq === freq && items[i].key === key) return name;
      }
    }
    return null;
  }

  /**
   * Apply a slider multiplier to base spending values and return scaled values.
   *
   * @param {string} sliderName
   * @param {number} multiplier
   * @param {Object} baseValues - Map of "freq:key" → base number
   * @returns {Object} { scaledValues: {"freq:key": number}, annualImpact: number }
   */
  function applySliderMultiplier(sliderName, multiplier, baseValues) {
    var items = SLIDER_MAPPING[sliderName] || [];
    var scaledValues = {};
    var annualImpact = 0;

    for (var i = 0; i < items.length; i++) {
      var freq = items[i].freq;
      var key = items[i].key;
      var baseKey = freq + ':' + key;
      var base = baseValues[baseKey];
      if (base === undefined) continue;
      var scaled = Math.round(base * multiplier);
      scaledValues[baseKey] = scaled;
      annualImpact += scaled * (FREQ_TO_ANNUAL[freq] || 1);
    }

    return { scaledValues: scaledValues, annualImpact: annualImpact };
  }

  // ── Public API ────────────────────────────────────────────

  return {
    // Data
    CONSTANTS: CONSTANTS,
    FEDERAL_SINGLE: FEDERAL_SINGLE,
    FEDERAL_MARRIED: FEDERAL_MARRIED,
    NY_STATE: NY_STATE,
    NYC_TAX: NYC_TAX,
    FREQ_TO_ANNUAL: FREQ_TO_ANNUAL,
    SLIDER_MAPPING: SLIDER_MAPPING,
    PRESETS: PRESETS,

    // Formatting
    fmt: fmt,
    fmtk: fmtk,
    pct: pct,
    parseInputValue: parseInputValue,

    // Computation
    calcBrackets: calcBrackets,
    computeTaxes: computeTaxes,
    computeSpending: computeSpending,
    computeBudget: computeBudget,

    // Sliders
    sliderToMultiplier: sliderToMultiplier,
    getSliderForItem: getSliderForItem,
    applySliderMultiplier: applySliderMultiplier,
  };
})();

// Node.js / CommonJS export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = TaxEngine;
}
