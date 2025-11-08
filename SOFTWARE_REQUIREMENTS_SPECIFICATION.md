# Three-Metric Crisis Prediction System: Software Requirements Specification
**Critical Evaluation of a Financial Crisis Prediction Framework**

Evaluator: Technical Analysis
Date: 2025-11-08

---

## EXECUTIVE SUMMARY

### What is Being Claimed

A proposed financial crisis prediction system claims to predict market crashes through three coupled dynamical metrics combined with an entropy-adaptive memory parameter (λ). The system has been tested on multiple assets (S&P 500, Bitcoin, Gold, Bonds) across historical crises (2008 GFC, COVID-19 crash, 2022 bear market).

**Key Claim**: The system can identify impending market crashes by combining:
1. Market entropy (disorder in returns)
2. Temporal correlation (autocorrelation patterns)
3. Returns-Reversal Precedence (volume-volatility relationship)
4. Adaptive memory scaling (λ parameter)

### Critical Context

**IMPORTANT**: The same author has published a methodologically rigorous paper admitting that a similar momentum-based prediction system suffered from severe **data leakage**, resulting in a claimed 45% improvement that was actually a 31% degradation when properly validated.

> "Initial testing suggested 45% improvement over baseline on chaotic systems. However, rigorous validation with reversed-time testing revealed this result stemmed from data leakage." (honest_paper, p.1)

This raises immediate skepticism about the current system's claims, demanding rigorous validation.

---

## CORE HYPOTHESES

Stripping away the obtuse language, here are the actual testable hypotheses:

### H1: Entropy Spike Hypothesis
**Claim**: Market entropy (RERL) increases before crashes.

**Translation**: When returns become more unpredictable (higher disorder), a crash is coming.

**Testable**: Does RERL statistically increase in the period 20-60 days before known crashes compared to normal periods?

**From code** (implementation file 1:68-69):
```python
histo, _ = np.histogram(recent, bins=ent_bins, density=True)
h = entropy(histo + 1e-12) / np.log(len(histo))
```

### H2: Correlation Breakdown Hypothesis
**Claim**: Autocorrelation patterns (RTCL) change before crises.

**Translation**: Markets stop following their recent trends before crashes.

**Testable**: Does RTCL show statistically significant changes before crashes?

**From code** (implementation file 2:51-60):
```python
def rolling_autocorr(returns, window=252, lag=1):
    # Computes correlation between returns[t] and returns[t-lag]
    out[i] = np.corrcoef(seg[:-lag], seg[lag:])[0,1]
```

### H3: Volume-Volatility Precedence Hypothesis
**Claim**: Volume surges precede volatility spikes (RRP metric).

**Translation**: Trading volume predicts future price swings.

**Testable**: Does lagged volume correlation with future volatility increase before crashes?

**From code** (implementation file 2:62-73):
```python
def rolling_rrp(volume, returns_abs, window=252, lag=LAG_RRP):
    # Correlates lagged volume with current volatility
    vseg = v.iloc[i-window-lag:i-lag].values
    aseg = a.iloc[i-window:i].values
    out[i] = np.corrcoef(vseg, aseg)[0,1]
```

### H4: Adaptive Memory Hypothesis
**Claim**: Market "memory" (λ parameter) should lengthen during high entropy (chaos).

**Translation**: When markets are chaotic, you need to look further back in history.

**From code** (implementation file 1:71):
```python
lam += alpha if h > thresh else -alpha * decay
```

**Testable**: Does increasing λ during high-entropy periods improve prediction accuracy?

### H5: Composite Crisis Index (CCI) Hypothesis
**Claim**: Combining λ, PTE (phase transition entropy), and FCI creates a crisis predictor.

**Translation**: The weighted combination of these metrics predicts forward drawdowns.

**From code** (implementation file 2:135-136):
```python
CCI = zscore(df["Lambda"]) + zscore(df["PTE"]) - zscore(df["FCI"])
```

**Testable**: Does CCI correlate with future drawdowns better than naive baselines?

**Author's claim** (implementation file 2:203-204):
```
Global Spearman(CCI_global, FwdDD_avg) = {rho:.3f}  (p = {pval:.1e})
Global PR-AUC (CCI_global → any-asset crash) = {pr_auc:.3f}
```

---

## MATHEMATICAL FRAMEWORK

### 1. Rolling Entropy (RERL)
**Purpose**: Measure disorder in return distribution

**Formula**:
```
For window of returns r[i-W:i]:
  - Bin returns into histogram with B bins
  - Compute probabilities: p_k = count_k / total
  - RERL(i) = -Σ(p_k * log₂(p_k))
```

**Python Implementation** (implementation file 2:40-49):
```python
def rolling_entropy(returns, window=252, bins=21):
    r = pd.Series(returns).fillna(0.0)
    ent = np.full(len(r), np.nan)
    for i in range(window, len(r)):
        seg = r.iloc[i-window:i].values
        counts, _ = np.histogram(seg, bins=bins)
        p = counts / max(counts.sum(), 1)
        p = np.clip(p, 1e-12, 1.0)
        ent[i] = -(p * np.log2(p)).sum()
    return pd.Series(ent, index=r.index)
```

**Units**: bits (information theory)
**Range**: [0, log₂(bins)] ≈ [0, 4.4] for 21 bins

### 2. Rolling Temporal Correlation (RTCL)
**Purpose**: Measure autocorrelation at lag-1

**Formula**:
```
RTCL(i) = corr(r[i-W:i-lag], r[i-W+lag:i])
```

**Python Implementation** (implementation file 2:51-60):
```python
def rolling_autocorr(returns, window=252, lag=1):
    r = pd.Series(returns).fillna(0.0)
    out = np.full(len(r), np.nan)
    for i in range(window, len(r)):
        seg = r.iloc[i-window:i].values
        if np.std(seg[:-lag]) < 1e-12 or np.std(seg[lag:]) < 1e-12:
            out[i] = np.nan
        else:
            out[i] = np.corrcoef(seg[:-lag], seg[lag:])[0,1]
    return pd.Series(out, index=r.index)
```

**Units**: Pearson correlation coefficient
**Range**: [-1, 1]

### 3. Returns-Reversal Precedence (RRP)
**Purpose**: Measure how past volume predicts current volatility

**Formula**:
```
RRP(i) = corr(volume[i-W-L:i-L], |returns|[i-W:i])
```
Where L = lag (default 20 days)

**Python Implementation** (implementation file 2:62-73):
```python
def rolling_rrp(volume, returns_abs, window=252, lag=20):
    v = pd.Series(volume).fillna(method="ffill").fillna(0.0)
    a = pd.Series(returns_abs).fillna(0.0)
    out = np.full(len(v), np.nan)
    for i in range(window+lag, len(v)):
        vseg = v.iloc[i-window-lag:i-lag].values  # Past volume
        aseg = a.iloc[i-window:i].values           # Current volatility
        if np.std(vseg) < 1e-12 or np.std(aseg) < 1e-12:
            out[i] = np.nan
        else:
            out[i] = np.corrcoef(vseg, aseg)[0,1]
    return pd.Series(out, index=v.index)
```

**Units**: Pearson correlation coefficient
**Range**: [-1, 1]

### 4. Financial Chaos Index (FCI)
**Purpose**: Combine entropy and correlation

**Formula**:
```
FCI = z(-RERL) + z(RTCL)
```
Where z(x) = (x - mean(x)) / std(x)

**Interpretation**:
- High FCI = Low entropy + High correlation = Stable market
- Low FCI = High entropy + Low correlation = Chaotic market

**Python Implementation** (implementation file 2:123):
```python
FCI = zscore(-RERL) + zscore(RTCL)
```

### 5. Phase Transition Entropy (PTE)
**Purpose**: Measure rate of change in the three metrics

**Formula**:
```
PTE = sqrt( (ΔRERL² + ΔRTCL² + ΔRRP²) / 3 )
```
Where Δ represents first difference (change from previous timestep)

**Python Implementation** (implementation file 2:124):
```python
PTE = np.sqrt((RERL.diff()**2 + RTCL.diff()**2 + RRP.diff()**2)/3.0)
```

**Interpretation**: High PTE = Rapid changes in market structure = Potential phase transition

### 6. Lambda (λ) - Adaptive Memory Parameter
**Purpose**: Dynamically adjust lookback window based on market entropy

**Formula** (implementation file 1:61-72):
```
weights[τ] = exp(-τ / λ)

If entropy > threshold:
    λ ← λ + α              # Lengthen memory
Else:
    λ ← λ - α * decay      # Shorten memory

λ ← clip(λ, λ_min, λ_max)
```

**Alternative Formula** (implementation file 2:75-88):
```
For each window of prices:
    errors = prediction_errors_over_window
    error_volatility = std(errors)
    error_mean = mean(errors)

    λ = 2.0 + (error_volatility / error_mean) * scale
    λ = clip(λ, 2.0, 30.0)
```

**Critical Question**: Which formula is actually being used? The two implementations differ significantly.

### 7. Composite Crisis Index (CCI)
**Purpose**: Unified crisis predictor

**Formula** (implementation file 2:135-136):
```
CCI = z(λ) + z(PTE) - z(FCI)
```

**Interpretation**:
- High CCI = Long memory + Rapid change + Low stability = CRISIS WARNING
- Low CCI = Short memory + Slow change + High stability = NORMAL

### 8. Forward Drawdown (Validation Metric)
**Purpose**: Measure actual future losses

**Formula** (implementation file 2:90-96):
```
For each time i:
    future_prices = prices[i : i+horizon]
    DD(i) = (min(future_prices) - prices[i]) / prices[i]
```

**Crash Definition** (implementation file 2:26):
```
Crash = (forward_drawdown ≤ -10%)
```

---

## SYSTEM ARCHITECTURE

### Data Flow

```
RAW DATA (OHLCV)
    ↓
LOG RETURNS + VOLUME
    ↓
PARALLEL COMPUTATION:
    ├─→ RERL (Rolling Entropy)
    ├─→ RTCL (Autocorrelation)
    └─→ RRP (Volume-Volatility Lag)
    ↓
DERIVED METRICS:
    ├─→ FCI = z(-RERL) + z(RTCL)
    ├─→ PTE = sqrt(ΔRERL² + ΔRTCL² + ΔRRP²)/√3
    └─→ λ = f(error_volatility)
    ↓
COMPOSITE CRISIS INDEX:
    CCI = z(λ) + z(PTE) - z(FCI)
    ↓
VALIDATION:
    Spearman(CCI, ForwardDrawdown)
    PR-AUC(CCI, CrashLabel)
```

### Component Dependencies

1. **Base Layer**: Returns and volume calculation
2. **Three-Metric Layer**: RERL, RTCL, RRP (independent, can run in parallel)
3. **Synthesis Layer**: FCI, PTE, λ (depends on Three-Metric)
4. **Index Layer**: CCI (depends on Synthesis)
5. **Validation Layer**: Forward-looking metrics (independent)

### Multi-Asset Fusion (implementation file 2:159-182)

```
Per-Asset Processing:
    Each asset → CCI_asset + ForwardDD_asset

Global Aggregation:
    CCI_global = mean(CCI_asset1, CCI_asset2, ...)
    Crash_any = any(ForwardDD_asset ≤ -10%)

Validation:
    Spearman(CCI_global, avg(ForwardDD))
    PR-AUC(CCI_global, Crash_any)
```

---

## DATA LEAKAGE RISKS

### Critical Issues from Author's Data Leakage Paper

The author's admission of data leakage in a related recursive momentum system raises red flags for the current system:

> "Root cause: Comparing predictions to current rather than t+h creates zero-step 'forecasting' that measures current-state tracking rather than future prediction." (honest_paper, p.1)

### Potential Leakage Points in Current System

#### 1. **Normalization Leakage**
**Risk**: Computing z-scores on entire dataset including test period

**From data leakage paper** (p.2):
> "Normalization leakage. Computing statistics (mean, standard deviation) on entire datasets including test data, then applying to training data, leaks information about future distributions backward in time."

**In Current Code** (implementation file 2:36-38):
```python
def zscore(s):
    s = pd.Series(s)
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)  # ← ENTIRE SERIES
```

**CRITICAL**: This z-score function operates on the entire time series, violating causal prediction.

#### 2. **Forward-Looking Window Overlap**
**Risk**: Using future data in rolling windows

**From data leakage paper** (p.2):
> "Window overlap. When training and test sequences share overlapping observations, recent training data contaminates initial test predictions."

**In Current Code**: Rolling windows of 252 days mean the first 252 days of any "test" period contain training data.

#### 3. **Horizon Confusion**
**Risk**: Unclear prediction horizon in λ computation

**From code** (implementation file 2:75-88):
The λ computation uses errors "over recent window" but doesn't specify if these are in-sample or out-of-sample errors.

#### 4. **Autocorrelation Artifacts**
**From data leakage paper** (p.2):
> "Autocorrelation artifacts. Chaotic systems exhibit strong short-term autocorrelation. Methods can appear predictive by exploiting this structure without genuine forecasting."

Financial markets have strong autocorrelation. The system may be measuring synchronous correlation, not predictive power.

### Required Leakage Tests

Based on the author's data leakage paper methodology (p.2), the current system MUST implement:

1. **Reversed-Time Test**:
```python
r_forward = corr(CCI(t), ForwardDD(t+h))
r_reversed = corr(CCI(T-t), ForwardDD(T-t-h))  # Process backward

if r_reversed > r_forward:
    print("DATA LEAKAGE DETECTED")
```

2. **Shuffled Future Test**:
```python
# Randomly permute future drawdown values
# Correlation should approach zero
r_shuffled = corr(CCI(t), shuffle(ForwardDD(t+h)))
if r_shuffled > 0.1:
    print("POSSIBLE SPURIOUS CORRELATION")
```

3. **Walk-Forward Validation**:
```python
# Train on [0:T], normalize on [0:T] ONLY
# Test on [T:], apply normalization from training period
# NO access to test statistics
```

---

## TESTABLE REQUIREMENTS

### REQ-1: Data Integrity
**Requirement**: Strict temporal separation between training and testing

**Test**:
```python
assert train_end < test_start
assert all(normalization_params computed from train_data only)
assert no(rolling_windows[test_period] overlap with train_period)
```

**Pass Criteria**: Zero temporal contamination

### REQ-2: Entropy-Crisis Correlation
**Hypothesis H1 Test**

**Requirement**: RERL increases before crashes

**Test**:
```python
pre_crash_entropy = RERL[-60:-20 days before crash]
normal_entropy = RERL[normal periods]

t_stat, p_value = ttest_ind(pre_crash_entropy, normal_entropy)
```

**Pass Criteria**: p < 0.05 with pre_crash > normal

### REQ-3: CCI Predictive Power
**Hypothesis H5 Test**

**Requirement**: CCI correlates with forward drawdowns

**Test**:
```python
rho, p_val = spearmanr(CCI[t], ForwardDD[t+20])
```

**Pass Criteria**:
- rho < -0.3 (negative correlation, higher CCI → worse future returns)
- p < 0.01

### REQ-4: Lambda Adaptation Benefit
**Hypothesis H4 Test**

**Requirement**: Adaptive λ outperforms fixed λ

**Test**:
```python
# Compare prediction accuracy
rmse_adaptive = evaluate(adaptive_lambda)
rmse_fixed = evaluate(lambda_fixed=5.0)

improvement = (rmse_fixed - rmse_adaptive) / rmse_fixed
```

**Pass Criteria**: improvement > 10%

### REQ-5: Multi-Asset Robustness
**Requirement**: CCI works across asset classes

**Test**:
```python
for asset in [SP500, BTC, GOLD, BONDS]:
    rho, p = spearmanr(CCI[asset], ForwardDD[asset])
    assert rho < -0.2 and p < 0.05
```

**Pass Criteria**: Significant correlation for ≥ 3 of 4 assets

### REQ-6: Crisis Period Performance
**Requirement**: CCI elevates before known crises

**Test**:
```python
crisis_windows = {
    "2008 GFC": ("2007-10-01", "2009-03-01"),
    "COVID": ("2020-02-15", "2020-05-15"),
    "2022 Bear": ("2022-01-01", "2022-10-31")
}

for crisis, (start, end) in crisis_windows.items():
    pre_crisis_CCI = CCI[start - 60days : start]
    assert mean(pre_crisis_CCI) > threshold
```

**Pass Criteria**: CCI > 75th percentile before each crisis

### REQ-7: Permutation Test Significance
**Requirement**: Results exceed chance levels

**Test** (from data leakage paper, p.2):
```python
observed_diff = mean(RERL[crisis]) - mean(RERL[baseline])
null_distribution = []

for i in range(10000):
    shuffled = shuffle(concat(RERL[crisis], RERL[baseline]))
    null_distribution.append(permutation_difference)

p_value = (sum(abs(null_distribution) >= abs(observed_diff)) + 1) / (10000 + 1)
```

**Pass Criteria**: p < 0.01

---

## CRITICAL QUESTIONS

### Methodological Concerns

**Q1**: Why are there TWO different λ formulas?
- Implementation file 1 uses entropy-driven adaptation: `λ += α if entropy > thresh`
- Implementation file 2 uses error-ratio: `λ = 2 + (std/mean) * scale`

Which one is the actual system?

**Q2**: How is causality enforced?
The code shows:
```python
def zscore(s):
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)
```
This computes statistics on the ENTIRE series, including future data. This is the EXACT mistake admitted in the data leakage paper.

**Q3**: What is the prediction horizon?
- Forward drawdown uses horizon = 20 days
- But CCI is computed using current values
- Is this predicting t+20 or just correlating with simultaneous drawdown?

**Q4**: Why does the data leakage paper show ML outperforms recursive methods by 21%, while the current system uses a recursive approach?

**Q5**: What is the naive baseline comparison?
- Data leakage paper uses persistence: `forecast[t+h] = value[t]`
- Current code doesn't show baseline comparison
- How do we know CCI beats random?

### Statistical Concerns

**Q6**: Multiple hypothesis testing correction?
Testing 4 assets × 3 crises × 5 metrics = 60 hypotheses
Without Bonferroni correction, p=0.05 becomes meaningless.

**Q7**: Survivorship bias?
Testing on "known" crises (2008, COVID, 2022) that are already in the historical record.
Would it have predicted 2025 crisis BEFORE it happened?

**Q8**: Asset selection bias?
Why SP500, BTC, Gold, TLT? Were other assets tested and excluded for poor performance?

**Q9**: Parameter overfitting?
The code has many hardcoded values:
- `WIN_ENT = 252`
- `ENT_BINS = 21`
- `LAG_RRP = 20`
- `WIN_LAMB = 50`
- `LAMB_SCALE = 15.0`
- `CRISIS_DD = -0.10`

Were these tuned on the same data being tested?

**Q10**: Why are permutation tests only done per-asset (N=3000) and not globally?

---

## IMPLEMENTATION CHECKLIST

### Phase 1: Data Preparation (No Leakage)
- [ ] Download OHLCV data for all assets
- [ ] Split into train/test with strict temporal separation
- [ ] Compute normalization parameters ON TRAIN DATA ONLY
- [ ] Store test data in separate, untouched structure

### Phase 2: Core Metric Implementation
- [ ] Implement `rolling_entropy()` with causal windows
- [ ] Implement `rolling_autocorr()` with causal windows
- [ ] Implement `rolling_rrp()` with causal lag
- [ ] Unit test each function with synthetic data

### Phase 3: Derived Metrics
- [ ] Implement FCI with train-only z-score normalization
- [ ] Implement PTE (first differences)
- [ ] Implement λ computation (CLARIFY which formula)
- [ ] Implement CCI composition

### Phase 4: Leakage Testing
- [ ] Reversed-time test
- [ ] Shuffled-future test
- [ ] Walk-forward cross-validation
- [ ] Permutation significance testing

### Phase 5: Hypothesis Testing
- [ ] Test H1: Entropy before crashes
- [ ] Test H2: Correlation breakdown
- [ ] Test H3: Volume-volatility precedence
- [ ] Test H4: Lambda adaptation benefit
- [ ] Test H5: CCI predictive power

### Phase 6: Validation
- [ ] Compare to naive persistence baseline
- [ ] Compare to simple momentum baseline
- [ ] Compare to ML methods (per data leakage paper)
- [ ] Multiple hypothesis testing correction
- [ ] Out-of-sample forward testing on NEW data

---

## REFERENCE QUOTES

### From implementation file 1:
```
"# Entropy–Driven λ Adaptation on S&P 500"
(Line 2)
```

### From implementation file 2:
```
"# Multi-asset crisis index with publication-grade plots & stats"
(Lines 2-3)

"Global Spearman(CCI_global, FwdDD_avg) = {rho:.3f}  (p = {pval:.1e})"
(Line 203)

"Global PR-AUC (CCI_global → any-asset crash) = {pr_auc:.3f}"
(Line 204)
```

### From data leakage paper:
```
"Initial testing suggested 45% improvement over baseline. However, rigorous
validation with reversed-time testing revealed this result stemmed from data
leakage." (Abstract, p.1)

"Proper causal testing showed the method performed 31% worse than naive baseline"
(Abstract, p.1)

"The method was comparing predictions to time t rather than t+h, measuring
synchronous tracking rather than forecasting." (Results, p.3)

"Reversed-time testing provides practical leakage detection. Implementation
requires <50 lines of code but catches contamination missed by standard
validation." (Discussion, p.3)

"Always test naive baselines. Persistence forecasts are surprisingly strong
due to autocorrelation." (Recommendations, p.4)
```

### From original positive results paper:
```
"45.5% RMSE reduction over naive persistence baseline on Duffing oscillator
10-step ahead forecasting (p < 0.0001, permutation test)" (Abstract, p.1)

"This performance variation establishes boundary conditions for recursive
prediction: effectiveness correlates with dynamical smoothness and momentum
continuity in continuous flows." (Conclusion, p.5)
```

---

## RECOMMENDED TESTING APPROACH

### 1. Reproduce Existing Code First
- Run implementation files as-is
- Generate all plots and statistics
- Document exactly what results are produced

### 2. Implement Leakage Tests
Based on author's data leakage paper methodology:

```python
def test_reversed_time_leakage(CCI, ForwardDD):
    """
    From data leakage paper (p.2):
    'Process data backward: if correlation with "future" (actually past)
    exceeds forward correlation, leakage is present.'
    """
    T = len(CCI)
    r_forward = np.corrcoef(CCI[:-20], ForwardDD[20:])[0,1]

    # Reverse time
    CCI_rev = CCI[::-1]
    DD_rev = ForwardDD[::-1]
    r_reversed = np.corrcoef(CCI_rev[:-20], DD_rev[20:])[0,1]

    print(f"Forward correlation: {r_forward:.3f}")
    print(f"Reversed correlation: {r_reversed:.3f}")

    if r_reversed > r_forward:
        print("⚠️  DATA LEAKAGE DETECTED")
        return False
    return True
```

### 3. Implement Causal Walk-Forward Validation

```python
def walk_forward_validation(data, train_size=2000, test_size=400):
    """
    From data leakage paper (p.2):
    'Training data strictly precedes test data with no overlap.
    Normalization parameters computed only on training data.'
    """
    results = []

    for i in range(0, len(data) - train_size - test_size, test_size):
        # Train period
        train_data = data[i : i+train_size]
        train_mean = train_data.mean()
        train_std = train_data.std()

        # Test period (strictly after train)
        test_data = data[i+train_size : i+train_size+test_size]

        # Apply train normalization to test (NO LEAKAGE)
        test_normalized = (test_data - train_mean) / train_std

        # Compute metrics
        CCI_test = compute_cci(test_normalized)
        DD_test = forward_drawdown(test_data, horizon=20)

        rho, p = spearmanr(CCI_test[:-20], DD_test[20:])
        results.append({'rho': rho, 'p': p})

    return results
```

### 4. Baseline Comparisons

```python
def compare_baselines(data):
    """Compare CCI against naive baselines"""

    # Baseline 1: Persistence (no change)
    baseline_1 = persistence_forecast(data)

    # Baseline 2: Historical volatility
    baseline_2 = historical_vol_forecast(data)

    # Baseline 3: Random walk
    baseline_3 = random_walk_forecast(data)

    # Proposed system CCI
    proposed = compute_full_pipeline(data)

    # Compare RMSE
    for name, forecast in [('Persistence', baseline_1),
                           ('HistVol', baseline_2),
                           ('Random', baseline_3),
                           ('Proposed', proposed)]:
        rmse = compute_rmse(forecast)
        print(f"{name}: RMSE = {rmse:.4f}")
```

---

## CONCLUSION

The proposed three-metric crisis prediction system makes testable claims about financial crisis prediction through entropy, correlation, and volume dynamics. However, the author's admission of data leakage in a parallel system raises serious methodological concerns.

### Required Before Acceptance:
1. ✅ **Leakage testing**: Reversed-time, shuffled-future, walk-forward
2. ✅ **Baseline comparison**: Must beat naive persistence
3. ✅ **Causal validation**: Strict temporal separation, train-only normalization
4. ✅ **Statistical rigor**: Permutation tests, multiple hypothesis correction
5. ✅ **Out-of-sample testing**: Apply to NEW data not used in development

### Key Red Flags:
- ⚠️ Z-score computed on entire series (potential leakage)
- ⚠️ Two contradictory λ formulas (unclear which is real)
- ⚠️ No naive baseline comparison shown
- ⚠️ Testing on known crises (selection bias)
- ⚠️ Many hardcoded parameters (overfitting risk)

### Verdict:
**SKEPTICAL BUT TESTABLE**. The hypotheses are clearly stated and mathematically specified. The system can be rigorously validated IF proper causal methodology is enforced. The author's honesty about past failures is encouraging, but demands extra scrutiny of this new system.
