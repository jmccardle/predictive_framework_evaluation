# Three-Metric Crisis Prediction System: Test Program Specification
**Rigorous Validation Framework with Leakage Detection**

Based on methodologies from:
- Data leakage detection framework (honest_paper)
- Statistical validation methods (original_paper)
- Python implementations (implementation files 1 & 2)

---

## OVERVIEW

This document specifies a rigorous test program to validate (or refute) a proposed financial crisis prediction system. The testing framework is designed to catch data leakage, spurious correlations, and overfitting.

### Testing Philosophy

> "Publishing this failure serves multiple purposes: Methodological contribution. Reversed-time testing provides practical leakage detection. Implementation requires <50 lines of code but catches contamination missed by standard validation."
>
> — Data leakage paper, Section 5.1, p.3

### Critical Requirement

> "Always test naive baselines. Persistence forecasts are surprisingly strong due to autocorrelation."
>
> — Data leakage paper, Recommendations #1, p.4

---

## TEST SUITE ARCHITECTURE

```
TEST_SUITE/
├── 1_data_integrity_tests/
│   ├── test_temporal_separation.py
│   ├── test_no_future_leakage.py
│   └── test_normalization_causality.py
│
├── 2_leakage_detection_tests/
│   ├── test_reversed_time.py
│   ├── test_shuffled_future.py
│   ├── test_permutation_significance.py
│   └── test_autocorrelation_artifact.py
│
├── 3_hypothesis_validation_tests/
│   ├── test_h1_entropy_spike.py
│   ├── test_h2_correlation_breakdown.py
│   ├── test_h3_volume_volatility.py
│   ├── test_h4_lambda_adaptation.py
│   └── test_h5_cci_prediction.py
│
├── 4_baseline_comparison_tests/
│   ├── test_vs_persistence.py
│   ├── test_vs_moving_average.py
│   ├── test_vs_random_walk.py
│   └── test_vs_ml_methods.py
│
├── 5_robustness_tests/
│   ├── test_walk_forward_cv.py
│   ├── test_parameter_sensitivity.py
│   ├── test_multi_asset_consistency.py
│   └── test_out_of_sample.py
│
└── 6_statistical_rigor_tests/
    ├── test_multiple_hypothesis_correction.py
    ├── test_effect_size.py
    └── test_confidence_intervals.py
```

---

## 1. DATA INTEGRITY TESTS

### 1.1 Temporal Separation Test

**Purpose**: Ensure training data never contaminates test data

**Reference**:
> "Proper train/test split. Training data (n = 2000) strictly precedes test data (n = 400) with no overlap. Normalization parameters computed only on training data."
>
> — Data leakage paper, Section 3.3, p.2

**Implementation**:
```python
def test_temporal_separation():
    """
    Verify strict temporal ordering and no overlap
    """
    # Load data
    df, src = load_sp500(start="1998-01-01")
    prices = df["Adj Close"].values
    dates = pd.to_datetime(df["Date"])

    # Define split
    TRAIN_SIZE = 2000
    TEST_SIZE = 400
    train_end_idx = TRAIN_SIZE
    test_start_idx = TRAIN_SIZE  # No gap, but no overlap

    # Test 1: No temporal overlap
    train_dates = dates[:train_end_idx]
    test_dates = dates[test_start_idx:test_start_idx+TEST_SIZE]

    assert train_dates.max() <= test_dates.min(), \
        f"FAIL: Train data extends to {train_dates.max()}, " \
        f"test starts at {test_dates.min()}"

    # Test 2: No index overlap
    train_indices = set(range(train_end_idx))
    test_indices = set(range(test_start_idx, test_start_idx+TEST_SIZE))

    overlap = train_indices & test_indices
    assert len(overlap) == 0, \
        f"FAIL: {len(overlap)} indices overlap between train/test"

    print("✓ PASS: Temporal separation verified")
    return True
```

**Expected Result**: PASS

**Failure Mode**: If test data leaks into training period

---

### 1.2 Normalization Causality Test

**Purpose**: Detect normalization leakage

**Reference**:
> "Normalization leakage. Computing statistics (mean, standard deviation) on entire datasets including test data, then applying to training data, leaks information about future distributions backward in time."
>
> — Data leakage paper, Section 2.1, p.2

**Problem in Existing Code**:
```python
# From implementation file 2
def zscore(s):
    s = pd.Series(s)
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)  # ← BUG: Uses entire series!
```

**Correct Implementation**:
```python
def test_normalization_causality():
    """
    Verify z-score normalization uses ONLY training statistics
    """
    df, _ = load_sp500()
    prices = df["Adj Close"].values

    TRAIN_SIZE = 2000

    # CORRECT: Compute stats on train only
    train_prices = prices[:TRAIN_SIZE]
    train_mean = np.mean(train_prices)
    train_std = np.std(train_prices, ddof=0)

    # Apply to test
    test_prices = prices[TRAIN_SIZE:TRAIN_SIZE+400]
    test_normalized_correct = (test_prices - train_mean) / train_std

    # INCORRECT: Compute stats on entire series (LEAKAGE)
    all_mean = np.mean(prices[:TRAIN_SIZE+400])
    all_std = np.std(prices[:TRAIN_SIZE+400], ddof=0)
    test_normalized_leaked = (test_prices - all_mean) / all_std

    # These should be DIFFERENT
    difference = np.abs(test_normalized_correct - test_normalized_leaked).mean()

    print(f"Mean difference: {difference:.6f}")
    assert difference > 0.01, \
        "WARNING: Normalization appears to use train-only stats (good), " \
        "but verify implementation"

    # Check if code uses leaky version
    import inspect
    zscore_source = inspect.getsource(zscore)
    if ".mean()" in zscore_source and "[:TRAIN_SIZE]" not in zscore_source:
        print("⚠️  WARNING: zscore() function may have normalization leakage")
        print("   Check that it receives train_mean/train_std as parameters")
        return False

    print("✓ PASS: Normalization uses train-only statistics")
    return True
```

**Expected Result**: Should detect leakage in current implementation

---

## 2. LEAKAGE DETECTION TESTS

### 2.1 Reversed-Time Test

**Purpose**: Detect temporal leakage by processing data backward

**Reference**:
> "Reversed-time leakage test (Novel). Process data backward: if correlation with 'future' (actually past) exceeds forward correlation, leakage is present."
>
> — Data leakage paper, Section 3.3, p.2

**Mathematical Definition**:
> "Compare: r_forward = corr(x̂(t), x(t+h)) vs r_reversed = corr(x̂(T-t), x(T-t-h)).
> If r_reversed > r_forward, the method accesses unavailable information."
>
> — Data leakage paper, Equations 7-8, p.2

**Implementation**:
```python
def test_reversed_time_leakage():
    """
    Novel leakage detection via time reversal

    If the system works equally well or BETTER when time runs backward,
    it's using information from the "future" (which is actually the past
    in reversed time).
    """
    # Load data and compute CCI
    df, _ = load_sp500(start="2000-01-01")
    prices = df["Adj Close"].values
    dates = pd.to_datetime(df["Date"])

    # Compute returns, metrics, CCI
    returns = np.diff(np.log(prices))
    volume = df["Volume"].values[1:]  # Align with returns

    # FORWARD TIME: Normal computation
    RERL_fwd = rolling_entropy(returns, window=252, bins=21)
    RTCL_fwd = rolling_autocorr(returns, window=252, lag=1)
    RRP_fwd = rolling_rrp(volume, np.abs(returns), window=252, lag=20)

    # Compute CCI forward
    CCI_fwd = compute_cci(RERL_fwd, RTCL_fwd, RRP_fwd)

    # Compute forward drawdown
    FWD_DD = forward_drawdown(prices[1:], horizon=20)

    # Forward correlation (CAUSAL)
    valid_fwd = ~(np.isnan(CCI_fwd) | np.isnan(FWD_DD))
    r_forward, p_forward = spearmanr(
        CCI_fwd[valid_fwd][:-20],
        FWD_DD[valid_fwd][20:]
    )

    # REVERSED TIME: Process backward
    returns_rev = returns[::-1]
    volume_rev = volume[::-1]

    RERL_rev = rolling_entropy(returns_rev, window=252, bins=21)
    RTCL_rev = rolling_autocorr(returns_rev, window=252, lag=1)
    RRP_rev = rolling_rrp(volume_rev, np.abs(returns_rev), window=252, lag=20)

    CCI_rev = compute_cci(RERL_rev, RTCL_rev, RRP_rev)

    # In reversed time, "forward" drawdown is actually looking backward
    FWD_DD_rev = forward_drawdown(prices[::-1][1:], horizon=20)

    valid_rev = ~(np.isnan(CCI_rev) | np.isnan(FWD_DD_rev))
    r_reversed, p_reversed = spearmanr(
        CCI_rev[valid_rev][:-20],
        FWD_DD_rev[valid_rev][20:]
    )

    # CRITICAL TEST
    print(f"Forward correlation:  ρ = {r_forward:.4f} (p = {p_forward:.4f})")
    print(f"Reversed correlation: ρ = {r_reversed:.4f} (p = {p_reversed:.4f})")
    print(f"Difference: {abs(r_reversed) - abs(r_forward):.4f}")

    # From data leakage paper case study:
    # Forward (causal): r = 0.569
    # Reversed: r = 0.959  ← LEAKAGE!

    if abs(r_reversed) > abs(r_forward):
        print("⚠️  FAIL: REVERSED CORRELATION HIGHER → DATA LEAKAGE DETECTED")
        print("   System performs better in reversed time = using future information")
        return False

    if abs(r_reversed) > 0.8 * abs(r_forward):
        print("⚠️  WARNING: Reversed correlation suspiciously high")
        print("   May indicate subtle leakage or strong autocorrelation artifact")
        return False

    print("✓ PASS: Forward correlation > Reversed correlation (no leakage)")
    return True
```

**Expected Result**: Unknown. If FAIL → serious leakage problem

---

### 2.2 Shuffled Future Test

**Purpose**: Verify predictions don't work with randomized future

**Reference**:
> "Shuffled future test: Randomly permute future values; correlation should approach zero"
>
> — Data leakage paper, Section 5.2, p.4

**Implementation**:
```python
def test_shuffled_future():
    """
    If CCI truly predicts forward drawdown, shuffling future values
    should destroy the correlation.

    If correlation remains high with shuffled future, the system is
    correlating with CURRENT state, not predicting FUTURE state.
    """
    # Compute CCI and forward drawdown
    CCI = compute_full_pipeline()
    FWD_DD = forward_drawdown(prices, horizon=20)

    # Original correlation
    valid = ~(np.isnan(CCI) | np.isnan(FWD_DD))
    r_original, p_original = spearmanr(CCI[valid][:-20], FWD_DD[valid][20:])

    # Shuffle future values (destroy temporal structure)
    rng = np.random.default_rng(42)
    FWD_DD_shuffled = FWD_DD.copy()
    FWD_DD_shuffled[20:] = rng.permutation(FWD_DD[20:])

    # Correlation with shuffled future
    r_shuffled, p_shuffled = spearmanr(
        CCI[valid][:-20],
        FWD_DD_shuffled[valid][20:]
    )

    print(f"Original correlation:  ρ = {r_original:.4f} (p = {p_original:.4f})")
    print(f"Shuffled correlation:  ρ = {r_shuffled:.4f} (p = {p_shuffled:.4f})")
    print(f"Correlation destroyed: {abs(r_original) - abs(r_shuffled):.4f}")

    # Shuffled correlation should be near zero
    if abs(r_shuffled) > 0.1:
        print("⚠️  FAIL: Shuffled future still shows correlation")
        print("   System may be correlating with current state, not predicting")
        return False

    if p_shuffled < 0.05:
        print("⚠️  FAIL: Shuffled future is statistically significant")
        print("   This should not happen if prediction is genuinely forward-looking")
        return False

    print("✓ PASS: Shuffled future destroys correlation")
    return True
```

**Expected Result**: Should PASS if system is truly predictive

---

### 2.3 Permutation Significance Test

**Purpose**: Verify results exceed chance levels

**Reference**:
> "Permutation testing. 10,000 random permutations establish null distribution for correlation differences, computing exact p-values."
>
> — Data leakage paper, Section 3.3, p.2

**From Implementation**:
```python
# From implementation file 2, lines 142-156
# Per-asset entropy permutation test
pool = np.concatenate([g,b]); n_g=len(g)
rng  = np.random.default_rng(0)
diffs=[]
for _ in range(N_PERM):  # N_PERM = 3000
    rng.shuffle(pool)
    diffs.append(np.nanmean(pool[:n_g]) - np.nanmean(pool[n_g:]))
obs   = np.nanmean(g) - np.nanmean(b)
pval  = (np.sum(np.abs(diffs) >= np.abs(obs)) + 1)/(N_PERM+1)
```

**Implementation**:
```python
def test_permutation_significance(n_permutations=10000):
    """
    Test if CCI-drawdown correlation exceeds chance levels
    """
    # Compute observed correlation
    CCI = compute_full_pipeline()
    FWD_DD = forward_drawdown(prices, horizon=20)

    valid = ~(np.isnan(CCI) | np.isnan(FWD_DD))
    CCI_valid = CCI[valid][:-20]
    DD_valid = FWD_DD[valid][20:]

    observed_rho, _ = spearmanr(CCI_valid, DD_valid)

    # Generate null distribution by permuting CCI labels
    rng = np.random.default_rng(42)
    null_distribution = []

    for i in range(n_permutations):
        # Permute CCI values (break temporal structure)
        CCI_permuted = rng.permutation(CCI_valid)
        rho_null, _ = spearmanr(CCI_permuted, DD_valid)
        null_distribution.append(rho_null)

    null_distribution = np.array(null_distribution)

    # Compute p-value (two-tailed)
    p_value = (np.sum(np.abs(null_distribution) >= np.abs(observed_rho)) + 1) / (n_permutations + 1)

    print(f"Observed ρ: {observed_rho:.4f}")
    print(f"Null mean: {null_distribution.mean():.4f}")
    print(f"Null std: {null_distribution.std():.4f}")
    print(f"Permutation p-value: {p_value:.6f}")

    # Plot null distribution
    plt.figure(figsize=(10, 6))
    plt.hist(null_distribution, bins=50, alpha=0.7, label='Null distribution')
    plt.axvline(observed_rho, color='red', linestyle='--', linewidth=2, label=f'Observed ρ = {observed_rho:.4f}')
    plt.xlabel('Spearman ρ')
    plt.ylabel('Frequency')
    plt.title(f'Permutation Test: p = {p_value:.6f}')
    plt.legend()
    plt.savefig('permutation_test_result.png', dpi=150)

    if p_value > 0.01:
        print(f"⚠️  FAIL: p = {p_value:.4f} > 0.01 (not significant)")
        return False

    print(f"✓ PASS: p = {p_value:.6f} < 0.01 (significant)")
    return True
```

**Expected Result**: Should PASS with p < 0.01

---

### 2.4 Autocorrelation Artifact Test

**Purpose**: Detect spurious correlation from autocorrelation

**Reference**:
> "Autocorrelation artifacts. Chaotic systems exhibit strong short-term autocorrelation. Methods can appear predictive by exploiting this structure without genuine forecasting."
>
> — Data leakage paper, Section 2.1, p.2

> "Autocorrelation dominance. At h=5 steps, Duffing oscillator exhibits r=0.83 autocorrelation. Simple persistence already captures 83% of predictable variance."
>
> — Data leakage paper, Section 4.3, p.3

**Implementation**:
```python
def test_autocorrelation_artifact():
    """
    Check if CCI is just measuring autocorrelation rather than
    predicting future changes
    """
    df, _ = load_sp500()
    prices = df["Adj Close"].values
    returns = np.diff(np.log(prices))

    # Measure autocorrelation at various lags
    lags = [1, 5, 10, 20, 40, 60]
    autocorrs = []

    for lag in lags:
        if lag < len(returns):
            r, p = pearsonr(returns[:-lag], returns[lag:])
            autocorrs.append(r)
            print(f"Lag {lag:2d}: r = {r:.4f}")

    # Compute CCI
    CCI = compute_full_pipeline()
    FWD_DD = forward_drawdown(prices, horizon=20)

    valid = ~(np.isnan(CCI) | np.isnan(FWD_DD))
    rho_cci, _ = spearmanr(CCI[valid][:-20], FWD_DD[valid][20:])

    # Compare CCI correlation to simple autocorrelation
    # If they're similar, CCI is just capturing autocorrelation
    autocorr_20 = autocorrs[lags.index(20)]

    print(f"\nAutocorrelation at lag 20: {autocorr_20:.4f}")
    print(f"CCI-DrawDown correlation: {rho_cci:.4f}")
    print(f"Improvement over autocorr: {abs(rho_cci) - abs(autocorr_20):.4f}")

    if abs(rho_cci) <= abs(autocorr_20) * 1.2:
        print("⚠️  WARNING: CCI provides minimal improvement over autocorrelation")
        print("   System may be exploiting temporal structure without genuine prediction")
        return False

    print("✓ PASS: CCI exceeds simple autocorrelation")
    return True
```

**Expected Result**: Should show CCI beats autocorrelation

---

## 3. HYPOTHESIS VALIDATION TESTS

### 3.1 Test H1: Entropy Spike Hypothesis

**Hypothesis**: Market entropy (RERL) increases before crashes

**Reference**:
```python
# From implementation file 2:40-49
def rolling_entropy(returns, window=252, bins=21):
    # Computes Shannon entropy of return distribution
    ent[i] = -(p * np.log2(p)).sum()
```

**Implementation**:
```python
def test_h1_entropy_spike():
    """
    Test if RERL statistically increases before crashes
    """
    df, _ = load_sp500(start="2000-01-01")
    prices = df["Adj Close"].values
    dates = pd.to_datetime(df["Date"])
    returns = np.diff(np.log(prices))

    # Compute RERL
    RERL = rolling_entropy(returns, window=252, bins=21)
    RERL_series = pd.Series(RERL, index=dates[1:])

    # Define crisis periods
    CRISES = {
        "2008 GFC":  ("2007-10-01", "2009-03-01"),
        "COVID":     ("2020-02-15", "2020-05-15"),
        "2022 Bear": ("2022-01-01", "2022-10-31"),
    }

    results = []

    for crisis_name, (start, end) in CRISES.items():
        # Pre-crisis period (60 days before)
        pre_start = pd.to_datetime(start) - pd.Timedelta(days=60)
        pre_end = pd.to_datetime(start)

        pre_crisis_mask = (RERL_series.index >= pre_start) & (RERL_series.index < pre_end)
        pre_crisis_entropy = RERL_series[pre_crisis_mask].dropna()

        # Normal period (2 years before pre-crisis)
        normal_start = pre_start - pd.Timedelta(days=730)
        normal_end = pre_start - pd.Timedelta(days=60)

        normal_mask = (RERL_series.index >= normal_start) & (RERL_series.index < normal_end)
        normal_entropy = RERL_series[normal_mask].dropna()

        if len(pre_crisis_entropy) > 10 and len(normal_entropy) > 10:
            # T-test
            t_stat, p_val = ttest_ind(pre_crisis_entropy, normal_entropy)

            mean_diff = pre_crisis_entropy.mean() - normal_entropy.mean()
            effect_size = mean_diff / normal_entropy.std()  # Cohen's d

            print(f"\n{crisis_name}:")
            print(f"  Normal entropy:     {normal_entropy.mean():.4f} ± {normal_entropy.std():.4f}")
            print(f"  Pre-crisis entropy: {pre_crisis_entropy.mean():.4f} ± {pre_crisis_entropy.std():.4f}")
            print(f"  Difference:         {mean_diff:.4f}")
            print(f"  Effect size (d):    {effect_size:.4f}")
            print(f"  t-statistic:        {t_stat:.4f}")
            print(f"  p-value:            {p_val:.6f}")

            results.append({
                'crisis': crisis_name,
                'mean_diff': mean_diff,
                'p_value': p_val,
                'effect_size': effect_size,
                'passed': mean_diff > 0 and p_val < 0.05
            })

    # Overall result
    passed_count = sum(r['passed'] for r in results)
    total_count = len(results)

    print(f"\n{'='*60}")
    print(f"H1 RESULT: {passed_count}/{total_count} crises show significant entropy increase")

    if passed_count >= 2:
        print("✓ PASS: Entropy spike hypothesis supported")
        return True
    else:
        print("⚠️  FAIL: Entropy spike hypothesis NOT supported")
        return False
```

**Pass Criteria**: ≥2 of 3 crises show significant increase (p < 0.05)

---

### 3.2 Test H5: CCI Predictive Power

**Hypothesis**: CCI correlates with forward drawdowns

**Reference**:
```python
# From implementation file 2:203-204
print(f"Global Spearman(CCI_global, FwdDD_avg) = {rho:.3f}  (p = {pval:.1e})")
print(f"Global PR-AUC (CCI_global → any-asset crash) = {pr_auc:.3f}")
```

**Implementation**:
```python
def test_h5_cci_prediction():
    """
    Test if CCI predicts forward drawdowns with proper causal validation
    """
    # Multi-asset test
    ASSETS = ["^GSPC", "BTC-USD", "GC=F", "TLT"]

    results = []

    for symbol in ASSETS:
        print(f"\nTesting {symbol}...")

        # Download data
        raw = yf.download(symbol, start="2000-01-01", end="2024-11-01",
                         progress=False)[["Close", "Volume"]].dropna()

        if raw.empty:
            print(f"  Skipping (no data)")
            continue

        prices = raw["Close"].values
        volume = raw["Volume"].values
        dates = pd.to_datetime(raw.index)

        # WALK-FORWARD VALIDATION (no leakage)
        train_size = 2000
        test_size = 400

        if len(prices) < train_size + test_size:
            print(f"  Skipping (insufficient data)")
            continue

        # Train period
        train_prices = prices[:train_size]
        train_volume = volume[:train_size]

        # Compute CCI on train data
        train_returns = np.diff(np.log(train_prices))
        train_RERL = rolling_entropy(train_returns, window=252, bins=21)
        train_RTCL = rolling_autocorr(train_returns, window=252, lag=1)
        train_RRP = rolling_rrp(train_volume[1:], np.abs(train_returns), window=252, lag=20)

        # Get normalization parameters from TRAIN only
        train_CCI = compute_cci(train_RERL, train_RTCL, train_RRP)
        cci_mean = np.nanmean(train_CCI)
        cci_std = np.nanstd(train_CCI)

        # Test period (apply train normalization)
        test_prices = prices[train_size:train_size+test_size]
        test_volume = volume[train_size:train_size+test_size]

        test_returns = np.diff(np.log(test_prices))
        test_RERL = rolling_entropy(test_returns, window=252, bins=21)
        test_RTCL = rolling_autocorr(test_returns, window=252, lag=1)
        test_RRP = rolling_rrp(test_volume[1:], np.abs(test_returns), window=252, lag=20)

        test_CCI_raw = compute_cci(test_RERL, test_RTCL, test_RRP)
        test_CCI = (test_CCI_raw - cci_mean) / cci_std  # Apply TRAIN normalization

        # Compute forward drawdown on test
        test_FWD_DD = forward_drawdown(test_prices, horizon=20)

        # Correlation (CAUSAL: CCI at t, DD at t+20)
        valid = ~(np.isnan(test_CCI) | np.isnan(test_FWD_DD))
        if valid.sum() < 50:
            print(f"  Skipping (insufficient valid points)")
            continue

        rho, p_val = spearmanr(test_CCI[valid][:-20], test_FWD_DD[valid][20:])

        # PR-AUC for crash prediction
        crash_labels = (test_FWD_DD <= -0.10).astype(int)
        if crash_labels[valid][20:].sum() > 5:  # At least 5 crashes
            prec, rec, _ = precision_recall_curve(
                crash_labels[valid][20:],
                test_CCI[valid][:-20]
            )
            pr_auc = auc(rec, prec)
        else:
            pr_auc = np.nan

        print(f"  Spearman ρ: {rho:.4f} (p = {p_val:.6f})")
        print(f"  PR-AUC:     {pr_auc:.4f}" if not np.isnan(pr_auc) else "  PR-AUC:     N/A")

        results.append({
            'asset': symbol,
            'rho': rho,
            'p_value': p_val,
            'pr_auc': pr_auc,
            'passed': (rho < -0.2) and (p_val < 0.05)  # Negative correlation expected
        })

    # Overall assessment
    passed_count = sum(r['passed'] for r in results)
    total_count = len(results)

    print(f"\n{'='*60}")
    print(f"H5 RESULT: {passed_count}/{total_count} assets show significant CCI prediction")

    if passed_count >= 3:
        print("✓ PASS: CCI predictive power hypothesis supported")
        return True
    else:
        print("⚠️  FAIL: CCI predictive power hypothesis NOT supported")
        return False
```

**Pass Criteria**: ≥3 of 4 assets show significant negative correlation

---

## 4. BASELINE COMPARISON TESTS

### 4.1 Naive Persistence Baseline

**Purpose**: Beat the simplest possible baseline

**Reference**:
> "Naive baseline. Persistence forecast x̂(t+h) = x(t) provides challenging baseline due to strong autocorrelation in chaotic systems."
>
> — Data leakage paper, Section 3.3, p.2

**Implementation**:
```python
def test_vs_persistence():
    """
    Compare CCI-based crash prediction vs naive persistence

    Persistence forecast: "Tomorrow will be like today"
    If returns are negative today, predict negative future drawdown
    """
    df, _ = load_sp500(start="2000-01-01")
    prices = df["Adj Close"].values

    # Compute CCI
    CCI = compute_full_pipeline(prices)

    # Compute forward drawdown
    FWD_DD = forward_drawdown(prices, horizon=20)

    # Naive persistence: use recent return as predictor
    returns = np.diff(np.log(prices))
    persistence_signal = -returns  # Negative return → positive signal (warning)

    # Align lengths
    min_len = min(len(CCI), len(persistence_signal), len(FWD_DD))
    CCI = CCI[:min_len]
    persistence_signal = persistence_signal[:min_len]
    FWD_DD = FWD_DD[:min_len]

    # Remove NaNs
    valid = ~(np.isnan(CCI) | np.isnan(persistence_signal) | np.isnan(FWD_DD))

    # Correlations
    rho_cci, p_cci = spearmanr(CCI[valid][:-20], FWD_DD[valid][20:])
    rho_persistence, p_persistence = spearmanr(
        persistence_signal[valid][:-20],
        FWD_DD[valid][20:]
    )

    # RMSE for continuous prediction
    from sklearn.metrics import mean_squared_error
    rmse_cci = np.sqrt(mean_squared_error(
        FWD_DD[valid][20:],
        -CCI[valid][:-20]  # Negative because high CCI = bad
    ))
    rmse_persistence = np.sqrt(mean_squared_error(
        FWD_DD[valid][20:],
        persistence_signal[valid][:-20]
    ))

    improvement = (rmse_persistence - rmse_cci) / rmse_persistence * 100

    print(f"Persistence:  ρ = {rho_persistence:.4f}, RMSE = {rmse_persistence:.4f}")
    print(f"CCI:          ρ = {rho_cci:.4f}, RMSE = {rmse_cci:.4f}")
    print(f"Improvement:  {improvement:.2f}%")

    if improvement < 5:
        print("⚠️  FAIL: CCI does not beat naive persistence baseline")
        return False

    print("✓ PASS: CCI beats persistence baseline")
    return True
```

**Pass Criteria**: >5% improvement over persistence

---

### 4.2 Machine Learning Comparison

**Purpose**: Compare to ML methods (per data leakage paper findings)

**Reference**:
> "Machine learning methods: 21% better than naive baseline [...] Random Forest (100 trees, depth 10), Gradient Boosting (100 estimators, depth 5), XGBoost (100 estimators, depth 5), Neural Network (32-16 hidden units, 200 epochs)"
>
> — Data leakage paper, Section 3.4, p.2

> "ML methods achieve genuine 21% improvements when tested with identical validation."
>
> — Data leakage paper, Abstract, p.1

**Implementation**:
```python
def test_vs_ml_methods():
    """
    Compare proposed CCI vs ML methods using identical validation
    """
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor

    # Load data
    df, _ = load_sp500(start="2000-01-01")
    prices = df["Adj Close"].values

    # Create features (lookback window)
    lookback = 20
    X = []
    y = []

    for i in range(lookback, len(prices) - 20):
        X.append(prices[i-lookback:i])
        y.append((prices[i+20:i+21].min() - prices[i]) / prices[i])  # Forward DD

    X = np.array(X)
    y = np.array(y)

    # Train/test split (causal)
    train_size = 2000
    X_train, X_test = X[:train_size], X[train_size:train_size+400]
    y_train, y_test = y[:train_size], y[train_size:train_size+400]

    # Train ML models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rho, p = spearmanr(y_pred, y_test)

        results[name] = {'rmse': rmse, 'rho': rho, 'p': p}
        print(f"  RMSE: {rmse:.4f}, ρ: {rho:.4f}, p: {p:.6f}")

    # Compare to proposed CCI
    # (Need to align CCI with same test period)
    CCI = compute_full_pipeline(prices)
    CCI_test = CCI[train_size+lookback:train_size+lookback+400]

    rmse_cci = np.sqrt(mean_squared_error(y_test, -CCI_test[:len(y_test)]))
    rho_cci, p_cci = spearmanr(-CCI_test[:len(y_test)], y_test)

    results['Proposed System'] = {'rmse': rmse_cci, 'rho': rho_cci, 'p': p_cci}
    print(f"\nProposed System:")
    print(f"  RMSE: {rmse_cci:.4f}, ρ: {rho_cci:.4f}, p: {p_cci:.6f}")

    # Ranking
    ranked = sorted(results.items(), key=lambda x: x[1]['rmse'])
    print(f"\n{'='*60}")
    print("RANKING (by RMSE):")
    for i, (name, metrics) in enumerate(ranked, 1):
        print(f"{i}. {name:20s}  RMSE={metrics['rmse']:.4f}  ρ={metrics['rho']:.4f}")

    # Check if CCI is competitive (top 3)
    cci_rank = [name for name, _ in ranked].index('Proposed System') + 1

    if cci_rank <= 3:
        print(f"\n✓ PASS: Proposed system ranks #{cci_rank} (competitive with ML)")
        return True
    else:
        print(f"\n⚠️  FAIL: Proposed system ranks #{cci_rank} (underperforms ML)")
        return False
```

**Pass Criteria**: CCI ranks in top 3 of 5 methods

---

## 5. ROBUSTNESS TESTS

### 5.1 Walk-Forward Cross-Validation

**Purpose**: Test on multiple out-of-sample periods

**Reference**:
> "Walk-forward cross-validation across independent folds validates robustness."
>
> — Original paper, Abstract, p.1

**Implementation**:
```python
def test_walk_forward_cv(n_folds=5):
    """
    Sliding window cross-validation with strict causality
    """
    df, _ = load_sp500(start="2000-01-01")
    prices = df["Adj Close"].values

    train_size = 2000
    test_size = 200
    step_size = 100  # Overlap allowed in rolling window design

    fold_results = []

    for fold in range(n_folds):
        start_idx = fold * step_size

        if start_idx + train_size + test_size > len(prices):
            break

        # Train and test splits
        train_prices = prices[start_idx:start_idx+train_size]
        test_prices = prices[start_idx+train_size:start_idx+train_size+test_size]

        print(f"\nFold {fold+1}: Train [{start_idx}:{start_idx+train_size}], "
              f"Test [{start_idx+train_size}:{start_idx+train_size+test_size}]")

        # Compute metrics (with train-only normalization)
        # ... (implementation details)

        # Evaluate
        rho, p = evaluate_fold(train_prices, test_prices)

        fold_results.append({'fold': fold+1, 'rho': rho, 'p': p})
        print(f"  Result: ρ = {rho:.4f}, p = {p:.6f}")

    # Aggregate results
    rhos = [r['rho'] for r in fold_results]
    ps = [r['p'] for r in fold_results]

    mean_rho = np.mean(rhos)
    std_rho = np.std(rhos)
    significant_folds = sum(1 for p in ps if p < 0.05)

    print(f"\n{'='*60}")
    print(f"Cross-validation results:")
    print(f"  Mean ρ: {mean_rho:.4f} ± {std_rho:.4f}")
    print(f"  Significant folds: {significant_folds}/{len(fold_results)}")

    if significant_folds >= len(fold_results) // 2:
        print("✓ PASS: Majority of folds show significant correlation")
        return True
    else:
        print("⚠️  FAIL: Insufficient fold consistency")
        return False
```

**Pass Criteria**: ≥50% of folds significant

---

## 6. STATISTICAL RIGOR TESTS

### 6.1 Multiple Hypothesis Testing Correction

**Purpose**: Correct for multiple comparisons

**Implementation**:
```python
def test_multiple_hypothesis_correction():
    """
    Apply Bonferroni correction for multiple hypothesis testing

    Testing: 4 assets × 5 hypotheses × 3 crises = 60 tests
    Corrected α = 0.05 / 60 = 0.000833
    """
    from statsmodels.stats.multitest import multipletests

    # Collect all p-values from hypothesis tests
    all_p_values = []
    test_names = []

    # Run all hypothesis tests and collect p-values
    # ... (abbreviated for brevity)

    # Apply Bonferroni correction
    reject, p_corrected, _, _ = multipletests(all_p_values, alpha=0.05, method='bonferroni')

    print(f"Total tests: {len(all_p_values)}")
    print(f"Significant (uncorrected): {sum(p < 0.05 for p in all_p_values)}")
    print(f"Significant (Bonferroni):  {sum(reject)}")

    # Family-wise error rate
    if sum(reject) >= len(all_p_values) // 3:
        print("✓ PASS: Results survive multiple testing correction")
        return True
    else:
        print("⚠️  WARNING: Many results lost after correction")
        return False
```

**Pass Criteria**: ≥33% of tests remain significant

---

## EXECUTION PLAN

### Phase 1: Immediate Tests (Run First)
```bash
python -m pytest tests/1_data_integrity_tests/ -v
python -m pytest tests/2_leakage_detection_tests/ -v
```

**CRITICAL**: If ANY leakage test fails, STOP and fix before proceeding.

### Phase 2: Hypothesis Tests (After Leakage Clean)
```bash
python -m pytest tests/3_hypothesis_validation_tests/ -v
```

### Phase 3: Comparison Tests
```bash
python -m pytest tests/4_baseline_comparison_tests/ -v
```

### Phase 4: Robustness
```bash
python -m pytest tests/5_robustness_tests/ -v
python -m pytest tests/6_statistical_rigor_tests/ -v
```

---

## EXPECTED OUTCOMES

### Scenario 1: Full Pass
- All leakage tests pass
- ≥4 of 5 hypotheses supported
- Beats naive baseline by >10%
- Competitive with ML methods

**Conclusion**: System is genuinely predictive

### Scenario 2: Leakage Detected
- Reversed-time test fails
- Normalization leakage detected

**Conclusion**: Results are artifacts, not genuine prediction

### Scenario 3: Weak Performance
- Leakage tests pass
- Hypotheses marginally supported
- Barely beats baseline

**Conclusion**: System has minor predictive power, not practically useful

### Scenario 4: Negative Results
- Leakage tests pass
- Hypotheses NOT supported
- Underperforms baseline

**Conclusion**: Honest negative result (like the data leakage paper)

---

## FINAL CHECKLIST

Before claiming the system works:

- [ ] **Temporal separation verified** (test 1.1)
- [ ] **No normalization leakage** (test 1.2)
- [ ] **Reversed-time test passed** (test 2.1)
- [ ] **Shuffled future test passed** (test 2.2)
- [ ] **Permutation test significant** (test 2.3, p < 0.01)
- [ ] **Entropy spike hypothesis supported** (test 3.1, ≥2 crises)
- [ ] **CCI prediction hypothesis supported** (test 3.2, ≥3 assets)
- [ ] **Beats naive persistence** (test 4.1, >5% improvement)
- [ ] **Competitive with ML** (test 4.2, top 3 ranking)
- [ ] **Cross-validation robust** (test 5.1, ≥50% folds)
- [ ] **Multiple testing correction applied** (test 6.1)

**Only if ALL boxes checked can we claim the system works.**

---

## REFERENCES

1. Data leakage paper - "Data Leakage in Chaotic Time Series Prediction"
2. Original positive results paper - "Recursive Momentum-Based Prediction in Nonlinear Oscillators"
3. Implementation file 1 - Entropy-driven lambda implementation
4. Implementation file 2 - Multi-asset crisis index implementation

---

**END OF TEST SPECIFICATION**
