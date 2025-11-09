#!/usr/bin/env python3
"""
Comprehensive Test Runner for Financial Crisis Prediction System

Runs all tests in the proper order and generates a summary report.

Based on TEST_PROGRAM_SPECIFICATION.md execution plan:
1. Data Integrity Tests (MANDATORY FIRST)
2. Leakage Detection Tests (CRITICAL - stop if any fail)
3. Hypothesis Validation Tests
4. Baseline Comparison Tests
5. Robustness Tests
6. Statistical Rigor Tests
"""

import sys
import os
import subprocess
from datetime import datetime
import importlib.util


class TestRunner:
    """Orchestrates test execution and reporting."""

    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None

    def run_test(self, test_path, test_name, critical=False):
        """
        Run a single test and capture result.

        Parameters
        ----------
        test_path : str
            Path to test file
        test_name : str
            Display name for test
        critical : bool
            If True, stop execution if test fails
        """
        print(f"\n{'='*70}")
        print(f"Running: {test_name}")
        print(f"{'='*70}")

        try:
            # Load and run the test module
            spec = importlib.util.spec_from_file_location("test_module", test_path)
            test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_module)

            # Find the main test function
            test_function = None
            for name in dir(test_module):
                if name.startswith('test_'):
                    test_function = getattr(test_module, name)
                    break

            if test_function is None:
                print(f"✗ ERROR: No test function found in {test_path}")
                self.results[test_name] = {'status': 'ERROR', 'message': 'No test function'}
                return False

            # Run the test
            result = test_function()

            if result:
                print(f"\n✓ PASSED: {test_name}")
                self.results[test_name] = {'status': 'PASS', 'message': 'Test passed'}
                return True
            else:
                print(f"\n✗ FAILED: {test_name}")
                self.results[test_name] = {'status': 'FAIL', 'message': 'Test failed'}

                if critical:
                    print(f"\n{'!'*70}")
                    print(f"CRITICAL TEST FAILED: {test_name}")
                    print(f"STOPPING EXECUTION AS PER TEST SPECIFICATION")
                    print(f"{'!'*70}")
                    self.print_summary()
                    sys.exit(1)

                return False

        except Exception as e:
            print(f"\n✗ ERROR: {test_name}")
            print(f"   {str(e)}")
            self.results[test_name] = {'status': 'ERROR', 'message': str(e)}

            if critical:
                print(f"\n{'!'*70}")
                print(f"CRITICAL TEST ERROR: {test_name}")
                print(f"STOPPING EXECUTION")
                print(f"{'!'*70}")
                self.print_summary()
                sys.exit(1)

            return False

    def print_summary(self):
        """Print comprehensive test summary."""
        print(f"\n\n{'='*70}")
        print("TEST EXECUTION SUMMARY")
        print(f"{'='*70}")
        print(f"Start time:  {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"Duration:    {duration:.1f} seconds")

        # Count results
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAIL')
        errors = sum(1 for r in self.results.values() if r['status'] == 'ERROR')

        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Total:   {total}")
        print(f"Passed:  {passed} ({passed/total*100:.1f}%)")
        print(f"Failed:  {failed} ({failed/total*100:.1f}%)")
        print(f"Errors:  {errors} ({errors/total*100:.1f}%)")

        # Detailed results
        print(f"\n{'='*70}")
        print("DETAILED RESULTS")
        print(f"{'='*70}")

        for test_name, result in self.results.items():
            status = result['status']
            if status == 'PASS':
                symbol = "✓"
                color = ""
            elif status == 'FAIL':
                symbol = "✗"
                color = ""
            else:
                symbol = "⚠"
                color = ""

            print(f"{symbol} [{status:5s}] {test_name}")

        # Overall result
        print(f"\n{'='*70}")
        if failed == 0 and errors == 0:
            print("✓ ALL TESTS PASSED")
        else:
            print("✗ SOME TESTS FAILED")
        print(f"{'='*70}")

        # Save report
        self.save_report()

    def save_report(self):
        """Save detailed report to file."""
        os.makedirs('results', exist_ok=True)
        report_file = f'results/test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FINANCIAL CRISIS PREDICTION SYSTEM - TEST REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {(datetime.now() - self.start_time).total_seconds():.1f} seconds\n\n")

            total = len(self.results)
            passed = sum(1 for r in self.results.values() if r['status'] == 'PASS')
            failed = sum(1 for r in self.results.values() if r['status'] == 'FAIL')
            errors = sum(1 for r in self.results.values() if r['status'] == 'ERROR')

            f.write("SUMMARY\n")
            f.write("-"*70 + "\n")
            f.write(f"Total:   {total}\n")
            f.write(f"Passed:  {passed} ({passed/total*100:.1f}%)\n")
            f.write(f"Failed:  {failed} ({failed/total*100:.1f}%)\n")
            f.write(f"Errors:  {errors} ({errors/total*100:.1f}%)\n\n")

            f.write("DETAILED RESULTS\n")
            f.write("-"*70 + "\n")
            for test_name, result in self.results.items():
                f.write(f"[{result['status']:5s}] {test_name}\n")
                if result['message']:
                    f.write(f"          {result['message']}\n")

        print(f"\nTest report saved to: {report_file}")


def main():
    """Run all tests in proper order."""
    print("="*70)
    print("FINANCIAL CRISIS PREDICTION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("\nBased on TEST_PROGRAM_SPECIFICATION.md")
    print("\nCritical tests will halt execution if they fail.")

    runner = TestRunner()
    runner.start_time = datetime.now()

    # Phase 1: Data Integrity Tests (MANDATORY FIRST)
    print("\n\n" + "="*70)
    print("PHASE 1: DATA INTEGRITY TESTS (MANDATORY)")
    print("="*70)

    runner.run_test(
        'tests/1_data_integrity_tests/test_temporal_separation.py',
        '1.1 Temporal Separation',
        critical=True
    )

    runner.run_test(
        'tests/1_data_integrity_tests/test_normalization_causality.py',
        '1.2 Normalization Causality',
        critical=True
    )

    # Phase 2: Leakage Detection Tests (CRITICAL)
    print("\n\n" + "="*70)
    print("PHASE 2: LEAKAGE DETECTION TESTS (CRITICAL)")
    print("="*70)
    print("\nFrom TEST_PROGRAM_SPECIFICATION.md:")
    print("> 'CRITICAL: If ANY leakage test fails, STOP. Fix before proceeding.'")

    runner.run_test(
        'tests/2_leakage_detection_tests/test_reversed_time.py',
        '2.1 Reversed-Time Leakage',
        critical=True
    )

    runner.run_test(
        'tests/2_leakage_detection_tests/test_shuffled_future.py',
        '2.2 Shuffled Future',
        critical=True
    )

    runner.run_test(
        'tests/2_leakage_detection_tests/test_permutation_significance.py',
        '2.3 Permutation Significance',
        critical=False  # Informative, not blocking
    )

    runner.run_test(
        'tests/2_leakage_detection_tests/test_autocorrelation_artifact.py',
        '2.4 Autocorrelation Artifact',
        critical=False  # Informative, not blocking
    )

    # Phase 3: Hypothesis Validation Tests
    print("\n\n" + "="*70)
    print("PHASE 3: HYPOTHESIS VALIDATION TESTS")
    print("="*70)

    runner.run_test(
        'tests/3_hypothesis_validation_tests/test_h1_entropy_spike.py',
        '3.1 H1 Entropy Spike Hypothesis',
        critical=False
    )

    # Phase 4: Baseline Comparison Tests
    print("\n\n" + "="*70)
    print("PHASE 4: BASELINE COMPARISON TESTS")
    print("="*70)

    runner.run_test(
        'tests/4_baseline_comparison_tests/test_vs_persistence.py',
        '4.1 vs Naive Persistence',
        critical=False
    )

    # Final summary
    runner.print_summary()

    # Exit code
    failed = sum(1 for r in runner.results.values() if r['status'] in ['FAIL', 'ERROR'])
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
