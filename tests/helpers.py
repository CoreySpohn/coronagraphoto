"""This file contains reusable helper functions for the test suite."""

import numpy as np
from scipy.stats import chisquare, poisson


def assert_poisson_distribution(
    sample_generator, expected_lambda, n_samples=2000, significance_level=0.01
):
    """Asserts that samples from a generator follow a Poisson distribution.

    Performs a Pearson chi-squared test to validate that a series of
    generated samples conforms to a theoretical Poisson distribution.

    Args:
        sample_generator (callable): A function that, when called, returns a
            single sample from the distribution to be tested.
        expected_lambda (float): The theoretical mean (lambda) of the Poisson
            distribution.
        n_samples (int): The number of samples to generate for the test.
        significance_level (float): The p-value threshold below which the
            test fails.
    """
    # Generate samples
    samples = [sample_generator() for _ in range(n_samples)]

    # Determine bins for histogram. Use a high quantile of the theoretical
    # distribution plus the observed max to ensure the tail is captured.
    max_observed = np.max(samples) if samples else 0
    # Use a high quantile to ensure we capture the vast majority of the PDF
    k_max_theoretical = poisson.ppf(0.9999, expected_lambda)
    k_max = int(max(max_observed, k_max_theoretical, 10))
    bins = np.arange(k_max + 2) - 0.5
    obs_counts, _ = np.histogram(samples, bins=bins)

    # Calculate expected counts based on the Poisson PMF
    bin_centers = np.arange(len(obs_counts))
    exp_counts = n_samples * poisson.pmf(bin_centers, expected_lambda)

    # Normalize expected counts to match the total number of observations.
    # This is the critical fix to account for the truncated tail of the
    # theoretical PMF and satisfy the chisquare function's sum check.
    if np.sum(exp_counts) > 0:
        exp_counts *= np.sum(obs_counts) / np.sum(exp_counts)

    # Filter out bins where expected counts are essentially zero
    valid = exp_counts > 1e-9
    if np.sum(valid) < 2:
        assert np.all(obs_counts[~valid] == 0)
        return

    # Perform the chi-squared test
    stat, p_value = chisquare(obs_counts[valid], exp_counts[valid])

    # Fail only if the Poisson hypothesis is rejected
    assert p_value > significance_level
