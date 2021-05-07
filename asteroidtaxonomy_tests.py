# Purpose: Assortment of algorithms used for testing purposes

# Imported modules/functions
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def OptimumWindowSizeTesting(x, y, deg=3, plot=False):
    # Retrieve length of x array
    n = len(x)

    # Iterate through different window sizes
    SAD = []  # Holds sum of absolute difference values
    window_sizes = []  # Holds window sizes used
    for i in range(deg + 1, n):
        # Ensure window size is odd
        if (n % 2) == 0:
            continue

        # Calculate smoothed function
        y_smoothed = signal.savgol_filter(y, n, deg)

        # Calculate sum of absolute differences
        s = np.sum(np.abs(y - y_smoothed))

        # Plot original function and smoothed function
        if (plot):
            plt.figure(figsize=(12, 6))
            plt.title('Window Size: ' + str(n))
            plt.xlabel('X-Axis')
            plt.ylabel('Y-Axis')
            sns.scatterplot(x=x, y=y, label='Original Data')
            sns.scatterplot(x=x, y=y_smoothed, label='Smoothed Data')
            plt.show()

        # Record window size and sum of absolute differences
        window_sizes.append(n)
        SAD.append(s)

    # Plot sum of absolute differences function
    plt.title('')
    plt.xlabel('Window Size')
    plt.ylabel('Sum of Absolute Differences')
    sns.scatterplot(x=window_sizes, y=SAD)
    plt.show()
