import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from statsmodels.tsa.stattools import adfuller


class TimeSeriesRandomWalkTest:
    def __init__(self, time_series, plot_autocorrelation=False):
        self.time_series = time_series
        self.adf_test_result = None
        self.p_value = None
        self.calculate_differenced_time_series()
        self.adf_test()
        if plot_autocorrelation:
            self.plot_autocorrelation()

    def calculate_differenced_time_series(self):
        """Get the differenced time series, which can be used to
        test for stationarity."""
        self.differenced_time_series = np.diff(self.time_series, n=1)

    def adf_test(self):
        adf_test_result = adfuller(self.differenced_time_series)
        self.adf_statistic = adf_test_result[0]
        self.p_value = adf_test_result[1]
        if self.p_value <= 0.05:
            print("p_value is less than 0.05 -> time series is stationary.")

    def plot_autocorrelation(self):
        plt.figure(figsize=(12, 8))
        plt.subplot(211)
        plt.plot(self.differenced_time_series)
        plt.subplot(212)
        plt.acorr(self.differenced_time_series, maxlags=100, normed=True)
        plt.show()

def main():
    # Test data set for Weekly carbon-dioxide concentration averages derived
    # from continuous air samples for the Mauna Loa Observatory
    # https://www.openml.org/search?type=data&status=active&id=41187
    data = fetch_openml(data_id=41187)
    year_float = data.data.year + (data.data.month - 1 ) / 12
    target = data.target
    window = 50
    target_moving_average = moving_average(target, window)
    plt.plot(year_float, target)
    plt.plot(year_float[window-1:], target_moving_average)
    plt.show()
    TimeSeriesRandomWalkTest(target, plot_autocorrelation=True)


def moving_average(a, n=3):
    a = a.to_numpy()
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == '__main__':
    main()
