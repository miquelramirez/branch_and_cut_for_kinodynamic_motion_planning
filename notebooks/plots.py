import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def compare_with_scatter_chart(x, y, title='untitled', x_name='x', y_name='y', x_log=False, y_log=False, paper=False):
    """
    Creates simple scatter plot to compare x vs. y
    :param x:
    :param y:
    :param title:
    :param x_name:
    :param y_name:
    :param x_log:
    :param y_log:
    :return:
    """
    lin_model = stats.linregress(x, y)

    plt.figure(figsize=(9, 9))

    plt.scatter(x, y)

    if not paper:
        plt.plot(x, lin_model.intercept + lin_model.slope * x, 'r', linestyle='--', label='lin. fit')

    plt.xlim(min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))
    plt.ylim(min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))

    if not paper:
        plt.title(title)
        plt.xlabel(x_name)
        plt.ylabel(y_name)

    if paper:
        plt.tight_layout()

    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')

    if not paper:
        plt.legend()

    plt.show()

    return lin_model