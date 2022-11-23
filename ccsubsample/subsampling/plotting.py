import matplotlib.pyplot as plt
import numpy as np

def plot_kde(x, y_log_density, title):
    # add (0, 0) to the beginning of the (x, y) pairs to avoid plotting weirdness
    # with plt.fill
    plt.fill(np.insert(x, 0, 0), np.insert(np.exp(y_log_density), 0, 0))
    plt.title("{} KDE".format(title))
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.axis()
    plt.show()



