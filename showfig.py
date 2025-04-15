import pickle
import matplotlib.pyplot as plt

with open("myplot.fig.pickle", "rb") as f:
    fig = pickle.load(f)


# Now show it properly
plt.show()
