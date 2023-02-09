import pdb
import matplotlib.pyplot as plt
from . import subsample, SubsampleMethods
import numpy as np
from datetime import datetime
from meteostat import Point, Hourly

vancouver = Point(49.2497, -123.1193, 70)
data = Hourly(vancouver, start=datetime(
    2019, 1, 1), end=datetime(2021, 12, 31))
data = data.fetch()

x_vals = (data.index.values - np.datetime64('1970-01-01T00:00:00Z')
          ) / np.timedelta64(1, 'h')
x_vals = x_vals - x_vals[0]
y_vals = data['temp'].values

x_sub, y_sub = subsample(SubsampleMethods.MinMax, x_vals, y_vals, 250)

fig, axes = plt.subplot_mosaic("AB;AC")
main = axes["A"]
side_top = axes["B"]
side_bottom = axes["C"]

fig.suptitle("Hourly Temperature in Vancouver")

main.set_title("Combined Plot")
main.set_xlabel("Time (s)")
main.set_ylabel("Temperature (K)")
main.plot(x_vals, y_vals)
main.plot(x_sub, y_sub)
main.legend(["Original Data", "Subsampled Data"])

side_top.set_title(f"Original Data, {len(x_vals)} Points")
side_top.plot(x_vals, y_vals)

side_bottom.set_title(f"Subsampled Data, {len(x_sub)} Points")
side_bottom.plot(x_sub, y_sub)

plt.show()

# pdb.set_trace()
