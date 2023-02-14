import numpy as np
from matplotlib import pyplot as plt


def compare_two_fields(field_a, field_b, a_title, b_title, x_coord, y_coord, title, tstamp, diff_factor=0.01, figsize=(15, 4)):

    cmin = np.min([field_a.min(), field_b.min()])
    cmax = np.max([field_a.max(), field_b.max()])
    print(f"cmin = {cmin}, cmax = {cmax}")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    cs1 = ax1.contourf(x_coord, y_coord, field_a, np.linspace(cmin, cmax, 31), cmap='rainbow')
    ax1.set_title(a_title)
    cbar1 = fig.colorbar(cs1)
    cs2 = ax2.contourf(x_coord, y_coord, field_b, np.linspace(cmin, cmax, 31), cmap='rainbow')
    ax2.set_title(b_title)
    cbar2 = fig.colorbar(cs2)
    if diff_factor:
        cs3 = ax3.contourf(x_coord, y_coord, np.abs(field_a-field_b),
                           np.linspace(0, diff_factor * max([np.abs(cmin), np.abs(cmax)]), 31), cmap='rainbow')
    else:
        cs3 = ax3.contourf(x_coord, y_coord, np.abs(field_a - field_b), cmap='rainbow')
    ax3.set_title(f'Abs difference')
    cbar3 = fig.colorbar(cs3)
    plt.show()

