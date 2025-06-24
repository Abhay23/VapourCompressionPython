
def OP(points,line,Enthalpy_points,Pressure_Points):

    import matplotlib.pyplot as plt
    import numpy as np

    # Slightly update y-coordinates for animation (example logic)

    # Update both the scatter and line
    points.set_offsets(np.c_[Enthalpy_points,Pressure_Points])
    line.set_data(Enthalpy_points,Pressure_Points)

    plt.draw()
    plt.pause(0.01)

    # Final hold
    plt.show(block=False)

    