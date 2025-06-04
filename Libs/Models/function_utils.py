import numpy as np

def nice_range(x_min : float, x_max : float, num_steps=20):
    """
    """
    # Calculate raw step size
    span = x_max - x_min
    raw_step = span / num_steps

    # Round step to nearest "nice" number
    magnitude = 10 ** int(np.floor(np.log10(raw_step)))
    nice_factors = [1, 2, 2.5, 5, 10]
    step = min(f * magnitude for f in nice_factors if f * magnitude >= raw_step)

    # Round min and max to the nearest multiple of step
    start = int(np.floor(x_min / step) * step)
    end = int(np.ceil(x_max / step) * step)

    return np.arange(start, end + step, step).astype(int)