import numpy as np

def read_cisbp_pwm(pwm_path):
    return np.loadtxt(pwm_path, skiprows=1, dtype=float,
                      usecols=(1, 2, 3, 4))


def pwms2conv_filters(pwm_paths):
    return [read_cisbp_pwm(path) for path in pwm_paths]


def pad_conv_filters(conv_filters):
    max_width = max(k.shape[0] for k in conv_filters)
    padded_filters = np.zeros((len(conv_filters), max_width, 4))
    for i, conv_filter in enumerate(conv_filters):
        padded_filter = 0.25 * np.ones((max_width, 4))
        width = conv_filter.shape[0]
        offset = int((max_width - width) / 2) 
        padded_filter[offset:(offset + width), :] = conv_filter
        padded_filters[i, :, :] = padded_filter

    return padded_filters


def pwms2conv_weights(pwm_paths):
    conv_filters = pwms2conv_filters(pwm_paths)
    padded_conv_filters = pad_conv_filters(conv_filters)
    normalized_conv_weights = np.log((0.0001 + padded_conv_filters) / 1.0004)

    return normalized_conv_weights
