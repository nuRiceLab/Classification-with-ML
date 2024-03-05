from scipy.stats import beta as beta_distribution
import numpy as np
import warnings
from scipy.special import loggamma


def compute_acceptance_pdf(k, n, epsilon=np.arange(0, 1+10**-5, 10**-5)):
    """
    Function to display acceptance pdf as a function of the acceptance.

    :param k: Number of events in bin after applying the cut.
    :param n: Number of events in bin before applying the cut.
    :param epsilon: Acceptance range for which the pdf values
        should be computed.
    :returns: Numpy array containing the acceptance pdf for the given 
        bin. Must be normalized by the user.
    """
    if (k == n == 0) or k > n:
        raise ValueError('Function is not defined for k == n == 0 or '
                         ' k > n!')
    power = loggamma(n + 2)
    power -= loggamma(k + 1)
    power -= loggamma(n - k + 1)
    
    epsilon_to_the_power_of_k = k*np.log(epsilon)
    if k == 0:
        if isinstance(epsilon_to_the_power_of_k, np.ndarray):
            epsilon_to_the_power_of_k[0] = 0
        else:
            epsilon_to_the_power_of_k = 0
    power += epsilon_to_the_power_of_k
    
    n_k_to_the_power_1_epsilon = (n - k)*np.log(1 - epsilon)
    if n == k:
        if isinstance(epsilon_to_the_power_of_k, np.ndarray):
            n_k_to_the_power_1_epsilon[-1] = 0
        else:
            n_k_to_the_power_1_epsilon = 0
    power += n_k_to_the_power_1_epsilon
    
    return np.exp(power)

def highest_density_region(data, fractions_desired, only_upper_part=False, _buffer_size=10):
    """Computes for a given sampled distribution the highest density region of the desired
    fractions. Does not assume anything on the normalisation of the data.

    :param data: Sampled distribution
    :param fractions_desired: numpy.array Area/probability for which
        the hdr should be computed.
    :param _buffer_size: Size of the result buffer. The size is
        equivalent to the maximal number of allowed intervals.
    :param only_upper_part: Boolean, if true only computes
        area/probability between maximum and current height.
    :return: two arrays: The first one stores the start and inclusive
        endindex of the highest density region. The second array holds
        the amplitude for which the desired fraction was reached.
    Note:
        Also goes by the name highest posterior density. Please note,
        that the right edge corresponds to the right side of the sample.
        Hence the corresponding index is -= 1.

    """
    fi = 0  # number of fractions seen
    # Buffer for the result if we find more then _buffer_size edges the function fails.
    # User can then manually increase the buffer if needed.
    res = np.zeros((len(fractions_desired), 2, _buffer_size), dtype=np.int32)
    res_amp = np.zeros(len(fractions_desired), dtype=np.float32)

    area_tot = np.sum(data)
    if area_tot <= 0:
        raise ValueError(
            "Highest density regions are not defined for distributions "
            "with a total probability of less-equal 0."
        )

    # Need an index which sorted by amplitude
    max_to_min = np.argsort(data, kind="mergesort")[::-1]

    lowest_sample_seen = np.inf
    for j in range(1, len(data)):
        # Loop over indices compute fractions from max to min
        if lowest_sample_seen == data[max_to_min[j]]:
            # We saw this sample height already, so no need to repeat
            continue

        lowest_sample_seen = data[max_to_min[j]]
        lowest_sample_seen *= int(only_upper_part)
        sorted_data_max_to_j = data[max_to_min[:j]]
        fraction_seen = np.sum(sorted_data_max_to_j - lowest_sample_seen) / area_tot

        # Check if this height step exceeded at least one of the desired
        # fractions
        m = fractions_desired[fi:] <= fraction_seen
        if not np.any(m):
            # If we do not exceed go to the next sample.
            continue

        for fraction_desired in fractions_desired[fi : fi + np.sum(m)]:
            # Since we loop always to the height of the next highest sample
            # it might happen that we overshoot the desired fraction. Similar
            # to the area deciles algorithm we have now to figure out at which
            # height we actually reached the desired fraction and store the
            # corresponding height:
            g = fraction_desired / fraction_seen

            # The following gives the true height, to get here one has to
            # solve for h:
            # 1. fraction_seen = sum_{i=0}^j (y_i - y_j) / a_total
            # 2. fraction_desired = sum_{i=0}^j (y_i - h) / a_total
            # 3. g = fraction_desired/fraction_seen
            # j == number of seen samples
            # n == number of total samples in distribution
            true_height = (1 - g) * np.sum(sorted_data_max_to_j) / j + g * lowest_sample_seen
            res_amp[fi] = true_height

            # Find gaps and get edges of hdr intervals:
            ind = np.sort(max_to_min[:j])
            gaps = np.arange(1, len(ind) + 1)

            g0 = 0
            g_ind = -1
            diff = ind[1:] - ind[:-1]
            gaps = gaps[:-1][diff > 1]

            if len(gaps) > _buffer_size:
                # This signal has more boundaries than the buffer can hold
                # hence set all entries to -1 instead.
                res[fi, 0, :] = -1
                res[fi, 1, :] = -1
                fi += 1
            else:
                for g_ind, g in enumerate(gaps):
                    # Loop over all gaps and get outer edges:
                    interval = ind[g0:g]
                    res[fi, 0, g_ind] = interval[0]
                    res[fi, 1, g_ind] = interval[-1] + 1
                    g0 = g

                # Now we have to do the last interval:
                interval = ind[g0:]
                res[fi, 0, g_ind + 1] = interval[0]
                res[fi, 1, g_ind + 1] = interval[-1] + 1
                fi += 1

        if fi == (len(fractions_desired)):
            # Found all fractions so we are done
            return res, res_amp

    # If we end up here this might be due to an offset
    # of the distribution with respect to zero. In that case it can
    # happen that we do not find all desired fractions.
    # Hence we have to enforce to compute the last step from the last
    # lowest hight we have seen to zero.
    # Left and right edge is by definition 0 and len(data):
    res[fi:, 0, 0] = 0
    res[fi:, 1, 0] = len(data)
    # Now we have to compute the heights for the fractions we have not
    # seen yet, since lowest_sample_seen == 0 and j == len(data)
    # the formula above reduces to:
    for ind, fraction_desired in enumerate(fractions_desired[fi:]):
        res_amp[fi + ind] = (1 - fraction_desired) * np.sum(data) / len(data)
    return res, res_amp
    
def compute_acceptance_uncertainty_bayesian(k, n,
                                            coverage=0.6827,
                                            precision=10**-5):
    """
    Function which computes the uncertainty of an acceptance bin based
    on a Bayesian method from FERMILAB-TM-2286-CD
    (https://inspirehep.net/literature/669498)

    The coverage of the distribution is computed using a highest
    density region method. This is done on sampled data.

    :param k: Number of events in bin after applying the cut.
    :param n: Number of events in bin before applying the cut.
    :param coverage: Coverage to be used for uncertainty estimate.
    :param precision: Stepping to be used to compute the acceptance pdf.
        Smaller value leads to more precise results but increases
        computing time.

    :returns: Numpy array with the lower and upper bound.
        Squared-qudratic difference of the asked coverage and the
        coverage used to estimate the bounds.
    """
    if isinstance(coverage, (float, int)):
        coverage = np.array([coverage])
        
    if not ((coverage > 0) and (coverage <= 1)):
        raise ValueError('Coverage must be between 0 < c <= 1!')
    
    if (k == n == 0) or k > n:
        warnings.warn('Acceptance not defined for k == n == 0 or k > n. '
                     'Retunrning nans instead.')
        return (np.nan, np.nan), np.nan
    
    _epsilon = np.arange(0, 1+precision, precision)
    data = compute_acceptance_pdf(k, n, _epsilon)
    n = len(data)
    dx = (max(_epsilon) - min(_epsilon))
    data *= dx/n
    
    index, _ = highest_density_region(data,
                                            coverage,
                                            )
    a_index = index[0, 0, 0]
    b_index = index[0, 1, 0]-1
    area = np.sum(data[a_index:b_index])

    error = np.sqrt((coverage - area)**2)
    return _epsilon[[a_index, b_index]], error

def compute_acceptance_uncertainty_CP(k, n, coverage=0.6827):
    """
    Returns the Clopper-Pearson interval of the probability of success
    for a binomial process. See
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    Typically, Clopper Pearson intervals are a conservative estimate
    (though for some small N I saw undercoverage).

    :param k: Number of successes (e.g. events passing cuts).
    :param n: Number of tries (e.g. events before passing cut).
    :param coverage: Coverage to be used for uncertainty estimate.

    :returns: Numpy array with the lower and upper bound on the
        probability of success.
    """
    if (coverage <= 0.) or (1. <= coverage):
        raise ValueError("Coverage is a float between 0 and 1")
    if (k > n) or (k < 0) or (n < 0):
        raise ValueError('Function is not defined for negative integers or k > n')

    alpha = 1. - coverage

    if k == 0:
        lo = 0.
    else:
        lo = beta_distribution.ppf(alpha/2, k, n-k+1)
    if k == n:
        hi = 1.
    else:
        hi = beta_distribution.ppf(1 - alpha/2, k+1, n-k)

    return np.array([lo, hi])