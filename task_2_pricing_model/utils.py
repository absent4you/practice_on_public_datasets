from matplotlib import pyplot
import numpy as np
import pandas as pd

def plot_distributions(aval, occup, recommended_price = False, title=False):
    """ Plots distributions & recommended price if specified """
    bins = np.linspace(min(aval.min(), occup.min()), max(aval.max(), occup.max()), 30)
    pyplot.hist(aval, bins, alpha=0.5, label="available")
    pyplot.hist(occup, bins, alpha=0.5, label="occupied")
    if recommended_price:
        pyplot.axvline(x=recommended_price, color="black", linestyle="dashed", label='recommended_price')
    pyplot.legend(loc="upper left")
    pyplot.xlabel("Per night price")
    pyplot.ylabel("Amount of places")
    if title:
        pyplot.title(title)
    pyplot.show()
    
def find_acceptable_price(aval, occup):
    """
        Find acceptable price as the highest difference between cumulative distributions.
        Log scaled prices to be provided as input
    """
    bins = np.linspace(min(aval.min(), occup.min()), max(aval.max(), occup.max()), 1000)
    distributions = pd.DataFrame({"lower_bound": bins[:-1], "upper_bound": bins[1:]})
    distributions["avail_count"] = distributions.apply(lambda x: count_in_range(x.lower_bound, x.upper_bound, aval), axis=1)
    distributions["occup_count"] = distributions.apply(lambda x: count_in_range(x.lower_bound, x.upper_bound, occup), axis=1)
    distributions["occup_%"] = distributions["occup_count"] / len(occup)
    distributions["avail_%"] = distributions["avail_count"] / len(aval)
    distributions["avail_cumul_prcnt"] = distributions["avail_%"].cumsum()
    distributions["occup_cumul_prcnt"] = distributions["occup_%"].cumsum()
    distributions["diff_cumul"] = distributions["occup_cumul_prcnt"] - distributions["avail_cumul_prcnt"]
    distributions = distributions[distributions["diff_cumul"] == distributions["diff_cumul"].max()]
    lower_bound = distributions.loc[distributions.index[0], "lower_bound"]
    upper_bound = distributions.loc[distributions.index[-1], "upper_bound"]
    lower_bound_euro = np.exp(lower_bound) - 1
    upper_bound_euro = np.exp(upper_bound) - 1
    acceptable_price_scaled = (lower_bound + upper_bound) / 2  # For plot creation (recommended price in log scale)
    acceptable_price_euro = (lower_bound_euro + upper_bound_euro) / 2  # Final recommendation

    return acceptable_price_euro, acceptable_price_scaled

def count_in_range(lb, ub, vector):
    """ Help function to find number of vector occurrences in specified range """
    return ((lb < vector) & (vector < ub)).sum()