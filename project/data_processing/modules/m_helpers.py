# utitility functions for building a multiple linear regression model
# ------------------------------------------------------------------------------

def create_prior_config_dict(mu_intercept_prior, sigma_intercept_prior, mu_slope_prior, sigma_slope_prior, mu_intercept_business_days_prior, sigma_intercept_business_days_prior, mu_slope_interaction_prior, sigma_slope_interaction_prior, truncate_intercept_prior=False):
    '''
    Create a dictionary that contains parameter values for mu and sigma of the
    prior distributions (normal distributions).

    Parameters
    ----------
    mu_intercept_prior : number
        Expected value of normal distribution for intercept prior.
    sigma_intercept_prior : number
        Standard deviation of normal distribution for intercept prior.
    TODO weiter kommentieren
    truncate_intercept_prior : bool
        indicates whether to use the truncated normal distribution for the
        intercept prior; truncated with left bound = 0.

    Returns
    -------
    dict
    '''

    # create dict with given prior configuration (parameter values for normal
    # distributions)
    prior_config_dict = {
        'mu_intercept_prior': mu_intercept_prior,
        'sigma_intercept_prior': sigma_intercept_prior,
        'truncate_intercept_prior': truncate_intercept_prior,
        'mu_slope_prior': mu_slope_prior,
        'sigma_slope_prior': sigma_slope_prior,
        'mu_intercept_business_days_prior': mu_intercept_business_days_prior,
        'sigma_intercept_business_days_prior': sigma_intercept_business_days_prior,
        'mu_slope_interaction_prior': mu_slope_interaction_prior,
        'sigma_slope_interaction_prior': sigma_slope_interaction_prior
    }

    return prior_config_dict


def translate_var_name_to_prior_config_key(var_name):
    '''
    Each prior with a normal distribution has two parameters: mu and sigma.
    They can be extracted from a models prior config dictionary.
    The keys of the mu and sigma within a config dict can be identified with
    this function.

    Parameters
    ----------
    var_name : str
        name of the parameter for which to get prior config keys
    
    Returns
    -------
    (str, str)
        keys of mu and sigma for config dictionary
    '''
    translation_dict = {
        'beta_intercept': ('mu_intercept_prior', 'sigma_intercept_prior'),
        'beta_slope': ('mu_slope_prior', 'sigma_slope_prior'),
        'beta_intercept_business_days':('mu_intercept_business_days_prior', 'sigma_intercept_business_days_prior'),
        'beta_slope_interaction': ('mu_slope_interaction_prior', 'sigma_slope_interaction_prior')
    }
    return translation_dict[var_name]