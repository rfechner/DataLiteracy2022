import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import arviz as az
import pymc3 as pm
import m_data_loading as mdl
import m_helpers

# plotting
# ------------------------------------------------------------------------------

def plot_data(data=mdl.get_data_for_location(), ylabel='Daily number of bike riders'):
    '''
    Plot daily bikerider count by mean day temperature, separate between business days and weekends.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with columns 'temperature', 'is_busday' and 'rider_count'.
    ylabel : str
        Label for y-axis.

    Returns
    -------
    (plot)
    '''
    # plot data
    plt.scatter(data[data['is_busday'] == True]['temperature'], data[data['is_busday'] == True]['rider_count'], label='business day', color='orange')
    plt.scatter(data[data['is_busday'] == False]['temperature'], data[data['is_busday'] == False]['rider_count'], label='weekend', color='blue')

    # legend
    plt.xlabel('Average daily temperature [°C]')
    plt.ylabel(ylabel)
    plt.legend();



# utility functions to plot the data with model fits
# --------------------------------------------------

def plot_linear_fit(intercept, slope, intercept_business_days, slope_interaction, label_non_busday_fit=None, label_busday_fit=None, alpha=1):
    '''
    Plot linear fit.

    Parameters
    ----------
    intercept : number
        Intercept of the regression line for non-business days.
    slope : number
        Slope of the regression line for non-business days.
    intercept_business_days : number
        Additional intercept of the regression line for business days.
    slope_interaction : number
        Additional slope of the regression line for business days.
    label_non_busday_fit : str
        Label for fitted line to non-business days.
    label_busday_fit : str
        Label for fitted line to business days.
    alpha : number
        Opacitiy of plotted lines.
    
    Returns
    -------
    (plot)
    '''
    # plot fit
    x_temps = np.linspace(-5, 25, 10)
    plt.plot(x_temps, intercept +  x_temps * slope, label=label_non_busday_fit, color='blue', alpha=alpha)
    plt.plot(x_temps, intercept + intercept_business_days +  x_temps * (slope + slope_interaction), label=label_busday_fit, color='orange', alpha=alpha)
    plt.legend();

def plot_exponential_fit(intercept, slope, intercept_business_days, slope_interaction, label_non_busday_fit=None, label_busday_fit=None, alpha=1):
    '''
    Plot linear fit.

    Parameters
    ----------
    intercept : number
        Intercept of the regression line for non-business days.
    slope : number
        Slope of the regression line for non-business days.
    intercept_business_days : number
        Additional intercept of the regression line for business days.
    slope_interaction : number
        Additional slope of the regression line for business days.
    label_non_busday_fit : str
        Label for fitted line to non-business days.
    label_busday_fit : str
        Label for fitted line to business days.
    alpha : number
        Opacitiy of plotted lines.
    
    Returns
    -------
    (plot)
    '''
    # plot fit
    x_temps = np.linspace(-5, 25, 10)
    plt.plot(x_temps, np.exp(intercept +  x_temps * slope), label=label_non_busday_fit, color='blue', alpha=alpha)
    plt.plot(x_temps, np.exp(intercept + intercept_business_days +  x_temps * (slope + slope_interaction)), label=label_busday_fit, color='orange', alpha=alpha)
    plt.legend();


def plot_data_and_fit(intercept, slope, intercept_business_days, slope_interaction, plot_fun_fit, is_bayes=False):
    '''
    Plot data and fit. Fitted function has the given parameter values and the
    form defined by plot_fun_fit.

    Parameters
    ----------
    intercept : number
        Intercept of the regression line for non-business days.
    slope : number
        Slope of the regression line for non-business days.
    intercept_business_days : number
        Additional intercept of the regression line for business days.
    slope_interaction : number
        Additional slope of the regression line for business days.
    plot_fun_fit : function
        Function to use for plotting the fitted regression lines.
        Either plot_linear_fit or plot_exponential_fit.
    is_bayes : boolean
        indicates whether fit derived with Bayes, adjusts labels accordingly.
    
    Returns
    -------
    (plot)
    '''
    # plot data
    plot_data()

    # adjust labels
    if (is_bayes):
        label_busday_fit = 'busday mean posterior'
        label_non_busday_fit = 'non_busday mean posterior'
    else:
        label_busday_fit = 'busday_fit'
        label_non_busday_fit = 'non_busday_fit'

    # plot fit
    plot_fun_fit(intercept, slope, intercept_business_days, slope_interaction, label_non_busday_fit=label_non_busday_fit, label_busday_fit=label_busday_fit)


def plot_data_and_linear_fit(intercept, slope, intercept_business_days, slope_interaction, is_bayes=False):
    plot_data_and_fit(intercept, slope, intercept_business_days, slope_interaction, plot_linear_fit, is_bayes)


def plot_data_and_exponential_fit(intercept, slope, intercept_business_days, slope_interaction, is_bayes=False):
    plot_data_and_fit(intercept, slope, intercept_business_days, slope_interaction, plot_exponential_fit, is_bayes)




# utility functions for showing the results of a Bayesian analysis
# ----------------------------------------------------------------

# plot data, MAP fit and example posteriors
def show_data_MAP_and_posts(idata, plot_fun_fit):
    '''
    Plot data, MAP and example posteriors.

    Parameters
    ----------
    idata : idata
        inference data
    plot_fun_fit
        Function to use for plotting the fitted regression lines.
        Either plot_linear_fit or plot_exponential_fit.
    '''
    # extract betas
    postbetas_intercept = idata.posterior['beta_intercept'].values.flatten()
    postbetas_slope = idata.posterior["beta_slope"].values.flatten()
    postbetas_intercept_business_days = idata.posterior["beta_intercept_business_days"].values.flatten()
    postbetas_slope_interaction = idata.posterior["beta_slope_interaction"].values.flatten()

    # plot some posteriors as examples

    num_posts_to_plot = 40 # number of posteriors to plot

    # draw random indices (--> random examples)
    rand_indices = np.random.choice(a=len(postbetas_intercept), size=num_posts_to_plot, replace=False)

    # plot example posteriors
    for i, idx in enumerate(rand_indices):

        # temperature range
        x_temps = np.linspace(-10, 30, 10)

        if (i == 0): # add label for first plotted posterior
            plot_fun_fit(postbetas_intercept[idx], postbetas_slope[idx], postbetas_intercept_business_days[idx], postbetas_slope_interaction[idx], 'non_busday posteriors', 'busday posteriors', alpha=0.3)
        else:
            plot_fun_fit(postbetas_intercept[idx], postbetas_slope[idx], postbetas_intercept_business_days[idx], postbetas_slope_interaction[idx], alpha=0.3)
    
    # plot data and MAP
    plot_data_and_fit(postbetas_intercept.mean(),
            postbetas_slope.mean(),
            postbetas_intercept_business_days.mean(),
            postbetas_slope_interaction.mean(),
            plot_fun_fit,
            True)

def show_results(idata, plot_fun_fit):
    '''
    Print and plot fit results.
    '''
    # show results
    az.plot_trace(idata)
    display(az.summary(idata, hdi_prob=0.95))

    az.plot_posterior(idata, var_names=['beta_intercept', 'beta_slope', 'beta_intercept_business_days', 'beta_slope_interaction'], hdi_prob=0.95)
    
    # plot data, MAP and example posteriors
    plt.figure(figsize=(9,7))
    show_data_MAP_and_posts(idata, plot_fun_fit)


def show_results_linear_fit(idata):
    '''
    Print and plot fit results for linear fit.
    '''
    show_results(idata, plot_linear_fit)


def show_results_exponential_fit(idata):
    '''
    Print and plot fit results for exponential fit.
    '''
    show_results(idata, plot_exponential_fit)


# plot regression lines with parameters sampled from prior
def prior_pred_check(prior_checks, title="Prior predictive checks"):

    for i, (intercept, slope, intercept_business_days, slope_interaction) in enumerate(zip(prior_checks["beta_intercept"], prior_checks["beta_slope"], prior_checks["beta_intercept_business_days"], prior_checks["beta_slope_interaction"])):
    
        if (i==0): # add labels
            plot_linear_fit(intercept, slope, intercept_business_days, slope_interaction, label_non_busday_fit='non business day', label_busday_fit='business day', alpha=0.3)
        else:
            plot_linear_fit(intercept, slope, intercept_business_days, slope_interaction, label_non_busday_fit=None, label_busday_fit=None, alpha=0.3)

    plt.xlabel("Average daily temperature [°C]")
    plt.ylabel("Daily bike rider count")
    plt.title(title);


def plot_posterior(idata, var_name, kind='kde', color='blue', alpha=0.5, label=None, title='posterior distribution', add_title=True, plot_mean=False, plot_hdi=False, label_hdi=False):
    '''
    Plot posterior distribution.

    Parameters
    ----------
    idata : arviz.InferenceData
        samples from the posterior
    var_name : str
        name of the parameter for which to plot the posterior distribution
    kind : str
        type of plot: 'hist' or 'kde'
    color : str
        color to use for plot
    alpha : number in [0, 1]
        opacity (for histogram)
    label : str
        label for posterior
    title : str
        title
    add_title : bool
        indicates whether to add title
    plot_mean : bool
        indicates whether to plot the mean
    plot_hdi : bool
        indidcates whether to plot the HDI
    label_hdi : bool
        indicates whether to add a label for the HDI
    
    Returns
    -------
    (plot)
    '''
    # extract posterior
    posterior = idata.posterior[var_name].values
    posterior = np.asarray(posterior).flatten()

    # plot
    if (kind=='hist'):
        plt.hist(posterior, bins=100, density=True, color=color, alpha=alpha, label=label)
    elif (kind=='kde'):
        az.plot_kde(posterior, plot_kwargs={'color': color}, label=label)

    # add mean
    if (plot_mean):
        plt.axvline(x=posterior.mean(), color=color, label='mean='+str(round(posterior.mean())))

    # add 95% HDI
    if (plot_hdi):
        lower_bound = az.hdi(idata, hdi_prob=0.95, var_names=[var_name])[var_name].values[0][0]
        upper_bound = az.hdi(idata, hdi_prob=0.95, var_names=[var_name])[var_name].values[0][1]
        if (label_hdi):
            plt.axvline(x=lower_bound, color=color, linestyle='--', label='95% HDI')
        else:
            plt.axvline(x=lower_bound, color=color, linestyle='--')
        plt.axvline(x=upper_bound, color=color, linestyle='--')
    
    # add legend
    if (plot_mean or (label is not None) or label_hdi):
        plt.legend()

    # labels
    if (add_title):
        plt.title(title)
    plt.xlabel(var_name)
    plt.ylabel('p(' + var_name + ')');


def plot_posteriors(models_dat, model_indices, var_names, kind='kde', alpha=0.5, plot_mean=False, plot_hdi=False, label_hdi=False):
    '''
    Plot posterior distributions.

    Parameters
    ----------
    models_dat : list of dict
        List of dictionaries. Each dict contains data of one model.
    model_indices : list of int
        Indices of the models for which to plot posterior distribution.
    var_names : list of str
        names of the parameters for which to plot the posterior distributions
    kind : str
        type of plot: 'hist' or 'kde'
    alpha : number in [0, 1]
        opacity (for histogram)
    plot_mean : bool
        indicates whether to plot the mean
    plot_hdi : bool
        indidcates whether to plot the HDI
    label_hdi : bool
        indicates whether to add a label for the HDI
    
    Returns
    -------
    (plot)
    '''
    if (len(model_indices) > 8):
        print('To many models for one plot.')
        return

    # initialize figure
    plt.figure(figsize=(20,4))

    # colors for models
    colors = ['royalblue', 'lime', 'darkviolet', 'magenta', 'tomato', 'cyan', 'peru', 'darkgreen']

    for var_idx, var_name in enumerate(var_names):

        # initialize new figure for each parameter
        plt.subplot(1, len(var_names), var_idx+1)
        
        for idx, model_idx in enumerate(model_indices):
            plot_posterior(models_dat[model_idx]['idata'], var_name, kind=kind, color=colors[idx], alpha=alpha, label=models_dat[model_idx]['name'], plot_hdi=plot_hdi, label_hdi=label_hdi)
            
    plt.tight_layout()


# plot density
# taken from https://www.youtube.com/watch?v=tKJDIN4Y7LE, edited
def plot_cont(self, label=None, color='blue'):
    samples = self.random(size=1000)
    x = np.linspace(np.min(samples), np.max(samples), 1000)
    plt.plot(x, np.exp(self.logp(x)).eval(), label=label, color=color)

# assign this functionality to the abstract class Continuous
pm.Continuous.plot = plot_cont

def plot_prior_and_posterior(models_dat, model_idx, var_name, title=None):
    '''
    Plot posterior distribution together with prior (truncated) normal
    distribution.

    Parameters
    ----------
    models_dat : list of dict
        List of dictionaries. Each dict contains data of one model.
    model_idx : int
        Index of the model for which to compare prior and posterior.
    var_name : str
        name of the parameter for which to plot the prior and posterior distribution
    title : str
        title
    
    Returns
    -------
    (plot)
    '''
    # translate var_name to key
    mu_prior_key, sigma_prior_key = m_helpers.translate_var_name_to_prior_config_key(var_name)

    # extract model data
    prior_config = models_dat[model_idx]['prior_config']
    idata = models_dat[model_idx]['idata']

    # plot prior
    # check whether intercept prior distribution is truncated
    if ((var_name == 'beta_intercept') & (prior_config['truncate_intercept_prior'])):
        pm.TruncatedNormal.dist(mu=prior_config[mu_prior_key], sd=prior_config[sigma_prior_key], lower=0).plot(label='prior', color='lime')
    else:
        pm.Normal.dist(mu=prior_config[mu_prior_key], sd=prior_config[sigma_prior_key]).plot(label='prior', color='lime')

    # plot posterior
    plot_posterior(idata, var_name, label='posterior', add_title=False)
    plt.legend()
    if (title is not None):
        plt.title(title);
    else:
        plt.title('Prior and posterior distribution for ' + var_name);


def compare_models_priors_and_posteriors(models_dat, model_indices, var_name):
    '''
    Plot posterior distributions together with prior normal distributions.

    Parameters
    ----------
    models_dat : list of dict
        List of dictionaries. Each dict contains data of one model.
    model_indices : list of int
        Indices of the models for which to compare prior and posterior.
    var_name : str
        name of the parameter for which to plot the prior and posterior distributions
    
    Returns
    -------
    (plot)
    '''
    # initialize figure
    plt.figure(figsize=(20,5))

    for idx, model_idx in enumerate(model_indices):

        # extract model data
        prior_config = models_dat[model_idx]['prior_config']
        idata = models_dat[model_idx]['idata']

        plt.subplot(1, len(model_indices), idx+1)
        plot_prior_and_posterior(models_dat, model_idx, var_name, title=models_dat[model_idx]['name'])

    plt.suptitle('Prior and posterior distribution for ' + var_name)

    plt.tight_layout()



# plot prediction intervals together with datapoints
def plot_data_with_pred_intervals(quants_per_datapoint_df, dat=mdl.get_data_for_location(), ylabel=None):
    '''
    Plot prediction intervals together with datapoints.

    Parameters
    ----------
    quants_per_datapoint_df : pd.DataFrame
        Dataframe with columns '2-5-quant' and '97-5-quant'
    dat : pd.DataFrame
        Dataframe with columns 'temperature', 'is_busday', 'rider_count'

    Returns
    -------
    (plot)
    '''
    # initialize figure
    plt.figure()

    # plot data
    if (ylabel is not None):
        plot_data(dat, ylabel=ylabel)
    else:
        plot_data(dat)

    # indices of business days / non-business days
    dat_copy = dat.reset_index(drop=True) # copy dat to reset index (and not change original dataframe)
    busday_indices = dat_copy[dat_copy['is_busday']].index
    nonbusday_indices = dat_copy[~dat_copy['is_busday']].index

    # plot 95% prediction intervals
    plt.vlines(x=dat.iloc[busday_indices]['temperature'], ymin=quants_per_datapoint_df.iloc[busday_indices]['2-5-quant'], ymax=quants_per_datapoint_df.iloc[busday_indices]['97-5-quant'], color='orange', alpha=0.3, label='95% prediction interval business day')
    plt.vlines(x=dat.iloc[nonbusday_indices]['temperature'], ymin=quants_per_datapoint_df.iloc[nonbusday_indices]['2-5-quant'], ymax=quants_per_datapoint_df.iloc[nonbusday_indices]['97-5-quant'], color='blue', alpha=0.3, label='95% prediction interval weekend')
    plt.legend();