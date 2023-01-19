import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
import m_data_loading as mdl
import m_plotting as mplot

# TODO describe what is in this file

# TODO rename function
def post_pred_check(model, model_idata, given_data=mdl.get_data_for_location(), change_dat=False, print_output=False, plot_error_hist=False, plot_pred_ints=False, plot_post_pred_means=False, quant0=True):
    '''
    Compare predicted values to observed values.
    Compute prediction intervals and MAE.
    TODO weiter kommentieren
    '''
    with model:

        if(change_dat):
            # change data
            pm.set_data({'dat_temps': given_data[['temperature']]})
            pm.set_data({'dat_busday': given_data[['is_busday']]})
            pm.set_data({'dat_rider_counts': given_data[['rider_count']]}) # TODO eigentlich nicht nötig, oder?
            # TODO check that data has the correct size - to test whether change of data worked

        # simulate: sample from sampled posterior
        #ppc = pm.sample_posterior_predictive(model_idata)
        # TODO use fast version?
        ppc = pm.fast_sample_posterior_predictive(model_idata)
        
        # compute posterior predictive means
        posterior_predicitive_means = ppc['observed'].mean(axis=0)
        # TODO flatten ppc['observed']?

        # compute errors
        errors = given_data['rider_count'] - posterior_predicitive_means.flatten()

        # compute absolute errors
        absolute_errors = np.abs(errors)

        # compute median absolute error
        median_absolute_error = np.median(absolute_errors)

        # compute scaled median absolute error
        standard_deviations = ppc['observed'].std(axis=0).flatten() # TODO ddof?
        scaled_MAE = np.median(absolute_errors / standard_deviations) # TODO Stimmt das?

        # calculate prediction intervals
        # for each datapoint, calculate 4 quantiles: 2.5%, 25%, 75%, 97.5% --> 95% and 50% prediction interval
        quantiles = np.quantile(ppc['observed'], [0.025, 0.25, 0.75, 0.975], axis=0).T # number of rows = number of datapoints in original dataset, number of columns = 4 = number of computed quantiles

        # convert numpy array to dataframe to make adding of columns and comparison of columns easier
        if (quant0): # TODO correct it
            quants_per_datapoint_df = pd.DataFrame(quantiles[0], columns=['2-5-quant', '25-quant', '75-quant', '97-5-quant'])
        else:
            quants_per_datapoint_df = pd.DataFrame(quantiles, columns=['2-5-quant', '25-quant', '75-quant', '97-5-quant'])

        # add original datapoints to df
        quants_per_datapoint_df['rider_count'] = given_data['rider_count'].values

        # check whether points in 50% / 95% prediction interval

        # point in 50% prediction interval?
        # TODO < / > or <=/>= Include equality?
        quants_per_datapoint_df['point_in_50_pred_int'] = ((quants_per_datapoint_df['rider_count'] <= quants_per_datapoint_df['75-quant']) & (quants_per_datapoint_df['rider_count'] >= quants_per_datapoint_df['25-quant']))

        # point in 95% prediction interval?
        # TODO < / > or <=/>= Include equality?
        quants_per_datapoint_df['point_in_95_pred_int'] = ((quants_per_datapoint_df['rider_count'] <= quants_per_datapoint_df['97-5-quant']) & (quants_per_datapoint_df['rider_count'] >= quants_per_datapoint_df['2-5-quant']))
        
        # compute percentage of original datapoints that lie within the prediction interval
        percent_within_50 = sum(quants_per_datapoint_df['point_in_50_pred_int']) / len(quants_per_datapoint_df)
        percent_within_95 = sum(quants_per_datapoint_df['point_in_95_pred_int']) / len(quants_per_datapoint_df)

        if (print_output):

            print('Median absolute error: ', median_absolute_error)
            print('Standard deviation of absolute errors: ', np.std(absolute_errors))

            print('scaled MAE:', scaled_MAE)
            
            #display(quants_per_datapoint_df) # TODO uncomment?

            print('percentage of datapoints in 50% prediction interval:', percent_within_50)
            print('percentage of datapoints in 95% prediction interval:', percent_within_95)
        
        if (plot_error_hist):

            # plot histogram of errors
            plt.hist(errors);
            plt.xlabel('error: difference between observed rider count and posterior predictive mean');
        
        if (plot_pred_ints):
            mplot.plot_data_with_pred_intervals(quants_per_datapoint_df, dat=given_data)

            # TODO re-structure code: make it also possible to plot posterior predictive means without prediction intervals
            if (plot_post_pred_means):
                plt.scatter(given_data['temperature'], posterior_predicitive_means, label='posterior predictive means', color='black', marker='x') # TODO Heißt das so? # TODO adjust color to busday/non-busday
                plt.legend()

        # TODO weniger Rückgabewerte 
        return ppc, posterior_predicitive_means, errors, median_absolute_error, scaled_MAE, quants_per_datapoint_df, percent_within_50, percent_within_95


def compare_sim_to_original(sims, original, xlabel='Daily number of bike riders'):
    '''
    Compare simulated and original data distributions visually.

    Parameters
    ----------
    sims : TODO
        simulated bike rider counts
    original : TODO
        original (observed) bike rider counts
    xlabel : str
        x-axis label

    Returns
    -------
    (plot)
    '''
    # initialize figure
    plt.figure()

    # simulated bike rider counts data
    for i in np.arange(50):
        if i==0: # with label
            az.plot_kde(sims[i,:], label='simulated', plot_kwargs={'color': 'grey'})
        else:
            az.plot_kde(sims[i,:], plot_kwargs={'color': 'grey'})

    # original bike rider counts data
    az.plot_kde(original, label='original', plot_kwargs={'color': 'red'})

    # add axis label and legend
    plt.xlabel(xlabel)
    plt.legend();


def cross_validation(model_build_fun, prior_config, num_folds=5, dat=mdl.get_data_for_location(), print_results=True):
    '''
    Fit and test the model num_folds times on different subsets of the data.
    
    Parameters
    ----------
    num_folds : int
        Number of sets to partition the data in.
    model_build_fun : function
        Function to use to build the model and sample from the posterior.
    prior_config : dict
        Dictionary specifying the prior distribution.
    num_folds : int
        Number of sets to partition the data in.
    dat : pd.DataFrame
        Dataset; pd.DataFrame with columns 'temperature', 'is_busday',
        'rider_count'.
    print_results : bool
        indicates whether to print the results
    
    Returns
    -------
    tuple with lists
        list of median absolute errors on test set per fold
        list of scaled median absolute errors on test set per fold
        list of percentages that indicate how many of the test data points lie
        within the 50% prediction interval (per fold)
        list of percentages that indicate how many of the test data points lie
        within the 95% prediction interval (per fold)
    '''

    # number of datapoints
    n_datapoints = len(dat)

    # number of folds
    num_folds = 5 # TODO adjust

    # determine indices to split data randomly into num_folds parts
    indices = np.arange(n_datapoints)
    # shuffle
    np.random.shuffle(indices)

    # number of datapoints in each of the  num_folds part
    num_datapoints_per_part = int(np.floor(n_datapoints / num_folds))

    # TODO später entfernen
    storage_fold_train_data = []
    storage_fold_test_data = []

    # initialize storage for results
    # TODO später einige davon entfernen (brauchen nicht alle)
    storage_idata = []
    storage_ppcs = []
    storage_posterior_predicitive_means = []
    storage_AEs = []
    storage_MAEs = []
    storage_scaledMAEs = []
    storage_percent_within_50 = []
    storage_percent_within_95 = []

    for fold_idx in np.arange(num_folds):

        # TODO später entfernen
        print('fold ' + str(fold_idx))

        # extract train and test data for fold

        # determine indices of train and test data points
        
        start_idx_test_data = fold_idx * num_datapoints_per_part
        end_idx_test_data = (fold_idx + 1) * num_datapoints_per_part

        indices_of_test_data = indices[start_idx_test_data : end_idx_test_data]
        # everything that is not in the test set is in the train set
        indices_of_train_data = np.setxor1d(indices, indices_of_test_data)

        # extract train and test data
        train_data = dat.iloc[indices_of_train_data]
        test_data = dat.iloc[indices_of_test_data]
        test_data.reset_index(drop=True, inplace=True) # TODO für train_data auch machen?

        # store data
        storage_fold_train_data.append(train_data)
        storage_fold_test_data.append(test_data)


        # train the model

        # build linear model with categories "business-day vs no-business-day"
        # TODO Gibt das Probleme, weil ich es immer wieder überschreibe?
        # TODO make sure to use the same parameter values as above
        model, model_idata = model_build_fun(prior_config, train_data)
        # TODO check that data has the correct size

        # simulate for test set
        ppc, posterior_predicitive_means, errors, median_absolute_error, scaled_MAE, quants_per_datapoint_df, percent_within_50, percent_within_95 = post_pred_check(model, model_idata, test_data, change_dat=True)
        # TODO check that data has the correct size - to test whether change of data worked

        # store results
        storage_idata.append(model_idata)
        storage_ppcs.append(ppc['observed'])
        storage_posterior_predicitive_means.append(posterior_predicitive_means) # TODO maybe append flattened version?
        storage_AEs.append(np.abs(errors))
        storage_MAEs.append(median_absolute_error)
        storage_scaledMAEs.append(scaled_MAE)
        storage_percent_within_50.append(percent_within_50)
        storage_percent_within_95.append(percent_within_95)
    
    # print results of cross validation:
    if(print_results):
        print('mean MAE over all folds:', np.mean(storage_MAEs)) # TODO also report standard deviation of MAEs
        print('mean scaled MAE over all folds:', np.mean(storage_scaledMAEs))
        print('percentage of datapoints within 50% prediction interval:', np.mean(storage_percent_within_50))
        print('percentage of datapoints within 95% prediction interval:', np.mean(storage_percent_within_95))

    # TODO return less stuff
    return storage_MAEs, storage_scaledMAEs, storage_percent_within_50, storage_percent_within_95