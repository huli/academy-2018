import numpy as np
import pandas as pd

def preprocess_data(data, samples):
    '''
    Function for removing outlier and applying some log transformations
    '''
    
    log_data = np.log(data)
    
    # Scale the sample data using the natural logarithm
    log_samples = np.log(samples)


    # Select the indices for data points you wish to remove
    outliers  = []

    # For each feature find the data points with extreme high or low values
    for feature in log_data.keys():

        # Calculate Q1 (25th percentile of the data) for the given feature
        Q1 = log_data[feature].quantile(.25)

        # Calculate Q3 (75th percentile of the data) for the given feature
        Q3 = log_data[feature].quantile(.75)

        # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
        step = 1.5 * (Q3 - Q1)

        outliers_of_feature = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]

        outliers.append(outliers_of_feature.index.tolist())

    outliers = [val for sublist in outliers for val in sublist]

    # Find outliers in more than one feature group
    outliers_series = pd.Series(outliers) 
    count_outliers = outliers_series.groupby(outliers_series).count()

    # Filter out outliers in more than one feature group
    outliers = [a for a in outliers if a not in count_outliers[count_outliers > 1]]

    # Remove the outliers, if any were specified
    good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
    print('Removed %s outliers and applied log transformation.' % len(outliers))
    return good_data, log_samples, outliers