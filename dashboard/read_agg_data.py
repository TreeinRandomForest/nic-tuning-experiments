from config import *
from utils import *
import pandas as pd

#def start_netpipe_analysis(drop_outliers=False, old=False)
#def start_nodejs_analysis(filename, drop_outliers=False, scale_requests=False)
#def start_mcd_analysis(filename, drop_outliers=False, scale_requests=True)
#def start_mcdsilo_analysis(filename, drop_outliers=False, scale_requests=False)

def start_analysis(workload, drop_outliers=False, **kwargs):
    if workload=='netpipe':
        if 'old' not in kwargs:
            kwargs['old'] = False

        df_comb, df, outlier_list = start_netpipe_analysis(drop_outliers=drop_outliers, old=kwargs['old'])

    return df_comb, df, outlier_list

def start_netpipe_analysis(drop_outliers=False, old=False):
    N_ROUNDS = 5000

    if old:
        df_fixed = pd.read_csv('jul20data.csv')
    else:
        df_fixed = pd.read_csv('aug11data.csv') #truncated logs for 512k fixed
    df_gov = pd.read_csv('jul20data_governor.csv')

    df_fixed = rename_cols(df_fixed)
    df_gov = rename_cols(df_gov)

    #outlier_list = identify_outliers(df, dfr)
    #df_fixed = filter_outliers(df_fixed)

    df_fixed['gov'] = 0
    df_gov['gov'] = 1

    df_fixed = convert_units(df_fixed)
    df_gov = convert_units(df_gov)

    df = pd.concat([df_fixed, df_gov], axis=0)
    df['tput'] = df['msg']*N_ROUNDS / df['time']

    df['eput'] = df['msg']*N_ROUNDS / df['joules']

    df_comb = prepare_scan_all_data(df)
    df_comb.reset_index(inplace=True)

    outlier_list = identify_outliers(df_comb, df)
    if drop_outliers:    
        print(f'Dropping {len(outlier_list)} outlier rows')

        print(f'Before: {df.shape[0]}')
        df = filter_outliers(outlier_list, df)
        print(f'After: {df.shape[0]}')

        df_comb = prepare_scan_all_data(df)

    return df_comb, df, outlier_list

def start_nodejs_analysis(filename, drop_outliers=False, scale_requests=False):
    df = pd.read_csv(filename, sep=' ')

    df.drop(['START_RDTSC', 'END_RDTSC'], axis=1, inplace=True)

    df = rename_cols(df)

    print('Dropping rows with time <= 29')
    print(f'Before: {df.shape[0]}')
    df = df[df['time']>29].copy()
    print(f'After: {df.shape[0]}\n')

    def scale_to_requests(d):
        COLS_TO_SCALE = ['requests',
                         'time',
                         'joules',
                         'instructions',
                         'cycles',
                         'refcyc',
                         'llc_miss',
                         'c3',
                         'c6',
                         'c7']
        SCALE_FACTOR = 100000. / d['requests']

        for c in COLS_TO_SCALE:
            d[c] = d[c] * SCALE_FACTOR

        return d

    if scale_requests:
        df = scale_to_requests(df)

    df['edp'] = 0.5 * (df['joules'] * df['time'])
    df['tput'] = df['requests'] / df['time']
    df['eput'] = df['requests'] / df['joules']

    dfr = df.copy() #raw data
    df = prepare_scan_all_data(dfr) #grouped by data
    df.reset_index(inplace=True)

    outlier_list = identify_outliers(df, dfr)
    if drop_outliers:    
        print(f'Dropping {len(outlier_list)} outlier rows')

        print(f'Before: {dfr.shape[0]}')
        dfr = filter_outliers(outlier_list, dfr)
        print(f'After: {dfr.shape[0]}')

        df = prepare_scan_all_data(dfr) #grouped by data    

        df.reset_index(inplace=True)

    return df, dfr, outlier_list

def start_mcd_analysis(filename, drop_outliers=False, scale_requests=True):
    '''TODO: Merge with start_nodejs_analysis'''
    df = pd.read_csv(filename, sep=' ')

    df = rename_cols(df)

    #fix QPS
    possible_qps_vals = np.array([200000, 400000, 600000])
    print(f"Possible QPS values : {possible_qps_vals}")
    def cluster_qps_values(df):
        df['QPS_uncorrected'] = df['QPS']
        
        df['QPS'] = df['QPS'].apply(lambda x: possible_qps_vals[np.argmin(np.abs((x - possible_qps_vals)))])

        return df

    df = cluster_qps_values(df)

    #drop time < 30s
    print('Dropping rows with time <= 29')
    print(f'Before: {df.shape[0]}')
    df = df[df['time']>29].copy()
    print(f'After: {df.shape[0]}\n')

    #drop 99th percentile > 500 latencies
    print('Dropping rows with read_99th latency > 500')
    print(f'Before: {df.shape[0]}')
    df = df[df['read_99th'] <= 500].copy()
    print(f'After: {df.shape[0]}\n')

    def scale_to_requests(d):
        print("Scaling to 5 million requests")
        COLS_TO_SCALE = ['time',
                         'joules',
                         'instructions',
                         'cycles',
                         'refcyc',
                         'llc_miss',
                         'c3',
                         'c6',
                         'c7']

        SCALE_FACTOR = 5000000. / (d['QPS_uncorrected']*d['time'])

        for c in COLS_TO_SCALE:
            d[c] = d[c] * SCALE_FACTOR

        return d

    if scale_requests:
        df = scale_to_requests(df)


    df['edp'] = 0.5 * (df['joules'] * df['time'])
    df['eput'] = df['QPS'] * df['time'] / df['joules']
    #df['tput'] = df['requests'] / df['time']
    #df['eput'] = df['requests'] / df['joules']

    dfr = df.copy() #raw data
    df = prepare_scan_all_data(dfr) #grouped by data
    df.reset_index(inplace=True)

    outlier_list = identify_outliers(df, dfr)
    if drop_outliers:    
        print(f'Dropping {len(outlier_list)} outlier rows')

        print(f'Before: {dfr.shape[0]}')
        dfr = filter_outliers(outlier_list, dfr)
        print(f'After: {dfr.shape[0]}')

        df = prepare_scan_all_data(dfr) #grouped by data    
        df.reset_index(inplace=True)

    return df, dfr, outlier_list

def start_mcdsilo_analysis(filename, drop_outliers=False, scale_requests=False):
    df = pd.read_csv('aug19_mcdsilologs/mcdsilo_combined.csv', sep=' ')

    df = rename_cols(df)

    possible_qps_vals = np.array([50000, 100000, 200000])
    print(f"Possible QPS values : {possible_qps_vals}")
    def cluster_qps_values(df):
        df['QPS_uncorrected'] = df['QPS']
        
        df['QPS'] = df['QPS'].apply(lambda x: possible_qps_vals[np.argmin(np.abs((x - possible_qps_vals)))])

        return df

    df = cluster_qps_values(df)

    #drop time < 30s
    print('Dropping rows with time <= 19')
    print(f'Before: {df.shape[0]}')
    df = df[df['time']>19].copy()
    print(f'After: {df.shape[0]}\n')

    #drop 99th percentile > 500 latencies
    print('Dropping rows with read_99th latency > 500')
    print(f'Before: {df.shape[0]}')
    df = df[df['read_99th'] <= 500].copy()
    print(f'After: {df.shape[0]}\n')

    def scale_to_requests(d):
        print("Scaling to 1 million requests")
        COLS_TO_SCALE = ['time',
                         'joules',
                         'instructions',
                         'cycles',
                         'refcyc',
                         'llc_miss',
                         #'c3',
                         #'c6',
                         #'c7'
                         ]

        SCALE_FACTOR = 1000000. / (d['QPS_uncorrected']*d['time'])

        for c in COLS_TO_SCALE:
            try:
                d[c] = d[c] * SCALE_FACTOR
            except:
                print("problem with column", c)                

        return d

    if scale_requests:
        df = scale_to_requests(df)

    df['edp'] = 0.5 * (df['joules'] * df['time'])
    #df['tput'] = df['requests'] / df['time']
    #df['eput'] = df['requests'] / df['joules']

    dfr = df.copy() #raw data
    df = prepare_scan_all_data(dfr) #grouped by data
    df.reset_index(inplace=True)

    outlier_list = identify_outliers(df, dfr)
    if drop_outliers:    
        print(f'Dropping {len(outlier_list)} outlier rows')

        print(f'Before: {dfr.shape[0]}')
        dfr = filter_outliers(outlier_list, dfr)
        print(f'After: {dfr.shape[0]}')

        df = prepare_scan_all_data(dfr) #grouped by data    
        df.reset_index(inplace=True)

    return df, dfr, outlier_list
    