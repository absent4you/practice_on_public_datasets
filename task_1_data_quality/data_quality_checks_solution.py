import pandas as pd
import statsmodels.stats.proportion as ssp
from scipy.stats import kstest, chi2_contingency

# Read the data
df_train = pd.read_csv(r'data/train.csv')
df_test = pd.read_csv(r'data/test.csv')

####################################
### 1: Compare shape of the data ###
####################################

# Check shapes of the data: What is length of train? test? Are their shapes equal?
print(f'Length of the train is {df_train.shape[0]}, length of the test is {df_test.shape[0]}')

if df_train.shape[1] == df_test.shape[1]:
    print('Amount of columns is equal')
else:
    print(f'Tables have different amount of columns. Difference in amount is {abs(df_train.shape[1] - df_test.shape[1])}')

# Find the differences between columns
intersection_btw_cols = list(set(df_train.columns) & set(df_test.columns))
presented_only_in_train = list(set(df_train.columns) - set(intersection_btw_cols))
presented_only_in_test = list(set(df_test.columns) - set(intersection_btw_cols))

# Conclusion: 'SalePrice' column is presented only in train which makes since since Kaggle task was to predict price in test :)

####################################
### 2: Compare column data types ###
####################################
columns_with_different_dtypes = []
for col in df_test.columns:
    if df_test[col].dtype != df_train[col].dtype:
        columns_with_different_dtypes.append([col, df_test[col].dtype, df_train[col].dtype])

print(f'There are {len(columns_with_different_dtypes)} columns with different data types.')

# Conclusion: from the comparison we see only one case: Float type in test, Int type in train.
# In our case the only difference is that numbers in test getting empty decimal part ".0" and it should not impact us.
# This can be explained by pandas read_csv method inconsistency and it's good to track and control dtypes, especially,
# for example, in banking industry where type of number representation in pc memory might impact further calculations.

#######################################
### 3: Compare missing values ratio ###
#######################################
# Check missing values in both datasets
percent_missing_train = df_train.isnull().mean() * 100
percent_missing_test = df_test.isnull().mean() * 100
percent_missing_train.name = 'train_miss_ratio'
percent_missing_test.name = 'test_miss_ratio'

mis_values_report = pd.concat([percent_missing_train, percent_missing_test], axis=1)
mis_values_report['abs_difference'] = abs(mis_values_report['train_miss_ratio'] - mis_values_report['test_miss_ratio'])

# Lets check columns with highest differences in missing value ratio
mis_values_report.sort_values(['abs_difference'], ascending=False).head()

# Should we care about the differences we see? How to check if difference is significant? A/B test!
# In our case we are comparing the difference in population proportions between 2 groups. Lets use Chi-Square test!
# p-value will tell us how likely null hypothesis is true (H0: proportions are equal in groups)
missing_values = pd.concat([df_train.isnull().sum(), df_test.isnull().sum()], axis=1)
missing_values.columns = ['train_mis_sum', 'test_mis_sum']
missing_values['train_length'] = len(df_train)
missing_values['test_length'] = len(df_test)
missing_values['p_val_prop_diff'] = missing_values.apply(lambda x: ssp.proportions_chisquare(x[:2], x[2:])[1], axis=1)
# Note: NaN values produced when there is no missing values in both datasets, thus, NaN is a good sign here.

# Join p_values back
mis_values_report = mis_values_report.join(missing_values['p_val_prop_diff'])

# Check columns with highest differences in missing value ratio
mis_values_report.sort_values(['abs_difference'], ascending=False).head()

# Conclusion: using p-value benchmark (e.x. 0.05) we can create identifier which will tell us whether ratio of missing
# values in new data differs from the already observed and based on that we can make a decision about data quality/stability.
# Absolute difference can be also used. For example benchmark will be dependent on combination of p-value & absolute diff

##########################################
### 4: Columns distribution comparison ###
##########################################
# To test whether new data has the same nature as already observed we can compare their distributions
# If distributions will differ a lot it can be a sign of new approach to data collection or data leakage, etc.
# Expectations: new data will have more-less same distribution as old one
# Comparison can be done using visualization or statistical tests
# I will not focus on visualizations since they are not rigor + with a big amount of data (new columns)
# they becoming irrelevant since we cannot check 100+ plots to make a decision

tests_output = {}
for col, type in zip(df_test.columns, df_test.dtypes):
    # Kolmogorov-Smirnov for continuous variables
    if type != 'object':
        tests_output[col] = kstest(df_train[col].dropna(), df_test[col].dropna())[1]
    # Contingency chi square for categorical variables (to handle multiclass variables)
    else:
        train = df_train[col].value_counts().reset_index()
        test = df_test[col].value_counts().reset_index()
        test.rename({col: 'test_col'}, axis=1, inplace=True)
        df_to_test = pd.merge(train, test, how='outer', on='index')
        df_to_test.fillna(0, inplace=True)
        tests_output[col] = chi2_contingency(df_to_test.drop(['index'], axis=1))[1]

tests_output = pd.DataFrame.from_dict(tests_output, orient='index', columns=['p-value']).sort_values(by='p-value')
tests_output.head(10)

# Conclusion: column 'Id' serves as good illustrator how should p-value look when distributions are totally different
# Based on test results we can see that distribution of few variables ('Fence', 'GarageQual',...) are differ
# significantly based on statistical tests and it could be sign for data quality/stability team to investigate them deeper