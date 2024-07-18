import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Final Project/ARBike.csv")
print("Shape of unfiltered data:", data.shape)
print("\nMissing value counts:\n",data.isna().sum())   # We can see there are just a few datapoints missing that we have to take care of

# This function will take a dataset and column name or list of names as input and fill NA values with the median
def fill_with_median(data, column_name):
    if isinstance(column_name, list):
        for col in column_name:
            impute_val = data[col].median()
            data[col] = data[col].fillna(impute_val)
    else:
        impute_val = data[column_name].median()
        data[column_name] = data[column_name].fillna(impute_val)

data.columns = data.columns.str.strip()  # Some columns had unnecessary extra whitespace characters
cols_with_NA = ["TMAX_10THC", "TMIN_10THC", "REGISTERED", "TTLCNT"]  
fill_with_median(data, cols_with_NA)
print("\nMissing value counts after cleaning:\n",data.isna().sum())   # Now we can see all the NA values are gone

# This function will take a dataset and a column name and filter outliers using IQR (interquartile range)
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    iqr = Q3 - Q1
    lower_bound = Q1 - 2.25 * iqr
    upper_bound = Q3 + 2.25 * iqr
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

outlier_potentials = ["PRCP_MM", "TMAX_10THC", "TMIN_10THC", "AWND_MTRSPERSEC", "REGISTERED", "CASUAL", "TTLCNT"]
for col in outlier_potentials:
    data = remove_outliers_iqr(data, col)
    
# We will use indexing options to create a new dataset that has only the columns we need
# Columns like station, latitude, longitude, elevation, and name are not needed because 
# they're the same throughout the entire dataset
data = data[['DATE', 'PRCP_MM', 'TMAX_10THC', 'TMIN_10THC', 'AWND_MTRSPERSEC', 'CASUAL', 'REGISTERED', 'TTLCNT']]
data.head()

print("\nShape of filtered data:", data.shape)  # Our function detected 73 outliers, but now we know our data is clean

# We can sort the dataset by registered users to see the what conditions brought the most registered users
highest_reg = data.sort_values(by='REGISTERED', ascending=False)
print(highest_reg.head())

# We can compute a boolean array to see which rows have a temp of at least 200
temp_bool_arr = data['TMAX_10THC'] >= 200
temp_bool_arr[temp_bool_arr == False].count()
print(temp_bool_arr.head())

# We can also demonstrate two different types of joins by creating two datasets and merging them together
df1 = data[['DATE','CASUAL', 'TMAX_10THC']].iloc[::2]
df2 = data[['DATE', 'REGISTERED']]

inner_join = pd.merge(df1, df2, on='DATE', how='left')
outer_join = pd.merge(df1, df2, on='DATE', how='right')
print("\n", inner_join.head())
print(outer_join.head())

# We can use 2 different groupby methods to view total number of users by month and average users per weekday
# First, convert 'DATE' column to datetime format
data['DATE'] = pd.to_datetime(data['DATE'])
# Total number of users by month
monthly_users = data.groupby(data['DATE'].dt.month)[['REGISTERED', 'CASUAL']].sum()
print("\nTotal Number of Users by Month:")
print(monthly_users)

# Average users per weekday
weekday_users = data.groupby(data['DATE'].dt.weekday)[['REGISTERED', 'CASUAL']].mean()
print("\nAverage Users per Weekday:")
print(weekday_users)

# We can slice the dataset to focus on a specific month with datetime
# We will look at August, since we saw from before it brought the highest registered customer attendance
august_data = data[data['DATE'].dt.month == 8]
print("\nUser Attendance in August:")
print(august_data[['DATE', 'PRCP_MM', 'TMAX_10THC', 'TMIN_10THC', 'CASUAL', 'REGISTERED']].head())

# We can create a period index for quarterly analysis
quarterly_index = pd.PeriodIndex(data['DATE'], freq='Q')
data['QUARTER'] = quarterly_index
print("\nQuarterly Analysis:")
print(data[['DATE', 'TMAX_10THC', 'TMIN_10THC', 'CASUAL', 'REGISTERED', 'QUARTER']].head())

# We can calculate the duration between two specific dates
# We will calculate the distance between a random date in the highest attended month and the lowest
date1 = pd.to_datetime('2021-08-21')
date2 = pd.to_datetime('2021-01-03')
duration = date1 - date2
print("\nDuration between", date1, "and", date2, ":", duration)

# We will start by splitting our data into 2 sets: casual and registered
casual = data[['DATE', 'PRCP_MM', 'TMAX_10THC', 'TMIN_10THC', 'AWND_MTRSPERSEC', 'CASUAL']]
registered = data[['DATE', 'PRCP_MM', 'TMAX_10THC', 'TMIN_10THC', 'AWND_MTRSPERSEC', 'REGISTERED']]

# Now we will create a line plot with point markers for registered customers
plt.figure(figsize=(10, 6))

# Plot registered customers
plt.plot(registered['DATE'], registered['REGISTERED'], marker='o', linestyle='-', color='green', label='Registered Customers')
# Plot casual customers
plt.plot(casual['DATE'], casual['CASUAL'], marker='o', linestyle='-', color='darkblue', label='Casual Customers')
# Add details like labels and a legend
plt.xlabel('Date')
plt.ylabel('Number of Customers')
plt.title('Registered and Casual Customer Counts over the Year')
plt.xticks(rotation=45)
plt.xticks(registered['DATE'][::15])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Vertical Bar Plot: Number of customers by wind speed
wind_users = data.groupby('AWND_MTRSPERSEC')[['REGISTERED', 'CASUAL']].sum()
wind_users.plot(kind='bar', figsize=(10, 6), color=['red', 'blue'])
plt.title('Number of Customers in Relation to Wind Speeds')
plt.xlabel('Wind Speed (meters per second)')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.legend(['Registered', 'Casual'])
plt.grid()
plt.tight_layout()
plt.show()

# Horizontal Bar Plot: Number of customers by amount of rain/snow
weather_users = data.groupby('PRCP_MM')[['REGISTERED', 'CASUAL']].sum()
weather_users.plot(kind='barh', figsize=(10, 6), color=['skyblue', 'purple'])
plt.title('Number of Customers in Relation to Amount of Precipitation')
plt.xlabel('Number of Customers')
plt.ylabel('Milimeters of Rain/Snow')
plt.legend(['Registered', 'Casual'])
plt.grid()
plt.show()

# Scatter Plot with regression lines for REGISTERED customers
sns.regplot(x='TMAX_10THC', y='REGISTERED', data=data, scatter_kws={'color': 'blue'}, line_kws={'color': 'blue'})
# Scatter Plot with regression lines for CASUAL customers
sns.regplot(x='TMAX_10THC', y='CASUAL', data=data, scatter_kws={'color': 'orange'}, line_kws={'color': 'orange'})

plt.title('Number of Total Customers in Relation to Increasing Temperatures')
plt.xlabel('Maximum Temperature')
plt.ylabel('Number of Customers')

# The legend wouldn't work right, so I had to jump through some hoops for it
legend_handles = [plt.Line2D([0], [0], color='blue', marker='o', linestyle='None', label='REGISTERED'),
                  plt.Line2D([0], [0], color='orange', marker='s', linestyle='None', label='CASUAL')]
plt.legend(handles=legend_handles)
plt.show()

# We first extract the month from the 'DATE' column
data['MONTH'] = data['DATE'].dt.month

# Then we define a helper function to group by season
def get_season(month):
    if month in [12,1,2]:
        return 'Winter'
    elif month in [3,4,5]:
        return 'Spring'
    elif month in [6,7,8]:
        return 'Summer'
    elif month in [9,10,11]:
        return 'Fall'
    else:
        raise Exception('Error, month data not within required range')

# Now we apply the function to create a 'SEASON' column
data['SEASON'] = data['MONTH'].apply(get_season)

# Finally, we create the box plot
plt.figure(figsize=(10,6))
sns.boxplot(x='SEASON', y='TTLCNT', data=data)
plt.title('Number of Total Customers by Season')
plt.xlabel('Season')
plt.ylabel('Number of Total Customers')
plt.show()

# Scatter Plot with regression lines for REGISTERED customers
sns.regplot(x='TMIN_10THC', y='REGISTERED', data=data, scatter_kws={'color': 'darkblue'}, line_kws={'color': 'darkblue'})
# Scatter Plot with regression lines for CASUAL customers
sns.regplot(x='TMIN_10THC', y='CASUAL', data=data, scatter_kws={'color': 'lightblue'}, line_kws={'color': 'lightblue'})

plt.title('Number of Total Customers in Relation to Low Temperatures')
plt.xlabel('Minimum Temperature')
plt.ylabel('Number of Customers')
legend_handles = [plt.Line2D([0], [0], color='darkblue', marker='o', linestyle='None', label='REGISTERED'),
                  plt.Line2D([0], [0], color='lightblue', marker='s', linestyle='None', label='CASUAL')]
plt.legend(handles=legend_handles)

# Reverse x-axis to go from high temps to low temps instead of low to high
plt.xlim(plt.xlim()[::-1])
plt.show()

# 'predictors' in X array
X = data[['PRCP_MM', 'TMAX_10THC', 'TMIN_10THC', 'AWND_MTRSPERSEC']]

# First we calculate mean and standard deviation of each predictor
mean_prcp = data['PRCP_MM'].mean()
std_prcp = data['PRCP_MM'].std()
mean_maxtemp = data['TMAX_10THC'].mean()
std_maxtemp = data['TMAX_10THC'].std()
mean_mintemp = data['TMIN_10THC'].mean()
std_mintemp = data['TMIN_10THC'].std()
mean_wind_speed = data['AWND_MTRSPERSEC'].mean()
std_wind_speed = data['AWND_MTRSPERSEC'].std()

# Threshold sets data to where it has to be within one standard deviation
threshold = 1 * std_wind_speed

# Now define a function to determine if weather attribute is close enough to the mean
def is_close_enough(value, mean_attribute, std_attribute):
    # Threshold sets data to where it has to be within one standard deviation
    threshold = 1 * std_attribute
    return abs(value - mean_attribute) <= threshold

def will_bike(row):
    # Now we use that function to generate boolean arrays for each attribute
    prcp_close = is_close_enough(row['PRCP_MM'], mean_prcp, std_prcp)
    maxtemp_close = is_close_enough(row['TMAX_10THC'], mean_maxtemp, std_maxtemp)
    mintemp_close = is_close_enough(row['TMIN_10THC'], mean_mintemp, std_mintemp)
    wind_close = is_close_enough(row['AWND_MTRSPERSEC'], mean_wind_speed, std_wind_speed)
    
    if prcp_close and maxtemp_close and mintemp_close and wind_close:
        return 1
    else:
        return 0

# Finally, apply the function and create a new column    
data['WILL_BIKE'] = data.apply(will_bike, axis=1)

# Now use our new column as our 'y'
y = data['WILL_BIKE']

# We use train_test_split function to split data into training and testing sets (80% train, 20% test)
# This is from sk_learn library, handy for when you don't have separate train and test datasets already
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now instantiate the logistic regression model
model = LogisticRegression(C=10)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Score the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display summary statistics
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
y_pred = model.predict(X_test)
print("R-squared:", r2_score(y_test, y_pred))