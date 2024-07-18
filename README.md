# DSCI 230 Final Project

## Project Overview
For this project, I was contacted by a mountain biking company in Bentonville, AR, where there are world-class mountain biking trails. The company has customers who register to go biking before arriving, and customers who walk in without a reservation. For this project, there are registered and casual users, with registered being those who made reservations and casual users who did not.

## Objectives
I aimed to answer several questions through my data analysis, including:
- If weather affects one group of customers more than another
- If there are thresholds in which a group may decide not to come (e.g., snow, rain, high temps)
- Exploring a relationship between weather attributes and registered/casual customers

## Methods
To answer these questions, I used a variety of tools and methods in Python, such as:
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Data Operations
After cleaning the data, I performed various operations, including:
1. Sorting the dataset by registered users to identify conditions attracting the most registered customers.
2. Computing a boolean array to filter rows with temperatures of at least 200.
3. Using join operations to merge two subsets of the data.
4. Using GroupBy methods to view total viewers by month and average users per weekday.
5. Slicing the dataset to look at a certain month using DateTime.
6. Creating a Period Index for quarterly analysis.
7. Calculating the distance between dates in the highest and lowest attended months.

## Findings
- August had the most registered customers, while July had the most casual customers.
- Tuesdays had the highest average registered customer attendance, while Fridays had the highest casual customer attendance.
- The company experienced the most customer traffic during the summer and fall months.

## Visualizations
- Line plots showing the relationship between registered and casual customer attendance over the year.
- Bar plots illustrating customer attendance in relation to wind speed and precipitation.
- Scatter plots with regression lines showing attendance correlation with higher and lower temperatures.
- Box plots displaying customer attendance by season.

## Model Building
I built a model to predict whether a potential customer would go biking on a given day based on weather attributes. The process included:
- Calculating the mean and standard deviation for each weather attribute.
- Creating a threshold value representing one standard deviation from the mean.
- Comparing data points to this threshold to predict attendance.
- Using linear regression to build and score the model, achieving approximately 75% accuracy.

## Conclusion
- No weather attribute significantly affects one group of customers more than another.
- Warm weather leads to higher customer traffic.
- The predictive model can help the company make cost-effective business decisions, such as adjusting prices or employee schedules based on weather predictions.

## Jupyter Notebook
The entire project, including data analysis and visualizations, is documented in the accompanying Jupyter Notebook.

---
