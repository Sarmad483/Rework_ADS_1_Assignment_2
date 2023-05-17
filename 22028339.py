# -*- coding: utf-8 -*-
"""22028339

#**Task 1: Data Ingestion and Manipulation:**
"""

import pandas as pd

def read_worldbank_data(url):
    # Read the CSV file, skipping the metadata rows
    df = pd.read_csv(url, skiprows=3)

    # Drop the columns with missing values
    df.dropna(axis=1, how='all', inplace=True)

    # Extract the years as column headers
    years = df.columns[4:]

    # Create a dataframe for years
    years_df = pd.DataFrame({'Year': years})

    # Transpose the dataframe to have countries as rows
    countries_df = df.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], value_vars=years, var_name='Year', value_name='Value')

    return years_df, countries_df

# Call the function and capture the returned dataframes
years_df, countries_df = read_worldbank_data('https://raw.githubusercontent.com/amna-sarwar/data/main/API_19_DS2_en_csv_v2_5455435.csv')

# Print the dataframes
print(years_df)
print(countries_df)

"""#**Task 2:Analysis of Statistical Properties and Correlations in Worldbank Data**

### **Summary Statistics and Mean Values:**
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a DataFrame
url = 'https://raw.githubusercontent.com/amna-sarwar/data/main/API_19_DS2_en_csv_v2_5455435.csv'
data = pd.read_csv(url, skiprows=4)

# Remove empty columns
data.dropna(axis=1, how='all', inplace=True)

# Select the indicators of interest
indicator1 = "Population growth (annual %)"
indicator2 = "Energy use (kg of oil equivalent per capita)"

# Filter the DataFrame for the selected indicators
subset = data.loc[data['Indicator Name'].isin([indicator1, indicator2])]

# Task 1: Summary Statistics
summary_stats = subset[['Indicator Name', '2019', '2020', '2021']].describe()
print("Summary Statistics:")
print(summary_stats)

# Task 2: Mean Values
mean_values = subset[['Indicator Name', '2019', '2020', '2021']].mean()
print("\nMean Values:")
print(mean_values)

# Visualization - Bar Plot for Mean Values
ax = mean_values.plot(kind='bar', rot=0, legend=False)
ax.set_ylabel('Mean Value')
ax.set_title('Mean Values of Indicators')

# Add labels to the bars
for i, p in enumerate(ax.patches):
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')

# Set the tick labels and axis labels
ax.set_xticklabels(mean_values.index)
ax.set_xlabel('Indicators')

plt.show()

"""### **Correlation Analysis:**"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a DataFrame
url = 'https://raw.githubusercontent.com/amna-sarwar/data/main/API_19_DS2_en_csv_v2_5455435.csv'
data = pd.read_csv(url, skiprows=4)

# Define the indicators of interest
indicator1 = 'Urban population (% of total population)'
indicator2 = 'Energy use (kg of oil equivalent per capita)'

# Filter the DataFrame for the indicators and countries of interest
countries = ['Aruba', 'Brazil']  # Specify the countries you want to analyze
subset = data[data['Country Name'].isin(countries)][['Country Name', 'Indicator Name', indicator1, indicator2]]

# Pivot the data to have years as columns and indicators as indices
pivoted_data = subset.pivot(index='Country Name', columns='Indicator Name')

# Drop unnecessary level in the column index
pivoted_data.columns = pivoted_data.columns.droplevel(0)

# Calculate the correlation between the two indicators for each country
correlations = pivoted_data[[indicator1, indicator2]].corr()

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
plt.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(countries)), countries, rotation=45)
plt.yticks(range(len(countries)), countries)
plt.title('Correlation Matrix')
plt.show()

"""#**Task 3:Analysis of Correlations and Trends in Worldbank Indicators**"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a DataFrame
url = 'https://raw.githubusercontent.com/amna-sarwar/data/main/API_19_DS2_en_csv_v2_5455435.csv'
data = pd.read_csv(url, skiprows=4)

# Remove empty columns
data.dropna(axis=1, how='all', inplace=True)

# Filter the DataFrame for the indicators of interest
indicators = ['Population growth (annual %)', 'Energy use (kg of oil equivalent per capita)', 'CO2 emissions (metric tons per capita)']
selected_data = data[data['Indicator Name'].isin(indicators)]

# Pivot the data to have indicators as columns and years as indices
pivoted_data = selected_data.pivot(index='Country Name', columns='Indicator Name')

# Drop the unnecessary level in the column index
pivoted_data.columns = pivoted_data.columns.droplevel(0)

# Extract the years from column names as strings
years = [str(col) for col in pivoted_data.columns]

# Convert the column names to integers
pivoted_data.columns = years

# Set up the figure and subplots
fig, axes = plt.subplots(nrows=3, figsize=(10, 18))

# Visualize time series data for population growth
pivoted_data['Population growth (annual %)'].plot(ax=axes[0], legend=False)
axes[0].set_ylabel('Population Growth (%)')
axes[0].set_title('Population Growth Over Time')
axes[0].tick_params(axis='x', rotation=45)

# Visualize time series data for energy consumption
pivoted_data['Energy use (kg of oil equivalent per capita)'].plot(ax=axes[1], legend=False)
axes[1].set_ylabel('Energy Consumption (kg of oil equivalent per capita)')
axes[1].set_title('Energy Consumption Over Time')
axes[1].tick_params(axis='x', rotation=45)

# Visualize time series data for CO2 emissions
pivoted_data['CO2 emissions (metric tons per capita)'].plot(ax=axes[2], legend=False)
axes[2].set_ylabel('CO2 Emissions (metric tons per capita)')
axes[2].set_title('CO2 Emissions Over Time')
axes[2].tick_params(axis='x', rotation=45)

# Adjust spacing between subplots
plt.subplots_adjust(hspace=1)

# Display the plots
plt.show()