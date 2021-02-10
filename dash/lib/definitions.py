# relevant variables 
relevant_variables = ["Police Spending",
'Percent Below Poverty (65 Years and Over)',
 'Percent Below Poverty (Worked Full-Time Past Twelve Months)',
 'Percent Inviduals Below Poverty Line (under 18)',
 'Percent Inviduals For Whom Poverty Status is Determined',
 'Percent of Individuals Under 18 Years of Age',
 'Percent of Individuals Below Poverty (Ages 18-64)',
 'Housing and Community Development',
 'Per Capita Personal Income Dollars',
 'Personal Income Thousands of Dollars',
 'Percent Educational Attainment (25 years and over)',
 'Percent of Individuals Aged 18-64',
 'Percent Labor Force (Age 16 and Over)']

vc_n = ['Housing and Community Development',
 'Per Capita Personal Income Dollars',
 'Percent Educational Attainment (25 years and over)',
 'Percent of Individuals Aged 18-64',
 'Percent Labor Force (Age 16 and Over)']

vc_p = ['Percent Inviduals Below Poverty Line (under 18)',
 'Percent Inviduals For Whom Poverty Status is Determined',
 'Percent of Individuals Below Poverty (Ages 18-64)',
 'Percent Below Poverty (65 Years and Over)',
 'Percent Below Poverty (Worked Full-Time Past Twelve Months)']

# Property Crime Positive Correlations 
pc_p = ['Percent Below Poverty (65 Years and Over)',
 'Percent Below Poverty (Worked Full-Time Past Twelve Months)',
 'Percent Inviduals Below Poverty Line (under 18)',
 'Percent Inviduals For Whom Poverty Status is Determined',
 'Population over 65 Years of Age',
 'Percent of Individuals Below Poverty (Ages 18-64)']

pc_n = ['Population',
 'Total Revenue',
 'Police Spending',
 'Education Services',
 'Education',
 'Higher Education',
 'Education (Elementary and Secondary)',
 'Social Services',
 'Public Welfare',
 'Welfare Vendors',
 'Welfare Other',
 'Correction',
 'Parks Recreation',
 'Housing and Community Development',
 'Per Capita Personal Income Dollars',
 'Personal Income Thousands of Dollars',
 'Population (ACS Estimate)',
 'Percent Educational Attainment (25 years and over)',
 'Percent of Individuals Aged 18-64',
 'Percent Labor Force (Age 16 and Over)',
 'Mean Income Deficit ($)']

races = ['Asian Population (%)',
 'Black/African-American Population (%)',
 'Hawaiian and Other Pacific Islander Population (%)',
 'White Population (%)',
 'Hispanic or Latino Population (%)']

regress_vars = {'Percent Inviduals Below Poverty Line (under 18)':
                               'youth_poverty','Hawaiian and Other Pacific Islander Population (%)':'percent_hawaiian',
                               'Mean Income Deficit ($)': 'income_deficit',
                               'Percent Educational Attainment (25 years and over)':'educational_attainment', 
                               'Percent of Total Workforce (16 years and over)': 'estimate_employed', 
                               'Black/African-American Population (%)':'percent_black',
                               'Violent Crime':'violent_crime',
                               'Total Revenue':'rev_total',
                               'Public Welfare':'public_welfare',
                                'Welfare Cash':'welfare_cash',
                                'Correction':'correction',
                                'Housing and Community Development':'housing_commdevt',   
                                'Percent of Individuals Under 18 Years of Age':'total_estimate_age_under_18_years',
                                'Percent Male Population':'total_estimate_sex_male',
                                'Population over 65 Years of Age':'total_estimate_age_65_years_and_over',
                                'Asian Population (%)':'percent_race_one_race_asian',  
                                'White Population (%)':'percent_race_one_race_white',
                                'Hispanic or Latino Population (%)':'percent_hispanic_or_latino'}