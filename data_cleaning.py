# Import necessary libraries
import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('your_input_data.csv')

# Data cleaning process
# Remove duplicate names
df = df.drop_duplicates(subset='Name')

# Clean 'Entrance Exam' column to accept only values between 0 and 100 and replace 'Failed' with NaN
df['Entrance Exam'] = pd.to_numeric(df['Entrance Exam'], errors='coerce')
df['Entrance Exam'] = df['Entrance Exam'].apply(lambda x: x if 0 <= x <= 100 else np.nan)

# Replace NaN values with 0 in 'Entrance Exam' column
df['Entrance Exam'].fillna(0, inplace=True)

# Define passing grade ranges for different courses
passing_grades = {
    'BS Computer Science': (65, 100),
    'BS Biology': (85, 100),
    'BS Civil Engineering': (85, 100),
    'BS Accountancy': (75, 100),
    'BS Criminology': (65, 100),
    'BS Information Technology': (65, 100),
    'BS Psychology': (85, 100),
    'BSED Mathematics': (80, 100)
}

# Convert 'Entrance Exam' column to numeric values
df['Entrance Exam'] = pd.to_numeric(df['Entrance Exam'], errors='coerce')

# Function to determine the course and remarks based on passing grade ranges
def determine_course_remarks(row):
    if pd.notnull(row['Entrance Exam']):
        for chosen_program in ['First Chosen Program', 'Second Chosen Program']:
            program = row[chosen_program]
            if program in passing_grades:
                grades_range = passing_grades[program]
                if grades_range[0] <= row['Entrance Exam'] <= grades_range[1]:
                    row['Course'] = program
                    row['Remarks'] = 'Passed'
                    return row

    if row['Physical Fitness Test'] == 'Failed':
        row['Interview'] = 'Failed'
        row['Remarks'] = 'Failed'
        return row

    row['Remarks'] = 'Failed'
    return row

# Apply the function to each row of the DataFrame
df = df.apply(determine_course_remarks, axis=1)

# Convert '1' to 'Passed' and '0' to 'Failed' in 'Interview' and 'Physical Fitness Test' columns
df['Interview'] = df['Interview'].map({'1': 'Passed', '0': 'Failed'})
df['Physical Fitness Test'] = df['Physical Fitness Test'].map({'1': 'Passed', '0': 'Failed'})
df.loc[df['Physical Fitness Test'] == 'Failed', ['Remarks', 'Interview', 'Entrance Exam']] = 'Failed', '0', 'Failed'

# Update 'Remarks' based on 'Interview'
df.loc[df['Interview'] == 'Failed', 'Remarks'] = 'Failed'

# Remove rows with values other than "1" and "0" in 'Interview' and 'Physical Fitness Test' columns
df = df[(df['Interview'].isin(['Passed', 'Failed'])) & (df['Physical Fitness Test'].isin(['Passed', 'Failed']))]

# Save cleaned data to a new CSV file
df.to_csv('cleaned_data.csv', index=False)