from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, redirect, url_for
import pandas as pd
import numpy as np
import csv
import os
import logging
from zipfile import ZipFile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV




# Define a static folder for serving files
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

app = Flask(__name__, static_url_path='/static', static_folder='static')



@app.route('/')
def index():
    return render_template('blank.html')


@app.route('/clean_data', methods=['POST'])
def clean_data():
    # Read your CSV file
    file_path = r'C:/Users/Huawie/Downloads/USER INTERFACE CART/StudentRankwithGWA/Passed Candidates with GWA/Freshmen_Student_Final_Data_with_Course_and_Rank.csv'  # Replace with your file path
    df = pd.read_csv(file_path)

    # Data cleaning
    # Remove duplicate names
    df = df.drop_duplicates(subset='Name')

    # Clean 'Entrance Exam' column to accept only values between 0 and 100 and replace 'Failed' with NaN
    df['Entrance Exam'] = pd.to_numeric(df['Entrance Exam'], errors='coerce')
    df['Entrance Exam'] = df['Entrance Exam'].apply(lambda x: x if 0 <= x <= 100 else np.nan)
    df['Interview'] = pd.to_numeric(df['Interview'], errors='coerce')
    df['Interview'] = df['Interview'].apply(lambda x: x if 0 <= x <= 100 else np.nan)
    df['GWA'] = pd.to_numeric(df['GWA'], errors='coerce')
    df['GWA'] = df['GWA'].apply(lambda x: x if 0 <= x <= 100 else np.nan)

    # Replace NaN values with 0 in 'Entrance Exam' column
    df['Entrance Exam'].fillna(0, inplace=True)
    df['GWA'].fillna(0, inplace=True)
    df['Interview'].fillna(0, inplace=True)

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

    def check_passed(row):
        entrance_exam_score = row['Entrance Exam']
        interview_score = row['Interview']
        gwa_score = row['GWA']
        final_grade = entrance_exam_score * 0.50 + interview_score * 0.25 + gwa_score * 0.25

        # Check if first chosen program's final grade is within passing range
        first_chosen_course = row['First Chosen Program']
        if first_chosen_course in passing_grades:
            min_grade, max_grade = passing_grades[first_chosen_course]
            if min_grade <= final_grade <= max_grade:
                remarks = "Passed"
                return first_chosen_course, final_grade, remarks

        # Check if second chosen program's final grade is within passing range
        second_chosen_course = row['Second Chosen Program']
        if second_chosen_course in passing_grades:
            min_grade, max_grade = passing_grades[second_chosen_course]
            if min_grade <= final_grade <= max_grade:
                remarks = "Passed"
                return second_chosen_course, final_grade, remarks

        remarks = "Failed"
        return "Failed", final_grade, remarks

    # Assuming 'First Chosen Program' and 'Second Chosen Program' columns exist in your DataFrame
    df[['Course', 'RAW scores', 'Remarks']] = df.apply(check_passed, axis=1, result_type='expand')

    # Sort DataFrame by 'RAW scores' in descending order to get the ranking
    df['Rank'] = df['RAW scores'].rank(method='min', ascending=False).astype(int)

    # Save the updated dataframe back to the CSV file
    save_path = r'C:\Users\Huawie\Downloads\USER INTERFACE CART\Clean Data\FinalStudentCoursesWithRank.csv'
    df.sort_values(by='Rank', inplace=True)
    df.to_csv(save_path, index=False)

    # Filter passed students
    p_students = df[df['Remarks'] == 'Passed']
    # Filter passed students
    f_students = df[df['Remarks'] == 'Failed']


    # Sort passed students by Entrance Exam score in descending order and assign ranks
    p_path = r'C:\Users\Huawie\Downloads\USER INTERFACE CART\StudentRankwithGWA\Passed Candidates\passed_students.csv'
    p_students = p_students.sort_values(by='RAW scores', ascending=False)
    p_students['Rank'] = range(1, len(p_students) + 1)
    f_path = r'C:\Users\Huawie\Downloads\USER INTERFACE CART\StudentRankwithGWA\Failed Candidates\failed_students.csv'
  

    # Save the updated passed students data with ranks to a CSV file
    p_students.to_csv(p_path, index=False)
    f_students.to_csv(f_path, index=False)

    course_counts = df['Course'].value_counts()

@app.route('/download_cleaned_data')
def download_cleaned_data():
    cleaned_file_path = r'C:\Users\Huawie\Downloads\USER INTERFACE CART\Clean Data\FinalStudentCoursesWithRank.csv'  # Path to cleaned data file
    files_to_download = [cleaned_file_path]
    zip_file_path = r'C:\Users\Huawie\Downloads\cleaned_data.zip'  # Path to save the zip file
    
    with ZipFile(zip_file_path, 'w') as zipf:
        for file in files_to_download:
            zipf.write(file, os.path.basename(file))
    
    return send_file(zip_file_path, as_attachment=True)

@app.route('/add-student', methods=['POST'])
def add_student():
    try:
        # Get form data
        data = request.get_json()
        print("Received data:", data)  # Debugging statement

        # Define CSV file path
        csv_file = "C:/Users/Huawie/Downloads/USER INTERFACE CART/StudentRankwithGWA/syntheticdata1.csv"

        # Define fieldnames for CSV
        fieldnames = ['Name', 'University', 'Physical Fitness Test', 'Entrance Exam', 'First Chosen Program', 'Second Chosen Program']

        # Write data to CSV file
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # Check if the file is empty and write header if needed
            if file.tell() == 0:
                writer.writeheader()

            # Write data to CSV
            writer.writerow(data)

        return jsonify({'message': 'Student added successfully!'})
    except Exception as e:
        return jsonify({'error': f'Error adding student: {str(e)}'}), 500
    
# Route to render the passed applicants page
@app.route('/passed_applicants')
def passed_applicants():
    # Read the CSV file
    csv_file = "C:\\Users\\Huawie\\Downloads\\USER INTERFACE CART\\StudentRankwithGWA\\Passed Candidates\\passed_students.csv"
    data = pd.read_csv(csv_file)
    
    # Pass the CSV data to the template
    return render_template('passed.html', data=data.to_dict('records'))

@app.route('/get_csv_data')

def get_csv_data():
    csv_file = "C:/Users/Huawie/Downloads/USER INTERFACE CART/StudentRankwithGWA/Passed Candidates with GWA/Freshmen_Student_Final_Data_with_Course_and_Rank.csv"
    data = pd.read_csv(csv_file)
    return data.to_csv(index=False)

@app.route('/failed_applicants')
def failed_applicants():
    # Read the CSV file
    csv_file1 = "C:\\Users\Huawie\\Downloads\\USER INTERFACE CART\\StudentRankwithGWA\\Failed Candidates\\failed_students.csv"
    data1 = pd.read_csv(csv_file1)
    
    # Pass the CSV data to the template
    return render_template('failed.html', data1=data1.to_dict('records')) 

@app.route('/failed_applicants1')
def get_csv_data1():
    csv_file1 = "C:\\Users\Huawie\\Downloads\\USER INTERFACE CART\\StudentRankwithGWA\\Failed Candidates\\failed_students.csv"
    data1 = pd.read_csv(csv_file1)
    return data1.to_csv(index=False)

@app.route('/blank.html')
def blank():
    return send_from_directory(os.path.join(app.root_path, 'templates'), 'blank.html')
@app.route('/train.html')
def train():
    return send_from_directory(os.path.join(app.root_path, 'templates'), 'train.html')

# Define a route to serve the bar plot image
@app.route('/serve_bar_plot')
def serve_bar_plot():
    # Generate the bar plot image
    plot_file_path = generate_bar_plot('C:/Users/Huawie/Downloads/USER INTERFACE CART/StudentRankwithGWA/Passed Candidates/passed_students.csv')
    # Return the image file
    return send_file(plot_file_path, mimetype='image/png')

def generate_bar_plot(csv_file_path):
    # Read the cleaned CSV file
    df = pd.read_csv(csv_file_path)

    # Extract counts of students in each course
    course_counts = df['Course'].value_counts()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=course_counts.index, y=course_counts.values, palette='viridis')
    plt.title('Number of Students in Each Course')
    plt.xlabel('Course')
    plt.ylabel('Number of Students')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
    plt.tight_layout()

    # Save the plot as an image to the desired directory
    plot_file_path = os.path.join("C:\\Users\\Huawie\\Downloads\\USER INTERFACE CART\\graph", 'bar_plot.png')
    plt.savefig(plot_file_path)

    # Close the plot to free up memory
    plt.close()

    return plot_file_path

@app.route('/decision_tree')
def decision_tree():
    # Generate the decision tree image and evaluation metrics
    decision_tree_image_path, evaluation_metrics = generate_decision_tree()

    # Pass the decision tree image path and evaluation metrics to the template
    return render_template('decision_tree.html', decision_tree_image_path=decision_tree_image_path, evaluation_metrics=evaluation_metrics)

@app.route('/serve_decision_tree')
def serve_decision_tree():
    # Generate the decision tree image and evaluation metrics
    decision_tree_image_path, evaluation_metrics = generate_decision_tree()

    # Return the decision tree image file
    return send_file(decision_tree_image_path, mimetype='image/png')

def generate_decision_tree():
    # Load CSV data
    df = pd.read_csv('C:/Users/Huawie/Downloads/USER INTERFACE CART/Clean Data/FinalStudentCoursesWithRank.csv')

    # Check column names
    print("Column Names:", df.columns)

    # Verify if 'Course' column is present
    if 'Course' not in df.columns:
        raise ValueError("Column 'Course' not found in the DataFrame.")

    # Define the target variable 'Course'
    target = 'Course'

    # Assuming 'Course' is the target variable and other columns are features
    numeric_features = ['RAW scores']
    categorical_features = ['First Chosen Program', 'Second Chosen Program']

    # Drop rows with missing values in numeric features or target column
    df_ml = df.dropna(subset=numeric_features + [target])

    # Encode categorical variables using one-hot encoding
    df_ml = pd.get_dummies(df_ml, columns=categorical_features)

    # Convert non-numeric values in 'Interview' column to binary
    df_ml['Interview'] = df_ml['Interview'].apply(lambda x: 1 if x == 'Passed' else 0)
    # Convert non-numeric values in 'Remarks' column to binary
    df_ml['Remarks'] = df_ml['Remarks'].apply(lambda x: 1 if x == 'Passed' else 0)

    # Extract the columns to use in the model
    columns_to_use = numeric_features + [col for col in df_ml.columns if col.startswith('First Chosen Program_') or col.startswith('Second Chosen Program_')]

    X = df_ml[columns_to_use]
    y = df_ml[target]

    # Set a minimum test size of 0.1
    min_test_size = 0.1

    # Calculate the test size based on dataset size
    test_size = min(min_test_size, max(0.1, min(0.2, 1.0 - len(X) / 10)))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)

    # Fit the model on the training data
    clf.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Save decision tree visualization
    plt.figure(figsize=(40, 30))
    plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
    decision_tree_image_path = "C:/Users/Huawie/Downloads/USER INTERFACE CART/graph/decision_tree.png"
    plt.savefig(decision_tree_image_path)

    # Construct evaluation metrics dictionary
    evaluation_metrics = {'accuracy': accuracy, 'f1_score': f1, 'confusion_matrix': conf_matrix}

    return decision_tree_image_path, evaluation_metrics

@app.route('/rank_students_by_course', methods=['POST'])
def rank_students_by_course():
    try:
        # Placeholder code for passed_students and passing_grades
        passed_students = pd.read_csv('C:\\Users\\Huawie\\Downloads\\USER INTERFACE CART\\StudentRankwithGWA\\Passed Candidates\\passed_students.csv')
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
        
        # Filter passed students based on courses, sort them by Entrance Exam score, and assign ranks
        passed_students_courses = {}
        for course in passing_grades.keys():
            passed_students_courses[course] = passed_students[passed_students['Course'] == course]
            passed_students_courses[course] = passed_students_courses[course].sort_values(by='RAW scores', ascending=False)
            
            # Assign ranks considering ties
            rank = 0
            prev_score = None
            ranks = []
            for score in passed_students_courses[course]['RAW scores']:
                if score != prev_score:
                    rank += 1
                ranks.append(rank)
                prev_score = score
            
            passed_students_courses[course]['Rank'] = ranks

            # Save the updated data to a CSV file
            course_file_path = f'C:\\Users\\Huawie\\Downloads\\USER INTERFACE CART\\StudentRankwithGWA\\Student Files\\Passed\\passed_students_{course.replace(" ", "_").lower()}_with_rank.csv'
            passed_students_courses[course].to_csv(course_file_path, index=False)
            print(f"Saved passed students for {course} to '{course_file_path}' with updated ranks.")
        
        return jsonify({'message': 'Students ranked by course and saved to CSV files successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if the username and password match the ones in the CSV file
        if check_credentials(username, password):
            return redirect(url_for('successful_login'))  # Redirect to successful login page
        else:
            error_message = "Invalid username or password"
            return render_template('login.html', error=error_message)
    else:
        return render_template('login.html')

# Function to check the credentials against the CSV file
def check_credentials(username, password):
    with open(r'C:\Users\Huawie\Downloads\USER INTERFACE CART\UserPass.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == username and row[1] == password:
                return True
    return False

@app.route('/successful_login')
def successful_login():
    return render_template('blank.html')

@app.route('/create_account', methods=['POST'])
def create_account():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Save the username and password to CSV file
        with open('C:/Users/Huawie/Downloads/USER INTERFACE CART/UserPass.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([username, password])
        return redirect(url_for('index'))
    
@app.route('/rank_candidates')
def rank_candidates():
    return render_template('Ranking.html')
    
@app.route('/view_applicants')
def view_applicants():
    # Define the path to the CSV file
    csv_path = "C:\\Users\\Huawie\\Downloads\\USER INTERFACE CART\\dataset\\syntheticdata1.csv"
    
    # Read the CSV file
    csv_data = pd.read_csv(csv_path)
    
    # Pass the CSV data to the template
    return csv_data.to_csv()  # Assuming you want to return CSV data

@app.route('/clean_data1', methods=['POST'])
def clean_data1():
    # Read your CSV file
    file_path = r'C:\Users\Huawie\Downloads\USER INTERFACE CART\StudentRankwithGWA\syntheticdata1.csv'  # Replace with your file path
    df = pd.read_csv(file_path)

    # Data cleaning
    # Remove duplicate names
    df = df.drop_duplicates(subset='Name')

    # Clean 'Entrance Exam' column to accept only values between 0 and 100 and replace 'Failed' with NaN
    df['Entrance Exam'] = pd.to_numeric(df['Entrance Exam'], errors='coerce')
    df['Entrance Exam'] = df['Entrance Exam'].apply(lambda x: x if 0 <= x <= 100 else np.nan)

    # Replace NaN values with 0 in 'Entrance Exam' column
    df['Entrance Exam'].fillna(0, inplace=True)

    # Convert 'Entrance Exam' column to numeric values
    df['Entrance Exam'] = pd.to_numeric(df['Entrance Exam'], errors='coerce')

    # Define function to determine course remarks
    def determine_course_remarks(row):
        if row['Physical Fitness Test'] == 'Failed':
            return True  # Keep the row if the condition is met
        else:
            return False  # Discard the row otherwise

    # Apply the function to each row of the DataFrame
    df = df[~df.apply(determine_course_remarks, axis=1)]  # Use ~ to negate the condition

    # Convert '1' to 'Passed' and '0' to 'Failed' in 'Interview' and 'Physical Fitness Test' columns
    df['Physical Fitness Test'] = df['Physical Fitness Test'].map({'1': 'Passed', '0': 'Failed'})

    # Remove rows with values other than "1" and "0" in 'Interview' and 'Physical Fitness Test' columns
    df = df[df['Physical Fitness Test'].isin(['Passed', 'Failed'])]

    # Reorder columns
    columns_order = ['Name', 'University', 'Physical Fitness Test', 'Entrance Exam', 'First Chosen Program',
                     'Second Chosen Program']
    df = df[columns_order]

    # Save cleaned data to CSV file
    cleaned_file_path = r'C:\Users\Huawie\Downloads\USER INTERFACE CART\\Clean Data\\Clean_Data1.csv'  # Replace with your desired file path
    df.to_csv(cleaned_file_path, index=False)

    # Convert cleaned data to dictionary format
    cleaned_data = df.to_dict('records')

    # Return JSON response with success message and cleaned data
    return jsonify({'message': 'Data cleaned successfully!', 'cleaned_data': cleaned_data})

@app.route('/download_cleaned_data1')
def download_cleaned_data1():
    cleaned_file_path = r'C:\Users\Huawie\Downloads\USER INTERFACE CART\\Clean\\Data Clean\\Clean_Data1.csv'  # Path to cleaned data file
    ranked_file_path = r'C:\Users\Huawie\Downloads\USER INTERFACE CART\Data Clean\\rank_students.csv'  # Path to passed students file
    
    files_to_download = [cleaned_file_path, ranked_file_path]
    zip_file_path = r'C:\Users\Huawie\Downloads\cleaned_data.zip'  # Path to save the zip file
    
    with ZipFile(zip_file_path, 'w') as zipf:
        for file in files_to_download:
            zipf.write(file, os.path.basename(file))
    
    return send_file(zip_file_path, as_attachment=True)





@app.route('/rank_students_by_course1', methods=['POST'])
def rank_students_by_course1():
    try:
        rank_students = pd.read_csv('C:\\Users\\Huawie\\Downloads\\USER INTERFACE CART\\Clean Data\\Clean_Data1.csv')
        
        # Get unique courses from the 'First Chosen Program' column
        courses = rank_students['First Chosen Program'].unique()

        # Initialize an empty DataFrame to store all passed students
        all_passed_students = pd.DataFrame()

        # Filter passed students based on courses, sort them by Entrance Exam score, and assign ranks
        rank_students_bycourses = {}
        for course1 in courses:
            rank_students_bycourses[course1] = rank_students[rank_students['First Chosen Program'] == course1]

            # Filter out students who failed Physical Fitness
            passed_students = rank_students_bycourses[course1][rank_students_bycourses[course1]['Physical Fitness Test'] != 'Failed']

            # If there are no students who passed, continue to the next course
            if passed_students.empty:
                continue
            
            passed_students = passed_students.sort_values(by='Entrance Exam', ascending=False)

            # Assign ranks considering ties
            rank1 = 0
            prev_score1 = None
            ranks1 = []
            for score, row in passed_students.iterrows():
                if prev_score1 is None or score != prev_score1:
                    rank1 += 1
                ranks1.append(rank1)
                prev_score1 = score

            passed_students['Rank'] = ranks1

            # Save the updated data to a CSV file for passed candidates
            passed_candidates_path = f'C:\\Users\\Huawie\\Downloads\\USER INTERFACE CART\\StudentRankwithGWA\\Passed Candidates\\Passed_candidates_{course1.replace(" ", "_").lower()}.csv'
            passed_students.to_csv(passed_candidates_path, index=False)
            print(f"Saved passed students for {course1} to '{passed_candidates_path}' with updated ranks.")
            
            # Append passed students to the DataFrame storing all passed students
            all_passed_students = pd.concat([all_passed_students, passed_students])

            # Filter out failed candidates
            failed_candidates = rank_students_bycourses[course1][rank_students_bycourses[course1]['Physical Fitness Test'] == 'Failed']
            if not failed_candidates.empty:
                # Create a separate CSV file for failed candidates
                failed_candidates_path = f'C:\\Users\\Huawie\\Downloads\\USER INTERFACE CART\\StudentRankwithGWA\\Failed Candidates\\Failed_candidate_{course1.replace(" ", "_").lower()}.csv'
                failed_candidates.to_csv(failed_candidates_path, index=False)
                print(f"Saved failed candidates for {course1} to '{failed_candidates_path}'.")

        # Save all passed students in one file
        all_passed_students_path = 'C:\\Users\\Huawie\\Downloads\\USER INTERFACE CART\\StudentRankwithGWA\\All Passed Students.csv'
        all_passed_students.to_csv(all_passed_students_path, index=False)
        print(f"Saved all passed students to '{all_passed_students_path}'.")

        # Save all failed students
        all_failed_students_path = 'C:\\Users\\Huawie\\Downloads\\USER INTERFACE CART\\StudentRankwithGWA\\All Failed Students.csv'
        all_failed_students = rank_students[rank_students['Physical Fitness Test'] == 'Failed']
        all_failed_students.to_csv(all_failed_students_path, index=False)
        print(f"Saved all failed students to '{all_failed_students_path}'.")

    except Exception as e:
        print(f"An error occurred: {e}")

    return "Process completed."
@app.route('/upload')
def upload():
    return render_template('GWA.html')

@app.route('/upload_gwa_file', methods=['POST'])
def upload_gwa_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            df = pd.read_csv(file)
            data = df.to_dict(orient='records')
            return jsonify({'success': True, 'data': data})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'File upload failed'})  
    
@app.route('/interview')
def interview():
    return render_template('Interview.html')

@app.route('/upload_interview_file', methods=['POST'])
def upload_interview_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            df = pd.read_csv(file)
            data = df.to_dict(orient='records')
            return jsonify({'success': True, 'data': data})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'File upload failed'})
    
@app.route('/save_csv', methods=['POST'])
def save_csv():
    try:
        # Receive CSV data from frontend
        data = request.get_json()
        csv_content = data.get('csvContent')

        # Specify the folder where you want to save the CSV file
        save_folder = "C:/Users/Huawie/Downloads/USER INTERFACE CART/StudentRankwithGWA/Passed Candidates with GWA"

        # Save the CSV file to the specified folder
        file_path = os.path.join(save_folder, "Freshmen_Student_Final_Data_with_Course_and_Rank.csv")
        with open(file_path, 'w') as file:
            file.write(csv_content)

        return jsonify({"message": "CSV file saved successfully."}), 200
    except Exception as e:
        print("Error saving CSV file:", e)
        return jsonify({"error": "Failed to save CSV file."}), 500
    
    
@app.route('/save_csv_ready_for_interview', methods=['POST'])
def save_csv_ready_for_interview():
    try:
        # Receive CSV data from frontend
        data = request.get_json()
        csv_content = data.get('csvContent')

        # Specify the folder where you want to save the CSV file
        save_folder = "C:/Users/Huawie/Downloads/USER INTERFACE CART/StudentRankwithGWA/Passed Candidates with GWA"

        # Save the CSV file to the specified folder
        file_path = os.path.join(save_folder, "candidatesreadyforinterview.csv")
        with open(file_path, 'w') as file:
            file.write(csv_content)

        return jsonify({"message": "CSV file saved successfully."}), 200
    except Exception as e:
        print("Error saving CSV file:", e)
        return jsonify({"error": "Failed to save CSV file."}), 500
    
    # Set up logging configuration
logging.basicConfig(filename='error.log', level=logging.DEBUG)

@app.route('/perform_cross_validation', methods=['GET'])
def perform_cross_validation():
    try:
        # Placeholder code to simulate cross-validation
        # Replace this with your actual cross-validation logic
        # Load the dataset
        df = pd.read_csv('C:/Users/Huawie/Downloads/USER INTERFACE CART/Clean Data/FinalStudentCoursesWithRank.csv')

        # Convert 'Passed' into 1 and 'Failed' into 0 in 'Interview' and 'Physical Fitness Test' columns
        df['Interview'] = df['Interview'].map({'Passed': 1, 'Failed': 0})
        df['Physical Fitness Test'] = df['Physical Fitness Test'].map({'Passed': 1, 'Failed': 0})

        # Label encode categorical columns
        label_encoder = LabelEncoder()
        categorical_cols = ['University', 'First Chosen Program', 'Second Chosen Program']
        numerical_cols = ['RAW scores']

        for col in categorical_cols:
            df[col] = label_encoder.fit_transform(df[col])

        # Splitting the data into features and target
        X = df.drop(['Name', 'Course', 'Remarks'], axis=1)
        y = df['Remarks']

        # Preprocessing pipelines
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        # Model pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', DecisionTreeClassifier())])

        # Define hyperparameters for grid search
        param_grid = {
            'classifier__max_depth': [3, 5, 7, 10],
            'classifier__min_samples_split': [2, 5, 10]
        }

        # Grid search with cross-validation
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
        cv_scores = cross_val_score(grid_search, X, y, cv=5, scoring='accuracy')

        # Calculate mean cross-validation score
        mean_cv_score = cv_scores.mean()

        # Fit grid search to find best parameters and best score
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Generate classification report
        y_pred = grid_search.predict(X)
        class_rep = classification_report(y, y_pred, output_dict=True)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)

        # Prepare data for CSV
        csv_data = {
            'Cross Validation Scores': cv_scores.tolist(),
            'Mean Cross Validation Score': mean_cv_score,
            'Best Parameters': best_params,
            'Best Score': best_score,
            'Classification Report': class_rep,
            'Confusion Matrix': conf_matrix.tolist()
        }

        # Convert data to DataFrame
        csv_df = pd.DataFrame(csv_data)

        # Save DataFrame to CSV
        csv_filename = 'cross_validation_results.csv'
        csv_df.to_csv(csv_filename, index=False)

        # Return the CSV file for download
        return send_file(csv_filename, as_attachment=True)

    except Exception as e:
        error_message = str(e)
        result = {
            'success': False,
            'error': error_message
        }
        # Log the error
        logging.exception('An error occurred:')
        return jsonify(result), 500
    
if __name__ == "__main__":
    app.run(debug=True)
