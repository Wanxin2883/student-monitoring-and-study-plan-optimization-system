from flask import Flask, jsonify, render_template, redirect, send_file, send_from_directory, url_for, request
from io import BytesIO
import pandas as pd
import db, ml 
import secrets
import os

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# When the user click on the website link
@app.route('/')
def login_page(): 
    return render_template('login.html')

# When the user click on the login button
@app.route('/login', methods = ['POST'])
def validateUser(): 
    username = request.form['username']
    password = request.form['password']

    if username.startswith('P'): 
        validate_pl = db.pl_login(username, password)
        if validate_pl: 
            return redirect(url_for('plDashboard', username = username))
        
    elif username.startswith('B'):
        validate_stu = db.stu_login(username, password)
        if validate_stu: 
            return redirect(url_for('stuDashboard', username = username))
        
    return render_template('login.html', error_msg = 'Invalid Username or Password!')

def format_date(date_str):
    # Split the string into year, month, and day
    year, month, day = date_str.split('-')
    
    # Only process JAN, MAY, AUG (i.e., months 01, 05, 08)
    if month == '01':
        month_abbr = 'JAN'
    elif month == '05':
        month_abbr = 'MAY'
    elif month == '08':
        month_abbr = 'AUG'
    else:
        month_abbr = 'Unknown'  # Handle months other than JAN, MAY, and AUG
    
    # Return the formatted string as "MON-YYYY"
    return f"{month_abbr} {year}"

# Get graduation year 
def get_graduation_year(username): 
    upcoming_sbj = db.get_upcoming_sbj(username, None, None)
    # Assuming upcoming_sbj is a list of subjects where each subject is a list with [name, description, date]
    furthest_date = None

    for subject in upcoming_sbj:
        current_date = subject[2]  # Assuming it's a string in "YYYY-MM-DD" format

        # Track the furthest date by comparing the strings lexicographically
        if furthest_date is None or current_date > furthest_date:
            furthest_date = current_date 

    return format_date(str(furthest_date))

# When program leader login successfully 
@app.route('/plDash/<username>', methods = ['GET', 'POST'])
def plDashboard(username): 

    if username == 'PL0001': 
        pgrm_code = 'HUBDA'
    else: 
        pgrm_code = 'HUDT'

    # Get the Unique Value in each features (for user selection)    
    sbj_year = [col[0] for col in db.get_uq_year(pgrm_code)]
    sem_name = [col[0] for col in db.get_uq_sem()]
    sbj_class = [col[0] for col in db.get_uq_class(pgrm_code)]
    sbj_status = [col[0] for col in db.get_uq_status()]

    # Set the initial value for the filters (All is subject filter)
    yFilter = request.form.get('yFilter', 1)
    semFilter = request.form.get('semFilter', 'JAN')
    cFilter = request.form.get('cFilter', 'CORE COMPUTING')
    statusFilter = request.form.get('statusFilter', None)

    # If there is None value in the chart data, replace it with empty string. 
    def replace_none(obj): 
        if isinstance(obj, dict): 
            return {k: replace_none(v) for k, v in obj.items()}
        elif isinstance(obj, list): 
            return [replace_none(v) for v in obj]
        elif obj is None: 
            return ""
        return obj

    # Extract chart data for visualization
    extracted_data_dict = {}
    extracted_data = db.stacked_chart(pgrm_code, yFilter, semFilter, cFilter, statusFilter)

    for col in extracted_data: 
        sbj_code = col[0]
        upcoming = str(col[1])
        total_students = col[2]

        if sbj_code not in extracted_data_dict:
            extracted_data_dict[sbj_code] = {}
        extracted_data_dict[sbj_code][upcoming] = total_students

    chart_data = replace_none(extracted_data_dict)

    # Extract the pending subject details 
    add_assess = db.get_pending_sbj(None, pgrm_code, yFilter, semFilter, cFilter, statusFilter, None, 'AA')
    add_exam = db.get_pending_sbj(None, pgrm_code, yFilter, semFilter, cFilter, statusFilter, None, 'AE')
    sbj_failed = db.get_pending_sbj(None, pgrm_code, yFilter, semFilter, cFilter, statusFilter, None, 'FL')

    # Calculate the total number of students after applied filters 
    ttl_students = db.get_ttl_stu(pgrm_code, yFilter, semFilter, cFilter, statusFilter)

    # Extract data for table beside chart 
    pending_grade = db.get_pending_grade(pgrm_code, yFilter, semFilter, cFilter, statusFilter)

    return render_template('plDash.html', pgrm_code = pgrm_code, sem_name = sem_name, sbj_status = sbj_status, sbj_year = sbj_year, 
                           sbj_class = sbj_class, selected_year = yFilter, selected_class = cFilter, selected_sem = semFilter, 
                           selected_status = statusFilter, chart_data = chart_data, add_assess = add_assess, add_exam = add_exam, 
                           pending_grade = pending_grade, sbj_failed = sbj_failed, ttl_students = ttl_students, username = username)

# When program leader wants to update student progress after a semester have been completed
@app.route('/updateProgress')
def updateProgress():
    return render_template('updateProgress.html')

# When program leader click the download button to view template to update student progress
@app.route('/downloadTemplate')
def downloadTemplate():
    return send_from_directory("uploads", "update_progress_template.csv", as_attachment=True)

# When program leader click the upload button to update student progress
@app.route('/upload', methods = ['POST'])
def uploadCSV(): 

    file = request.files['csvFile']
    filepath = os.path.join('uploads', file.filename)
    
    # Save the uploaded file in this filepath 
    file.save(filepath)

    # Process the uploaded file to predict the upcoming semester 
    new_df = ml.predict_upcoming_sem(filepath)
    
    filepath = os.path.join('uploads', 'help.csv')

    student_info = new_df[["stu_id", "intake_sem", "intake_year", "gender", "nationality", "curr_year", "bm_result", "pgrm_code"]]
    db.upload_table_with_headers(student_info, "student", 'stu_id')

    spr_info = new_df[["stu_grade", "sbj_status", "supposed_to_be_taken", "upcoming_sem", "sbj_code", "stu_id"]]
    db.upload_table_with_headers(spr_info, 'student_progress_report', 'stu_id', 'sbj_code')

    return jsonify({"message": "File uploaded. Processing started."}), 200

# When program leader wants to update student progress after a semester have been completed
@app.route('/updateSubject/<username>', methods=['GET'])
def updateSubject(username):

    if username == 'PL0001': 
        pgrm_code = 'HUBDA'
    else: 
        pgrm_code = 'HUDT'

    uq_sbj = db.get_uq_sbj()

    # Define the patterns to check
    patterns = ('DIP', 'MPU2', 'HGA1', 'GEN2')

    # Apply filtering based on program code
    if pgrm_code == "HUBDA":
        # Remove subjects that start with any of the patterns
        filtered_sbj = [sbj for sbj in uq_sbj if not sbj[0].startswith(patterns)]
    else:
        # Keep only subjects that start with the patterns
        filtered_sbj = [sbj for sbj in uq_sbj if sbj[0].startswith(patterns)]

    uq_class = db.get_uq_class(pgrm_code)
    uq_year = db.get_uq_year(pgrm_code)
    uq_sem = db.get_uq_sem()
    uq_cat = db.get_uq_cat(pgrm_code)

    return render_template('updateSubject.html', uq_sbj = filtered_sbj, uq_class = uq_class, 
                           uq_year = uq_year, uq_sem = uq_sem, uq_cat = uq_cat)

# After the program leader click submit button to update subject details 
@app.route('/manageSubject', methods=['POST'])
def manageSubject():
    data = request.form.to_dict()
    print("Received data:", data)  # Debugging

    action = request.form.get("action")  # Get action dynamically
    
    sem_name = request.form.get('semester_offered')
    if sem_name == 'AUG': 
        sem_id = 'SEM0003'
    elif sem_name == 'JAN': 
        sem_id = 'SEM0001'
    else: 
        sem_id = 'SEM0002'

    old_subject_codes = request.form.get('old_subject_code')

    subject_data = {
        'sbj_code': request.form['subject_code'],
        'sbj_name': request.form.get('subject_name'),
        'sbj_class': request.form.get('subject_class'),
        'credit_hours': request.form.get('credit_hours'),
        'sbj_year': request.form.get('subject_year'),
        'pre_req': request.form.get('pre_requisite'),
        'sem_id': sem_id,
        'sbj_cat': request.form.get('subject_category')
    }
    
    # Convert to DataFrame for processing
    df = pd.DataFrame([subject_data])
    
    # Call the function to process the request
    db.manageSubject(df, 'subject', action, old_subject_codes)
    
    return jsonify({"message": f"Successfully {action} {subject_data['sbj_code']}"}), 200

# When the program leader wants to view the student progress in overall 
@app.route('/downloadProgress/<username>', methods=['GET'])
def downloadProgress(username):

    if username == 'PL0001': 
        pgrm_code = 'HUBDA'
    else: 
        pgrm_code = 'HUDT'

    # Set filter values
    yFilter = request.form.get('yFilter', 1)
    semFilter = request.form.get('semFilter', 'JAN')
    cFilter = request.form.get('cFilter', 'CORE COMPUTING')
    statusFilter = request.form.get('statusFilter', None)

    # Extract student progress data
    stu_progress = db.get_all_progress(pgrm_code, yFilter, semFilter, cFilter, statusFilter,)

    # Convert data to list of dictionaries
    stu_progress_ls = []
    for col in stu_progress:
        stu_id, sbj_code, sbj_status, upcoming_sem = col

        # Use the tick mark for COMPLETE
        if sbj_status == 'COMPLETE':
            sbj_status = '✔'  # Tick symbol
        elif sbj_status == 'PENDING' and upcoming_sem:
            sbj_status = upcoming_sem  # Replace with upcoming semester date

        stu_progress_ls.append({
            'stu_id': stu_id, 
            'sbj_code': sbj_code, 
            'sbj_status': sbj_status
        })

    # Convert to DataFrame
    df = pd.DataFrame(stu_progress_ls)

    # Create pivot table (Student ID as index, Subjects as columns)
    pivot_table = df.pivot(index='stu_id', columns='sbj_code', values='sbj_status')

    # Reset index to include Student ID as a column
    pivot_table.reset_index(inplace=True)

    # Convert DataFrame to CSV using BytesIO
    output = BytesIO()
    pivot_table.to_csv(output, index=False, encoding='utf-8-sig')  # Fix encoding issue
    output.seek(0)

    return send_file(
        output,
        mimetype = "text/csv",
        as_attachment = True,
        download_name = "student_progress_pivot.csv"
    )

# When program leader wants to view the student progress in details 
@app.route('/stuDetails/<stu_id>/<username>', methods = ['GET'])
def stuDetails(stu_id, username):
    student_id = request.args.get('id') or stu_id  # Use given student ID or fallback to stu_id
    stu_details = db.get_stu_progress(student_id)

    if not stu_details: 
        return 'Student Not Found'

    ttl_hours = db.ttl_credit_hours(student_id)

    categories = [
        ("upcoming", "get_upcoming_sbj"),
        ("enrolled", "enrolled_and_completed", 'ENROL'),
        ("completed", "enrolled_and_completed", 'COMPLETE'),
        ("add_assess", "get_pending_sbj", "AA"),
        ("add_exam", "get_pending_sbj", "AE"),
        ("failed", "get_pending_sbj", "FL")
    ]

    years = [1, 2, 3, None]
    
    student_data = {}

    for category, method, *extra in categories: 
        for year in years: 
            key = f"y{year if year else 'mpu'}_{category}"
            args = [stu_id]
            # Define subject code filters based on username
            subject_filters = {
                "PL0001": ['BIT%', 'BDA%', 'FEC%'],  # PL0001: BIT, BDA, FEC
                "PL0002": ['DIP%'],  # PL0002: DIP only
            }

            # Determine the subject filter based on username
            if username in subject_filters:
                sbj_patterns = subject_filters[username]  # Get specific subject codes
            else:
                sbj_patterns = []  # Default to empty if username is not recognized

            # If the query is for MPU subjects, use MPU filters
            if year is None:
                sbj_patterns = ['HGA%', 'MPU%', 'GEN%']  # MPU subjects are common for all

            if method == "enrolled_and_completed":
                args.append(extra[0])  # Append 'ENROL' or 'COMPLETE'
            args.extend([year, sbj_patterns])
            if method == "get_pending_sbj":
                args = [stu_id, None, year, None, None, None, sbj_patterns, extra[0]]
            
            student_data[key] = getattr(db, method)(*args)

    grad_year = get_graduation_year(student_id)
            
    # Render the template with student data
    return render_template('stuDetails.html', **{
        "stu_id": stu_details[0], "intake_year": stu_details[1], "intake_sem": stu_details[2],
        "gender": stu_details[3], "nationality": stu_details[4], "curr_year": stu_details[5],
        "bm_result": stu_details[6], "pgrm_code": stu_details[7], "stu_grade": stu_details[8],
        "sbj_status": stu_details[9], "upcoming_sem": stu_details[10], "sbj_code": stu_details[11],
        "grad_year": grad_year, "ttl_hours": ttl_hours, "username": username, **student_data
    })

# When student login successfully, they will be directed to the dashboard
@app.route('/stuDash/<username>')
def stuDashboard(username): 

    columns = [
        "stu_id", "intake_year", "intake_sem", "gender", "nationality", "curr_year",
        "bm_result", "pgrm_code", "stu_grade", "sbj_status", "upcoming_sem", "sbj_code"]

    # Convert fetched data to a list of dictionaries
    stu_details_dict = dict(zip(columns, db.get_stu_progress(username))) if db.get_stu_progress(username) else None

    # Get the total credit hour by calculating the credit hours of completed subjects 
    ttl_credit_hours = db.ttl_credit_hours(username)
    
    # Get graduation year
    grad_year = get_graduation_year(username)

    # Get supposed to enrol subject (Notification)
    supposed_to_enrolFL = db.supposed_to_enrolFL(username)
    supposed_to_enrol = db.supposed_to_enrol(username)

    # Get subjects with different grades
    enrol_sbj = db.enrolled_and_completed(username, 'ENROL', None, None)
    completed_sbj = db.enrolled_and_completed(username, 'COMPLETE', None, None)
    upcoming_sbj = db.get_upcoming_sbj(username, None, None)
    add_assess = db.get_pending_sbj(username, None, None, None, None, None, None, 'AA')
    add_exam = db.get_pending_sbj(username, None, None, None, None, None, None, 'AE')
    sbj_failed = db.get_pending_sbj(username, None, None, None, None, None, None, 'FL')

    # Pass the fetched data to display in student dashboard
    return render_template('stuDash.html', stu = stu_details_dict, ttl_hours = ttl_credit_hours, grad_year = grad_year,
                           enrol_sbj = enrol_sbj, completed_sbj = completed_sbj, upcoming_sbj = upcoming_sbj, 
                           add_assess = add_assess, add_exam = add_exam, sbj_failed = sbj_failed, 
                           supposed_to_enrol = supposed_to_enrol, supposed_to_enrolFL = supposed_to_enrolFL)

# When student want to download their original study plan 
@app.route('/downloadStudyPlan/<stu_id>', methods=['GET'])
def downloadStudyPlan(stu_id): 
    # Get the original study plan 
    study_plan = db.get_study_plan(stu_id)

    # Define column headers
    headers = ["sbj_year", "sem_name", "sbj_code", "sbj_name", "sbj_class", "credit_hours"]

    # Convert to DataFrame
    df = pd.DataFrame(study_plan, columns=headers)

    # Merge duplicate values in "sbj_year" and "sem_name"
    for col in ["sbj_year", "sem_name"]:
        df[col] = df[col].where(df[col] != df[col].shift(), "")

    # Convert DataFrame to CSV
    output = BytesIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    output.seek(0)

    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="study_plan.csv"
    )

# When student want to download updated study plan generated based on their grades
@app.route('/downloadNewPlan/<stu_id>', methods=['GET'])
def downloadNewPlan(stu_id): 
    # Get the updated study plan 
    study_plan = db.get_reshuffle_plan(stu_id)

    # Define column headers
    headers = ["year", "sem_name", "spr.sbj_code", "sbj_name", "sbj_class", "credit_hours", 
               "supposed_to_be_taken", "stu_grade", "upcoming_sem", "sbj_status"]

    # Convert to DataFrame with headers
    df = pd.DataFrame(study_plan, columns = headers)

    df.drop(columns=["supposed_to_be_taken", "upcoming_sem", "sbj_status"], inplace=True)

    # Merge duplicate values by replacing consecutive duplicates with an empty string
    for col in ["year", "sem_name"]:  # Adjust columns as needed
        df[col] = df[col].where(df[col] != df[col].shift(), "")

    # Convert DataFrame to CSV using BytesIO
    output = BytesIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    output.seek(0)

    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="new_plan.csv"
    )

# When student wants to reshuffle their study plan 
@app.route('/studyPlan/<stu_id>', methods = ['GET'])
def studyPlan(stu_id): 

    study_plan = db.get_ori_plan(stu_id)
    study_plan = [list(row[:6]) + [str(row[6])] for row in study_plan]

    return render_template('studyPlan.html', study_plan = study_plan, stu_id = stu_id)

# When student wants to reshuffle their study plan 
@app.route('/reshufflePlan/<stu_id>', methods = ['GET'])
def reshufflePlan(stu_id): 

    study_plan = db.get_reshuffle_plan(stu_id)
    study_plan = [list(row[:6]) + [str(row[6])] + [row[7]] + [str(row[8])] + [row[9]] for row in study_plan]

    return render_template('reshufflePlan.html', study_plan = study_plan, stu_id = stu_id)

if __name__ == '__main__': 
    app.run(debug = True)




