from io import StringIO
import pandas as pd
import psycopg2

def db_connection(): 
    return psycopg2.connect(
        host = 'pg-3998d03d-helplive-bf40.e.aivencloud.com', 
        port = '20643', 
        database = 'defaultdb', 
        user = 'avnadmin', 
        password = 'AVNS_Nq9tap9LFaBnhE-8vxe'
    )

def execute_query(query, fetch, params=()):
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            if fetch == "one":
                return cursor.fetchone()
            elif fetch == "all":
                return cursor.fetchall()
            elif fetch == "none":
                return None
            
def filter(pgrm_code, sbj_year, sem_name, sbj_class, sbj_status):
    filter = []
    if sem_name: 
        filter.append(f"sem_name = '{sem_name.strip()}'")
    if sbj_class: 
        filter.append(f"sbj_class = '{sbj_class.strip()}'")
    if pgrm_code:
        filter.append(f"stu.pgrm_code = '{pgrm_code}'")
    if sbj_year != 'None' and sbj_year != '':
        filter.append(f"sbj_year = '{sbj_year}'")
    if sbj_status: 
        filter.append(f"sbj_status = '{sbj_status}'")
    return filter 
            
def stu_login(stu_id, password): 
    query = ''' 
        SELECT stu_id, intake_year
        FROM student 
        WHERE stu_id = %s 
        AND intake_year = %s;
    '''
    return execute_query(query, 'one', (stu_id, password))

def pl_login(pl_id, password): 
    query = '''
        SELECT pl_id, pgrm_code
        FROM program  
        WHERE pl_id = %s 
        AND pgrm_code = %s;
    '''
    return execute_query(query, 'one', (pl_id, password))

def get_stu_progress(stu_id): 
    query = '''
        SELECT spr.stu_id, intake_year, intake_sem, gender, nationality, curr_year, 
        bm_result, pgrm_code, stu_grade, sbj_status, upcoming_sem, sbj_code
        FROM student stu
        JOIN student_progress_report spr ON stu.stu_id = spr.stu_id
        WHERE stu.stu_id = %s;
    '''

    return execute_query(query, 'one', (stu_id, ))

def stacked_chart(pgrm_code, sbj_year, sem_name, sbj_class, sbj_status): 
    query = '''
        SELECT spr.sbj_code, upcoming_sem, 
               COUNT(DISTINCT spr.stu_id) AS ttl_stu
        FROM student_progress_report spr
        JOIN subject sbj ON sbj.sbj_code = spr.sbj_code
        JOIN student stu ON stu.stu_id = spr.stu_id
        JOIN semester sem ON sem.sem_id = sbj.sem_id
    '''

    query += 'WHERE ' + ' AND '.join(filter(pgrm_code, sbj_year, sem_name, sbj_class, sbj_status))
    query += """AND sbj_status IN ('PENDING', 'NOT ENROL') 
                GROUP BY spr.sbj_code, upcoming_sem"""

    return execute_query(query, 'all')

def get_all_progress(pgrm_code, sbj_year, sem_name, sbj_class, sbj_status): 
    query = '''
        SELECT spr.stu_id, spr.sbj_code, sbj_status, upcoming_sem
        FROM student_progress_report spr
        JOIN student stu ON stu.stu_id = spr.stu_id
        JOIN subject sbj ON sbj.sbj_code = spr.sbj_code
        JOIN program pgrm ON pgrm.pgrm_code = stu.pgrm_code
        JOIN semester sem ON sem.sem_id = sbj.sem_id
    '''

    query += 'WHERE ' + ' AND '.join(filter(pgrm_code, sbj_year, sem_name, sbj_class, sbj_status))

    return execute_query(query, 'all')

def ttl_credit_hours(stu_id): 
    query = '''
        SELECT SUM(credit_hours)
        FROM subject sbj 
        JOIN student_progress_report spr ON spr.sbj_code = sbj.sbj_code
        JOIN student stu ON stu.stu_id = spr.stu_id
        WHERE sbj_status = 'COMPLETE'
        AND spr.stu_id = %s
    '''

    return execute_query(query, 'one', (stu_id,))

def get_pending_sbj(stu_id, pgrm_code, sbj_year, sem_name, sbj_class, sbj_status, sbj_code, stu_grade): 
    query = '''
        SELECT spr.sbj_code, sbj_name, upcoming_sem
        FROM student_progress_report spr
        JOIN student stu ON stu.stu_id = spr.stu_id
        JOIN subject sbj ON sbj.sbj_code = spr.sbj_code
        JOIN semester sem ON sem.sem_id = sbj.sem_id
    '''

    filter_conditions = []
    params = []

    if sbj_class:
        filter_conditions.append("sbj_class = %s")
        params.append(sbj_class)

    if sem_name:
        filter_conditions.append("sem_name = %s")
        params.append(sem_name)

    if pgrm_code:
        filter_conditions.append("stu.pgrm_code = %s")
        params.append(pgrm_code)

    if sbj_year and sbj_year != "None":
        filter_conditions.append("sbj_year = %s")
        params.append(sbj_year)

    if sbj_status:
        filter_conditions.append("sbj_status = %s")
        params.append(sbj_status)

    if stu_id:
        filter_conditions.append("spr.stu_id = %s")
        params.append(stu_id)

    if sbj_code:
        extracted_codes = [f"{code}%" for code in sbj_code]  # Convert to wildcard format
        filter_conditions.append("spr.sbj_code LIKE ANY(%s)")  # Expecting an array
        params.append(extracted_codes)  # Pass as a Python list
            
    if stu_grade:
        filter_conditions.append("stu_grade = %s")
        params.append(stu_grade)

    if filter_conditions:
        query += " WHERE " + " AND ".join(filter_conditions)

    return execute_query(query, 'all', tuple(params))  # Ensure params is a tuple

def get_upcoming_sbj(username, sbj_year, sbj_code): 
    query = '''
        SELECT spr.sbj_code, sbj_name, upcoming_sem 
        FROM student_progress_report spr
        JOIN subject sbj ON sbj.sbj_code = spr.sbj_code
        WHERE sbj_status = 'NOT ENROL' 
        AND sbj_status <> 'NOT REQUIRE'
        AND stu_id = %s
        AND (%s IS NULL OR spr.sbj_code LIKE ANY(%s))
        AND (%s IS NULL OR sbj_year = %s)
    ''' 

    return execute_query(query, 'all', (username, sbj_code, sbj_code, sbj_year, sbj_year))

def enrolled_and_completed(stu_id, sbj_status, sbj_year, sbj_code): 
    query = '''
        SELECT spr.sbj_code, sbj_name, sbj_class, credit_hours, stu_grade
        FROM student_progress_report spr
        JOIN subject sbj ON sbj.sbj_code = spr.sbj_code
        WHERE stu_id = %s 
        AND sbj_status = %s
        AND (%s IS NULL OR spr.sbj_code LIKE ANY(%s))
        AND (%s IS NULL OR sbj_year = %s)
    '''

    if sbj_code:
        extracted_codes = [f"{code}%" for code in sbj_code]  # Convert to wildcard format
    else: 
        extracted_codes = None

    return execute_query(query, 'all', (stu_id, sbj_status, extracted_codes, extracted_codes, sbj_year, sbj_year))

def get_ttl_stu(pgrm_code, sbj_year, sem_name, sbj_class, sbj_status): 
    query = '''
        SELECT COUNT(DISTINCT spr.stu_id)
        FROM student stu
        JOIN student_progress_report spr ON spr.stu_id = stu.stu_id
        JOIN subject sbj ON sbj.sbj_code = spr.sbj_code
        JOIN semester sem ON sem.sem_id = sbj.sem_id 
    '''
    query += 'WHERE ' + ' AND '.join(filter(pgrm_code, sbj_year, sem_name, sbj_class, sbj_status))

    return execute_query(query, 'one')


def get_pending_grade(pgrm_code, sbj_year, sem_name, sbj_class, sbj_status): 
    query = '''
        SELECT spr.stu_id, spr.sbj_code, stu_grade, upcoming_sem 
        FROM student_progress_report spr
        JOIN student stu ON stu.stu_id = spr.stu_id
        JOIN subject sbj ON sbj.sbj_code = spr.sbj_code
        JOIN semester sem ON sem.sem_id = sbj.sem_id
    '''

    query += 'WHERE ' + ' AND '.join(filter(pgrm_code, sbj_year, sem_name, sbj_class, sbj_status))
    query += """
        ORDER BY
            CASE 
                WHEN stu_grade = 'FL' THEN 1
                WHEN stu_grade = 'AA' THEN 2
                WHEN stu_grade = 'AE' THEN 3
            ELSE 4 
        END
    """

    return execute_query(query, 'all')

# To display the upcoming sem in the monitor student progress table 
def get_upcoming_sem(stu_id, sbj_code): 
    query = '''
        SELECT upcoming_sem
        FROM student_progress_report
        WHERE stu_id = %s
        AND sbj_code = %s
    '''

    return execute_query(query, 'one', (stu_id, sbj_code))

def get_uq_year(pgrm_code): 
    query = '''
        SELECT DISTINCT sbj_year
        FROM subject sbj
        JOIN student_progress_report spr ON spr.sbj_code = sbj.sbj_code
        JOIN student stu ON stu.stu_id = spr.stu_id
        WHERE pgrm_code = %s
        ORDER BY sbj_year
    '''

    return execute_query(query, 'all', (pgrm_code, ))

def get_uq_sem(): 
    query = '''
        SELECT DISTINCT sem_name
        FROM semester  
        ORDER BY sem_name
    '''

    return execute_query(query, 'all')

def get_uq_class(pgrm_code): 
    query = '''
        SELECT sbj_class 
        FROM (
            SELECT DISTINCT sbj_class 
            FROM subject sbj
            JOIN student_progress_report spr ON spr.sbj_code = sbj.sbj_code
            JOIN student stu ON stu.stu_id = spr.stu_id
            WHERE pgrm_code = %s
        ) AS subquery
        ORDER BY LENGTH(sbj_class)
    '''

    return execute_query(query, 'all', (pgrm_code, ))

def get_uq_status():
    query = '''
        SELECT DISTINCT sbj_status
        FROM student_progress_report
        ORDER BY sbj_status 
    '''

    return execute_query(query, 'all')

def get_uq_sbj(): 
    query = '''
        SELECT DISTINCT sbj.sbj_code, sbj.sbj_name
        FROM subject sbj
        ORDER BY sbj.sbj_code;
    '''

    return execute_query(query, 'all')

def get_uq_cat(pgrm_code): 
    query = '''
        SELECT sbj_cat 
        FROM (
            SELECT DISTINCT sbj_cat 
            FROM subject sbj
            JOIN student_progress_report spr ON spr.sbj_code = sbj.sbj_code
            JOIN student stu ON stu.stu_id = spr.stu_id
            WHERE pgrm_code = %s
        ) AS subquery
        ORDER BY LENGTH(sbj_cat)
    '''

    return execute_query(query, 'all', (pgrm_code, ))

def get_study_plan(stu_id):
    query = '''
        SELECT sbj_year || '(' || EXTRACT(YEAR FROM supposed_to_be_taken) || ')', 
               sem_name, spr.sbj_code, sbj_name, sbj_class, credit_hours
        FROM subject sbj
        JOIN student_progress_report spr ON spr.sbj_code = sbj.sbj_code
        JOIN student stu ON stu.stu_id = spr.stu_id
        JOIN semester sem ON sem.sem_id = sbj.sem_id
        WHERE spr.stu_id = %s
        AND sbj_status <> 'NOT REQUIRE'
        ORDER BY sbj_year,
        CASE 
            -- If intake is MAY, order semesters as MAY → AUG → JAN
            WHEN intake_sem = 'MAY' AND sem_name = 'MAY' THEN 1
            WHEN intake_sem = 'MAY' AND sem_name = 'AUG' THEN 2
            WHEN intake_sem = 'MAY' AND sem_name = 'JAN' THEN 3

            -- If intake is JAN, order semesters as JAN → MAY → AUG
            WHEN intake_sem = 'JAN' AND sem_name = 'JAN' THEN 1
            WHEN intake_sem = 'JAN' AND sem_name = 'MAY' THEN 2
            WHEN intake_sem = 'JAN' AND sem_name = 'AUG' THEN 3

            -- If intake is AUG, order semesters as AUG → JAN → MAY
            WHEN intake_sem = 'AUG' AND sem_name = 'AUG' THEN 1
            WHEN intake_sem = 'AUG' AND sem_name = 'JAN' THEN 2
            WHEN intake_sem = 'AUG' AND sem_name = 'MAY' THEN 3
        END;
    '''
    
    return execute_query(query, 'all', (stu_id, ))


def get_ori_plan(stu_id):
    query = '''
        SELECT sbj_year, sem_name, spr.sbj_code, sbj_name, sbj_class, 
               credit_hours, supposed_to_be_taken
        FROM subject sbj
        JOIN student_progress_report spr ON spr.sbj_code = sbj.sbj_code
        JOIN student stu ON stu.stu_id = spr.stu_id
        JOIN semester sem ON sem.sem_id = sbj.sem_id
        WHERE spr.stu_id = %s
        AND sbj_status <> 'NOT REQUIRE'
        ORDER BY sbj_year, EXTRACT(YEAR FROM supposed_to_be_taken),
        CASE 
            -- If intake is MAY, order semesters as MAY → AUG → JAN
            WHEN intake_sem = 'MAY' AND sem_name = 'MAY' THEN 1
            WHEN intake_sem = 'MAY' AND sem_name = 'AUG' THEN 2
            WHEN intake_sem = 'MAY' AND sem_name = 'JAN' THEN 3

            -- If intake is JAN, order semesters as JAN → MAY → AUG
            WHEN intake_sem = 'JAN' AND sem_name = 'JAN' THEN 1
            WHEN intake_sem = 'JAN' AND sem_name = 'MAY' THEN 2
            WHEN intake_sem = 'JAN' AND sem_name = 'AUG' THEN 3

            -- If intake is AUG, order semesters as AUG → JAN → MAY
            WHEN intake_sem = 'AUG' AND sem_name = 'AUG' THEN 1
            WHEN intake_sem = 'AUG' AND sem_name = 'JAN' THEN 2
            WHEN intake_sem = 'AUG' AND sem_name = 'MAY' THEN 3
        END;
    '''

    return execute_query(query, 'all', (stu_id, ))


def get_reshuffle_plan(stu_id): 
    query = '''
        SELECT EXTRACT(YEAR FROM upcoming_sem), sem_name, spr.sbj_code, sbj_name, sbj_class, 
               credit_hours, supposed_to_be_taken, stu_grade, upcoming_sem, sbj_status
        FROM subject sbj
        JOIN student_progress_report spr ON spr.sbj_code = sbj.sbj_code
        JOIN student stu ON stu.stu_id = spr.stu_id
        JOIN semester sem ON sem.sem_id = sbj.sem_id
        WHERE spr.stu_id = %s
        AND sbj_status <> 'NOT REQUIRE'
        ORDER BY EXTRACT(YEAR FROM upcoming_sem),
        CASE 
            -- If intake is MAY, order semesters as MAY → AUG → JAN
            WHEN intake_sem = 'MAY' AND sem_name = 'MAY' THEN 1
            WHEN intake_sem = 'MAY' AND sem_name = 'AUG' THEN 2
            WHEN intake_sem = 'MAY' AND sem_name = 'JAN' THEN 3

            -- If intake is JAN, order semesters as JAN → MAY → AUG
            WHEN intake_sem = 'JAN' AND sem_name = 'JAN' THEN 1
            WHEN intake_sem = 'JAN' AND sem_name = 'MAY' THEN 2
            WHEN intake_sem = 'JAN' AND sem_name = 'AUG' THEN 3

            -- If intake is AUG, order semesters as AUG → JAN → MAY
            WHEN intake_sem = 'AUG' AND sem_name = 'AUG' THEN 1
            WHEN intake_sem = 'AUG' AND sem_name = 'JAN' THEN 2
            WHEN intake_sem = 'AUG' AND sem_name = 'MAY' THEN 3
        END, 
        CASE 
            WHEN sbj_status = 'PENDING' THEN 1
            WHEN sbj_status = 'NOT ENROL' THEN 2
            WHEN sbj_status = 'ENROL' THEN 3
            ELSE 4
        END;
    '''
    
    return execute_query(query, 'all', (stu_id, ))

# Get records to extract part of the data for machine learning demonstration 
def get_all_records(): 
    query = '''
        SELECT 
            spr.stu_id, intake_sem, intake_year, gender, nationality, curr_year, bm_result, 
            stu.pgrm_code, pgrm_name, pgrm_duration, pgrm.pl_id, pl_name, spr.sbj_code, sbj_cat, 
            sbj_name, sbj_class, credit_hours, sbj_year, pre_req, sbj.sem_id, sem_name, 
            sem_duration, stu_grade, sbj_status, TO_CHAR(upcoming_sem, 'MON-YY') AS upcoming_sem 
        FROM student_progress_report spr
        JOIN student stu ON stu.stu_id = spr.stu_id
        JOIN subject sbj ON sbj.sbj_code = spr.sbj_code
        JOIN semester sem ON sem.sem_id = sbj.sem_id
        JOIN program pgrm ON pgrm.pgrm_code = stu.pgrm_code 
        JOIN program_leader pl ON pl.pl_id = pgrm.pl_id
        WHERE spr.stu_id BETWEEN 'B000001' AND 'B000082'
        OR spr.stu_id = 'B000121';
    '''

    return execute_query(query, 'all')

def supposed_to_enrolFL(stu_id): 
    query = '''
        SELECT sbj_code FROM student_progress_report
        WHERE upcoming_sem = '2025-01-01'
        AND sbj_status = 'PENDING'
        AND sbj_status <> 'ENROL'
        AND sbj_status <> 'NOT REQUIRE'
        AND stu_id = %s
    '''

    return execute_query(query, 'all', (stu_id, ))

def supposed_to_enrol(stu_id): 
    query = '''
        SELECT sbj_code FROM student_progress_report
        WHERE upcoming_sem = '2025-01-01'
        AND sbj_status <> 'PENDING'
        AND sbj_status <> 'ENROL'
        AND sbj_status <> 'NOT REQUIRE'
        AND stu_id = %s
    '''

    return execute_query(query, 'all', (stu_id, ))

def upload_table_with_headers(dataframe, table_name, primary_key, second_key=None):
    conn = db_connection()
    cursor = conn.cursor()

    # Process each row for update or insert
    for _, row in dataframe.iterrows():
        new_sbj_code = row[primary_key]  # New subject code
        # Construct WHERE condition based on keys
        where_clause = f"{primary_key} = %s"
        values = [row[primary_key]]

        if second_key:
            where_clause += f" AND {second_key} = %s"
            values.append(row[second_key])

        # Check if the record exists
        cursor.execute(f"SELECT {primary_key} FROM {table_name} WHERE {where_clause}", tuple(values))
        old_record = cursor.fetchone()

        if old_record:
            old_sbj_code = old_record[0]

            # If subject code has changed, update all references
            if old_sbj_code != new_sbj_code:
                print(f"Updating subject code {old_sbj_code} -> {new_sbj_code}")

                # List of tables that reference subject code
                related_tables = ["table1", "table2"]  # Replace with actual table names

                for rel_table in related_tables:
                    cursor.execute(f"UPDATE {rel_table} SET {primary_key} = %s WHERE {primary_key} = %s", (new_sbj_code, old_sbj_code))

            # Update existing record
            update_query = f"""
                UPDATE {table_name} SET 
                {', '.join([f"{col} = %s" for col in dataframe.columns if col not in [primary_key, second_key]])}
                WHERE {where_clause}
            """
            update_values = [row[col] for col in dataframe.columns if col not in [primary_key, second_key]] + values
            cursor.execute(update_query, update_values)
            print(f"Successfully updated {new_sbj_code}")
        else:
            # Insert new record
            print(f"Successfully insert {row[primary_key]}")
            buffer = StringIO()
            dataframe.loc[[_]].to_csv(buffer, index=False, header=False)  # Append only this row
            buffer.seek(0)
            cursor.copy_expert(f"COPY {table_name} FROM STDIN WITH CSV", buffer)

    conn.commit()
    cursor.close()
    conn.close()
    print(f"\nUploaded {table_name} successfully with updates for existing records and inserts for new ones!")

def manageSubject(dataframe, table_name, action, old_subject_codes = None):
    conn = db_connection()
    cursor = conn.cursor()

    primary_key = 'sbj_code'

    for _, row in dataframe.iterrows():
        if action == "deleted":
            # Delete the subject
            subject_code = row[primary_key]
            cursor.execute(f"DELETE FROM subject WHERE sbj_code = %s", (subject_code,))
            print(f"Successfully deleted {subject_code}")
        
        elif action == "updated":
            subject_code = row[primary_key]
            old_subject_code = old_subject_codes  # Old subject code passed in the request
            
            # Prepare update query dynamically based on available fields
            update_fields = []
            update_values = []
            for column, value in row.items():
                if pd.notna(value) and value != "":  # Only update non-empty values
                    update_fields.append(f"{column} = %s")
                    update_values.append(value)
            
            if update_fields:
                update_query = f"""
                    UPDATE {table_name}
                    SET {', '.join(update_fields)}
                    WHERE sbj_code = %s
                """
                update_values.append(old_subject_code)
                cursor.execute(update_query, update_values)
                print(f"Successfully updated {old_subject_code} to {subject_code if 'sbj_code' in row and row['sbj_code'] else old_subject_code}")
                    
        elif action == "added":
            subject_code = row[primary_key]
            # Insert new subject
            buffer = StringIO()
            dataframe.loc[[_]].to_csv(buffer, index=False, header=False)
            buffer.seek(0)
            cursor.copy_expert(f"COPY {table_name} FROM STDIN WITH CSV", buffer)
            print(f"Successfully added {subject_code}")

    conn.commit()
    cursor.close()
    conn.close()


    