# connect_DB.py

def getEmployee(name):
    # Implement the logic to retrieve employee information from the database based on the given name
    # For example, you might connect to the database and execute a query to get the employee information
    # Replace the following line with your actual implementation
    employee_data = {
        "Le Phuong Thu": {"name": "Thu", "fullName": "Le Phuong Thu", "position": "Manager"},
        "Le Ngoc Linh": {"name": "Linh", "fullName": "Le Ngoc Linh", "position": "Employee"},
        # Add more employee information as needed
    }

    if name in employee_data:
        return employee_data[name]
    else:
        return None
