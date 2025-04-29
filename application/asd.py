from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value

# Initialize the problem
prob = LpProblem("Study_Hours_Optimization", LpMaximize)

# Subjects
subjects = ['Calculus', 'Chemistry', 'Physics', 'Economics']

# Hours ranges and corresponding grades
grade_data = {
    'Calculus': {1: 75, 6: 84, 11: 93},
    'Chemistry': {1: 76, 6: 87, 11: 94},
    'Physics': {1: 65, 6: 81, 11: 91},
    'Economics': {1: 85, 6: 92, 11: 97}
}

# Create variables for hours and binary range indicators
hours_vars = {}
range_vars = {}

for subject in subjects:
    # Hours variable (integer)
    hours_vars[subject] = LpVariable(f"{subject}_hours", lowBound=1, upBound=15, cat='Integer')
    
    # Binary variables for each range (1-5, 6-10, 11-15)
    range_vars[subject] = {
        '1-5': LpVariable(f"{subject}_1_5", cat='Binary'),
        '6-10': LpVariable(f"{subject}_6_10", cat='Binary'),
        '11-15': LpVariable(f"{subject}_11_15", cat='Binary')
    }

# Objective function: Maximize GPA
grades = []
for subject in subjects:
    grades.append(
        grade_data[subject][1] * range_vars[subject]['1-5'] +
        grade_data[subject][6] * range_vars[subject]['6-10'] +
        grade_data[subject][11] * range_vars[subject]['11-15']
    )
prob += lpSum(grades)
 # Average GPA

# Constraints
for subject in subjects:
    # Only one range can be active per subject
    prob += (
        range_vars[subject]['1-5'] + 
        range_vars[subject]['6-10'] + 
        range_vars[subject]['11-15'] == 1
    )
    
    # Link hours to ranges using big-M constraints
    hours = hours_vars[subject]
    prob += hours >= 1 * range_vars[subject]['1-5'] + 6 * range_vars[subject]['6-10'] + 11 * range_vars[subject]['11-15']
    prob += hours <= 5 * range_vars[subject]['1-5'] + 10 * range_vars[subject]['6-10'] + 15 * range_vars[subject]['11-15']

# Total hours constraint
prob += lpSum([hours_vars[subject] for subject in subjects]) <= 35

# Subject-specific constraints
prob += hours_vars['Physics'] >= 2 * hours_vars['Economics']  # P ≥ 2E
prob += hours_vars['Economics'] <= hours_vars['Calculus']      # E ≤ C
prob += hours_vars['Calculus'] <= hours_vars['Physics']       # C ≤ P

# Solve the problem
prob.solve()

# Print results
print("Optimal Study Hours:")
for subject in subjects:
    print(f"{subject}: {int(value(hours_vars[subject]))} hours (Grade: {grade_data[subject][value(range_vars[subject]['1-5'])*1 + value(range_vars[subject]['6-10'])*6 + value(range_vars[subject]['11-15'])*11]})")

print(f"\nTotal Hours: {sum(int(value(hours_vars[subject])) for subject in subjects)}")
print(f"Maximized GPA: {value(prob.objective):.2f}")