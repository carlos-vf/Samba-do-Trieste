from pyomo.environ import *
import pandas as pd
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CAPACITY_PATH = SCRIPT_DIR.parent / "data" / "02_input_capacity.csv"
INPUT_TARGET_PATH = SCRIPT_DIR.parent / "data" / "02_input_target.csv"
PRODUCTION_PLAN_PATH = SCRIPT_DIR.parent / "output" / "03_output_productionPlan_8475.csv"
SHIPMENT_PLAN_PATH = SCRIPT_DIR.parent / "output" / "03_output_shipments_8475.csv"
SHIPMENT_COST_PATH = SCRIPT_DIR.parent / "data" / "02_03_input_shipmentsCost_example.csv"
PRODUCTION_COST_PATH = SCRIPT_DIR.parent / "data" / "03_input_productionCost.csv"

M_PENALTY = 100

# Create the model
model = ConcreteModel()

capacity = pd.read_csv(CAPACITY_PATH)
capacity_data = dict(zip(capacity["Country"], capacity["Monthly Capacity"]))
demand = pd.read_csv(INPUT_TARGET_PATH)
demand_data = {
    (row['Country'], row['Month'], row['Product']): row['Quantity']
    for _, row in demand.iterrows()
}
prodcost = pd.read_csv(PRODUCTION_COST_PATH)
prodcost_data = {
    (row['Country'], row['Product']): row['Unit Cost']
    for _, row in prodcost.iterrows()
}
prodtransp = pd.read_csv(SHIPMENT_COST_PATH)
prodtransp_data = {
    (row['Origin'], row['Destination']): row['Unit Cost']
    for _, row in prodtransp.iterrows()
}

nations = sorted(capacity['Country'].unique())
months = sorted(demand['Month'].unique())
products = sorted(demand['Product'].unique())

# Sets
model.N = Set(initialize=nations)  # Nations
model.M = Set(initialize=months)   # Months
model.P = Set(initialize=products) # Products

# Parameters
model.Capacity = Param(model.N, initialize=capacity_data) 
model.Demand = Param(model.N, model.M, model.P, initialize=demand_data)
model.ProdCost = Param(model.N,model.P,initialize= prodcost_data)
model.TranspCost = Param(model.N,model.N, initialize = prodtransp_data, default = 0)

# Variables
model.X = Var(model.N, model.M, model.P, domain=NonNegativeIntegers)  # Demand satisfaction
model.F = Var(model.N, model.M, model.P, domain = NonNegativeIntegers) # Factory production 
model.SHIP = Var(model.N, model.N, model.M, model.P, domain=NonNegativeIntegers)  # Shipment

# Constraint: factory production
def production_constraint_rule(model, n, m):
    return sum(model.F[n,m,p] for p in model.P) <= model.Capacity[n]
model.ProductionConstraint = Constraint(model.N,model.M ,rule=production_constraint_rule)

# Constraint: Available is given by capacity + in - out
def availability_constraint_rule(model, n, m, p):
    return model.X[n, m, p] == model.F[n,m,p]+sum(model.SHIP[n2,n,m,p] for n2 in model.N if n2!=n)-sum(model.SHIP[n,n2,m,p] for n2 in model.N if n2!=n)
model.AvailableConstraint = Constraint(model.N, model.M, model.P, rule=availability_constraint_rule)

# Constraint: Capacity exports 
def capacity_constraint_rule(model, n, m,p):
    total_out = sum(model.SHIP[n, n2, m, p] for n2 in model.N if n2 != n)
    return total_out <= (model.F[n,m,p]+sum(model.SHIP[n2,n,m,p] for n2 in model.N if n2!=n))
model.CapacityConstraint = Constraint(model.N, model.M, model.P,rule=capacity_constraint_rule)

# Shipment constraint
def shipment_constraint_rule(model, n, n2):
    if model.TranspCost[n,n2]==0:
        return sum(model.SHIP[n,n2, m, p] for m in model.M for p in model.P) == 0
    return Constraint.Skip
model.ShipmentConstraint = Constraint(model.N, model.N, rule= shipment_constraint_rule)

# Constraint: Satisfied demand cannot exceed actual demand
def demand_satisfaction_rule(model, n, m, p):
    if (n, m, p) in model.Demand:
        return model.X[n, m, p] <= model.Demand[n, m, p]
    else:
        return model.X[n, m, p] == 0 
model.DemandSatisfactionConstraint = Constraint(model.N, model.M, model.P, rule=demand_satisfaction_rule)

def objective_rule(model):
    return M_PENALTY * sum(model.Demand[n,m,p] - model.X[n, m, p] for n in model.N for m in model.M for p in model.P) + sum(model.ProdCost[n,p]*model.F[n,m,p] for n in model.N for m in model.M for p in model.P)+sum(model.TranspCost[n,n2]*model.SHIP[n,n2,m,p] for m in model.M for p in model.P  for n in model.N for n2 in model.N if n!=n2)
model.Obj = Objective(rule=objective_rule, sense=minimize)

solver = SolverFactory('glpk')
results = solver.solve(model, tee=False)  # 'tee=True' shows solver output

# Optional: check if it's optimal
from pyomo.opt import TerminationCondition
if results.solver.termination_condition == TerminationCondition.optimal:
    print("Solver found an optimal solution.")
else:
    print(f"Solver status: {results.solver.termination_condition}")

# Postprocess
def pyomo_postprocess(model, results):
    model.F.display()
    model.SHIP.display()

#pyomo_postprocess(model, results)

################################################################################
# Assuming model.X is your decision variable
# Create a list to store rows for the CSV
def create_csv():
    csv_data = []

    # Loop through the model.X values (for indexed variables)
    for index, value in model.F.items():
        # index is a tuple like ('Italy', 'Nov2004', 'SoothingSerenity Baby Oil')
        country, month, product = index
        quantity = value(value)  # The quantity is the value of the decision variable
        
        # Append formatted row to the csv_data list
        csv_data.append([country, product, month, quantity])

    # Define the CSV file path
    csv_file_path = PRODUCTION_PLAN_PATH

    # Write the data to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row
        writer.writerow(['Country', 'Product', 'Month', 'Quantity'])
        
        # Write the data rows
        writer.writerows(csv_data)

    print(f"CSV file '{csv_file_path}' has been created.")

    #############################################################################

    # Assuming model.X is your decision variable
    # Create a list to store rows for the CSV
    csv_data = []

    # Loop through the model.X values (for indexed variables)
    for index, value in model.SHIP.items():
        # index is a tuple like ('United Kingdom', 'United Kingdom', 'Sep2004', 'SunShield SPF 50 Lotion')
        origin, destination, month, product = index
        
        # Here, you may get 'None' values for some attributes if the model didn't assign a value
        quantity = value(value) if value(value) is not None else 0  # Assign 0 if no value exists for the variable

        # Append formatted row to the csv_data list
        csv_data.append([origin, destination, product, month, quantity])

    # Define the CSV file path
    csv_file_path = SHIPMENT_PLAN_PATH

    # Write the data to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row
        writer.writerow(['Origin', 'Destination', 'Product', 'Month', 'Quantity'])
        
        # Write the data rows
        writer.writerows(csv_data)

    print(f"CSV file '{csv_file_path}' has been created.")

create_csv()