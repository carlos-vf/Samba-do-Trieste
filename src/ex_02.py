from pyomo.environ import *
import pandas as pd
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CAPACITY_PATH = SCRIPT_DIR.parent / "data" / "02_input_capacity.csv"
INPUT_TARGET_PATH = SCRIPT_DIR.parent / "data" / "02_input_target.csv"
PRODUCTION_PLAN_PATH = SCRIPT_DIR.parent / "output" / "02_output_productionPlan_8475.csv"
SHIPMENT_PLAN_PATH = SCRIPT_DIR.parent / "output" / "02_output_shipments_8475.csv"

# Create the model
model = ConcreteModel()

capacity = pd.read_csv(CAPACITY_PATH)
capacity_data = dict(zip(capacity["Country"], capacity["Monthly Capacity"]))
demand = pd.read_csv(INPUT_TARGET_PATH)
demand_data = {
    (row['Country'], row['Month'], row['Product']): row['Quantity']
    for _, row in demand.iterrows()
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

# Constraint: Capacity limit per nation per month 
def capacity_constraint_rule(model, n, m,p):
    total_out = sum(model.SHIP[n, n2, m, p] for n2 in model.N if n2 != n)
    return total_out <= (model.F[n,m,p])
model.CapacityConstraint = Constraint(model.N, model.M, model.P,rule=capacity_constraint_rule)

# Constraint: Satisfied demand cannot exceed actual demand
def demand_satisfaction_rule(model, n, m, p):
    if (n, m, p) in model.Demand:
         return model.X[n, m, p] <= model.Demand[n, m, p]
    else:
        return model.X[n, m, p] == 0 
model.DemandSatisfactionConstraint = Constraint(model.N, model.M, model.P, rule=demand_satisfaction_rule)

def objective_rule(model):
    return sum(model.Demand[n,m,p] - model.X[n, m, p] for n in model.N for m in model.M for p in model.P)
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
# Create a list to store rows for the CSV
def create_csv():
    csv_data = []

    for index, value in model.F.items():
        country, month, product = index
        quantity = value(value)  
        
        csv_data.append([country, product, month, quantity])

    csv_file_path = PRODUCTION_PLAN_PATH

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(['Country', 'Product', 'Month', 'Quantity'])
        
        writer.writerows(csv_data)

    print(f"CSV file '{csv_file_path}' has been created.")

    #############################################################################

    # Create a list to store rows for the CSV
    csv_data = []

    for index, value in model.SHIP.items():
        origin, destination, month, product = index
        
        quantity = value(value) if value(value) is not None else 0  # Assign 0 if no value exists for the variable

        csv_data.append([origin, destination, product, month, quantity])

    csv_file_path = SHIPMENT_PLAN_PATH

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(['Origin', 'Destination', 'Product', 'Month', 'Quantity'])
        
        writer.writerows(csv_data)

    print(f"CSV file '{csv_file_path}' has been created.")

create_csv()
