import pandas as pd
import sys

def score(reference, proposal):
    """Calculate error metric for proposal

    Args:
        reference (Dataframe): Dataframe containing reference data
        proposal (Dataframe): Dataframe containing proposal data

    Returns:
        tuple: (error, dataframe)
              mean_error is the mean of ((Reference - Proposal)^2)/Reference
              dataframe contains merged data with additional 'err' column
    """

    # Merge dataframes on common columns (country, product, month)
    merged_df = reference.merge(proposal, left_on=['Country','Product','Month'],
                          right_on=['Country','Product','Month'], how='left')
    merged_df = merged_df.rename(columns={"Quantity_x": "Reference", "Quantity_y": "Proposal"})
    # Fill any NaN values in Proposal column with 0, if there is no value is is assumed that the proposal is 0
    merged_df.fillna({'Proposal':0}, inplace=True)

    # Calculate error metric
    # ((Reference - Proposal)^2) / Reference for each row
    newRef = merged_df['Reference']
    newRef = newRef.replace(0, 1)
    merged_df['Error'] = ((merged_df['Reference'] - merged_df['Proposal'])**2)/newRef

    # Calculate mean error across all rows
    mean_error = merged_df['Error'].mean()

    # Return results along with the merged dataframe
    return mean_error, merged_df

def checkCapacity(proposal, capacity):
    """
    Check if the proposed production is within capacity

    Args:
        proposal (DataFrame) Dataframe containing the proposed production
        capacity (DataFrame) Dataframe containing the monthly capacity for each country

    Returns:
        Boolean indicating if all proposed productions are within capacity, and a merged dataframe with check result column
    """

    result = proposal.groupby(['Country', 'Month'])['Quantity'].sum().reset_index()
    result = result.groupby(['Country'])['Quantity'].max().reset_index()
    merged_df = result.merge(capacity, left_on='Country', right_on='Country', how='left')
    merged_df['valid']=(merged_df['Quantity']<=merged_df['Monthly Capacity'])
    valid = merged_df['valid'].all().item()
    return valid, merged_df

def checkShipments(production, shipments):
    """
    Check if the proposed shipments are within production

    Args:
        production (DataFrame) Dataframe containing the proposed production
        shipments (DataFrame) Dataframe containing the shipments
    Returns:
        Boolean indicating if all proposed shipments are within production, and a merged dataframe with check result column
    """
    totalShipments = shipments.groupby(['Origin', 'Product', 'Month'])['Quantity'].sum().reset_index()
    totalProduction = production.groupby(['Country', 'Product', 'Month'])['Quantity'].sum().reset_index()
    merged_df = totalShipments.merge(totalProduction, left_on=['Origin', 'Product', 'Month'], right_on=['Country', 'Product', 'Month'], how='left')
    merged_df.drop(columns=['Origin'], inplace=True)
    merged_df.rename(columns={"Quantity_x": "Shipped", "Quantity_y": "Produced"}, inplace=True)
    merged_df['valid']=(merged_df['Shipped']<=merged_df['Produced'])
    valid = merged_df['valid'].all().item()
    merged_df = merged_df[['Country','Month', 'Product', 'Shipped','Produced','valid']]
    return valid, merged_df


def computeAvailability(production, shipments):
    """
    Compute the availability of each country based on the local production and shipments

    Args:
        production (DataFrame) Dataframe containing the production of each country
        shipments (DataFrame) Dataframe containing the shipments
    Returns:
        DataFrame with availability of each country

    """
    totalInboundShipments = shipments.groupby(['Destination','Product', 'Month'])['Quantity'].sum().reset_index()
    totalOutboundShipments = shipments.groupby(['Origin','Product', 'Month'])['Quantity'].sum().reset_index()
    availability = production.merge(totalInboundShipments, left_on=['Country','Product','Month'], right_on=['Destination', 'Product', 'Month'], how='outer')
    availability['Country'] = availability['Country'].mask(pd.isna, availability['Destination'])
    availability = availability.merge(totalOutboundShipments, left_on=['Country','Product','Month'], right_on=['Origin', 'Product', 'Month'], how='left')
    availability.fillna({'Quantity_x':0}, inplace=True)
    availability.fillna({'Quantity_y':0}, inplace=True)
    availability.fillna({'Quantity':0}, inplace=True)
    availability['AvailableQuantity']= availability['Quantity_x'] + availability['Quantity_y'] - availability['Quantity']
    availability.fillna({'AvailableQuantity':0}, inplace=True)
    result = availability[['Country','Product','Month','AvailableQuantity']]
    result = result.rename(columns={"AvailableQuantity": "Quantity"})
    return result

def computeCost(production, shipments, productionCost, shipmentsCost):
    """
    Compute the total cost based on the production and shipments
    Args:
        production (DataFrame) Dataframe containing the production of each country
        shipments (DataFrame) Dataframe containing the shipments
    Returns:
        number with value of total cost
    """
    totalProduction = production.groupby(['Country','Product'])['Quantity'].sum().reset_index()
    costs = totalProduction.merge(productionCost, left_on=['Country','Product'], right_on=['Country','Product'], how='left')
    costs['Cost']= costs['Quantity'] * costs['Unit Cost']
    totalProductionCost = costs['Cost'].sum()
    
    totalShipments = shipments.groupby(['Origin','Destination'])['Quantity'].sum().reset_index()
    s_costs = totalShipments.merge(shipmentsCost, left_on=['Origin','Destination'], right_on=['Origin','Destination'], how='left')
    s_costs['Cost'] = s_costs['Quantity'] * s_costs['Unit Cost']
    totalShipmentsCost= s_costs['Cost'].sum()
    
    return totalProductionCost + totalShipmentsCost

# Read data from command line arguments
df1 = pd.read_csv(sys.argv[2])
df2 = pd.read_csv( sys.argv[3])

# Execute score function with command line arguments
if(sys.argv[1]=="prediction"):
    s, df = score(df1, df2)
    # Print output results
    print(f"Evaluation: {s}")
    if (len(sys.argv)>4 and sys.argv[4] == "verbose"):
        print("Data:")
        print(df)
elif(sys.argv[1]=="balance"):
    proposal = pd.read_csv( sys.argv[4])
    shipments = pd.read_csv( sys.argv[5])
    check, dfv = checkCapacity(proposal, df2)
    checkS, dfs = checkShipments(proposal, shipments)
    availability = computeAvailability(proposal, shipments)
    s, dfA = score(df1, availability)
    if not check or not checkS:
        s = float("inf")
    print(f"Evaluation: {s}")
    if (len(sys.argv)>6 and sys.argv[6] == "verbose"):
        print(f"Valid capacity: {check}")
        print(f"Valid shipments: {checkS}")
        print(dfv)
        print(dfs)
        print(dfA)
else:
    productionCost = pd.read_csv( sys.argv[4])
    shipmentsCost = pd.read_csv( sys.argv[5])
    proposal = pd.read_csv( sys.argv[6])
    shipments = pd.read_csv( sys.argv[7])
    check, dfv = checkCapacity(proposal, df2)
    checkS, dfs = checkShipments(proposal, shipments)
    availability = computeAvailability(proposal, shipments)
    s, dfA = score(df1, availability)
    if not check or not checkS:
        s = float("inf")
    cost = computeCost(proposal, shipments, productionCost, shipmentsCost)
    print(f"Evaluation: {s}, {cost}")
    if (len(sys.argv)>8 and sys.argv[8] == "verbose"):
        print(f"Valid capacity: {check}")
        print(f"Valid shipments: {checkS}")
        print(dfv)
        print(dfs)
        print(dfA)

