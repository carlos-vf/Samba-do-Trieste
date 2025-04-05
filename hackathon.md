### Regression Model

**Objective**: Balancing production and transportation costs

-  Try to predict the monthly amount of production of a product in a certain State
    **Inputs**:
    - Amount required from a certain state
    - Production capacity of facilities (monthly constraints)
        - Analyze situations where market demand varies relative to production capacity
    **Constraints**:
    - Do not produce ahead in previous months
    - Cannot deliver late
    **Strategy**:
    - Produce in other facilities with lower workload, and then transport to the state where there is demand
