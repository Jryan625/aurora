# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 11:48:14 2024

This script takes energy grid pricing data from three markets and optimises
the charging/discharging of a battery on the grid to maximise profit and 
cater for battery degradation.

@author: jimji
"""
# Import libraries
import os
import pandas as pd
from pulp import LpVariable, LpProblem, LpMaximize, lpSum, value


# Data dir
datadir = os.path.join('C:\\', 'Users', 'jimji', 'Code', 'data')

# Filename
file = 'Second Round Technical Question - Attachment 2.xlsx'

# Create dataframe for half hourly data
half_hour_data = pd.read_excel(os.path.join(
    datadir, 
    file),
    sheet_name='Half-hourly data',
    converters={
        '': lambda x: pd.to_datetime(x)
    })

# Create dataframe for daily data
daily_data = pd.read_excel(os.path.join(
    datadir, 
    file),
    sheet_name='Daily data',
    converters={
        '': lambda x: pd.to_datetime(x)
    })

# Check for nulls
assert not half_hour_data.isnull().values.any(), "Null values in half-hour data"
assert not daily_data.isnull().values.any(), "Null values in daily data"

# Constants
max_charging_rate = 2  # MW
max_discharging_rate = 2  # MW
max_storage_volume = 4  # MWh
initial_storage = 2  # MWh (assuming battery has 50% initial charge)
efficiency = 0.95  # 95% efficiency
max_life_years = 10  # Years
max_life_cycles = 5000  # Cycles
storage_loss_per_cycle = 0.00001  # 0.001%
capex = 500000  # £ cost of purchasing and installing the battery
foc = 5000  # £ fixed operational costs
nhalf_hours = half_hour_data.shape[0]  # number of half-hour periods
ndays = daily_data.shape[0]  # number of days

# Define variables
power_charge_mk1 = LpVariable.dicts("power_charge_mk1", range(nhalf_hours), lowBound=0)
power_charge_mk2 = LpVariable.dicts("power_charge_mk2", range(nhalf_hours), lowBound=0)
power_charge_mk3 = LpVariable.dicts("power_charge_mk3", range(ndays), lowBound=0)
power_discharge_mk1 = LpVariable.dicts("power_discharge_mk1", range(nhalf_hours), lowBound=0)
power_discharge_mk2 = LpVariable.dicts("power_discharge_mk2", range(nhalf_hours), lowBound=0)
power_discharge_mk3 = LpVariable.dicts("power_discharge_mk3", range(ndays), lowBound=0)
soc = LpVariable.dicts("soc", range(nhalf_hours), lowBound=0, upBound=max_storage_volume)
cycles = LpVariable.dicts("cycles", range(nhalf_hours), lowBound=0, upBound=max_life_cycles)

# Auxiliary variables to track charging and discharging
charge_amount = LpVariable.dicts("charge_amount", range(nhalf_hours), lowBound=0)
discharge_amount = LpVariable.dicts("discharge_amount", range(nhalf_hours), lowBound=0)

# Create optimisation model
model = LpProblem("Optimise_Battery", LpMaximize)

# Define objective function, which is to maximise profit
revenue = lpSum(
    half_hour_data['Market 1 Price [£/MWh]'][t] * power_discharge_mk1[t] +
    half_hour_data['Market 2 Price [£/MWh]'][t] * power_discharge_mk2[t] +
    (daily_data['Market 3 Price [£/MWh]'][t // 48] * power_discharge_mk3[t // 48] )
    for t in range(nhalf_hours)
)
cost = lpSum(
    half_hour_data['Market 1 Price [£/MWh]'][t] * power_charge_mk1[t] +
    half_hour_data['Market 2 Price [£/MWh]'][t] * power_charge_mk2[t] +
    (daily_data['Market 3 Price [£/MWh]'][t // 48] * power_charge_mk3[t // 48] )
    for t in range(nhalf_hours)
)
model += revenue - cost - capex - (3 * foc)

# Define constraints

model += soc[0] == initial_storage  # Initial charge 
model += cycles[0] == 0  # Initial cycles

for t in range(1, nhalf_hours):
    
    # Loop increment for daily data 
    day = t // 48

    # Net charge and discharge for the current time step
    sum_of_charge = power_charge_mk1[t-1] + power_charge_mk2[t-1] + (power_charge_mk3[day])
    sum_of_discharge = power_discharge_mk1[t-1] + power_discharge_mk2[t-1] + (power_discharge_mk3[day])
    
    # Track charging and discharging amounts
    model += charge_amount[t] == sum_of_charge
    model += discharge_amount[t] == sum_of_discharge
    
    # State of charge
    model += soc[t] == soc[t-1] + (sum_of_charge * efficiency - sum_of_discharge / efficiency)
    
    # Update cycles
    model += cycles[t] == cycles[t-1] + (charge_amount[t] + discharge_amount[t]) / (2 * max_storage_volume)
    
for t in range(nhalf_hours):
    
    # Loop increment for daily data
    day = t // 48

    # Sum of charging from all three markets must not exceed max charging rate of battery    
    model += power_charge_mk1[t] + power_charge_mk2[t] + power_charge_mk3[day] <= max_charging_rate
    
    # Sum of discharging to all three markets must not exceed max discharging rate of battery
    model += power_discharge_mk1[t] + power_discharge_mk2[t] + power_discharge_mk3[day] <= max_discharging_rate
    
    # Number of cycles must not exceed the maximum cycles of the battery's lifetime
    model += cycles[t] <= max_life_cycles
    
    # State of charge must not exceed the max storage volume minus the accumulated storage loss per cycle
    model += soc[t] <= max_storage_volume * (1 - storage_loss_per_cycle * cycles[t])


# Solve model
model.solve()

# Extract results
results = {
    'time': [], 
    'power_charge_mk1': [], 
    'power_charge_mk2': [], 
    'power_charge_mk3': [], 
    'power_discharge_mk1': [], 
    'power_discharge_mk2': [], 
    'power_discharge_mk3': [], 
    'soc': [], 
    'cycles': [], 
    'profit': []
}
total_profit = 0

for t in range(nhalf_hours):
    day = t // 48
    results['time'].append(t)
    results['power_charge_mk1'].append(value(power_charge_mk1[t]))
    results['power_charge_mk2'].append(value(power_charge_mk2[t]))
    results['power_charge_mk3'].append(value(power_charge_mk3[day]) )
    results['power_discharge_mk1'].append(value(power_discharge_mk1[t]))
    results['power_discharge_mk2'].append(value(power_discharge_mk2[t]))
    results['power_discharge_mk3'].append(value(power_discharge_mk3[day]) )
    results['soc'].append(value(soc[t]))
    results['cycles'].append(value(cycles[t]))
    if value(power_discharge_mk1[t]) is not None:
        total_profit += half_hour_data['Market 1 Price [£/MWh]'][t] * value(power_discharge_mk1[t]) - half_hour_data['Market 1 Price [£/MWh]'][t] * value(power_charge_mk1[t])
    if value(power_discharge_mk2[t]) is not None:
        total_profit += half_hour_data['Market 2 Price [£/MWh]'][t] * value(power_discharge_mk2[t]) - half_hour_data['Market 2 Price [£/MWh]'][t] * value(power_charge_mk2[t])
    if value(power_discharge_mk3[day]) is not None:
        total_profit += daily_data['Market 3 Price [£/MWh]'][day] * value(power_discharge_mk3[day]) - daily_data['Market 3 Price [£/MWh]'][day] * value(power_charge_mk3[day])
    results['profit'].append(total_profit - capex / max_life_years - foc)

# Create dataframe from results
results_df = pd.DataFrame(results)
results_df['Datetime'] = half_hour_data['Unnamed: 0']
results_df.index = pd.to_datetime(results_df[results_df.columns[10]])
results_df =results_df.drop(['time', 'Datetime'], axis=1)

# Save results to Excel
results_df.to_excel(os.path.join(datadir, 'battery_optimization_results_final.xlsx'), index=True)

# Print total profit
print("Total Profit: £{:.2f}".format(total_profit))
