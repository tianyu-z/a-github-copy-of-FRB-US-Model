import pandas
from numpy import array, cumprod

from pyfrbus.frbus import Frbus
from pyfrbus.sim_lib import sim_plot
from pyfrbus.load_data import load_data


# Load data
data = load_data("../data/LONGBASE.TXT")

# Load model
frbus = Frbus("../models/model.xml")

# Specify dates
start = pandas.Period("2021Q3")
end = "2022Q3"

# Standard configuration, use surplus ratio targeting
data.loc[start:end, "dfpdbt"] = 0
data.loc[start:end, "dfpsrp"] = 1

# Solve to baseline with adds
with_adds = frbus.init_trac(start, end, data)

# Scenario based on 2021Q3 Survey of Professional Forecasters
with_adds.loc[start:end, "lurnat"] = 3.78

# Set up trajectories for mcontrol
with_adds.loc[start:end, "lur_t"] = [5.3, 4.9, 4.6, 4.4, 4.2]
with_adds.loc[start:end, "picxfe_t"] = [3.7, 2.2, 2.1, 2.1, 2.2]
with_adds.loc[start:end, "rff_t"] = [0.1, 0.1, 0.1, 0.1, 0.1]
with_adds.loc[start:end, "rg10_t"] = [1.4, 1.6, 1.6, 1.7, 1.9]

# Get GDP level as accumulated growth from initial period
gdp_growth = cumprod((array([6.8, 5.2, 4.5, 3.4, 2.7]) / 100 + 1) ** 0.25)
with_adds.loc[start:end, "xgdp_t"] = with_adds.loc[start - 1, "xgdp"] * gdp_growth

targ = ["xgdp", "lur", "picxfe", "rff", "rg10"]
traj = ["xgdp_t", "lur_t", "picxfe_t", "rff_t", "rg10_t"]
inst = ["eco_aerr", "lhp_aerr", "picxfe_aerr", "rff_aerr", "rg10p_aerr"]

# Run mcontrol
sim = frbus.mcontrol(start, end, with_adds, targ, traj, inst)

# View results
sim_plot(with_adds, sim, start, end)
