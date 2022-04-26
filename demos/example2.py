import pandas

from pyfrbus.frbus import Frbus
from pyfrbus.sim_lib import sim_plot
from pyfrbus.load_data import load_data


# Load data
data = load_data("../data/LONGBASE.TXT")

# Load model
frbus = Frbus("../models/model.xml", mce="mcap+wp")

# Specify dates
start = pandas.Period("2040q1")
end = start + 60 * 4

# Standard MCE configuration, use surplus ratio targeting, rstar endogenous in long run
data.loc[start:end, "dfpdbt"] = 0
data.loc[start:end, "dfpsrp"] = 1
data.loc[start:end, "drstar"] = 0
data.loc[(start + 39) : end, "drstar"] = 1

# Solve to baseline with adds
with_adds = frbus.init_trac(start, end, data)

# 100 bp monetary policy shock and solve
with_adds.loc[start, "rffintay_aerr"] += 1
sim = frbus.solve(start, end, with_adds)

# View results
sim_plot(with_adds, sim, start, end)
