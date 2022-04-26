import pandas

from pyfrbus.frbus import Frbus
from pyfrbus.sim_lib import sim_plot
from pyfrbus.load_data import load_data
from pyfrbus.time_series_data import TimeSeriesData


# Load data
data = load_data("../data/LONGBASE.TXT")

# Load model
frbus = Frbus("../models/model.xml")

# Specify dates
start = pandas.Period("2040Q1")
end = start + 24

# Standard configuration, use surplus ratio targeting
data.loc[start:end, "dfpdbt"] = 0
data.loc[start:end, "dfpsrp"] = 1

# Use non-inertial Taylor rule
data.loc[start:end, "dmptay"] = 1
data.loc[start:end, "dmpintay"] = 0

# Enable thresholds
data.loc[start:end, "dmptrsh"] = 1
# Arbitrary threshold values
data.loc[start:end, "lurtrsh"] = 6.0
data.loc[start:end, "pitrsh"] = 3.0

# Solve to baseline with adds
with_adds = frbus.init_trac(start, end, data)

# Zero tracking residuals for funds rate and thresholds
with_adds.loc[start:end, "rfftay_trac"] = 0
with_adds.loc[start:end, "rffrule_trac"] = 0
with_adds.loc[start:end, "rff_trac"] = 0
with_adds.loc[start:end, "dmptpi_trac"] = 0
with_adds.loc[start:end, "dmptlur_trac"] = 0
with_adds.loc[start:end, "dmptmax_trac"] = 0
with_adds.loc[start:end, "dmptr_trac"] = 0

# Shocks vaguely derived from historical residuals
with_adds.loc[start : start + 3, "eco_aerr"] = [-0.002, -0.0016, -0.0070, -0.0045]
with_adds.loc[start : start + 3, "ecd_aerr"] = [-0.0319, -0.0154, -0.0412, -0.0838]
with_adds.loc[start : start + 3, "eh_aerr"] = [-0.0512, -0.0501, -0.0124, -0.0723]
with_adds.loc[start : start + 3, "rbbbp_aerr"] = [0.3999, 2.7032, 0.3391, -0.7759]
with_adds.loc[start : start + 8, "lhp_aerr"] = [
    -0.0029,
    -0.0048,
    -0.0119,
    -0.0085,
    -0.0074,
    -0.0061,
    -0.0077,
    -0.0033,
    -0.0042,
]

# Set up time-series object
d = TimeSeriesData(with_adds)
# Set range
d.range = pandas.period_range(start + 4, end)
# Roll off residuals with 0.5 persistence
rho = 0.5
d.eco_aerr = rho * d.eco_aerr(-1)
d.ecd_aerr = rho * d.ecd_aerr(-1)
d.eh_aerr = rho * d.eh_aerr(-1)
d.rbbbp_aerr = rho * d.rbbbp_aerr(-1)
d.range = pandas.period_range(start + 9, end)
d.lhp_aerr = rho * d.lhp_aerr(-1)

# Adds so that thresholds do not trigger before shocks are felt
with_adds.loc[start, "dmptr_aerr"] = -1
with_adds.loc[start : start + 2, "dmptlur_aerr"] = -1

# Solve
sim = frbus.solve(start, end, with_adds)

# View results, unemployment threshold binds
sim_plot(with_adds, sim, start, end)
