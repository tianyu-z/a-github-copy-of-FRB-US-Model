import sys
sys.path.append("../")

from pyfrbus.frbus import Frbus
from pyfrbus.sim_lib import stochsim_plot
from pyfrbus.load_data import load_data


# Load data
data = load_data("../data/LONGBASE.TXT")

# Load model
frbus = Frbus("../models/model.xml")

# Specify dates and other params
residstart = "1975q1"
residend = "2018q4"
simstart = "2040q1"
simend = "2045q4"
# Number of replications
nrepl = 1000
# Run up to 5 extra replications, in case of failures
nextra = 5

# Policy settings
data.loc[simstart:simend, "dfpdbt"] = 0
data.loc[simstart:simend, "dfpsrp"] = 1

# Compute add factors
# Both for baseline tracking and over history, to be used as shocks
with_adds = frbus.init_trac(residstart, simend, data)

# Call FRBUS stochsim procedure
solutions = frbus.stochsim(
    nrepl, with_adds, simstart, simend, residstart, residend, nextra=nextra
)

stochsim_plot(with_adds, solutions, simstart, simend)
