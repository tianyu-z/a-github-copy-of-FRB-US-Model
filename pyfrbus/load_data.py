import pandas
import numpy

# For mypy typing
from pandas import DataFrame


# Function to load Public FRB/US dataset from text file
def load_data(filename: str) -> DataFrame:
    data = pandas.read_csv(filename, index_col=0)
    data.index = pandas.PeriodIndex(data.index, freq="Q")
    data.index.name = None
    data.columns = [col.lower() for col in data.columns]
    data = data.astype(numpy.float64)
    return data
