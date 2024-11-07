import os
import pandas as pd
from pgmpy.utils import get_example_model
from pgmpy.readwrite import BIFWriter

#TODO FIX NODE NAMES IN CORRECT ORDER

# bnfdata set you want to save
bnfdata = "asia"
path = "Tag-PC-using-LLM/additionalData"

# Ensure save path exists
os.makedirs(path, exist_ok=True)

# Load model from bnf repo here
model = get_example_model(bnfdata)
writer = BIFWriter(model)

# Sample some data from the model
data = model.simulate(n_samples=1000, seed=42)

# Extract node names
node_names = list(model.nodes())
# Insert node names as the first row
header_df = pd.DataFrame([node_names], columns=node_names)
data_with_header = pd.concat([header_df, data], ignore_index=True)

# Save data to a CSV file
csv_filename = os.path.join(path, (bnfdata + "_data.csv"))
data_with_header.to_csv(csv_filename, index=False, header=False)
print(f"CSV file saved at: {csv_filename}")