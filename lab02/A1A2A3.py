import pandas as pd
import numpy as np     

file_path = 'lab02/Lab Session Data.xlsx'
sheet_name = 'Purchase data'
data = pd.read_excel(file_path, sheet_name=sheet_name)

A = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values

C = data["Payment (Rs)"].values.reshape(-1, 1)

dimensionality = A.shape[1]
num_vectors = A.shape[0]
  
rank_A = np.linalg.matrix_rank(A)
pseudo_inverse_A = np.linalg.pinv(A)
model_vector_X = np.dot(pseudo_inverse_A, C)

print(f"Dimensionality of the vector space: {dimensionality}")
print(f"Number of vectors in the vector space: {num_vectors}")
print(f"Rank of Matrix A: {rank_A}")
print("Cost of each product:")  
i = 1
for cost in model_vector_X:
    print(f"Product {i}: {cost[0]:.2f}")
    i += 1 

classes = []

for payment in data["Payment (Rs)"]:
    if payment > 200:
        classes.append("RICH")
    else:
        classes.append("POOR")

data["Class"] = classes 

print(data[["Customer", "Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)", "Class"]])