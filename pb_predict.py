import tensorflow as tf
import struct2tensor as s2t
import pandas as pd
import numpy as np

# Load the TensorFlow model from the SavedModel directory
model_path = r"D:\JTech\Nairi\FGBL2019-2021-3tClass\predict\001"
loaded_model = tf.saved_model.load(model_path)
infer = loaded_model.signatures["serving_default"]

# Load the data to predict
csv_path = r"D:\JTech\Nairi\FGBL2019-2021-3.csv"
df = pd.read_csv(csv_path)

# Ensure the columns are numeric
columns = ['1', '2', '3', '4', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
df[columns] = df[columns].apply(pd.to_numeric)

# Prepare the data for prediction (assuming the model expects a dictionary of tensors)
data = {str(i): tf.convert_to_tensor(df[i].values, dtype=tf.float32) for i in columns}

# Make the predictions
predictions = infer(**data)['output_0'].numpy()  # Adjust 'output_0' based on your model's output tensor name

# Add predictions to your dataframe
df["prediction"] = predictions

# Save the updated dataframe to a new CSV file
output_path = r"D:\JTech\Nairi\FGBL2019-2021-3_Class_pred.csv"
df.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")
