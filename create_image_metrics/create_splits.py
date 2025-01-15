import pandas as pd
from sklearn.model_selection import train_test_split

# Load your aggregated data
df = pd.read_csv('/Users/rushil/Downloads/agg_masked_images_metrics.csv')

# Step 1: Extract unique patient IDs
unique_pids = df['PID'].unique()

# Step 2: Shuffle the list of unique patient IDs
shuffled_pids = pd.Series(unique_pids).sample(frac=1, random_state=42).tolist()

# Step 3: Determine the split sizes
num_train = int(0.7 * len(shuffled_pids))
num_test = int(0.1 * len(shuffled_pids))
num_val = len(shuffled_pids) - num_train - num_test

# Step 4: Split the shuffled list of patient IDs
train_pids = shuffled_pids[:num_train]
test_pids = shuffled_pids[num_train:num_train+num_test]
val_pids = shuffled_pids[num_train+num_test:]

# Step 5: Assign data to sets based on PID
train_df = df[df['PID'].isin(train_pids)]
test_df = df[df['PID'].isin(test_pids)]
val_df = df[df['PID'].isin(val_pids)]

# Verify the distribution
print(f"Training set: {len(train_df)} rows, {len(train_pids)} unique PIDs")
print(f"Testing set: {len(test_df)} rows, {len(test_pids)} unique PIDs")
print(f"Validation set: {len(val_df)} rows, {len(val_pids)} unique PIDs")

# Save the splits to new CSV files if needed
train_df.to_csv('/Users/rushil/Downloads/lr_train_data.csv', index=False)
test_df.to_csv('/Users/rushil/Downloads/lr_test_data.csv', index=False)
val_df.to_csv('/Users/rushil/Downloads/lr_val_data.csv', index=False)