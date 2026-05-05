import pandas as pd
import numpy as np
import os

def split_credit_card_data():
    input_file = 'data/creditcard.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Split into 6 batches
    chunk_size = int(np.ceil(len(df) / 6))
    batches = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    print(f"Total rows: {len(df)}")
    
    for i, batch in enumerate(batches):
        batch_num = i + 1
        output_file = f'data/batch_{batch_num:02d}.csv'
        batch.to_csv(output_file, index=False)
        print(f"Saved {output_file} with {len(batch)} rows.")

if __name__ == "__main__":
    split_credit_card_data()
