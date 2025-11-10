import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import traceback


def main():
    folder_path = "/home/nicit/nordsoen/nordsoen-i-wflo/results/DL/recorders/"

    recorders = {}
    # load pickle
    for random_state in tqdm(range(100), desc="Loading raw data", colour="green"):
        with open(folder_path + f"recorder_{random_state}.pkl", "rb") as f:
            recorders[random_state] = pickle.load(f)

    rows = []
    input_keys = ['d_x', 'd_y', 'ws_eff', 'ti_eff', 'ws']
    output_keys = ['yaw', 'aep', 'aep_opt']

    # Function to normalize arrays to shape (n_wt, -1)
    def normalize_array(arr, n_wt):
        if n_wt not in arr.shape:
            # Tile the array to match n_wt rows
            return np.tile(arr.flatten(), (n_wt, 1))
        else:
            axis = arr.shape.index(n_wt)
            arr = np.moveaxis(arr, axis, 0)
            return arr.reshape(n_wt, -1)

    # Define input and output keys
    input_keys = ['d_x', 'd_y', 'ws_eff', 'ti_eff', 'ws']
    output_keys = ['yaw', 'aep', 'aep_opt']

    rows = []

    # make directory "../data/recorders_df.h5" if it doens't exist
    os.makedirs("../data", exist_ok=True)
    output_folder = "../data"
    
    for recorder_id, results in tqdm(recorders.items(), desc="Processing recorders", colour="green"):
        rows = []
        for result_id, result in results.items():
            try:
                n_wt = result['yaw'].shape[0]
                input_array = np.concatenate(
                    [normalize_array(result[k], n_wt) for k in input_keys], axis=1
                ).flatten()
                output_array = np.concatenate(
                    [normalize_array(result[k], n_wt) for k in output_keys], axis=1
                ).flatten()

                row = {
                    "recorder_id": recorder_id,
                    "result_id": result_id
                }
                row.update({f"input_{i}": val for i, val in enumerate(input_array)})
                row.update({f"output_{i}": val for i, val in enumerate(output_array)})
                rows.append(row)

            except Exception as e:
                print(f"❌ Error processing recorder {recorder_id}, result {result_id}: {e}")
                traceback.print_exc()

        df_chunk = pd.DataFrame(rows)
        if len(df_chunk) > 0:
            df_chunk.to_parquet(f"{output_folder}/recorder_{recorder_id}.parquet", index=False)
            print(f"✅ Wrote {len(df_chunk)} rows for recorder {recorder_id}")
        else:
            print(f"⚠️ No data to write for recorder {recorder_id}")





if __name__ == "__main__":
    main()