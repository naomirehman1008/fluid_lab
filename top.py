import os
import shutil

num_scenes = 5

if __name__ == "__main__":
    shutil.rmtree("/Users/Tanya/Sim_data/examples", ignore_errors=True)
    for _ in range(num_scenes):
        os.system("python3 solver.py")

    os.system("python3 convert_nc_to_csv.py velocity_fields_all_times.csv")
