import sys
import os
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
import re
import xml.etree.ElementTree as ET


def convert_nc_to_csv(sim_directory, output_file='velocity_fields_all_times.csv'):
    if not os.path.isdir(sim_directory):
        raise ValueError(f"Directory not found: {sim_directory}")

    nc_files = sorted(glob.glob(f"{sim_directory}/state_phys_t*.nc"))

    print(f"Found {len(nc_files)} NetCDF files in {sim_directory}")

    params_file = os.path.join(sim_directory, 'params_simul.xml')
    if not os.path.exists(params_file):
        raise ValueError(f"params_simul.xml not found in {sim_directory}")

    tree = ET.parse(params_file)
    root = tree.getroot()
    oper = root.find('oper')

    nx = int(oper.get('nx'))
    ny = int(oper.get('ny'))
    Lx = float(oper.get('Lx'))
    Ly = float(oper.get('Ly'))

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)

    X, Y = np.meshgrid(x, y)
    print(f"Grid size: {nx} × {ny} = {nx * ny} points per time step")
    print(f"Domain size: Lx={Lx}, Ly={Ly}")

    all_frames = []
    for i, nc_file in enumerate(nc_files):
        with nc.Dataset(nc_file, 'r') as dataset:
            print(dataset.groups['state_phys'].variables["ux"][:])
            match = re.search(r't(\d+)\.\d+', nc_file)
            t = float(match.group(1)) if match else 0.0

            ux_field = dataset.groups['state_phys'].variables['ux'][:]
            uy_field = dataset.groups['state_phys'].variables['uy'][:]
            rot_field = dataset.groups['state_phys'].variables['rot'][:]

        frame_df = pd.DataFrame({
            't': t,
            'x': X.flatten(),
            'y': Y.flatten(),
            'ux': ux_field.flatten(),
            'uy': uy_field.flatten(),
            'rot': rot_field.flatten()
        })
        all_frames.append(frame_df)

        if (i + 1) % 10 == 0 or (i + 1) == len(nc_files):
            print(f"  Processed {i + 1}/{len(nc_files)} files...")

    all_fields_df = pd.concat(all_frames, ignore_index=True)

    print(f"Saving to {output_file}...")
    all_fields_df.to_csv(output_file, index=False)

    print(f"\nSaved!")

    return all_fields_df


def main():

    sim_root = "/Users/Tanya/Sim_data/examples/"

    sim_directories = os.listdir("/Users/Tanya/Sim_data/examples/")

    output_file = sys.argv[1] if len(sys.argv) > 1 else 'velocity_fields_all_times.csv'
    dfs = []

    for i, sim_dir in enumerate(sim_directories):
        sim_directory = sim_root + sim_dir
        df = convert_nc_to_csv(sim_directory, output_file)
        df['scene'] = i
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    main()
