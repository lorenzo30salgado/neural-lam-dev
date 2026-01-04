# Download script for ERA5 matching COSMO domain
# For LAM model project
# by Joel Oskarsson, joel.oskarsson@outlook.com
# adapted by Simon Adamov simon.adamov@meteoswiss.ch

import numcodecs
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

# ERA5 from weatherbench 2
era = xr.open_zarr(
    "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
)

# Precomputed limits (adapted to COSMO)
boundary_lon_range = (-16, 33)
boundary_lat_range = (27, 66)

time_slice = slice("2016-01-01T00", "2016-02-29T18")

boundary_vars = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
    "surface_pressure",
    "land_sea_mask",
    "geopotential_at_surface",
]

# Make new longitude coord for slicing
# Create a new longitude coordinate that goes from -180 to 180
longitude_new = np.where(
    era["longitude"] > 180, era["longitude"] - 360, era["longitude"]
)
# Assign the new longitude coordinate to the dataset
era = era.assign_coords(longitude=longitude_new)
# Sort the dataset by the new longitude coordinate
era = era.sortby("longitude")

# Slice
era_subset = era.sel(
    longitude=slice(*boundary_lon_range),
    latitude=slice(boundary_lat_range[1], boundary_lat_range[0]),
    time=time_slice,
)[boundary_vars]

# Change back longitude
longitude_back = (era_subset["longitude"] + 360) % 360
era_subset = era_subset.assign_coords(longitude=longitude_back)
era_subset = era_subset.sortby("longitude")

print("ERA Subset:")
print(era_subset)
print(era_subset["time"])
print(f"ERA subset has size {era_subset.nbytes / 1e9} GB uncompressed")

# Set up compression
compressor = numcodecs.Blosc(
    cname="zstd", clevel=9, shuffle=numcodecs.Blosc.SHUFFLE
)
print("Downloading and saving zarr...")
with ProgressBar():
    era_subset.to_zarr(
        "era_subset_FULL.zarr",
        encoding={
            var: {"compressor": compressor} for var in era_subset.data_vars
        },
    )
