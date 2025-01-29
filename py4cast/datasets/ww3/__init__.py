import datetime as dt
import time
from pathlib import Path
from typing import List, Literal

import numpy as np
import xarray as xr

from py4cast.datasets.access import (
    DataAccessor,
    Grid,
    GridConfig,
    ParamConfig,
    Timestamps,
    WeatherParam,
)
from py4cast.datasets.ww3.settings import FORMATSTR, METADATA, SCRATCH_PATH, DEFAULT_CONFIG


class WW3Accessor(DataAccessor):
    @staticmethod
    def get_weight_per_level(
        level: int=0,
         level_type: Literal["isobaricInhPa", "heightAboveGround", "surface", "meanSea"] = "meanSea"
    ):
        return 1.0

    #############################################################
    #                            GRID                           #
    #############################################################
    @staticmethod
    def load_grid_info(name: str) -> GridConfig:
        if name not in ["BRETAGNE0002"]:
            raise NotImplementedError("Grid must be in ['BRETAGNE0002'].")

        path = SCRATCH_PATH / f"conf_{name}.grib"
        conf_ds = xr.open_dataset(path)
        bathy = conf_ds.unknown.values
        landsea_mask = ~np.isnan(bathy)
        grid_conf = GridConfig(
            conf_ds.unknown.shape,
            conf_ds.latitude.values,
            conf_ds.longitude.values,
            bathy,
            landsea_mask,
        )
        return grid_conf

    @staticmethod
    def get_grid_coords(param: WeatherParam) -> List[int]:
        return METADATA["GRIDS"][param.grid.name]["extent"]

    #############################################################
    #                              PARAMS                       #
    #############################################################
    @staticmethod
    def load_param_info(name: str) -> ParamConfig:
        info = METADATA["WEATHER_PARAMS"][name]
        grid = info["grid"]
        if grid not in ["BRETAGNE0002"]:
            raise NotImplementedError(
                "Parameter native grid must be in ['BRETAGNE0002']"
            )
        return ParamConfig(
            unit=info["unit"],
            level_type=info["type_level"],
            long_name=info["long_name"],
            grid=grid,
            grib_name=None,
            grib_param=info["name"],
            )

    #############################################################
    #                              LOADING                      #
    #############################################################

    def cache_dir(self, name: str, grid: Grid):
        path = self.get_dataset_path(name, grid)
        path.mkdir(mode=0o777, exist_ok=True)
        return path

    @staticmethod
    def get_dataset_path(name: str, grid: Grid):
        return SCRATCH_PATH / "cache"

    @classmethod
    def get_filepath(
        cls,
        ds_name: str,
        param: WeatherParam,
        date: dt.datetime,
        file_format: Literal["zarr", "grib"],
    ) -> Path:
        """
        Returns the path of the file containing the parameter data.
        - in grib format, data is grouped by level type.
        - in npz format, data is saved as npz, rescaled to the wanted grid, and each
        2D array is saved as one file to optimize IO during training."""
        if file_format == "grib":
            return (
                SCRATCH_PATH
                / param.grid.name
                / file_format
                / f"{date.strftime(FORMATSTR)}_SWH-SHWW-MDWW-MPWW-SHPS-MDPS-MPPS_MER_BRETAGNE0002.{file_format}"
            )
        else:
            return (
                SCRATCH_PATH
                / param.grid.name
                / file_format
                / f"{date.strftime(FORMATSTR)}.{file_format}"
            )

    @classmethod
    def load_data_from_disk(
        cls,
        ds_name: str,
        param: WeatherParam,
        timestamps: Timestamps,
        # the member parameter is not accessed if irrelevant
        member: int = 0,
        file_format: Literal["zarr", "grib"] = "grib",
    ):
        """
        Function to load invidiual parameter and lead time from a file stored in disk
        """
        dates = timestamps.validity_times
        arr_list = []
        for date in dates:
            data_path = cls.get_filepath(ds_name, param, date, file_format)
            if file_format == "grib":
                arr = xr.open_dataset(data_path, engine="cfgrib")
                arr = arr.rename({"unknown": "shps"})
                arr.shps.attrs["long_name"] = "Significant height of primary wave"
                arr.shps.attrs["units"] = "m"

            else:
                arr = xr.open_zarr(data_path)
            arr = arr[param.grib_param].values
            arr = np.nan_to_num(arr, nan=0)
            # invert latitude
            arr = arr[::-1]
            arr_list.append(np.expand_dims(arr, axis=-1))
        return np.stack(arr_list)

    @classmethod
    def exists(
        cls,
        ds_name: str,
        param: WeatherParam,
        timestamps: Timestamps,
        file_format: Literal["zarr", "grib"] = "grib",
    ) -> bool:
        for date in timestamps.validity_times:
            filepath = cls.get_filepath(ds_name, param, date, file_format)
            if not filepath.exists():
                return False
        return True

    @staticmethod
    def parameter_namer(param: WeatherParam) -> str:
        return param.name