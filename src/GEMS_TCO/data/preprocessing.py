"""Reusable preprocessing primitives for raw GEMS orbit files."""

from __future__ import annotations

import calendar
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

DEFAULT_OBSERVATION_HOURS = tuple(range(8))
DEFAULT_QUALITY_FLAGS = (0, 2)


@dataclass(frozen=True)
class GeographicBounds:
    """Closed latitude-longitude bounds for a rectangular study region."""

    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.minimum_latitude,
                self.maximum_latitude,
                self.minimum_longitude,
                self.maximum_longitude,
            ),
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise ValueError("geographic bounds must be finite")
        if self.minimum_latitude > self.maximum_latitude:
            raise ValueError("minimum_latitude must not exceed maximum_latitude")
        if self.minimum_longitude > self.maximum_longitude:
            raise ValueError("minimum_longitude must not exceed maximum_longitude")

    def mask(self, frame: pd.DataFrame) -> pd.Series:
        """Return rows lying inside the closed geographic rectangle."""

        _require_columns(frame, ("Latitude", "Longitude"))
        return frame["Latitude"].between(self.minimum_latitude, self.maximum_latitude) & frame[
            "Longitude"
        ].between(self.minimum_longitude, self.maximum_longitude)


class GEMSOrbitReader:
    """Read one GEMS netCDF orbit and return quality-ready tabular data."""

    GEOLOCATION_VARIABLES = ("Latitude", "Longitude", "Time")
    DATA_VARIABLES = ("ColumnAmountO3", "FinalAlgorithmFlags")

    def __init__(self, bounds: GeographicBounds) -> None:
        self.bounds = bounds

    def read(self, path: str | Path) -> pd.DataFrame:
        """Read geolocation and ozone fields from one netCDF file."""

        path = Path(path)
        with xr.open_dataset(path, group="Geolocation Fields") as location_dataset:
            location_selected = location_dataset[list(self.GEOLOCATION_VARIABLES)]
            location_index_names = tuple(location_selected.sizes)
            location = location_selected.to_dataframe().reset_index()
        with xr.open_dataset(path, group="Data Fields") as data_dataset:
            observation_selected = data_dataset[list(self.DATA_VARIABLES)]
            observation_index_names = tuple(observation_selected.sizes)
            observations = observation_selected.to_dataframe().reset_index()

        if not location_index_names or set(location_index_names) != set(observation_index_names):
            raise ValueError(
                f"{path}: geolocation and data groups must share the same dimension-index "
                f"keys, got {location_index_names} and {observation_index_names}"
            )

        index_columns = list(location_index_names)
        if location.duplicated(index_columns).any() or observations.duplicated(index_columns).any():
            raise ValueError(f"{path}: netCDF dimension-index keys must be unique in both groups")

        row_order_column = "__gems_location_row_order__"
        location[row_order_column] = np.arange(len(location), dtype=np.int64)
        try:
            aligned = location.merge(
                observations,
                on=index_columns,
                how="outer",
                sort=False,
                validate="one_to_one",
                indicator=True,
            )
        except pd.errors.MergeError as exc:
            raise ValueError(f"{path}: netCDF groups cannot be aligned one-to-one") from exc
        if not aligned["_merge"].eq("both").all():
            raise ValueError(
                f"{path}: geolocation and data groups contain different dimension-index sets"
            )

        aligned = aligned.sort_values(row_order_column, kind="stable")
        return aligned[[*self.GEOLOCATION_VARIABLES, *self.DATA_VARIABLES]].reset_index(drop=True)

    def read_hour(self, path: str | Path) -> pd.DataFrame:
        """Read, crop, and assign the orbit's representative minute."""

        frame = self.read(path).dropna(subset=(*self.GEOLOCATION_VARIABLES, *self.DATA_VARIABLES))
        frame = frame.loc[self.bounds.mask(frame)].copy()
        frame = frame.loc[frame["ColumnAmountO3"] < 1000].copy()
        if frame.empty:
            return frame

        representative_hour = float(pd.to_numeric(frame["Time"]).mean())
        timestamp = pd.to_datetime(representative_hour, unit="h").floor("min")
        frame = frame.assign(Time=timestamp)
        return frame.reset_index(drop=True)


def monthly_orbit_paths(
    root_directory: str | Path,
    year: int,
    month: int,
    *,
    observation_hours: Sequence[int] = DEFAULT_OBSERVATION_HOURS,
    minute: int = 45,
) -> list[Path]:
    """Construct the expected GEMS orbit paths for one calendar month."""

    if not isinstance(year, int) or isinstance(year, bool) or year < 1:
        raise ValueError("year must be a positive integer")
    if not isinstance(month, int) or isinstance(month, bool) or not 1 <= month <= 12:
        raise ValueError("month must be an integer in 1..12")
    if not isinstance(minute, int) or isinstance(minute, bool) or not 0 <= minute <= 59:
        raise ValueError("minute must be an integer in 0..59")
    hours = tuple(observation_hours)
    if any(
        not isinstance(hour, int) or isinstance(hour, bool) or not 0 <= hour <= 23 for hour in hours
    ):
        raise ValueError("observation_hours must contain integers in 0..23")

    last_day = calendar.monthrange(year, month)[1]
    day_token = f"01{last_day:02d}"
    directory = Path(root_directory) / f"{year}{month:02d}{day_token}"
    return [
        directory / f"{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}.nc"
        for day in range(1, last_day + 1)
        for hour in hours
    ]


class MonthlyOrbitAggregator:
    """Aggregate hourly netCDF orbits after spatial and quality filtering."""

    def __init__(
        self,
        reader: GEMSOrbitReader,
        acceptable_quality_flags: Iterable[int] = DEFAULT_QUALITY_FLAGS,
    ) -> None:
        self.reader = reader
        flags = tuple(acceptable_quality_flags)
        if not flags:
            raise ValueError("acceptable_quality_flags must be nonempty")
        if any(
            isinstance(flag, (bool, np.bool_)) or not isinstance(flag, Integral) for flag in flags
        ):
            raise TypeError("acceptable_quality_flags must contain only integers")
        self.acceptable_quality_flags = tuple(int(flag) for flag in flags)

    def aggregate(self, paths: Iterable[str | Path]) -> pd.DataFrame:
        """Combine all readable orbit files into one quality-filtered frame."""

        hourly_frames: list[pd.DataFrame] = []
        for path_value in paths:
            path = Path(path_value)
            try:
                hourly_frames.append(self.reader.read_hour(path))
            except FileNotFoundError:
                logger.warning("Orbit file not found: %s", path)

        nonempty = [frame for frame in hourly_frames if not frame.empty]
        if not nonempty:
            raise ValueError("no nonempty orbit files were available for aggregation")

        combined = pd.concat(nonempty, ignore_index=True)
        _require_columns(combined, ("Time", "FinalAlgorithmFlags"))
        timestamps = pd.to_datetime(combined["Time"], errors="raise")
        combined["Hours_elapsed"] = timestamps.astype("int64") / 3_600_000_000_000
        quality_mask = combined["FinalAlgorithmFlags"].isin(self.acceptable_quality_flags)
        return combined.loc[quality_mask].reset_index(drop=True)

    @staticmethod
    def write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
        """Write an aggregated frame, creating its parent directory."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        return output


def group_by_orbit(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group a monthly frame by minute-resolution orbit timestamps."""

    _require_columns(frame, ("Time",))
    grouped_frame = frame.copy()
    timestamps = pd.to_datetime(grouped_frame["Time"], errors="raise").dt.floor("min")
    grouped_frame["Orbit"] = timestamps.dt.strftime("%Y-%m-%d %H:%M")

    orbit_frames: dict[str, pd.DataFrame] = {}
    for timestamp, indices in timestamps.groupby(timestamps).groups.items():
        key = timestamp.strftime("y%ym%mday%d_hm%H:%M")
        orbit_frames[key] = grouped_frame.loc[indices].reset_index(drop=True)
    return orbit_frames


def build_center_grid(
    bounds: GeographicBounds,
    *,
    latitude_step: float = 0.044,
    longitude_step: float = 0.063,
    latitude_edge_offset: float = 0.0,
    longitude_edge_offset: float = 0.0,
    latitude_drift_per_column: float = 0.0,
) -> pd.DataFrame:
    """Construct the regular center grid used by the forward-binning step.

    The two calibration terms are explicit because they encode a study-specific
    geometric correction. To reproduce the historical calibrated grid use
    ``latitude_edge_offset=0.0002``, ``longitude_edge_offset=0.0002``,
    and ``latitude_drift_per_column=0.00012``; the default is uncalibrated.
    """

    for name, value in (
        ("latitude_step", latitude_step),
        ("longitude_step", longitude_step),
    ):
        if not np.isfinite(float(value)) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
    for name, value in (
        ("latitude_edge_offset", latitude_edge_offset),
        ("longitude_edge_offset", longitude_edge_offset),
        ("latitude_drift_per_column", latitude_drift_per_column),
    ):
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")

    latitude_values = (
        np.arange(
            bounds.maximum_latitude - latitude_step - latitude_edge_offset,
            bounds.minimum_latitude - latitude_step,
            -latitude_step,
        )
        + latitude_step
    )
    longitude_values = (
        np.arange(
            bounds.maximum_longitude - longitude_step - longitude_edge_offset,
            bounds.minimum_longitude - longitude_step,
            -longitude_step,
        )
        + longitude_step
    )
    latitude_grid = latitude_values[:, None] + (
        np.arange(len(longitude_values)) * latitude_drift_per_column
    )
    return pd.DataFrame(
        {
            "Latitude": latitude_grid.ravel(order="C"),
            "Longitude": np.tile(longitude_values, len(latitude_values)),
        }
    )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
