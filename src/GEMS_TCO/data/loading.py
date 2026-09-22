"""Load processed GEMS grids and convert them to model-ready tensors."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# GEMS provides eight nominal daytime observations per day. The historical
# analyses express Unix time in hours and center it near 2024-07-01.
GEMS_TIME_SLOTS_PER_DAY = 8
DEFAULT_TIME_ORIGIN_HOURS = 477_700.0


class ProcessedDataLoader:
    """Read regular-grid pickle files produced by the GEMS preprocessing pipeline.

    Pickle files must be trusted; Python's pickle format is not safe for
    untrusted input.
    """

    def __init__(self, data_directory: str | Path) -> None:
        self.data_directory = Path(data_directory)

    def load_grid_frames(
        self, years: Sequence[int | str], months: Sequence[int]
    ) -> dict[str, pd.DataFrame]:
        """Load processed grid frames for the requested year-month pairs.

        Missing files are logged and skipped. Malformed files raise an error
        because silently accepting them would make fitted samples ambiguous.
        """

        _validate_years_and_months(years, months)
        grid_frames: dict[str, pd.DataFrame] = {}
        for year_value in years:
            year = str(year_value)
            for month in months:
                filename = f"tco_grid_{year[2:]}_{month:02d}.pkl"
                path = self.data_directory / f"pickle_{year}" / filename
                try:
                    with path.open("rb") as stream:
                        loaded = pickle.load(stream)
                except FileNotFoundError:
                    logger.warning("Processed grid file not found: %s", path)
                    continue

                if not isinstance(loaded, Mapping):
                    raise TypeError(f"{path} must contain a mapping of time keys to DataFrames")
                for key, frame in loaded.items():
                    if not isinstance(frame, pd.DataFrame):
                        raise TypeError(f"{path}: value for {key!r} is not a pandas DataFrame")
                    global_key = f"{year}_{month:02d}_{key}"
                    if global_key in grid_frames:
                        raise ValueError(f"duplicate processed-grid key: {global_key}")
                    grid_frames[global_key] = frame.reset_index(drop=True)
        return grid_frames

    def build_spatial_ordering(
        self, grid_frames: Mapping[str, pd.DataFrame], max_neighbors: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a max-min order and predecessor neighbors from a common grid.

        Every frame is required to contain the same ordered grid coordinates.
        This is an important assumption of the downstream Vecchia fits and is
        therefore checked here instead of being left implicit.
        """

        keys = sorted(grid_frames)
        if not keys:
            raise ValueError("grid_frames must contain at least one frame")

        from .. import orderings

        reference = _grid_coordinates(grid_frames[keys[0]], key=keys[0])
        for key in keys[1:]:
            candidate = _grid_coordinates(grid_frames[key], key=key)
            if candidate.shape != reference.shape or not np.allclose(
                candidate, reference, equal_nan=False
            ):
                raise ValueError(
                    "all grid frames must share identical ordered Latitude/Longitude coordinates"
                )

        spatial_order = orderings.maxmin_order(reference)
        neighbors = orderings.predecessor_neighbors(
            reference[spatial_order], max_neighbors=max_neighbors
        )
        return spatial_order, neighbors

    def load_monthly_grids(
        self,
        years: Sequence[int | str] = ("2024",),
        months: Sequence[int] = (7,),
        latitude_range: Sequence[float] | None = None,
        longitude_range: Sequence[float] | None = None,
        max_neighbors: int = 10,
        compute_ordering: bool = False,
    ) -> tuple[dict[str, pd.DataFrame], np.ndarray | None, np.ndarray | None, float]:
        """Load grid frames, optionally with the ordering used by Vecchia.

        The returned mean is calculated across every finite ``ColumnAmountO3``
        value in the selected files. When multiple months are requested this
        is a pooled mean, not a separate mean for each month.
        """

        _validate_years_and_months(years, months)
        latitude_bounds = _validate_bounds("latitude_range", latitude_range)
        longitude_bounds = _validate_bounds("longitude_range", longitude_range)
        if (latitude_bounds is None) != (longitude_bounds is None):
            raise ValueError("latitude_range and longitude_range must be supplied together")

        grid_frames = self.load_grid_frames(years, months)
        if not grid_frames:
            requested = ", ".join(
                f"{str(year)}-{int(month):02d}" for year in years for month in months
            )
            raise FileNotFoundError(
                f"no processed grid files were found for {requested} under {self.data_directory}"
            )

        if latitude_bounds is not None and longitude_bounds is not None:
            filtered: dict[str, pd.DataFrame] = {}
            for key, frame in grid_frames.items():
                _require_columns(frame, ("Latitude", "Longitude"), context=key)
                mask = frame["Latitude"].between(*latitude_bounds) & frame["Longitude"].between(
                    *longitude_bounds
                )
                filtered[key] = frame.loc[mask].reset_index(drop=True)
            grid_frames = filtered

        ozone_values: list[np.ndarray] = []
        for key, frame in grid_frames.items():
            _require_columns(frame, ("ColumnAmountO3",), context=key)
            ozone_values.append(pd.to_numeric(frame["ColumnAmountO3"], errors="coerce").to_numpy())
        concatenated = np.concatenate(ozone_values)
        finite_ozone = concatenated[np.isfinite(concatenated)]
        if finite_ozone.size == 0:
            raise ValueError("loaded grids contain no finite ColumnAmountO3 values")
        ozone_mean = float(finite_ozone.mean())
        logger.info("Pooled mean over loaded ColumnAmountO3 values: %.4f", ozone_mean)

        if not compute_ordering:
            return grid_frames, None, None, ozone_mean

        spatial_order, neighbors = self.build_spatial_ordering(
            grid_frames, max_neighbors=max_neighbors
        )
        return grid_frames, spatial_order, neighbors, ozone_mean

    def build_model_tensors(
        self,
        grid_frames: Mapping[str, pd.DataFrame],
        ozone_mean: float = 0.0,
        time_slice: Sequence[int] = (0, GEMS_TIME_SLOTS_PER_DAY),
        spatial_order: np.ndarray | None = None,
        dtype: torch.dtype = torch.double,
        use_source_coordinates: bool = True,
        time_origin_hours: float = DEFAULT_TIME_ORIGIN_HOURS,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Convert one day's ordered grid frames to model tensors.

        Each output row contains ``[latitude, longitude, centered_ozone,
        centered_time, D1, ..., D7]``. The seven indicators encode the eight
        nominal GEMS time slots with the first slot as the reference category.

        ``spatial_order`` is applied independently to every selected time. It
        must therefore be a complete permutation of the row indices shared by
        those frames. Missing responses remain ``NaN``; source coordinates may
        also be missing only on those unobserved rows. Regular-grid coordinates
        should be selected for Debiased Whittle fits.
        """

        if len(time_slice) != 2:
            raise ValueError("time_slice must contain exactly (start, stop)")
        start, stop = time_slice
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in (start, stop)
        ):
            raise TypeError("time_slice entries must be integers")
        if start < 0 or stop < start or stop > GEMS_TIME_SLOTS_PER_DAY:
            raise ValueError(
                f"time_slice must satisfy 0 <= start <= stop <= {GEMS_TIME_SLOTS_PER_DAY}"
            )
        if not np.isfinite(float(ozone_mean)):
            raise ValueError("ozone_mean must be finite")
        if not np.isfinite(float(time_origin_hours)):
            raise ValueError("time_origin_hours must be finite")
        if (
            not isinstance(dtype, torch.dtype)
            or not torch.empty((), dtype=dtype).is_floating_point()
        ):
            raise TypeError("dtype must be a real floating-point torch dtype")
        if not isinstance(use_source_coordinates, (bool, np.bool_)):
            raise TypeError("use_source_coordinates must be boolean")

        keys = sorted(grid_frames)
        if not keys:
            raise ValueError("grid_frames must contain one day's time slices")
        if len(keys) > GEMS_TIME_SLOTS_PER_DAY:
            raise ValueError(
                "build_model_tensors accepts one day at a time; "
                f"got {len(keys)} frames, expected at most {GEMS_TIME_SLOTS_PER_DAY}"
            )
        selected_keys = keys[int(start) : int(stop)]

        model_frames: dict[str, torch.Tensor] = {}
        stacked: list[torch.Tensor] = []
        expected_rows: int | None = None
        order: np.ndarray | None = None

        for selected_index, key in enumerate(selected_keys):
            frame = grid_frames[key].copy()
            coordinate_columns = (
                ("Source_Latitude", "Source_Longitude")
                if use_source_coordinates
                else ("Latitude", "Longitude")
            )
            required = (*coordinate_columns, "ColumnAmountO3", "Hours_elapsed")
            _require_columns(frame, required, context=key)

            if expected_rows is None:
                expected_rows = len(frame)
                order = _validate_spatial_order(spatial_order, expected_rows)
            elif len(frame) != expected_rows:
                raise ValueError("all selected grid frames must contain the same number of rows")

            hours = pd.to_numeric(frame["Hours_elapsed"], errors="coerce")
            if np.isinf(hours.to_numpy(dtype=np.float64)).any():
                raise ValueError(f"{key}: Hours_elapsed cannot contain infinity")
            if not hours.notna().any():
                raise ValueError(f"{key}: Hours_elapsed contains no finite values")
            frame["Hours_elapsed"] = hours.fillna(hours.median())
            frame["Hours_elapsed"] = np.round(frame["Hours_elapsed"] - float(time_origin_hours))

            if order is not None:
                frame = frame.iloc[order].reset_index(drop=True)

            selected = frame[[*coordinate_columns, "ColumnAmountO3", "Hours_elapsed"]].copy()
            response = pd.to_numeric(selected["ColumnAmountO3"], errors="coerce")
            if np.isinf(response.to_numpy(dtype=np.float64)).any():
                raise ValueError(f"{key}: ColumnAmountO3 cannot contain infinity")
            selected["ColumnAmountO3"] = response - float(ozone_mean)
            values = selected.to_numpy(dtype=np.float64)
            finite_coordinates = np.isfinite(values[:, :2]).all(axis=1)
            observed = np.isfinite(values[:, 2])
            if not finite_coordinates[observed].all():
                raise ValueError(f"{key}: observed responses require finite coordinates")
            if not np.isfinite(values[:, 3]).all():
                raise ValueError(f"{key}: centered time must be finite")
            base = torch.from_numpy(values).to(dtype=dtype)

            # Preserve the nominal slot when a strict subset is requested.
            # Re-indexing the selected frames from zero would silently change
            # the reference category and therefore the fitted mean model.
            time_slot = int(start) + selected_index
            indicators = F.one_hot(torch.tensor(time_slot), num_classes=GEMS_TIME_SLOTS_PER_DAY)[1:]
            indicators = indicators.repeat(len(base), 1).to(dtype=dtype)
            tensor = torch.cat((base, indicators), dim=1)
            model_frames[key] = tensor
            stacked.append(tensor)

        combined = (
            torch.cat(stacked, dim=0)
            if stacked
            else torch.empty(0, 4 + GEMS_TIME_SLOTS_PER_DAY - 1, dtype=dtype)
        )
        return model_frames, combined


class CoordinateDeviationFilter:
    """Filter regular-grid rows whose source coordinates deviate too far.

    The mask is shared across all time slots in a day. Aggregated tensors are
    assumed to be stacked in time-major order using the same row order as the
    hourly tensors.
    """

    @staticmethod
    def filter(
        hourly_grid: Mapping[int, Mapping[str, torch.Tensor]],
        hourly_source: Mapping[int, Mapping[str, torch.Tensor]],
        aggregated_grid: Mapping[int, torch.Tensor],
        aggregated_source: Mapping[int, torch.Tensor],
        latitude_tolerance: float = 0.025,
        longitude_tolerance: float = 0.04,
    ) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, torch.Tensor]]:
        """Keep rows within both coordinate tolerances at every daily time."""

        for name, tolerance in (
            ("latitude_tolerance", latitude_tolerance),
            ("longitude_tolerance", longitude_tolerance),
        ):
            if not np.isfinite(float(tolerance)) or tolerance < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")

        day_keys = set(hourly_grid)
        collections = (hourly_source, aggregated_grid, aggregated_source)
        if any(set(collection) != day_keys for collection in collections):
            raise ValueError("all four inputs must contain the same day keys")

        filtered_hourly: dict[int, dict[str, torch.Tensor]] = {}
        filtered_aggregated: dict[int, torch.Tensor] = {}
        for day in sorted(day_keys):
            grid_day = hourly_grid[day]
            source_day = hourly_source[day]
            time_keys = sorted(grid_day)
            if set(source_day) != set(time_keys):
                raise ValueError(f"day {day}: grid and source time keys differ")
            if not time_keys:
                if aggregated_grid[day].shape[0] != 0 or aggregated_source[day].shape[0] != 0:
                    raise ValueError(
                        f"day {day}: empty hourly mappings require empty aggregated tensors"
                    )
                filtered_hourly[day] = {}
                filtered_aggregated[day] = aggregated_grid[day][:0]
                continue

            reference = _coordinate_tensor(grid_day[time_keys[0]], day, time_keys[0])
            row_count = reference.shape[0]
            mask = torch.ones(row_count, dtype=torch.bool, device=reference.device)
            for time_key in time_keys:
                grid_tensor = _coordinate_tensor(grid_day[time_key], day, time_key)
                source_tensor = _coordinate_tensor(
                    source_day[time_key], day, time_key, allow_missing=True
                )
                if grid_tensor.shape[0] != row_count or source_tensor.shape[0] != row_count:
                    raise ValueError(f"day {day}, time {time_key}: inconsistent row count")
                comparable_reference = reference[:, :2].to(
                    device=grid_tensor.device,
                    dtype=grid_tensor.dtype,
                )
                if not torch.allclose(
                    grid_tensor[:, :2],
                    comparable_reference,
                    rtol=1e-7,
                    atol=1e-10,
                ):
                    raise ValueError(
                        f"day {day}, time {time_key}: regular-grid coordinates or row order "
                        "differ from the day's reference grid"
                    )
                source_coordinates = source_tensor[:, :2].to(device=grid_tensor.device)
                difference = torch.abs(grid_tensor[:, :2] - source_coordinates)
                current = (difference[:, 0] <= latitude_tolerance) & (
                    difference[:, 1] <= longitude_tolerance
                )
                mask &= current.to(device=mask.device)

            expected_rows = row_count * len(time_keys)
            grid_aggregate = aggregated_grid[day]
            source_aggregate = aggregated_source[day]
            if grid_aggregate.shape[0] != expected_rows:
                raise ValueError(
                    f"day {day}: aggregated_grid has {grid_aggregate.shape[0]} rows; "
                    f"expected {expected_rows}"
                )
            if source_aggregate.shape[0] != expected_rows:
                raise ValueError(
                    f"day {day}: aggregated_source has {source_aggregate.shape[0]} rows; "
                    f"expected {expected_rows}"
                )

            expected_grid_aggregate = torch.cat(
                [grid_day[time_key] for time_key in time_keys], dim=0
            )
            expected_source_aggregate = torch.cat(
                [source_day[time_key] for time_key in time_keys], dim=0
            )
            _validate_aggregate_order(
                grid_aggregate,
                expected_grid_aggregate,
                day=day,
                name="aggregated_grid",
                allow_missing_coordinates=False,
            )
            _validate_aggregate_order(
                source_aggregate,
                expected_source_aggregate,
                day=day,
                name="aggregated_source",
                allow_missing_coordinates=True,
            )

            filtered_hourly[day] = {
                time_key: grid_day[time_key][mask.to(grid_day[time_key].device)]
                for time_key in time_keys
            }
            expanded_mask = mask.repeat(len(time_keys)).to(grid_aggregate.device)
            filtered_aggregated[day] = grid_aggregate[expanded_mask]
            logger.info(
                "Coordinate filter day %s: retained %d of %d spatial rows",
                day,
                int(mask.sum().item()),
                row_count,
            )

        return filtered_hourly, filtered_aggregated


def _validate_years_and_months(years: Sequence[int | str], months: Sequence[int]) -> None:
    if not years or not months:
        raise ValueError("years and months must be nonempty")
    for year in years:
        text = str(year)
        if len(text) != 4 or not text.isdigit():
            raise ValueError(f"invalid four-digit year: {year!r}")
    for month in months:
        if not isinstance(month, (int, np.integer)) or isinstance(month, (bool, np.bool_)):
            raise TypeError("months must contain integers")
        if not 1 <= int(month) <= 12:
            raise ValueError(f"month must lie in 1..12, got {month}")


def _validate_bounds(name: str, bounds: Sequence[float] | None) -> tuple[float, float] | None:
    if bounds is None:
        return None
    if len(bounds) != 2:
        raise ValueError(f"{name} must contain exactly (minimum, maximum)")
    lower, upper = (float(value) for value in bounds)
    if not np.isfinite((lower, upper)).all() or lower > upper:
        raise ValueError(f"{name} must contain finite values with minimum <= maximum")
    return lower, upper


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, context: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{context}: missing required columns {missing}")


def _grid_coordinates(frame: pd.DataFrame, *, key: str) -> np.ndarray:
    if frame.empty:
        raise ValueError(f"{key}: grid frame is empty")
    _require_columns(frame, ("Longitude", "Latitude"), context=key)
    coordinates = frame[["Longitude", "Latitude"]].to_numpy(dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{key}: grid coordinates must be finite")
    return coordinates


def _validate_spatial_order(spatial_order: np.ndarray | None, row_count: int) -> np.ndarray | None:
    if spatial_order is None:
        return None
    order = np.asarray(spatial_order)
    if order.ndim != 1 or order.shape[0] != row_count:
        raise ValueError("spatial_order must be a one-dimensional full-row permutation")
    if not np.issubdtype(order.dtype, np.integer):
        raise TypeError("spatial_order must contain integers")
    order = order.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(order), np.arange(row_count)):
        raise ValueError("spatial_order must be a permutation of all row indices")
    return order


def _coordinate_tensor(
    tensor: torch.Tensor,
    day: int,
    time_key: str,
    *,
    allow_missing: bool = False,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"day {day}, time {time_key}: expected a torch.Tensor")
    if tensor.ndim != 2 or tensor.shape[1] < 2:
        raise ValueError(f"day {day}, time {time_key}: tensor must have at least two columns")
    coordinates = tensor[:, :2]
    if allow_missing and torch.isinf(coordinates).any():
        raise ValueError(f"day {day}, time {time_key}: coordinates cannot contain infinity")
    if not allow_missing and not torch.isfinite(coordinates).all():
        raise ValueError(f"day {day}, time {time_key}: coordinates must be finite")
    return tensor


def _validate_aggregate_order(
    aggregate: torch.Tensor,
    expected: torch.Tensor,
    *,
    day: int,
    name: str,
    allow_missing_coordinates: bool,
) -> None:
    """Require an aggregate to equal the hourly time-major concatenation."""

    if not isinstance(aggregate, torch.Tensor):
        raise TypeError(f"day {day}: {name} must be a torch.Tensor")
    if aggregate.ndim != 2 or aggregate.shape[1] < 2:
        raise ValueError(f"day {day}: {name} must be two-dimensional with at least two columns")
    if expected.ndim != 2 or expected.shape[1] < 2:
        raise ValueError(f"day {day}: hourly tensors must have at least two columns")
    if aggregate.shape != expected.shape:
        raise ValueError(
            f"day {day}: {name} shape {tuple(aggregate.shape)} must match the hourly "
            f"concatenation shape {tuple(expected.shape)}"
        )

    # Coordinates identify spatial rows.  When available, time additionally
    # identifies the time-major block order even though every block shares the
    # same regular grid.
    identity_columns = [0, 1, 3] if min(aggregate.shape[1], expected.shape[1]) >= 4 else [0, 1]
    actual_identity = aggregate[:, identity_columns].to(dtype=torch.float64)
    expected_identity = expected[:, identity_columns].to(
        device=aggregate.device,
        dtype=torch.float64,
    )
    if torch.isinf(actual_identity).any():
        raise ValueError(f"day {day}: {name} identity columns cannot contain infinity")
    if not allow_missing_coordinates and not torch.isfinite(actual_identity).all():
        raise ValueError(f"day {day}: {name} identity columns must be finite")
    if not torch.allclose(
        actual_identity,
        expected_identity,
        rtol=1e-7,
        atol=1e-10,
        equal_nan=allow_missing_coordinates,
    ):
        raise ValueError(
            f"day {day}: {name} coordinate/time order must match the hourly concatenation"
        )

    actual_values = aggregate.to(dtype=torch.float64)
    expected_values = expected.to(device=aggregate.device, dtype=torch.float64)
    if not torch.allclose(
        actual_values,
        expected_values,
        rtol=1e-7,
        atol=1e-10,
        equal_nan=True,
    ):
        raise ValueError(f"day {day}: {name} values must match the hourly concatenation")
