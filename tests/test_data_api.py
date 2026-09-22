from __future__ import annotations

import calendar
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from GEMS_TCO import data
from GEMS_TCO.data import loading, preprocessing


class DataApiTests(unittest.TestCase):
    def test_public_exports_are_canonical(self):
        expected = {
            "CoordinateDeviationFilter",
            "GEMSOrbitReader",
            "GeographicBounds",
            "MonthlyOrbitAggregator",
            "ProcessedDataLoader",
            "build_center_grid",
            "group_by_orbit",
            "monthly_orbit_paths",
        }
        self.assertEqual(set(data.__all__), expected)
        self.assertIs(data.ProcessedDataLoader, loading.ProcessedDataLoader)
        self.assertIs(data.CoordinateDeviationFilter, loading.CoordinateDeviationFilter)
        self.assertIs(data.GEMSOrbitReader, preprocessing.GEMSOrbitReader)

    def test_monthly_paths_honor_leap_years(self):
        paths = preprocessing.monthly_orbit_paths("/data", 2024, 2)
        self.assertEqual(len(paths), calendar.monthrange(2024, 2)[1] * 8)
        self.assertEqual(paths[0], Path("/data/2024020129/20240201_0045.nc"))
        self.assertEqual(paths[-1], Path("/data/2024020129/20240229_0745.nc"))

    def test_center_grid_exposes_calibration(self):
        bounds = preprocessing.GeographicBounds(-0.1, 0.1, 120.0, 120.2)
        plain = preprocessing.build_center_grid(bounds, latitude_step=0.1, longitude_step=0.1)
        calibrated = preprocessing.build_center_grid(
            bounds,
            latitude_step=0.1,
            longitude_step=0.1,
            latitude_edge_offset=0.001,
            longitude_edge_offset=0.002,
            latitude_drift_per_column=0.002,
        )
        self.assertEqual(list(plain.columns), ["Latitude", "Longitude"])
        self.assertNotEqual(plain.iloc[0]["Latitude"], calibrated.iloc[0]["Latitude"])
        self.assertNotEqual(plain.iloc[0]["Longitude"], calibrated.iloc[0]["Longitude"])
        self.assertAlmostEqual(calibrated.iloc[0]["Latitude"], 0.099)
        self.assertAlmostEqual(calibrated.iloc[0]["Longitude"], 120.198)

    def test_center_grid_reproduces_historical_calibration_recipe(self):
        bounds = preprocessing.GeographicBounds(5.0, 5.2, 120.0, 120.2)
        grid = preprocessing.build_center_grid(
            bounds,
            latitude_step=0.044,
            longitude_step=0.063,
            latitude_edge_offset=0.0002,
            longitude_edge_offset=0.0002,
            latitude_drift_per_column=0.00012,
        )
        self.assertAlmostEqual(grid.iloc[0]["Latitude"], 5.1998, places=12)
        self.assertAlmostEqual(grid.iloc[0]["Longitude"], 120.1998, places=12)
        self.assertAlmostEqual(
            grid.iloc[1]["Latitude"] - grid.iloc[0]["Latitude"],
            0.00012,
            places=12,
        )

    def test_group_by_orbit_does_not_mutate_input(self):
        frame = pd.DataFrame(
            {
                "Time": ["2024-07-01 00:45:30", "2024-07-01 00:45:50"],
                "ColumnAmountO3": [1.0, 2.0],
            }
        )
        grouped = preprocessing.group_by_orbit(frame)
        self.assertNotIn("Orbit", frame.columns)
        self.assertEqual(list(grouped), ["y24m07day01_hm00:45"])
        self.assertEqual(len(grouped["y24m07day01_hm00:45"]), 2)

    def test_orbit_reader_aligns_groups_by_dimension_indices(self):
        bounds = preprocessing.GeographicBounds(-90.0, 90.0, -180.0, 180.0)
        reader = preprocessing.GEMSOrbitReader(bounds)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "orbit.nc"
            scan = np.array([10, 20])
            pixel = np.array([1, 2, 3])
            expected = 100.0 * scan[:, None] + pixel[None, :]
            location = xr.Dataset(
                {
                    "Latitude": (("scan", "pixel"), expected / 1000.0),
                    "Longitude": (("scan", "pixel"), 120.0 + expected / 1000.0),
                    "Time": (("scan", "pixel"), np.full(expected.shape, 477_700.0)),
                },
                coords={"scan": scan, "pixel": pixel},
            )
            observations = xr.Dataset(
                {
                    "ColumnAmountO3": (("pixel", "scan"), expected.T),
                    "FinalAlgorithmFlags": (
                        ("pixel", "scan"),
                        np.zeros(expected.T.shape, dtype=np.int16),
                    ),
                },
                coords={"pixel": pixel, "scan": scan},
            )
            location.to_netcdf(path, group="Geolocation Fields", engine="netcdf4")
            observations.to_netcdf(
                path,
                group="Data Fields",
                mode="a",
                engine="netcdf4",
            )

            frame = reader.read(path)

        np.testing.assert_array_equal(frame["ColumnAmountO3"].to_numpy(), expected.ravel())
        np.testing.assert_allclose(
            frame["Latitude"].to_numpy(),
            frame["ColumnAmountO3"].to_numpy() / 1000.0,
        )

    def test_orbit_reader_rejects_different_dimension_index_sets(self):
        bounds = preprocessing.GeographicBounds(-90.0, 90.0, -180.0, 180.0)
        reader = preprocessing.GEMSOrbitReader(bounds)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "orbit.nc"
            location = xr.Dataset(
                {
                    "Latitude": (("scan",), [0.0, 1.0]),
                    "Longitude": (("scan",), [120.0, 121.0]),
                    "Time": (("scan",), [477_700.0, 477_700.0]),
                },
                coords={"scan": [0, 1]},
            )
            observations = xr.Dataset(
                {
                    "ColumnAmountO3": (("scan",), [300.0, 301.0]),
                    "FinalAlgorithmFlags": (("scan",), [0, 0]),
                },
                coords={"scan": [0, 2]},
            )
            location.to_netcdf(path, group="Geolocation Fields", engine="netcdf4")
            observations.to_netcdf(
                path,
                group="Data Fields",
                mode="a",
                engine="netcdf4",
            )

            with self.assertRaisesRegex(ValueError, "different dimension-index sets"):
                reader.read(path)

    def test_orbit_aggregator_rejects_noninteger_quality_flags(self):
        bounds = preprocessing.GeographicBounds(-90.0, 90.0, -180.0, 180.0)
        reader = preprocessing.GEMSOrbitReader(bounds)
        with self.assertRaisesRegex(TypeError, "quality_flags"):
            preprocessing.MonthlyOrbitAggregator(reader, acceptable_quality_flags=(0, 1.5))
        with self.assertRaisesRegex(TypeError, "quality_flags"):
            preprocessing.MonthlyOrbitAggregator(reader, acceptable_quality_flags=(False, 2))


class ProcessedDataLoaderTests(unittest.TestCase):
    @staticmethod
    def _frame(hour: float, ozone: tuple[float, float] = (10.0, 12.0)) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Latitude": [0.0, 1.0],
                "Longitude": [120.0, 121.0],
                "Source_Latitude": [0.01, 1.01],
                "Source_Longitude": [120.01, 121.01],
                "ColumnAmountO3": ozone,
                "Hours_elapsed": [hour, hour],
            }
        )

    def test_build_model_tensors_preserves_documented_layout(self):
        loader = loading.ProcessedDataLoader("/unused")
        frames = {"h0": self._frame(477_700), "h1": self._frame(477_701)}
        hourly, combined = loader.build_model_tensors(
            frames,
            ozone_mean=11.0,
            time_slice=(0, 2),
            spatial_order=np.array([1, 0]),
            use_source_coordinates=False,
        )
        self.assertEqual(combined.shape, (4, 11))
        np.testing.assert_allclose(hourly["h0"][:, :4], [[1, 121, 1, 0], [0, 120, -1, 0]])
        np.testing.assert_array_equal(hourly["h0"][:, 4:], torch.zeros((2, 7)))
        np.testing.assert_array_equal(
            hourly["h1"][:, 4:],
            torch.tensor([[1, 0, 0, 0, 0, 0, 0]]).repeat(2, 1),
        )

    def test_build_model_tensors_rejects_invalid_order(self):
        loader = loading.ProcessedDataLoader("/unused")
        with self.assertRaises(ValueError):
            loader.build_model_tensors({"h0": self._frame(477_700)}, spatial_order=np.array([0, 0]))

    def test_build_model_tensors_rejects_ambiguous_multi_day_and_invalid_types(self):
        loader = loading.ProcessedDataLoader("/unused")
        nine_frames = {
            f"h{slot}": self._frame(477_700 + slot)
            for slot in range(loading.GEMS_TIME_SLOTS_PER_DAY + 1)
        }
        with self.assertRaisesRegex(ValueError, "one day at a time"):
            loader.build_model_tensors(nine_frames)
        with self.assertRaisesRegex(ValueError, "one day's time slices"):
            loader.build_model_tensors({})
        with self.assertRaisesRegex(TypeError, "time_slice entries"):
            loader.build_model_tensors({"h0": self._frame(477_700)}, time_slice=(False, 1))
        with self.assertRaisesRegex(TypeError, "floating-point"):
            loader.build_model_tensors({"h0": self._frame(477_700)}, dtype=torch.int64)
        with self.assertRaisesRegex(TypeError, "use_source_coordinates"):
            loader.build_model_tensors(
                {"h0": self._frame(477_700)}, use_source_coordinates="source"
            )

    def test_time_slice_preserves_nominal_hour_indicators(self):
        loader = loading.ProcessedDataLoader("/unused")
        frames = {
            f"h{slot}": self._frame(477_700 + slot)
            for slot in range(loading.GEMS_TIME_SLOTS_PER_DAY)
        }
        hourly, _ = loader.build_model_tensors(frames, time_slice=(2, 5))

        self.assertEqual(list(hourly), ["h2", "h3", "h4"])
        np.testing.assert_array_equal(
            hourly["h2"][:, 4:],
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]).repeat(2, 1),
        )
        np.testing.assert_array_equal(
            hourly["h4"][:, 4:],
            torch.tensor([[0, 0, 0, 1, 0, 0, 0]]).repeat(2, 1),
        )

        with self.assertRaisesRegex(ValueError, "time_slice"):
            loader.build_model_tensors(frames, time_slice=(0, 9))

    def test_missing_response_may_carry_missing_source_coordinates(self):
        frame = self._frame(477_700)
        frame.loc[1, ["Source_Latitude", "Source_Longitude", "ColumnAmountO3"]] = np.nan
        loader = loading.ProcessedDataLoader("/unused")
        hourly, _ = loader.build_model_tensors({"h0": frame})
        self.assertTrue(torch.isnan(hourly["h0"][1, :3]).all())

        frame.loc[1, "ColumnAmountO3"] = 12.0
        with self.assertRaisesRegex(ValueError, "observed responses require finite coordinates"):
            loader.build_model_tensors({"h0": frame})

    def test_coordinate_filter_uses_all_times_and_validates_aggregate_shape(self):
        grid = {
            0: {
                "h0": torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]]),
                "h1": torch.tensor([[0.0, 0.0, 3.0], [1.0, 1.0, 4.0]]),
            }
        }
        source = {
            0: {
                "h0": torch.tensor([[0.01, 0.01, 1.0], [1.01, 1.01, 2.0]]),
                "h1": torch.tensor([[0.01, 0.01, 3.0], [1.50, 1.01, 4.0]]),
            }
        }
        aggregate_grid = {0: torch.cat(tuple(grid[0].values()))}
        aggregate_source = {0: torch.cat(tuple(source[0].values()))}

        hourly, combined = loading.CoordinateDeviationFilter.filter(
            grid,
            source,
            aggregate_grid,
            aggregate_source,
            latitude_tolerance=0.025,
            longitude_tolerance=0.04,
        )
        self.assertEqual(hourly[0]["h0"].shape[0], 1)
        self.assertEqual(hourly[0]["h1"].shape[0], 1)
        self.assertEqual(combined[0].shape[0], 2)

        with self.assertRaises(ValueError):
            loading.CoordinateDeviationFilter.filter(
                grid, source, {0: aggregate_grid[0][:-1]}, aggregate_source
            )

        permuted_grid = {0: dict(grid[0])}
        permuted_grid[0]["h1"] = permuted_grid[0]["h1"].flip(0)
        with self.assertRaisesRegex(ValueError, "row order"):
            loading.CoordinateDeviationFilter.filter(
                permuted_grid,
                source,
                {0: torch.cat(tuple(permuted_grid[0].values()))},
                aggregate_source,
            )

        permuted_aggregate_source = {0: aggregate_source[0].flip(0)}
        with self.assertRaisesRegex(ValueError, "coordinate/time order"):
            loading.CoordinateDeviationFilter.filter(
                grid,
                source,
                aggregate_grid,
                permuted_aggregate_source,
            )

        changed_aggregate_grid = {0: aggregate_grid[0].clone()}
        changed_aggregate_grid[0][0, 2] = -999.0
        with self.assertRaisesRegex(ValueError, "values must match"):
            loading.CoordinateDeviationFilter.filter(
                grid,
                source,
                changed_aggregate_grid,
                aggregate_source,
            )

    def test_loader_reads_only_existing_months(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "pickle_2024"
            target.mkdir()
            frames = {"orbit": self._frame(477_700)}
            pd.to_pickle(frames, target / "tco_grid_24_07.pkl")
            loader = loading.ProcessedDataLoader(root)
            loaded = loader.load_grid_frames((2024,), (7, 8))
            self.assertEqual(list(loaded), ["2024_07_orbit"])

    def test_monthly_loader_rejects_an_all_missing_request(self):
        with tempfile.TemporaryDirectory() as directory:
            loader = loading.ProcessedDataLoader(directory)
            with self.assertRaisesRegex(FileNotFoundError, "no processed grid files"):
                loader.load_monthly_grids(years=(2024,), months=(7,))


if __name__ == "__main__":
    unittest.main()
