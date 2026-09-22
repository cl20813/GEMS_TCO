"""Compatibility checks for the clearer ``GEMS_TCO.data`` import paths."""

import unittest

from GEMS_TCO import data_loader as legacy_loading
from GEMS_TCO import data_preprocess as legacy_preprocessing
from GEMS_TCO.data import loading, preprocessing


class DataCompatibilityTests(unittest.TestCase):
    def test_loading_exports_are_the_existing_implementations(self):
        self.assertIs(
            loading.load_data_dynamic_processed,
            legacy_loading.load_data_dynamic_processed,
        )
        self.assertIs(loading.exact_location_filter, legacy_loading.exact_location_filter)

    def test_preprocessing_exports_are_the_existing_implementations(self):
        self.assertIs(preprocessing.GemsORITocsvHour, legacy_preprocessing.GemsORITocsvHour)
        self.assertIs(preprocessing.file_path_list, legacy_preprocessing.file_path_list)
        self.assertIs(preprocessing.MonthAggregatedCSV, legacy_preprocessing.MonthAggregatedCSV)
        self.assertIs(
            preprocessing.MonthAggregatedHashmap,
            legacy_preprocessing.MonthAggregatedHashmap,
        )
        self.assertIs(
            preprocessing.center_matching_hour,
            legacy_preprocessing.center_matching_hour,
        )


if __name__ == "__main__":
    unittest.main()
