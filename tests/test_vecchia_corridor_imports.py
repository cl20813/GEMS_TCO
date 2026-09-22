"""Compatibility checks for the reorganized corridor-neighbor imports."""

import importlib
import unittest


class CorridorImportCompatibilityTests(unittest.TestCase):
    def test_canonical_corridor_modules_reexport_legacy_api(self) -> None:
        module_pairs = [
            (
                "GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag432",
                "GEMS_TCO.vecchia.corridor_neighbors.local_lag432",
            ),
            (
                "GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag643",
                "GEMS_TCO.vecchia.corridor_neighbors.amarel_lag643",
            ),
        ]

        for legacy_name, canonical_name in module_pairs:
            with self.subTest(canonical_name=canonical_name):
                legacy = importlib.import_module(legacy_name)
                canonical = importlib.import_module(canonical_name)

                self.assertEqual(canonical.__all__, legacy.__all__)
                for public_name in legacy.__all__:
                    self.assertIs(
                        getattr(canonical, public_name),
                        getattr(legacy, public_name),
                    )

    def test_local_and_amarel_conditioning_budgets_remain_distinct(self) -> None:
        local = importlib.import_module(
            "GEMS_TCO.vecchia.corridor_neighbors.local_lag432"
        )
        amarel = importlib.import_module(
            "GEMS_TCO.vecchia.corridor_neighbors.amarel_lag643"
        )

        self.assertEqual(local.LAG_COUNTS, (4, 3, 2))
        self.assertEqual(amarel.LAG_COUNTS, (6, 4, 3))
        self.assertEqual(local.model_spec()["lag_pattern"], "4/3/2")
        self.assertEqual(amarel.model_spec()["lag_pattern"], "6/4/3")


if __name__ == "__main__":
    unittest.main()
