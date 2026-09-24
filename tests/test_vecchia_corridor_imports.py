"""Public-import checks for the supported corridor-neighbor Vecchia API."""

import importlib
import importlib.util
import inspect
import unittest


class CorridorImportTests(unittest.TestCase):
    def test_lag_conditioning_budgets_remain_distinct(self) -> None:
        lag432 = importlib.import_module("GEMS_TCO.vecchia.corridor_neighbors.corridor_lag432")
        lag643 = importlib.import_module("GEMS_TCO.vecchia.corridor_neighbors.corridor_lag643")
        directional432 = importlib.import_module(
            "GEMS_TCO.vecchia.corridor_neighbors.directional_lag432"
        )
        directional643 = importlib.import_module(
            "GEMS_TCO.vecchia.corridor_neighbors.directional_lag643"
        )

        self.assertEqual(lag432.LAG_COUNTS, (4, 3, 2))
        self.assertEqual(lag643.LAG_COUNTS, (6, 4, 3))
        self.assertEqual(lag432.SPEC_NAME, "corridor_width_4x4_lag432")
        self.assertEqual(lag643.SPEC_NAME, "corridor_width_4x4_lag643")
        self.assertEqual(lag432.model_spec()["lag_pattern"], "4/3/2")
        self.assertEqual(lag643.model_spec()["lag_pattern"], "6/4/3")
        for spec in (
            lag432.model_spec(),
            lag643.model_spec(),
            directional432.directional_model_spec(0.1, -0.2),
            directional643.directional_model_spec(0.1, -0.2),
        ):
            self.assertIn("conditioning_mode", spec)
            self.assertNotIn("strategy", spec)

        for spec in (
            directional432.directional_model_spec(0.1, -0.2),
            directional643.directional_model_spec(0.1, -0.2),
        ):
            self.assertEqual(spec["conditioning_mode"], "directional_corridor_width")
            self.assertEqual(spec["past_step_lat"], -0.1)
            self.assertEqual(spec["past_step_lon"], 0.2)

    def test_active_variants_import_from_canonical_namespace(self) -> None:
        modules_and_classes = {
            "directional_lag432": ("DirectionalLag432CorridorVecchia",),
            "directional_lag643": ("DirectionalLag643CorridorVecchia",),
            "generalized_cauchy": (
                "GeneralizedCauchyLag643CorridorVecchia",
                "GeneralizedCauchyLag432CorridorVecchia",
            ),
            "spline": (
                "SplineMaternLag643CorridorVecchia",
                "NoNuggetSplineMaternLag643CorridorVecchia",
            ),
            "separable_exponential": (
                "AdvectedSeparableExponentialLag643CorridorVecchia",
                "NoNuggetAdvectedSeparableExponentialLag643CorridorVecchia",
            ),
        }

        for module_name, class_names in modules_and_classes.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(
                    f"GEMS_TCO.vecchia.corridor_neighbors.{module_name}"
                )
                for class_name in class_names:
                    self.assertTrue(hasattr(module, class_name))

    def test_corridor_package_exports_canonical_models(self) -> None:
        package = importlib.import_module("GEMS_TCO.vecchia.corridor_neighbors")
        expected = {
            "corridor_lag432",
            "corridor_lag643",
            "directional_lag432",
            "directional_lag643",
            "Lag432CorridorVecchia",
            "DirectionalLag432CorridorVecchia",
            "Lag643CorridorVecchia",
            "DirectionalLag643CorridorVecchia",
            "GeneralizedCauchyLag432CorridorVecchia",
            "NoNuggetGeneralizedCauchyLag432CorridorVecchia",
            "GeneralizedCauchyLag643CorridorVecchia",
            "NoNuggetGeneralizedCauchyLag643CorridorVecchia",
            "SplineMaternLag643CorridorVecchia",
            "NoNuggetSplineMaternLag643CorridorVecchia",
            "separable_exponential",
            "AdvectedSeparableExponentialLag643CorridorVecchia",
            "NoNuggetAdvectedSeparableExponentialLag643CorridorVecchia",
        }
        self.assertEqual(set(package.__all__), expected)
        for name in expected:
            self.assertIsNotNone(getattr(package, name))

    def test_prototype_public_names_and_modules_are_removed(self) -> None:
        package = importlib.import_module("GEMS_TCO.vecchia.corridor_neighbors")
        retired_names = (
            "RealDataCorridorWidth4x4Lag432VecchiaFit",
            "RealDataCorridorWidth4x4Lag643VecchiaFit",
            "AdaptedRealDataCorridorWidth4x4Lag643VecchiaFit",
        )
        for name in retired_names:
            self.assertFalse(hasattr(package, name))
        fixed_432 = importlib.import_module("GEMS_TCO.vecchia.corridor_neighbors.corridor_lag432")
        self.assertFalse(hasattr(fixed_432, "DirectionalLag432CorridorVecchia"))
        self.assertFalse(hasattr(fixed_432, "directional_model_spec"))
        for module_name in ("local_lag432", "amarel_lag643", "adapted_lag643"):
            self.assertIsNone(
                importlib.util.find_spec(f"GEMS_TCO.vecchia.corridor_neighbors.{module_name}")
            )

    def test_family_variant_signatures_expose_only_effective_options(self) -> None:
        from GEMS_TCO.vecchia.corridor_neighbors.generalized_cauchy import (
            GeneralizedCauchyLag643CorridorVecchia,
        )
        from GEMS_TCO.vecchia.corridor_neighbors.spline import SplineMaternLag643CorridorVecchia

        for model_class in (
            GeneralizedCauchyLag643CorridorVecchia,
            SplineMaternLag643CorridorVecchia,
        ):
            with self.subTest(model=model_class.__name__):
                parameters = inspect.signature(model_class).parameters
                self.assertIn("reference_advec_lon_abs", parameters)
                self.assertIn("second_lag_stride", parameters)
                self.assertNotIn("daily_stride", parameters)
                self.assertNotIn("lag1_lon_offset", parameters)
                self.assertNotIn("lag2_lon_offset", parameters)
                self.assertFalse(
                    any(
                        parameter.kind is inspect.Parameter.VAR_KEYWORD
                        for parameter in parameters.values()
                    )
                )

    def test_private_corridor_base_has_no_retired_strategy_switches(self) -> None:
        geometry_module = importlib.import_module("GEMS_TCO.vecchia.corridor_neighbors._geometry")
        self.assertFalse(hasattr(geometry_module, "STRATEGIES"))
        self.assertFalse(hasattr(geometry_module, "_StrategyClusterVecchia"))
        self.assertIsNone(importlib.util.find_spec("GEMS_TCO.vecchia.corridor_neighbors._strategy"))

        parameters = inspect.signature(geometry_module._CorridorClusterVecchia).parameters
        for retired_name in (
            "strategy",
            "lag1_keep_fraction",
            "lag2_keep_fraction",
        ):
            self.assertNotIn(retired_name, parameters)


if __name__ == "__main__":
    unittest.main()
