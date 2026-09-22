from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path

import GEMS_TCO
from GEMS_TCO.data import CoordinateDeviationFilter, ProcessedDataLoader


class PublicNamespaceTests(unittest.TestCase):
    def test_root_namespace_is_small_and_dependency_free(self):
        self.assertEqual(GEMS_TCO.__all__, ["__version__"])
        self.assertEqual(GEMS_TCO.__version__, "0.3.0")

    def test_data_implementations_live_in_the_data_subpackage(self):
        self.assertEqual(ProcessedDataLoader.__module__, "GEMS_TCO.data.loading")
        self.assertEqual(CoordinateDeviationFilter.__module__, "GEMS_TCO.data.loading")

    def test_removed_data_compatibility_modules_are_not_importable(self):
        self.assertIsNone(importlib.util.find_spec("GEMS_TCO.data_loader"))
        self.assertIsNone(importlib.util.find_spec("GEMS_TCO.data_preprocess"))


class SourceTreeHygieneTests(unittest.TestCase):
    def test_package_does_not_mutate_python_path_or_embed_user_home_paths(self):
        package_root = Path(__file__).parents[1] / "src" / "GEMS_TCO"
        forbidden = (
            "sys.path.append",
            "sys.path.insert",
            "/Users/joonwonlee",
            "/home/jl2815",
            "/home/ec2-user",
            "C:\\Users",
        )

        violations = []
        for path in package_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                if marker in text:
                    violations.append(f"{path.relative_to(package_root)}: {marker}")

        self.assertEqual(violations, [])

    def test_package_tree_has_only_canonical_top_level_modules(self):
        package_root = Path(__file__).parents[1] / "src" / "GEMS_TCO"
        top_level_modules = {path.name for path in package_root.glob("*.py")}
        self.assertEqual(top_level_modules, {"__init__.py", "orderings.py"})

    def test_generated_native_artifacts_are_not_tracked_as_source(self):
        repository_root = Path(__file__).parents[1]
        generated_suffixes = {".so", ".pyd", ".obj", ".lib", ".exp", ".pyc"}
        result = subprocess.run(
            ["git", "ls-files", "src/GEMS_TCO"],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            self.skipTest("source checkout has no readable Git index")
        generated = [
            path
            for path in result.stdout.splitlines()
            if Path(path).suffix in generated_suffixes and (repository_root / path).exists()
        ]
        self.assertEqual(generated, [])


if __name__ == "__main__":
    unittest.main()
