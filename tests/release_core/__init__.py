"""Keep pytest's test package from shadowing the clean production package."""

from pathlib import Path

_production_package = Path(__file__).resolve().parents[2] / "release_core"
if str(_production_package) not in __path__:
    __path__.insert(0, str(_production_package))
