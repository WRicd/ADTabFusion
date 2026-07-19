"""
TODO:
After inventory confirms available tables, build a subject-visit-level master table
using RID + VISCODE + VISDATE/EXAMDATE/SCANDATE.

Potential sources:
- diagnosis
- demographics
- cognitive
- apoe
- MRI measurements
- PET measurements
- biofluid / CSF

This stage intentionally does not implement cross-table merging.
"""


def main() -> None:
    raise SystemExit(
        "Not implemented: run inspect_raw_adni_csvs.py and confirm the inventory first."
    )


if __name__ == "__main__":
    main()
