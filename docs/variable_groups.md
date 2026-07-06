# Variable Groups

This stage uses only fields available in the local TADPOLE D1_D2 CSV.

## Active Modalities

### demographic

- `AGE`
- `PTGENDER`
- `PTEDUCAT`

### cognitive

- `MMSE`
- `ADAS11`
- `ADAS13`
- `CDRSB`
- `RAVLT_immediate`
- `RAVLT_learning`
- `RAVLT_forgetting`
- `RAVLT_perc_forgetting`
- `FAQ_bl`

`FAQ_bl` is a baseline-only FAQ covariate. It is not treated as current-visit `FAQ`.

### mri_derived

- `Ventricles`
- `Hippocampus`
- `WholeBrain`
- `Entorhinal`
- `Fusiform`
- `MidTemp`
- `ICV`

### genetic

- `APOE4`

## Unavailable In Current CSV

- PET: `FDG`, `AV45`, `PIB`
- CSF: `ABETA`, `TAU`, `PTAU`
- Demographic extensions: `PTETHCAT`, `PTRACCAT`

Unavailable modalities are not shown as active dashboard modalities and are not used in missing-modality experiments.

