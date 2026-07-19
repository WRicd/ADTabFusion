# Transition Error Analysis

- Locked temporal-test predictions: 893
- Errors: 104
- Validation-frozen confidence threshold: 0.7431
- High-confidence errors: 70

## Required Error Groups

| Group | Rows |
|---|---:|
| MCI to AD missed conversions | 46 |
| MCI to MCI false progression | 20 |
| CN progression missed | 26 |
| AD non AD reversions | 1 |

## Error Distribution

### By Future Class

FUTURE_DX
AD     49
MCI    45
CN     10

### By Source Diagnosis

SOURCE_DX
MCI    76
CN     26
AD      2

### By Forecast Horizon

count    104.000000
mean      21.511294
std       10.570190
min        6.012320
25%       12.279261
50%       20.648871
75%       29.034908
max       48.624230

### Confidence

count    104.000000
mean       0.822412
std        0.161466
min        0.502837
25%        0.663669
50%        0.886809
75%        0.961548
max        0.997006
