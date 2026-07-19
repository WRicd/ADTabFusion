# Phase C Verification Gate

Overall status: **PASS**

| Check | Status | Detail |
|---|---|---|
| primary_model_exists | PASS | outputs\phase_c\models\primary_pipeline.joblib |
| primary_manifest_exists | PASS | outputs\phase_c\manifests\primary_model_manifest.json |
| primary_model_hash | PASS | 9c381d831126920d7bba7322786a97bfed13902561719077deef7cea2ef675e7 |
| primary_config_hash | PASS | 8a2545592bc3dd22c3da5cb833589e22fff1478933033c2d477c5f5d3db53c80 |
| primary_feature_order | PASS | 20 features |
| sensitivity_model_exists | PASS | outputs\phase_c\models\sensitivity_pipeline.joblib |
| sensitivity_manifest_exists | PASS | outputs\phase_c\manifests\sensitivity_model_manifest.json |
| sensitivity_model_hash | PASS | 419904735ae0ae40b48d9cfde38dbb7f6304fa37ede5244008f6431c8b8cb240 |
| sensitivity_config_hash | PASS | 8a2545592bc3dd22c3da5cb833589e22fff1478933033c2d477c5f5d3db53c80 |
| sensitivity_feature_order | PASS | 69 features |
| horizon_aware_model_exists | PASS | outputs\phase_c\models\horizon_aware_pipeline.joblib |
| horizon_aware_manifest_exists | PASS | outputs\phase_c\manifests\horizon_aware_manifest.json |
| horizon_aware_model_hash | PASS | c9d96f64b7dd45ac120d272ab585dcd8a246fe18245eb7c2e552d421dc16048f |
| horizon_aware_config_hash | PASS | fe7bc511ac7fa75ce909c2c70661bb3cc586e1de7a9005a5e3a9b33426fc34fb |
| horizon_aware_feature_order | PASS | 21 features |
| primary_d3_reproduced | PASS | 896 rows |
| primary_probability_simplex | PASS | sum=1 |
| sensitivity_d3_reproduced | PASS | 896 rows |
| sensitivity_probability_simplex | PASS | sum=1 |
| matched_rows | PASS | 210 |
| matched_subjects | PASS | 197 |
| first_follow_up_unique | PASS | 197 |
| horizon_d3_d4_reproduced | PASS | 210 rows |
| artifact_creation_order | PASS | evaluation created after frozen models |
