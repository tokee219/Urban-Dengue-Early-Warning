import pandas as pd
from predict import predict_ward

# Latest 5 consecutive weekly records for ONE ward (oldest first).
# The last row is the current week. Example values from the dataset.
recent = pd.DataFrame({
    "date": ["2021-09-06", "2021-09-13", "2021-09-20", "2021-09-27", "2021-10-04"],
    "dengue_cases": [12, 9, 15, 4, 5],
    "rainfall_mm": [84.46, 93.65, 91.43, 111.98, 0.00],
    "avg_temp_c": [27.86, 26.71, 28.11, 24.12, 24.66],
    "relative_humidity_pct": [87.46, 91.74, 82.94, 92.63, 65.84],
})

result = predict_ward("CCC_Ward_01", recent)

print("Ward: CCC_Ward_01 | current week: 2021-10-04")
for horizon, info in result.items():
    print(f"  {horizon} weeks ahead: {info['predicted_cases']} cases -> {info['risk']}")
