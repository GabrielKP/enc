from pathlib import Path

import pandas as pd
from great_tables import GT

from encoders.utils import ROOT_FOLDER

REGRESSION_CSV = Path(ROOT_FOLDER, "data", "regression_params.csv")
PRACTICES_CSV = Path(ROOT_FOLDER, "data", "software_practices.csv")

# table containing regression parameters
df_regression_params = pd.read_csv(REGRESSION_CSV)

gt_regression_params = GT(df_regression_params, rowname_col="Parameter").tab_stubhead(
    "Parameter"
)

# rse table
df_software_practices = pd.read_csv(PRACTICES_CSV)

df_software_practices_styled = df_software_practices.copy()
df_software_practices_styled["Readability"] = df_software_practices_styled[
    "Readability"
].replace({"Yes": "✅", "No": ""})
df_software_practices_styled["Reuse"] = df_software_practices_styled["Reuse"].replace(
    {"Yes": "✅", "No": ""}
)
df_software_practices_styled["Resilience"] = df_software_practices_styled[
    "Resilience"
].replace({"Yes": "✅", "No": ""})

COLORS = ["#ABE0B0", "#FFBF00", "#FF7D00"]
COLORS_LABELS = ["mid", "low", "high"]
gt_software_practices = (
    GT(df_software_practices_styled, rowname_col="Practice")
    .tab_stubhead("Practice")
    .tab_header(title="Software Engineering Practices in Research")
    .tab_spanner(
        label="Reproducibility Gain (3R)",
        columns=["Readability", "Reuse", "Resilience"],
    )
    .tab_spanner(label="Costs", columns=["Technical_Overhead", "Time_Investment"])
    .data_color(
        columns=["Technical_Overhead", "Time_Investment"],
        palette=COLORS,
        domain=COLORS_LABELS,
    )
    .cols_align(
        align="center",
        columns=[
            "Readability",
            "Reuse",
            "Resilience",
            "Technical_Overhead",
            "Time_Investment",
            "Example_Tools",
        ],
    )
    .cols_label(
        Time_Investment="Time Investment",
        Technical_Overhead="Technical Overhead",
        Example_Tools="Example Tools",
    )
)
