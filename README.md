# README

Xingyu Dai (Star19950818@foxmail.com)

## Step 1. Software and Package Requirements

The software versions used in this study are as follows:

- Python: 3.14.6
- R: 4.4.3

These versions correspond to the project environments inspected on October 30, 2026.

Required Python packages:

- arch 8.0.0
- numpy 2.5.3
- pandas 3.0.6
- portion 2.6.2
- Pygments 2.21.0
- PySDKit 0.5.0
- scikit-learn 1.9.1
- scipy 1.18.1

Required R packages:

- dplyr 1.1.4
- ggplot2 4.0.1
- ggrepel 0.9.8
- ggthemes 5.2.0
- grid 4.4.3 (included with R)
- gridExtra 2.3
- patchwork 1.3.2
- pheatmap 1.0.13
- RColorBrewer 1.1-3
- reshape2 1.4.4
- scales 1.4.0
- tidyr 1.3.1
- tidyverse 2.0.0
- xts 0.14.1

Third-party Python packages imported by each script:

- `Decomposition_Algorithm_03.py`: numpy, Pygments, PySDKit
- `Economics_Application_04.py`: arch, numpy, pandas, scipy
- `Interval_Forecast_Fun_02.py`: arch, numpy, portion, scikit-learn, scipy
- `Main_01.py`: numpy, pandas
- `save.py`: standard-library modules only

Only packages explicitly referenced by the project scripts are listed. Standard-library modules, local project modules, and transitive dependencies are omitted.

##  Step 2. Python Environment Setup

Open the following four Python files in the project:

- `Main_01.py`
- `Interval_Forecast_Fun_02.py`
- `Decomposition_Algorithm_03.py`
- `Economics_Application_04.py`

Please note that the Python scripts in this project are designed to be executed interactively in PyCharm by selecting and running specific sections of code. Do not execute an entire Python script as a standalone program. In `Main_01.py`, modify line 9: `os.chdir(r"...\Code and Data")`. Replace the existing path with the corresponding directory on your local computer.

##  Step 3. Main Computation

In `Main_01.py`, select all lines of code and execute the selected code in the current PyCharm environment.

##  Step 4. Economic Evaluation

Without closing or restarting the current PyCharm execution environment, open `Economics_Application_04.py`. Select all lines of code in this file and execute the selected code within the same environment.

##  Step 5. Figure Reproduction

Please execute only the selected lines of code or code blocks in RStudio, rather than running the entire script at once, as the script contains commands that close the graphics device.

Open `Figures.R` in RStudio and load the required R packages specified in lines 2–15. Modify the working directories specified in lines 26, 102, 221, 343, 467, 592, 665, 738, 812, 885, 990, 1114, 1204, 1355, 1462, and 1565. These directories correspond to the locations used for reading plotting data and saving generated figures. Please ensure that the modified paths are consistent with the folder structure of the project. After completing these steps, all figures presented in the paper can be reproduced. The generated figures will be automatically saved in the figures folder within the working directory.

Please note that each occurrence of `dev.off()` in the R script closes the current graphics device. If you wish to keep the graphics window open, do not execute this line. Comments following `dev.off()`, such as `#1500*2000`, indicate the resolution settings (in pixels) used for exporting the figures.

## Step 6. Table Reproduction

All numerical results reported in the tables of the paper are generated through the computations described in Steps 3 and 4. The following instructions explain how to locate and organize the corresponding results.

### 6.1 Notes on the MCS Test

The significance level of the Model Confidence Set (MCS) test in all tables is set at 5%, and 50,000 bootstrap replications are used to approximate the empirical distribution of the test statistic. It should be noted that the MCS test inherently relies on random bootstrap resampling. Consequently, even with a relatively large number of bootstrap replications (50,000 in this study), the resulting p-values may exhibit minor variations across different runs. After repeating the computations multiple times, we observe that the MCS p-values of different forecasting models fluctuate only slightly, generally around the third decimal place. Importantly, the MCS p-value of the proposed ISVD-VC-ACI model consistently remains equal to 1. These results indicate that the MCS findings exhibit good reproducibility and robustness. The use of 50,000 bootstrap replications represents a relatively large and robust computational setting.

### 6.2 Table 1

Table 1 is calculated using the code in region 21 of `Main_01.py`. The results are exported to: `tables/table1/Table 1.csv`. The first row of `Table 1.csv` contains the column headers.

After rounding to three decimal places, columns A, D, G, and J of `Table 1.csv` correspond to columns 1, 4, 7, and 10 of Table 1 in the paper, respectively. Columns B, E, H, and K of `Table 1.csv` contain the corresponding MCS test results.

As explained above, the MCS test relies on bootstrap resampling, meaning that the resulting MCS p-values may vary slightly across different runs. To improve the stability of these results, we employ 50,000 bootstrap replications, a substantially larger number than those commonly adopted in empirical financial forecasting studies.

When interpreting the MCS results, we focus specifically on forecasting models with p-values equal to 1. Across repeated computations, the set of models with MCS p-values equal to 1 remains unchanged.

### 6.3 Table 2

Table 2 is calculated using the code in region 25 of `Main_01.py`.

The results are exported to:

`tables/table2/crv.csv`

### 6.4 Tables 3 and 4

Tables 3 and 4 are calculated using the code in region 27 of `Main_01.py`.

The results are exported to:

- `tables/table3/Table 3.csv`
- `tables/table4/Table 4.csv`

For Table 3, the following cell ranges in `Table 3.csv` correspond to the respective columns of Table 3 in the paper:

- B2–B12
- E2–E12
- H2–H12
- K2–K12
- B13–B23
- E13–E23
- H13–H23
- K13–K23

These eight cell ranges are used to construct the corresponding columns of Table 3.

The relationship between `Table 4.csv` and Table 4 follows the same structure.

### 6.5 Tables 5 and 6

Tables 5 and 6 are calculated using the code in region 1 of `Economics_Application_04.py`. The results are exported to:

`figures/figure10/vol_loss_0.csv`

For convenience, this CSV file can be copied to the `tables/table5&6` folder.

Construction of Table 5:

Horizontally concatenate the following two cell ranges from `vol_loss_0.csv`:

- B2–C15
- B16–C29

The resulting data form Table 5 in the paper. Construction of Table 6:

Horizontally concatenate the following two cell ranges from `vol_loss_0.csv`:

- E2–F15
- E16–F29

The resulting data form Table 6 in the paper. As noted previously, the MCS test relies on bootstrap resampling, and its p-values may therefore exhibit minor variations across different runs. To enhance the stability of the results, we employ 50,000 bootstrap replications, substantially exceeding the number commonly used in empirical financial forecasting studies. Our interpretation of the MCS results focuses on forecasting models with p-values equal to 1. Across repeated computations, the set of models with MCS p-values equal to 1 remains unchanged.

### 6.6 Table C1

Table C1 is calculated using the code in region 22 of `Main_01.py`.

The results are exported to the following files:

- `figures/figure5/All_asset0_w84.csv`
- `figures/figure5/All_asset1_w84.csv`
- `figures/figure5/All_asset2_w84.csv`
- `figures/figure5/All_asset3_w84.csv`

These four CSV files correspond to the following asset categories in Table C1, respectively:

| CSV file           | Asset   |
|--------------------|---------|
| All_asset0_w84.csv | T-bond  |
| All_asset1_w84.csv | USDX    |
| All_asset2_w84.csv | S&P 500 |
| All_asset3_w84.csv | WTI     |

For an explanation of the minor variations in MCS p-values across different runs, please refer to the preceding discussion.

### 6.7 Tables C2 and C3

Tables C2 and C3 are calculated using the code in region 23 of `Main_01.py`.

The results are exported to the following directories:

- `tables/tableC2`
- `tables/tableC3`
