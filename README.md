# Mortgage Analytics & Structured Finance Models

This repository contains a collection of Python modules for mortgage analytics and structured finance modeling. The modules are designed to help you:

1. **Analyze standard mortgage cashflows**  
2. **Model pass-through mortgage-backed securities (MBS)**  
3. **Structure sequential pay and PAC bonds**  
4. **Compute OAS (Option-Adjusted Spreads) for pools**  
5. **Examine refinancing and prepayment dynamics**  
6. **Model a simplified CLO (Collateralized Loan Obligation)**  

Each module has classes and functions that handle different aspects of the mortgage and structured finance models. Below is an overview of each module and some basic usage guidelines.

---

## Table of Contents
1. [Project Structure](#project-structure)  
2. [Modules Overview](#modules-overview)  
   1. [mortgage_calculator.py](#mortgage_calculatorpy)  
   2. [passthrough.py](#passthroughpy)  
   3. [sequential_pac.py](#sequential_pacpy)  
   4. [oas_model.py](#oas_modelpy)  
   5. [refi_model.py](#refi_modelpy)  
   6. [clo_cashflow.py (CLO)](#clo_cashflowpy-clo)  
3. [Getting Started](#getting-started)  
4. [Examples](#examples)  
5. [Dependencies](#dependencies)  
6. [License](#license)  

---

## Project Structure

```
your-repo/
├── modules/
│   ├── mortgage_calculator.py
│   ├── passthrough.py
│   ├── sequential_pac.py
│   ├── oas_model.py
│   ├── refi_model.py
│   ├── clo_cashflow.py
│   └── __init__.py
├── data/
│   ├── (sample CSV files and rate curve data, if any)
│   └── ...
├── notebooks/
│   ├── example_usage.ipynb
│   └── ...
├── README.md  <-- (this file)
└── requirements.txt
```

- **modules/**: The folder containing Python modules.  
- **data/**: Contains any CSV or reference data for examples and tests.  
- **notebooks/**: Contains Jupyter notebooks illustrating code usage.  
- **README.md**: Documentation for how to use the modules.  
- **requirements.txt**: Python package dependencies.

---

## Modules Overview

### 1. `mortgage_calculator.py`

**Purpose**  
A class-based implementation to analyze mortgage cashflows on a single-loan level. Given:
- Principal  
- Annual coupon / interest rate  
- Maturity in months  

It calculates the monthly payment, interest/principal split, outstanding balance over time, Weighted Average Life (WAL), and cumulative interest.

**Key Classes**  
1. **`MortgageCalculator`**  
   - **Inputs**:
     - `initial_principal`  
     - `annual_coupon` (expressed as a decimal, e.g., 0.04 = 4%)  
     - `maturity` in months  
   - **Methods**:
     - `show()`: Prints a nicely formatted table of monthly cashflows and key statistics.  
     - `plots()`: Generates plots for the beginning principal, cumulative interest, scheduled principal, and interest payments over time.  

2. **`ScenarioAnalyses`**  
   - Used to automate scenario analyses for different maturities or coupon rates.  
   - **Methods**:
     - `WAL()`: Plots Weighted Average Life vs. different maturities.  
     - `plot_2d_col_chart()`: Example 2D charting of interest paid vs. time under different coupon/maturity combos.

---

### 2. `passthrough.py`

**Purpose**  
Simulates how a pool of identical or similar loans pass principal and interest to an MBS pass-through vehicle. Supports different prepayment assumptions (standard or user-defined Single Monthly Mortality—SMM).

**Key Classes**  
1. **`PoolPassthrough`**  
   - **Inputs**:
     - `num_loan`: number of loans in the pool  
     - `maturity`: maturity in years (converted to months internally)  
     - `wac`: Weighted Average Coupon for the pool  
     - `wala`: Weighted Average Loan Age  
     - `smm`: Single Monthly Mortality (assumed constant unless set to a function)  
     - `mode`: If `"Prepayment"`, triggers a different prepayment logic for partial payoffs.  
     - `init_principal`: starting principal per loan  
   - **Methods**:
     - `show()`: Displays DataFrame with monthly principal & interest breakdown, WAL, WAC, SMM, etc.  
     - `save()`: Save the data to CSV.

2. **`Passthrough`**  
   - Aggregates multiple `PoolPassthrough` objects into a combined pass-through.  
   - Tracks overall WAC, WAL, SMM, and net coupon (after fees).  
   - **Methods**:
     - `feed(pool)`: Adds a new `PoolPassthrough` to the aggregator.  
     - `show()`: Outputs the aggregated pass-through results.  
     - `save()`: Export to CSV.

---

### 3. `sequential_pac.py`

**Purpose**  
Implements sequential pay structures and Planned Amortization Class (PAC) bond structures commonly seen in mortgage-backed securities.  

**Key Classes**  
1. **`PoolSequentialPAC`**  
   - Models a single pass-through with **PSA-based** prepayment (`psa_func`) and derived SMM.  
   - **Methods**:
     - `show()`, `plot()`: Output or visualize the principal and interest flows, WAL, etc.

2. **`Sequential`**  
   - Takes a single pool’s principal payments and distributes them in **sequential** order to multiple tranches.  
   - **Methods**:
     - `plot()`: Shows distribution of principal to each tranche over time.  
     - `wals()`: Returns the WAL of each tranche.

3. **`PAC`**  
   - Implements a simplified PAC structure with a specified PSA band.  
   - Creates a **PAC** tranche and a **companion** tranche.  
   - **Methods**:
     - `plot()`: Visualizes principal payments allocated to the PAC vs. the companion.

---

### 4. `oas_model.py`

**Purpose**  
Implements an Option-Adjusted Spread (OAS) analysis for pass-through pools:
- Takes multiple interest rate paths/scenarios.
- Incorporates prepayment assumptions.  
- Computes present value of each scenario’s cashflows, averages them, and solves for the OAS that equates the model price to a market price.

**Key Components**  
- **`PoolOAS`**: A pass-through structure with a single SMM or a function-based prepayment assumption.  
- **`PassthroughOAS`**: Aggregates multiple pools or runs many paths, computing average PV across paths.  
- **`factor_to_smm`, `cpr_to_smm`, `pv(...)`**: Helper functions to solve for the SMM that yields a certain factor, convert CPR to monthly SMM, and compute present value given discounting.

---

### 5. `refi_model.py`

**Purpose**  
Focuses on the **refinancing** dimension: how borrowers refinance in response to interest rate changes. Also includes utility functions to compute mortgage rates or shapes of S-curve-based refi incentives.  

**Key Classes & Functions**  
1. **`PoolRefi`**: A pass-through style structure that re-computes prepayments based on dynamic refi incentives.  
2. **`PassthroughRefi`**: Aggregates `PoolRefi` objects and allows for scenario testing.  
3. **`factor_to_smm`**, **`swap_rate(...)`**, **`S_fit(...)`**: Example helper functions:
   - `factor_to_smm`: Solve for the monthly SMM that gives a certain pool factor.  
   - `swap_rate`: Illustrative function to compute 10yr swap rates from discount factors.  
   - `S_fit`: Example cubic spline or S-curve refi function for building a dynamic prepayment model.

---

### 6. `clo_cashflow.py` (CLO)

**Purpose**  
A simplified model for **Collateralized Loan Obligation** (CLO) cashflows:  
- **`CLO`** class:  
  - Models a capital stack with an A class, B class, and Equity.  
  - Allows specifying an annual default rate, then computes monthly defaults.  
  - Allows toggling Over-Collateralization (OC) tests that can cause principal to be diverted to higher tranches if tests fail.  
  - Outputs monthly interest and principal flows, leftover equity cashflows, and the final balances.

---

## Getting Started

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Install Dependencies**  
   - Create or activate a virtual environment (recommended).  
   - Install packages:
     ```bash
     pip install -r requirements.txt
     ```
3. **Explore Example Notebooks**  
   - Look in `notebooks/` for usage examples:
     - `example_usage.ipynb`  

4. **Run a Simple Example**  
   ```python
   from modules.mortgage_calculator import MortgageCalculator

   mc = MortgageCalculator(initial_principal=200000, annual_coupon=0.03, maturity=360)
   mc.show()
   mc.plots()
   ```

---

## Examples

Here are some high-level examples to get you started:

1. **Single Mortgage Calculation**  
   ```python
   from modules.mortgage_calculator import MortgageCalculator

   mc = MortgageCalculator(100000, 0.04, 180)  # 100k principal, 4% interest, 15 years
   mc.show()   # Print out schedule
   mc.plots()  # Plot principal, interest, etc.
   ```

2. **Pass-Through Pool**  
   ```python
   from modules.passthrough import PoolPassthrough, Passthrough

   p1 = PoolPassthrough(num_loan=1000, maturity=15, wac=0.064, wala=1, smm=0.005, init_principal=200000)
   p1.show()

   aggregator = Passthrough(fee=0.0075)
   aggregator.feed(p1)
   aggregator.show()
   ```

3. **Sequential Structure**  
   ```python
   from modules.sequential_pac import Sequential

   seq = Sequential(psa=100,
                    maturity=30,
                    wac=0.065,
                    init_principal=100000000,
                    tranches=[45000000, 25000000, 20000000, 10000000])
   seq.plot()  # Plot principal allocations to each sequential tranche
   print(seq.wals())  # Display WAL for each tranche
   ```

4. **CLO Example**  
   ```python
   from modules.clo_cashflow import CLO

   clo = CLO(A_bal=75, B_bal=15, Eq_bal=10,
             default_rate=0.02, maturity=10,
             asset_rate=0.07, A_rate=0.06, B_rate=0.09,
             A_trigger=1.2, B_trigger=1.05,
             A_OC=100/75, B_OC=100/(75+15), OC=True)
   clo.show()
   ```

5. **OAS Model**  
   ```python
   from modules.oas_model import PoolOAS, PassthroughOAS, cpr_to_smm, pv
   import numpy as np

   # Build a single pool
   pool = PoolOAS(smm=0.01, maturity=360, wac=0.04, wala=0, init_principal=500000)

   # Create multiple interest-rate paths & compute average PV
   pass_oas = PassthroughOAS(fee=0.0075).feed(pool)

   # Suppose we have a user-defined CPR path
   cpr_path = np.full(360, 5.0)  # 5% CPR each month
   pass_oas.cpr(cpr_path)
   pass_oas.show()
   ```

---

## Dependencies

- Python 3.7+  
- `numpy`  
- `pandas`  
- `matplotlib`  
- `scipy` (for `fsolve` and interpolation)  
- `IPython.display` (for in-notebook HTML display)  
- `locale` (optional, for currency formatting)  

Check `requirements.txt` for the exact versions used.

---

**Happy Modeling!**  

For further questions or extended usage instructions, please refer to:
- The docstrings in each module.  
- Example Jupyter notebooks.  
- Or open an issue in the repository.  
