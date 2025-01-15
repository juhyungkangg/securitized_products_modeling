import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import locale
from IPython.display import display, HTML

class MortgageCalculator(object):
    def __init__(self, initial_principal, annual_coupon, maturity):
        self.initial_principal = initial_principal
        self.annual_coupon = annual_coupon
        self.maturity = maturity

        # set variables
        c = annual_coupon / 12
        n = maturity
        M0 = initial_principal

        # Monthly payment: B
        B = (c * (1 + c) ** n * M0) / ((1 + c) ** n - 1)
        self.B = B

        # Principal (balance): M
        M = M0 * ((1 + c) ** n - (1 + c) ** np.linspace(1, n, n)) / ((1 + c) ** n - 1)
        BeginPrincipal = np.append(np.array([M0]), M[:-1])

        # Paid interest: I
        I = np.append(np.array([M0 * c]), M[:-1] * c)

        # Paid principal
        P = B - I

        # Weighted Average Life
        WAL = np.zeros(n)

        for i in range(n):
            WAL[i] = np.sum(P[i:] * np.linspace(1, n - (i), n - (i))) / BeginPrincipal[i]
        WAL = WAL / 12

        # Set class variables
        self.Month = [x for x in range(1, n + 1)]
        self.BeginPrincipal = BeginPrincipal
        self.MthlyPymt = np.full(n, B)
        self.InterestPymt = I
        self.ScheduledPrincipalPymt = P
        self.EndPrincipal = self.BeginPrincipal - self.ScheduledPrincipalPymt
        self.CumulativeInterestPaid = np.cumsum(I)
        self.WAL = WAL

    def show(self):
        locale.setlocale(locale.LC_ALL, '')

        # Save data in dataframe
        self.table = pd.DataFrame({
            'Month': self.Month,
            'Begin Principal': self.BeginPrincipal,
            'Mthly Pymt': self.MthlyPymt,
            'Interest Pymt': self.InterestPymt,
            'Scheduled Principal Pymt': self.ScheduledPrincipalPymt,
            'End Principal': self.EndPrincipal,
            'Cumulative Interest Paid': self.CumulativeInterestPaid,
            'WAL (yrs)': self.WAL
        })

        # Show result
        print("**Input**")
        print("Initla Principal:", locale.currency(self.initial_principal, grouping=True))
        print("Annual Coupon:", f'{self.annual_coupon:.2%}')
        print("Maturity (mths):", self.maturity, "\n")

        print("**Output**")
        print("Monthly Payment:", locale.currency(self.B, grouping=True))
        print("WAL (yrs):", np.round(self.WAL[0], 2), "\n")

        styled_df = self.table.style.format({
            'Begin Principal': '${:,.2f}',
            'Mthly Pymt': '${:,.2f}',
            'Interest Pymt': '${:,.2f}',
            'Scheduled Principal Pymt': '${:,.2f}',
            'End Principal': '${:,.2f}',
            'Cumulative Interest Paid': '${:,.2f}',
            'WAL (yrs)': '{:,.3f}'
        })

        display(HTML(styled_df.to_html()))

    def plots(self):
        # Beginning Principal
        plt.figure(figsize=(8, 6))  # Adjust the figure size
        plt.plot(self.Month, self.BeginPrincipal, color='black', label='Begin Principal')
        plt.grid(True)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Begin Principal', fontsize=16)
        plt.show()

        # Cumulative Interest Paid
        plt.figure(figsize=(8, 6))  # Adjust the figure size
        plt.plot(self.Month, self.CumulativeInterestPaid, color='black', label='Begin Principal')
        plt.grid(True)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Cumulative Interest Paid', fontsize=16)
        plt.show()

        # Scheduled Principal Pymt
        plt.figure(figsize=(8, 6))  # Adjust the figure size
        plt.plot(self.Month, self.ScheduledPrincipalPymt, color='black', label='Begin Principal')
        plt.grid(True)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Scheduled Principal Payment', fontsize=16)
        plt.show()

        # Interest Pymt
        plt.figure(figsize=(8, 6))  # Adjust the figure size
        plt.plot(self.Month, self.InterestPymt, color='black', label='Begin Principal')
        plt.grid(True)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Interest Payment', fontsize=16)
        plt.show()


def format_as_currency(val):
    return '${:,.2f}'.format(val) if ~np.isnan(val) else val


class ScenarioAnalyses(object):
    def __init__(self, initial_principal, annual_coupon, maturity):
        self.initial_principal = initial_principal
        self.annual_coupon = annual_coupon
        self.maturity = maturity

    def WAL(self, n=360):
        maturities = [x for x in range(1, n + 1)]
        WAL = np.zeros(n)

        for i in range(n):
            WAL[i] = MortgageCalculator(self.initial_principal, self.annual_coupon, maturities[i]).WAL[0]

        self.WAL = WAL

        print("**Input**")
        print("Initla Principal:", locale.currency(self.initial_principal, grouping=True))
        print("Annual Coupon:", f'{self.annual_coupon:.2%}')
        print("Maturity (mths):", self.maturity, "\n")

        # Interest Pymt
        plt.figure(figsize=(8, 6))  # Adjust the figure size
        plt.plot(maturities, WAL, label='WAL')
        plt.grid(True)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('WAL as a function of Maturity', fontsize=16)
        plt.show()

    def plot_2d_col_chart(self, coupon_rate_li, maturity_li):
        for i in range(len(coupon_rate_li)):
            for j in range(len(maturity_li)):
                mc = MortgageCalculator(self.initial_principal, coupon_rate_li[i], maturity_li[j])
                maturities = mc.Month
                cum_int = mc.CumulativeInterestPaid

                # Cumulative Interest Paid
                plt.figure(figsize=(8, 6))  # Adjust the figure size
                plt.bar(maturities, cum_int, label='Cumulative Interest Paid')
                plt.grid(True)
                plt.xlabel('Month', fontsize=12)
                plt.ylabel('Value', fontsize=12)
                plt.title(
                    f'[Maturity={maturity_li[j] / 12} years, Coupon Rate={np.round(coupon_rate_li[i] * 100, 2)}%] Cumulative Interest Paid',
                    fontsize=16)
                plt.show()
