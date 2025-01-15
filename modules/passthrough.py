# modules/passthrough.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import locale
from IPython.display import display, HTML


class PoolPassthrough(object):
    def __init__(self, num_loan, maturity, wac, wala, smm, mode=None, init_principal=100000):
        # Save inputs
        self.num_loan = num_loan
        self.maturity = maturity
        self.wac = wac
        self.wala = wala
        self.smm = smm
        self.mode = mode
        self.init_principal = init_principal

        # Set variables
        c = wac / 12
        n = maturity * 12
        M0 = init_principal

        # Monthly payment for each loan: B
        B = (c * (1 + c) ** n * M0) / ((1 + c) ** n - 1)
        self.B = B

        ### Calculate each month

        BeginPrincipal = np.zeros(n)
        InterestPymt = np.zeros(n)
        ScheduledPrincipalPymt = np.zeros(n)
        ActualPymt = np.zeros(n)
        UnscheduledPrincipalPymt = np.zeros(n)
        EndPrincipal = np.zeros(n)
        CumulativeInterestPaid = np.zeros(n)
        WAL = np.zeros(n)
        WAC = np.zeros(n)
        SMM = np.zeros(n)
        nums_loan = np.zeros(n)

        trunc_point = False

        for i in range(n):
            # The number of loans: nums_loan
            if i == 0:
                nums_loan[i] = num_loan
            else:
                nums_loan[i] = nums_loan[i - 1]

            # Unpaid balance: BeginPrincipal
            if i == 0:
                BeginPrincipal[i] = init_principal * num_loan
            else:
                BeginPrincipal[i] = EndPrincipal[i - 1]

                # WAC: WAC
            WAC[i] = c * 12

            # Interest expense: InterestPymt
            InterestPymt[i] = BeginPrincipal[i] * WAC[i] / 12

            # Scheduled principal payments: ScheduledPrincipalPymt
            ScheduledPrincipalPymt[i] = B * nums_loan[i] - InterestPymt[i]

            if ScheduledPrincipalPymt[i] > BeginPrincipal[i]:
                ScheduledPrincipalPymt[i] = BeginPrincipal[i]
                trunc_point = i

            # SMM: SMM
            SMM[i] = smm

            # The number of loans: nums_loan
            # Actual Payment: ActualPymt
            if mode == "Prepayment":
                n_prepaid = int(nums_loan[i] * smm)

                ActualPymt[i] = B * (nums_loan[i] - n_prepaid) + BeginPrincipal[i] / nums_loan[i] * (
                            1 + WAC[i] / 12) * n_prepaid

                nums_loan[i] -= n_prepaid
            else:
                ActualPymt[i] = SMM[i] * np.maximum(BeginPrincipal[i] - ScheduledPrincipalPymt[i], 0) + \
                                ScheduledPrincipalPymt[i] + InterestPymt[i]

            # Unscheduled Principal Payment: UnscheduledPrincipalPymt
            UnscheduledPrincipalPymt[i] = ActualPymt[i] - ScheduledPrincipalPymt[i] - InterestPymt[i]

            # End Principal: EndPrincipal
            EndPrincipal[i] = BeginPrincipal[i] - ActualPymt[i] + InterestPymt[i]

            # Cumulative Interest Paid: CumulativeInterestPaid
            CumulativeInterestPaid[i] = np.sum(InterestPymt[:i + 1])

            if trunc_point:
                break

        # Truncate unnecessary columns
        if trunc_point:
            BeginPrincipal = BeginPrincipal[:trunc_point + 1][:trunc_point + 1]
            InterestPymt = InterestPymt[:trunc_point + 1]
            ScheduledPrincipalPymt = ScheduledPrincipalPymt[:trunc_point + 1]
            ActualPymt = ActualPymt[:trunc_point + 1]
            UnscheduledPrincipalPymt = UnscheduledPrincipalPymt[:trunc_point + 1]
            EndPrincipal = EndPrincipal[:trunc_point + 1]
            CumulativeInterestPaid = CumulativeInterestPaid[:trunc_point + 1]
            WAL = WAL[:trunc_point + 1]
            WAC = WAC[:trunc_point + 1]
            SMM = SMM[:trunc_point + 1]
            nums_loan = nums_loan[:trunc_point + 1]

        P = ScheduledPrincipalPymt + UnscheduledPrincipalPymt
        n = len(P)

        # Weighted Average Life: WAL
        for i in range(n):
            WAL[i] = np.sum(P[i:] * np.linspace(1, n - i, n - i)) / BeginPrincipal[i]
        WAL = WAL / 12

        # SMM_real
        ScheduledBalance = BeginPrincipal - ScheduledPrincipalPymt
        ActualBalance = BeginPrincipal - (ActualPymt - InterestPymt)

        self.realized_SMM = (ScheduledBalance - ActualBalance) / ScheduledBalance

        # Set class variables
        self.Month = [x for x in range(1, n + 1)]
        self.BeginPrincipal = BeginPrincipal
        self.InterestPymt = InterestPymt
        self.ScheduledPrincipalPymt = ScheduledPrincipalPymt
        self.ActualPymt = ActualPymt
        self.UnscheduledPrincipalPymt = UnscheduledPrincipalPymt
        self.EndPrincipal = EndPrincipal
        self.CumulativeInterestPaid = CumulativeInterestPaid
        self.WAL = WAL
        self.WAC = WAC
        self.SMM = SMM
        self.nums_loan = nums_loan

    def show(self):

        self.table = pd.DataFrame({
            'Month': self.Month,
            'BeginPrincipal': self.BeginPrincipal,
            'InterestPymt': self.InterestPymt,
            'ScheduledPrincipalPymt': self.ScheduledPrincipalPymt,
            'ActualPymt': self.ActualPymt,
            'UnscheduledPrincipalPymt': self.UnscheduledPrincipalPymt,
            'EndPrincipal': self.EndPrincipal,
            'CumulativeInterestPaid': self.CumulativeInterestPaid,
            'WAL': self.WAL,
            'WAC': self.WAC,
            'SMM': self.SMM,
            'nums_loan': self.nums_loan,
            'realized_SMM': self.realized_SMM
        })

        styled_df = self.table.style.format({
            'BeginPrincipal': '${:,.2f}',
            'ActualPymt': '${:,.2f}',
            'InterestPymt': '${:,.2f}',
            'UnscheduledPrincipalPymt': '${:,.2f}',
            'ScheduledPrincipalPymt': '${:,.2f}',
            'EndPrincipal': '${:,.2f}',
            'CumulativeInterestPaid': '${:,.2f}',
            'WAC': '{:,.3f}',
            'SMM': '{:,.3f}',
            'nums_loan': '{:,.0f}',
            'realized_SMM': '{:,.3f}'
        })

        display(HTML(styled_df.to_html()))

    def save(self, title):
        self.table.to_csv(title)


class Passthrough(object):
    def __init__(self, fee=0.0075):
        self.fee = fee
        self.pools = []
        self.passthrough = None

    def feed(self, pool):
        self.pools.append(pool)
        pool_df = pool.table[pool.wala:].reset_index()

        if self.passthrough is not None:
            passthrough_aligned, pool_aligned = self.passthrough.align(pool_df, fill_value=0)

            columns_to_sum = ['BeginPrincipal', 'InterestPymt', 'ScheduledPrincipalPymt',
                              'ActualPymt', 'UnscheduledPrincipalPymt', 'EndPrincipal',
                              'nums_loan']

            passthrough = passthrough_aligned[columns_to_sum] + pool_aligned[columns_to_sum]
            passthrough.insert(loc=0, column='Month', value=[x for x in range(1, len(passthrough) + 1)])

            # CumulativeInterestPaid
            passthrough['CumulativeInterestPaid'] = passthrough['InterestPymt'].cumsum()

            # WAC
            bal1 = passthrough_aligned['BeginPrincipal']
            wac1 = passthrough_aligned['WAC']

            bal2 = pool_aligned['BeginPrincipal']
            wac2 = pool_aligned['WAC']

            wac = (bal1 * wac1 + bal2 * wac2) / (bal1 + bal2)
            passthrough['WAC'] = wac

            # WAL
            P = passthrough['ScheduledPrincipalPymt'] + passthrough['UnscheduledPrincipalPymt']
            n = len(P)

            WAL = np.zeros(n)
            for i in range(n):
                WAL[i] = np.sum(P[i:] * np.linspace(1, n - i, n - i)) / passthrough['BeginPrincipal'].iloc[i]
            WAL = WAL / 12

            passthrough['WAL'] = WAL

            # SMM
            ScheduledBalance = passthrough['BeginPrincipal'] - passthrough['ScheduledPrincipalPymt']
            ActualBalance = passthrough['BeginPrincipal'] - (passthrough['ActualPymt'] - passthrough['InterestPymt'])

            passthrough['SMM'] = (ScheduledBalance - ActualBalance) / ScheduledBalance

            # Net Coupon
            fees = passthrough['BeginPrincipal'] * self.fee / 12
            NetCoupon = passthrough['InterestPymt'] - fees

            passthrough['NetCoupon'] = NetCoupon

            self.passthrough = passthrough
        else:
            self.passthrough = pool.table[pool.wala:].reset_index()

    def show(self):
        styled_df = self.passthrough.style.format({
            'BeginPrincipal': '${:,.2f}',
            'ActualPymt': '${:,.2f}',
            'InterestPymt': '${:,.2f}',
            'UnscheduledPrincipalPymt': '${:,.2f}',
            'ScheduledPrincipalPymt': '${:,.2f}',
            'EndPrincipal': '${:,.2f}',
            'CumulativeInterestPaid': '${:,.2f}',
            'WAC': '{:,.3f}',
            'SMM': '{:,.3f}',
            'nums_loan': '{:,.0f}',
            'NetCoupon': '${:,.2f}'
        })

        display(HTML(styled_df.to_html()))

    def save(self, title):
        self.passthrough.to_csv(title)