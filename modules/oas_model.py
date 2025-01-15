# modules/oas_model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import locale
from IPython.display import display, HTML
from scipy.optimize import fsolve

def factor_to_smm(factor):
    init_principal = 1000000

    def obj_func(smm):
        pool_temp = PoolOAS(smm, maturity=358, wac=0.04075, wala=33, init_principal=init_principal)
        curr_curr_amt = pool_temp.BeginPrincipal[33]

        return curr_curr_amt - factor * init_principal

    return fsolve(obj_func, 0.001)[0]


def cpr_to_smm(cpr):
    return 1 - (1 - cpr / 100) ** (1 / 12)


def pv(cf, r, spread):
    pv = 0

    for i in range(len(cf)):
        pv += cf[i] / np.prod(1 + (r[:i + 1] / 100 + spread) / 12)

    return pv





class PoolOAS(object):
    def __init__(self, smm, maturity, wac, wala, init_principal=1000000):
        # Save inputs
        self.maturity = maturity
        self.wac = wac
        self.wala = wala
        self.smm = smm
        self.init_principal = init_principal

        # Set variables
        c = wac / 12
        n = maturity
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

        trunc_point = False

        for i in range(n):
            # Unpaid balance: BeginPrincipal
            if i == 0:
                BeginPrincipal[i] = init_principal  # Added
            else:
                BeginPrincipal[i] = EndPrincipal[i - 1]

                # WAC: WAC
            WAC[i] = c * 12

            # Interest expense: InterestPymt
            InterestPymt[i] = BeginPrincipal[i] * WAC[i] / 12

            # Scheduled principal payments: ScheduledPrincipalPymt
            ScheduledPrincipalPymt[i] = B - InterestPymt[i]

            if ScheduledPrincipalPymt[i] > BeginPrincipal[i]:
                ScheduledPrincipalPymt[i] = BeginPrincipal[i]
                trunc_point = i

            # SMM: SMM
            SMM[i] = smm

            # Actual Payment: ActualPymt
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

        P = ScheduledPrincipalPymt + UnscheduledPrincipalPymt
        n = len(P)

        # Weighted Average Life: WAL
        for i in range(n):
            WAL[i] = np.sum(P[i:] * np.linspace(1, n - i, n - i)) / BeginPrincipal[i]
        WAL = WAL / 12

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
            'SMM': self.SMM
        })

    def show(self):

        styled_df = self.table.style.format({
            'BeginPrincipal': '${:,.2f}',
            'ActualPymt': '${:,.2f}',
            'InterestPymt': '${:,.2f}',
            'UnscheduledPrincipalPymt': '${:,.2f}',
            'ScheduledPrincipalPymt': '${:,.2f}',
            'EndPrincipal': '${:,.2f}',
            'CumulativeInterestPaid': '${:,.2f}',
            'WAC': '{:,.3f}',
            'SMM': '{:,.3f}'
        })

        display(HTML(styled_df.to_html()))

    def save(self, title):
        self.table.to_csv(title)


class PassthroughOAS(object):
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

            passthrough['Fee'] = fees
            passthrough['NetCoupon'] = NetCoupon

            self.passthrough = passthrough
        else:
            passthrough = pool_df

            # Net Coupon
            fees = passthrough['BeginPrincipal'] * self.fee / 12
            NetCoupon = passthrough['InterestPymt'] - fees

            passthrough['Fee'] = fees
            passthrough['NetCoupon'] = NetCoupon

            self.passthrough = passthrough

        return self

    def cpr(self, cpr):
        B = self.pools[0].B
        n = self.pools[0].maturity

        # SMM
        smm = cpr_to_smm(cpr)

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
        CPR = np.zeros(n)

        nums_loan = 1

        trunc_point = False

        for i in range(n):
            # Unpaid balance: BeginPrincipal
            if i == 0:
                BeginPrincipal[i] = self.passthrough['BeginPrincipal'][0]
            else:
                BeginPrincipal[i] = EndPrincipal[i - 1]

            # WAC: WAC
            WAC[i] = self.pools[0].wac

            # Interest expense: InterestPymt
            InterestPymt[i] = BeginPrincipal[i] * WAC[i] / 12

            # Scheduled principal payments: ScheduledPrincipalPymt
            ScheduledPrincipalPymt[i] = B * nums_loan - InterestPymt[i]

            if ScheduledPrincipalPymt[i] > BeginPrincipal[i]:
                ScheduledPrincipalPymt[i] = BeginPrincipal[i]
                trunc_point = i

            # SMM: SMM
            SMM[i] = smm[i]

            # CPR: CPR
            CPR[i] = cpr[i]

            # Actual Payment: ActualPymt
            ActualPymt[i] = SMM[i] * np.maximum(BeginPrincipal[i] - ScheduledPrincipalPymt[i], 0) + \
                            ScheduledPrincipalPymt[i] + InterestPymt[i]

            # Curtailment
            #             nums_loan = nums_loan * (1 - SMM[i])

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
            BeginPrincipal = BeginPrincipal[:trunc_point + 1]
            InterestPymt = InterestPymt[:trunc_point + 1]
            ScheduledPrincipalPymt = ScheduledPrincipalPymt[:trunc_point + 1]
            ActualPymt = ActualPymt[:trunc_point + 1]
            UnscheduledPrincipalPymt = UnscheduledPrincipalPymt[:trunc_point + 1]
            EndPrincipal = EndPrincipal[:trunc_point + 1]
            CumulativeInterestPaid = CumulativeInterestPaid[:trunc_point + 1]
            WAL = WAL[:trunc_point + 1]
            WAC = WAC[:trunc_point + 1]
            SMM = SMM[:trunc_point + 1]
            CPR = CPR[:trunc_point + 1]

        P = ScheduledPrincipalPymt + UnscheduledPrincipalPymt
        n = len(P)

        # Weighted Average Life: WAL
        for i in range(n):
            WAL[i] = np.sum(P[i:] * np.linspace(1, n - i, n - i)) / BeginPrincipal[i]
        WAL = WAL / 12

        # Set class variables
        self.Month = [x for x in range(self.pools[0].wala, self.pools[0].wala + len(BeginPrincipal))]
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
        self.CPR = CPR

        self.passthrough = pd.DataFrame({
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
            'CPR': self.CPR
        })

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
            'Fee': '${:,.2f}',
            'NetCoupon': '${:,.2f}'
        })

        display(HTML(styled_df.to_html()))

    def save(self, title):
        self.passthrough.to_csv(title)