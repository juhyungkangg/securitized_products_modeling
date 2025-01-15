# modules/sequential_pac.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import locale
from IPython.display import display, HTML

def psa_func(val=100, period=120):
    psa = np.linspace(0.2, 6, 30) * val / 100 / 100
    return np.append(psa, np.full(period - 30, psa[-1]))


def cpr_to_smm(cpr):
    return 1 - (1 - cpr) ** (1 / 12)


class PoolSequentialPAC(object):
    def __init__(self, psa, maturity=30, wac=0.065, init_principal=100000000):
        # Save inputs
        self.psa = psa
        self.maturity = maturity
        self.wac = wac
        self.init_principal = init_principal

        # Set variables
        c = wac / 12
        n = maturity * 12
        M0 = init_principal

        # Monthly payment for each loan: B
        B = (c * (1 + c) ** n * M0) / ((1 + c) ** n - 1)
        self.B = B

        # CPR by PSA
        cpr = psa_func(val=psa, period=n)

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
                BeginPrincipal[i] = init_principal
            else:
                BeginPrincipal[i] = EndPrincipal[i - 1]

            # WAC: WAC
            WAC[i] = c * 12

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
            ActualPymt[i] = B * nums_loan * (1 - SMM[i]) + BeginPrincipal[i] * (1 + WAC[i] / 12) * SMM[i]

            nums_loan = nums_loan * (1 - SMM[i])

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
        self.CPR = CPR

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
            'CPR': self.CPR

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
            'CPR': '{:,.3f}'
        })

        display(HTML(styled_df.to_html()))

    def save(self, title):
        self.table.to_csv(title)

    def plot(self):
        plt.bar(self.Month, self.BeginPrincipal, width=1, label='BeginPrincipal', alpha=0.7)
        plt.bar(self.Month, self.CumulativeInterestPaid, width=1, label='CumulativeInterestPaid', alpha=0.7)
        plt.legend()
        plt.show()

        plt.bar(self.Month, self.ActualPymt - self.InterestPymt, width=1, label='PrincipalPymt', alpha=1)
        plt.bar(self.Month, self.ScheduledPrincipalPymt, width=1, label='ScheduledPrincipalPymt', alpha=0.7)
        plt.bar(self.Month, self.InterestPymt, width=1, label='InterestPymt', alpha=0.7)
        plt.legend()
        plt.show()

        plt.plot(self.Month, self.WAL, label='WAL')
        plt.legend()
        plt.show()


class Sequential(object):
    def __init__(self, psa, maturity=30,
                 wac=0.065, init_principal=100000000,
                 tranches=[45000000, 25000000, 20000000, 10000000]):
        # Save inputs
        self.psa = psa
        self.maturity = maturity
        self.wac = wac
        self.init_principal = init_principal

        if sum(tranches) != init_principal:  # Sanity Check
            print("sum(tranches) != pool.init_principal")
            self.tranches = None
        else:
            self.tranches = tranches

        # Make Pool
        self.pool = PoolSequentialPAC(psa, maturity, wac, init_principal)

        # Principal Payment
        PrincipalPymt = self.pool.ActualPymt - self.pool.InterestPymt

        # Beginning Principal of Tranches
        n = maturity * 12
        tranch1_BegP = np.zeros(n)
        tranch2_BegP = np.zeros(n)
        tranch3_BegP = np.zeros(n)
        tranch4_BegP = np.zeros(n)

        tranch1_BegP[0] = tranches[0]
        tranch2_BegP[0] = tranches[1]
        tranch3_BegP[0] = tranches[2]
        tranch4_BegP[0] = tranches[3]

        tranch_BegP_li = [tranch1_BegP, tranch2_BegP, tranch3_BegP, tranch4_BegP]

        # Actual Principal of Tranches
        tranch1_P = np.zeros(n)
        tranch2_P = np.zeros(n)
        tranch3_P = np.zeros(n)
        tranch4_P = np.zeros(n)

        tranch_P_li = [tranch1_P, tranch2_P, tranch3_P, tranch4_P]

        # Threshold
        thresh = np.cumsum(tranches)

        num_tranches = len(tranch_BegP_li)
        tranch_iter = 0
        cumul_P = 0

        # Calculate Beginning Principal and Monthly Principal Payment
        for i in range(len(PrincipalPymt)):
            # Set Beginning Balance
            for j in range(num_tranches):
                if i != 0:
                    tranch_BegP_li[j][i] = max(tranch_BegP_li[j][i - 1] - tranch_P_li[j][i - 1], 0)

            # Calculate and input principal payment
            principal_payment = PrincipalPymt[i]

            if cumul_P + principal_payment > thresh[tranch_iter]:
                # Fill up the principal
                tranch_P_li[tranch_iter][i] = thresh[tranch_iter] - cumul_P

                # Move to next tranch
                tranch_iter += 1
                if tranch_iter > 3:
                    break

                # Fill next tranch with remaining
                tranch_P_li[tranch_iter][i] = principal_payment - (thresh[tranch_iter - 1] - cumul_P)

                # Add cumulative principal
                cumul_P += principal_payment
            else:
                # Fill up the principal
                tranch_P_li[tranch_iter][i] = principal_payment

                # Add cumulative principal
                cumul_P += principal_payment

        def WAL_func(P, BegP):
            n = len(P)
            WAL = np.zeros(n)
            # Weighted Average Life: WAL
            for i in range(n):
                if BegP[i] == 0:
                    WAL[i] = 0
                else:
                    WAL[i] = np.sum(P[i:] * np.linspace(1, n - i, n - i)) / BegP[i]

            WAL = WAL / 12

            return WAL

        # Save variables
        self.tranch1_WAL = WAL_func(tranch1_P, tranch1_BegP)
        self.tranch2_WAL = WAL_func(tranch2_P, tranch2_BegP)
        self.tranch3_WAL = WAL_func(tranch3_P, tranch3_BegP)
        self.tranch4_WAL = WAL_func(tranch4_P, tranch4_BegP)

        self.tranch1_BegP = tranch1_BegP
        self.tranch2_BegP = tranch2_BegP
        self.tranch3_BegP = tranch3_BegP
        self.tranch4_BegP = tranch4_BegP

        self.tranch1_P = tranch1_P
        self.tranch2_P = tranch2_P
        self.tranch3_P = tranch3_P
        self.tranch4_P = tranch4_P

    def plot(self):
        Month = [x for x in range(1, len(self.tranch1_P) + 1)]
        plt.bar(Month, self.tranch1_P, width=1, label='Sequential 1', alpha=0.7)
        plt.bar(Month, self.tranch2_P, width=1, label='Sequential 2', alpha=0.7)
        plt.bar(Month, self.tranch3_P, width=1, label='Sequential 3', alpha=0.7)
        plt.bar(Month, self.tranch4_P, width=1, label='Sequential 4', alpha=0.7)
        plt.legend()
        plt.xlabel('Month')
        plt.ylabel('Principal')
        plt.show()

        pool_wal = np.zeros_like(self.tranch1_WAL)
        pool_wal[:len(self.pool.WAL)] = self.pool.WAL
        plt.plot(Month, self.tranch1_WAL, label='tranch1_WAL')
        plt.plot(Month, self.tranch2_WAL, label='tranch2_WAL')
        plt.plot(Month, self.tranch3_WAL, label='tranch3_WAL')
        plt.plot(Month, self.tranch4_WAL, label='tranch4_WAL')
        plt.plot(Month, pool_wal, label='Pool WAL')
        plt.legend()
        plt.xlabel('Month')
        plt.ylabel('WAL')
        plt.show()

    def wals(self):
        w1 = self.tranch1_WAL[0]
        w2 = self.tranch2_WAL[0]
        w3 = self.tranch3_WAL[0]
        w4 = self.tranch4_WAL[0]

        return [w1, w2, w3, w4]


class PAC(object):
    def __init__(self, psa, band=(100, 250), maturity=30,
                 wac=0.065, init_principal=100000000):
        # Save inputs
        self.psa = psa
        self.band = band
        self.maturity = maturity
        self.wac = wac
        self.init_principal = init_principal

        # Make pool with bands
        pool1 = PoolSequentialPAC(band[0])
        pool2 = PoolSequentialPAC(band[1])

        # PAC principal
        pool1_P = pool1.ActualPymt - pool1.InterestPymt
        pool2_P = pool2.ActualPymt - pool2.InterestPymt

        len1 = len(pool1_P)
        len2 = len(pool2_P)

        max_len = max(len1, len2)

        aligned_pool1_P = np.pad(pool1_P, (0, max_len - len1), mode='constant')
        aligned_pool2_P = np.pad(pool2_P, (0, max_len - len2), mode='constant')

        pac_P = np.minimum(aligned_pool1_P, aligned_pool2_P)

        # Sanity Check
        A = aligned_pool2_P - aligned_pool1_P
        A[A < 0] = 0

        B = pac_P

        C = aligned_pool1_P - aligned_pool2_P
        C[C < 0] = 0

        print("A + B =", np.round(np.sum(A + B)))
        print("B + C =", np.round(np.sum(B + C)))

        # Companion notional
        comp_notional = init_principal - np.sum(pac_P)

        # Pool
        pool0 = PoolSequentialPAC(psa)
        self.pool = pool0

        P = pool0.ActualPymt - pool0.InterestPymt
        BegP = pool0.BeginPrincipal
        I = pool0.InterestPymt

        n = len(P)

        pac_P_real = np.zeros(n)
        companion_P_real = np.zeros(n)

        pac_BegP_real = np.zeros(n)
        companion_BegP_real = np.zeros(n)

        for i in range(n):
            if i == 0:
                pac_BegP_real[i] = np.sum(pac_P)
                companion_BegP_real[i] = comp_notional
            else:
                pac_BegP_real[i] = max(pac_BegP_real[i - 1] - pac_P_real[i - 1], 0)
                companion_BegP_real[i] = max(companion_BegP_real[i - 1] - companion_P_real[i - 1], 0)

            if P[i] > pac_P[i]:
                if comp_notional > 0:
                    companion_P_real[i] = min(P[i] - pac_P[i], comp_notional)
                    pac_P_real[i] = P[i] - companion_P_real[i]
                    comp_notional -= companion_P_real[i]
                else:
                    pac_P_real[i] = P[i]
            elif P[i] <= pac_P[i]:
                pac_P_real[i] = P[i]

        def WAL_func(P, BegP):
            n = len(P)
            WAL = np.zeros(n)
            # Weighted Average Life: WAL
            for i in range(n):
                if BegP[i] == 0:
                    WAL[i] = 0
                else:
                    WAL[i] = np.sum(P[i:] * np.linspace(1, n - i, n - i)) / BegP[i]

            WAL = WAL / 12

            return WAL

        # Save variables
        self.pac_WAL = WAL_func(pac_P_real, pac_BegP_real)
        self.companion_WAL = WAL_func(companion_P_real, companion_BegP_real)

        self.pac_BegP_real = pac_BegP_real
        self.companion_BegP_real = companion_BegP_real

        self.pac_P_real = pac_P_real
        self.companion_P_real = companion_P_real

    def plot(self):
        Month = [x for x in range(1, len(self.pac_P_real) + 1)]
        plt.bar(Month, self.companion_P_real + self.pac_P_real, width=1, label='Companion', alpha=1)
        plt.bar(Month, self.pac_P_real, width=1, label='PAC', alpha=1)
        plt.legend()
        plt.xlabel('Month')
        plt.ylabel('Principal')
        plt.title(f'[PSA={self.psa}] PAC & Companion Principal Plot')
        plt.show()

        pool_wal = np.zeros_like(self.pac_WAL)
        pool_wal[:len(self.pool.WAL)] = self.pool.WAL
        plt.plot(Month, self.pac_WAL, label='PAC WAL')
        plt.plot(Month, self.companion_WAL, label='Companion_WAL')
        plt.plot(Month, pool_wal, label='Pool WAL')
        plt.legend()
        plt.xlabel('Month')
        plt.ylabel('WAL')
        plt.title(f'[PSA={self.psa}] PAC, Companion, Pool WAL Plot')
        plt.show()