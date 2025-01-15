# modules/refi_model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import locale
from IPython.display import display, HTML

import pandas as pd
import numpy as np


class CLO(object):
    def __init__(self, A_bal, B_bal, Eq_bal,
                 default_rate, maturity,
                 asset_rate, A_rate, B_rate,
                 A_trigger, B_trigger, A_OC, B_OC, OC=False):
        self.A_bal = A_bal
        self.B_bal = B_bal
        self.Eq_bal = Eq_bal
        self.default_rate = default_rate
        self.maturity = maturity
        self.asset_rate = asset_rate
        self.A_rate = A_rate
        self.B_rate = B_rate
        self.A_trigger = A_trigger
        self.B_trigger = B_trigger
        self.A_OC = A_OC
        self.B_OC = B_OC
        self.OC = OC

        months = maturity * 12
        months_li = range(1, months + 1)
        total_asset = A_bal + B_bal + Eq_bal

        total_df = pd.DataFrame(columns=['Month', "Total Not'l", "Total Not'l after Defaults",
                                         'Total Princ', 'Total Int', "Total Not'l after Def & Amort"])

        A_df = pd.DataFrame(columns=['Month', "A Not'l", "A Int Paid",
                                     'A Princ Paid', "A End Not'l"])

        B_df = pd.DataFrame(columns=['Month', "B Not'l", "B Int Paid",
                                     'B Princ Paid', "B End Not'l"])

        Eq_df = pd.DataFrame(columns=['Month', "Excess Int", "Excess Princ",
                                      'Resid Int Paid', "Resid CFs", "Equity Balance"])
        # Set Months
        total_df['Month'] = months_li
        A_df['Month'] = months_li
        B_df['Month'] = months_li
        Eq_df['Month'] = months_li

        # Calculate monthly default rates
        def monthly_default_rate_func(annual_default_rate):
            return 1 - (1 - annual_default_rate) ** (1 / 12)

        monthly_default_rate = monthly_default_rate_func(default_rate)

        # Total Not'l
        total_notl = np.zeros(months)
        total_notl[0] = total_asset

        for i in range(1, months):
            total_notl[i] = total_notl[i - 1] * (1 - monthly_default_rate)
        total_df["Total Not'l"] = total_notl

        # Total Not'l after Defaults
        total_notl_after_defaults = np.append(total_notl[1:], total_notl[-1] * (1 - monthly_default_rate))
        total_df["Total Not'l after Defaults"] = total_notl_after_defaults

        # Total Princ
        total_princ = np.zeros(months)
        total_princ[-1] = total_notl_after_defaults[-1]
        total_df["Total Princ"] = total_princ

        # Total Int
        total_int = total_notl * asset_rate / 12
        total_df['Total Int'] = total_int

        # Total Not'l after Def & Amort
        temp = np.copy(total_notl_after_defaults)
        temp[-1] = 0
        total_df["Total Not'l after Def & Amort"] = temp

        ### A

        # A Bond Not'l
        A_Notl = np.zeros(months)
        A_Notl[0] = A_bal

        # A Bond Int Paid
        A_Int = np.zeros(months)

        # A Bond Princ Paid
        A_Princ = np.zeros(months)

        # A End Bond Not'l
        A_end_Notl = np.zeros(months)

        ### B

        # B Bond Not'l
        B_Notl = np.zeros(months)
        B_Notl[0] = B_bal

        # B Bond Int Paid
        B_Int = np.zeros(months)

        # B Bond Princ Paid
        B_Princ = np.zeros(months)

        # B End Bond Not'l
        B_end_Notl = np.zeros(months)

        ### Eq

        # Excess Int
        ExcessInt = np.zeros(months)

        # Excess Princ
        ExcessPrinc = np.zeros(months)

        # Resid Int Paid
        ResidIntPaid = np.zeros(months)

        # Resid CFs
        ResidCFs = np.zeros(months)

        # Equity Balance
        EquityBalance = np.zeros(months)
        EquityBalance[0] = Eq_bal

        for i in range(months):
            if i == 0:
                pass
            else:
                A_Notl[i] = A_end_Notl[i - 1]
                B_Notl[i] = B_end_Notl[i - 1]

            if OC:
                if i == months - 1:  # Last
                    A_Princ[i] = min(A_Notl[i], total_notl_after_defaults[i])
                    B_Princ[i] = min(B_Notl[i], total_notl_after_defaults[i] - A_Princ[i])

                    #                     A_Int[i] = min(A_Notl[i] * A_rate / 12, total_int[i])
                    #                     B_Int[i] = min(B_Notl[i] * B_rate / 12, total_int[i] - A_Int[i])
                    #                     ExcessInt[i] = max(total_int[i] - A_Int[i] - B_Int[i], 0)

                    A_oc_ratio = total_notl_after_defaults[i] / A_Notl[i]
                    B_oc_ratio = total_notl_after_defaults[i] / (A_Notl[i] + B_Notl[i])
                    if total_int[i] > A_Notl[i] * A_rate / 12:  # Excess > 0
                        A_Int[i] = A_Notl[i] * A_rate / 12

                        if A_oc_ratio < A_trigger:  # A Fail
                            if A_Princ[i] + total_int[i] - A_Int[i] < A_Notl[i]:
                                A_Princ[i] += total_int[i] - A_Int[i]
                        else:  # A Pass
                            B_Int[i] = min(B_Notl[i] * B_rate / 12, total_int[i] - A_Int[i])
                            if B_oc_ratio < B_trigger:  # A Pass & B Fail
                                if A_Princ[i] + total_int[i] - A_Int[i] < A_Notl[i]:
                                    A_Princ[i] += total_int[i] - A_Int[i]
                            else:  # A Pass & B Pass
                                ExcessInt[i] = max(total_int[i] - A_Int[i] - B_Int[i], 0)
                                ResidIntPaid[i] = ExcessInt[i]
                    else:  # Excess = 0
                        A_Int[i] = total_int[i]

                    ExcessPrinc[i] = max(0, total_princ[i] - A_Princ[i] - B_Princ[i])
                    ResidCFs[i] = ExcessPrinc[i] + ResidIntPaid[i]
                else:  # Not last one
                    A_oc_ratio = total_notl_after_defaults[i] / A_Notl[i]
                    B_oc_ratio = total_notl_after_defaults[i] / (A_Notl[i] + B_Notl[i])
                    if total_int[i] > A_Notl[i] * A_rate / 12:  # Excess > 0
                        A_Int[i] = A_Notl[i] * A_rate / 12

                        if A_oc_ratio < A_trigger:  # A Fail
                            A_Princ[i] = total_int[i] - A_Int[i]
                            B_Princ[i] = 0
                            B_Int[i] = 0
                            ExcessInt[i] = max(total_int[i] - A_Int[i] - B_Int[i] - A_Princ[i], 0)
                            ResidIntPaid[i] = ExcessInt[i]
                        else:  # A Pass
                            B_Int[i] = min(B_Notl[i] * B_rate / 12, total_int[i] - A_Int[i])
                            if B_oc_ratio < B_trigger:  # A Pass & B Fail
                                A_Princ[i] = total_int[i] - A_Int[i]
                                B_Princ[i] = 0
                                ExcessInt[i] = max(total_int[i] - A_Int[i] - B_Int[i] - A_Princ[i], 0)
                                ResidIntPaid[i] = ExcessInt[i]
                            else:  # A Pass & B Pass
                                A_Princ[i] = 0
                                B_Princ[i] = 0
                                ExcessInt[i] = max(total_int[i] - A_Int[i] - B_Int[i] - A_Princ[i], 0)
                                ResidIntPaid[i] = ExcessInt[i]
                    else:  # Excess = 0
                        A_Int[i] = total_int[i]
                        B_Int[i] = 0
                        ExcessInt[i] = max(total_int[i] - A_Int[i] - B_Int[i], 0)

                    ExcessPrinc[i] = max(0, total_princ[i] - A_Princ[i] - B_Princ[i])
                    ResidCFs[i] = ExcessPrinc[i] + ResidIntPaid[i]
            else:
                A_Int[i] = min(A_Notl[i] * A_rate / 12, total_int[i])
                B_Int[i] = min(B_Notl[i] * B_rate / 12, total_int[i] - A_Int[i])
                ExcessInt[i] = max(total_int[i] - A_Int[i] - B_Int[i], 0)

                if i == months - 1:
                    A_Princ[i] = min(A_Notl[i], total_notl_after_defaults[i])
                    B_Princ[i] = min(B_Notl[i], total_notl_after_defaults[i] - A_Princ[i])
                    ExcessPrinc[i] = max(0, total_princ[i] - A_Princ[i] - B_Princ[i])
                    ResidIntPaid[i] = ExcessInt[i]
                    ResidCFs[i] = ExcessPrinc[i] + ResidIntPaid[i]
                else:
                    A_Princ[i] = 0
                    B_Princ[i] = 0
                    ExcessPrinc[i] = 0
                    ResidIntPaid[i] = ExcessInt[i]
                    ResidCFs[i] = ExcessPrinc[i] + ResidIntPaid[i]

            if i == 0:
                EquityBalance[i] = ResidCFs[i]
            else:
                EquityBalance[i] = EquityBalance[i - 1] + ResidCFs[i]

            A_end_Notl[i] = A_Notl[i] - A_Princ[i]
            B_end_Notl[i] = B_Notl[i] - B_Princ[i]

        A_df["A Not'l"] = A_Notl
        A_df["A Int Paid"] = A_Int
        A_df['A Princ Paid'] = A_Princ
        A_df["A End Not'l"] = A_end_Notl

        B_df["B Not'l"] = B_Notl
        B_df["B Int Paid"] = B_Int
        B_df['B Princ Paid'] = B_Princ
        B_df["B End Not'l"] = B_end_Notl

        Eq_df["Excess Int"] = ExcessInt
        Eq_df["Excess Princ"] = ExcessPrinc
        Eq_df["Resid Int Paid"] = ResidIntPaid
        Eq_df["Resid CFs"] = ResidCFs
        Eq_df["Equity Balance"] = EquityBalance

        self.table = total_df.join(A_df.set_index("Month"), on="Month").join(B_df.set_index("Month"), on="Month").join(
            Eq_df.set_index("Month"), on="Month")

    def show(self):
        styled_df = self.table.style.format({
            "Total Not'l": '${:,.2f}',
            "Total Not'l after Defaults": '${:,.2f}',
            'Total Princ': '${:,.2f}',
            'Total Int': '${:,.2f}',
            "Total Not'l after Def & Amort": '${:,.2f}',
            "A Not'l": '${:,.2f}',
            "A Int Paid": '${:,.2f}',
            'A Princ Paid': '${:,.2f}',
            "A End Not'l": '${:,.2f}',
            "B Not'l": '${:,.2f}',
            "B Int Paid": '${:,.2f}',
            'B Princ Paid': '${:,.2f}',
            "B End Not'l": '${:,.2f}',
            "Excess Int": '${:,.2f}',
            "Excess Princ": '${:,.2f}',
            'Resid Int Paid': '${:,.2f}',
            "Resid CFs": '${:,.2f}',
            "Equity Balance": '${:,.2f}',
        })

        display(HTML(styled_df.to_html()))

    def save(self, title):
        self.table.to_csv(title)



def A_lose_point(default_rate, oc=False):
    clo = CLO(A_bal=75, B_bal=15, Eq_bal=10,
              default_rate=default_rate, maturity=10,
              asset_rate=.07, A_rate=.06, B_rate=.09,
              A_trigger=1.2, B_trigger=1.05, A_OC=100 / 75, B_OC=100 / (75 + 15), OC=oc)
    A_ending_bal = clo.table["A End Not'l"].iloc[-1]

    return A_ending_bal


def B_lose_point(default_rate, oc=False):
    clo = CLO(A_bal=75, B_bal=15, Eq_bal=10,
              default_rate=default_rate, maturity=10,
              asset_rate=.07, A_rate=.06, B_rate=.09,
              A_trigger=1.2, B_trigger=1.05, A_OC=100 / 75, B_OC=100 / (75 + 15), OC=oc)
    B_ending_bal = clo.table["B End Not'l"].iloc[-1]

    return B_ending_bal