import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('max_columns',99)

class Model:
    def __init__(self, mtg, dp, i, amortization=25):
        self.mtg = mtg
        self.dp = dp
        if self.dp/self.mtg < .20:
            self.borrowing = (self.mtg-self.dp)*0.028 + (self.mtg-self.dp)
        else:
            self.borrowing = self.mtg-self.dp
        self.i = i
        self.amortization = amortization
        self.monthly_income = pd.DataFrame(dict(gross=[10433.00],net=[6955.00])).T
        self.monthly_expenses = pd.DataFrame(dict(housing=[1075.00],
                                    loans = [374.00],
                                    groceries=[160.00],
                                    clothing=[20.00],
                                    alcohol=[10.00],
                                    gifts=[100.00],
                                    cable=[14.00],
                                    internet=[84.00],
                                    mobile=[175.00],
                                    gas=[200.00],
                                    vehicle_maintenance=[150.00],
                                    public_transit=[150.00],
                                    tenant_insurance=[30.00],
                                    vehicle_insurance=[210.00],
                                    savings_investments=[450.00],
                                    dining=[60.00],
                                    entertainment=[92.00],
                                    hoobies=[21.00],
                                    dental=[40.00],
                                    eywear=[5.00])).T


    def __repr__(self):
        return f"<AMOUNT:${self.mtg:,.2f}| DOWN:{self.dp*100/self.mtg:,.2f}% {f'| INSURANCE:0.0280 ' if self.dp/self.mtg < .20 else ''}| INTEREST:{self.i:.4f}>"

    def calc_amortization(self):
        years = np.array(range(1,self.amortization+1))

        self.df = pd.DataFrame(dict(years=years))

        for index,data in self.df.iterrows():
            if data['years'] == 1:
                self.df.loc[0,'outsanding'] = self.borrowing
                self.df.loc[:,'principal & interest payment'] = np.pmt(self.i/12,self.amortization*12,self.df.loc[0,'outsanding'])*12
                self.df.loc[0,'interest'] = self.df.loc[0,'outsanding']*self.i
                self.df.loc[0,'principal'] = abs(self.df.loc[0,'interest']+self.df.loc[0,'principal & interest payment'])
                self.df.loc[0,'remainder'] = self.df.loc[0,'outsanding']-self.df.loc[0,'principal']

            else:
                self.df.loc[index,'outsanding'] = self.df.loc[index-1,'remainder']
                self.df.loc[index,'interest'] = self.df.loc[index,'outsanding']*self.i
                self.df.loc[index,'principal'] = abs(self.df.loc[index,'interest']+self.df.loc[index,'principal & interest payment'])
                self.df.loc[index,'remainder'] = self.df.loc[index,'outsanding']-self.df.loc[index,'principal']

        return self.df.set_index('years', inplace=True)

    def calculate_monthly_payments(self):
        return -np.pmt(self.i/12,self.amortization*12,self.borrowing)

    def calculate_property_taxes(self, assessed_value = None):
        # https://www.calgary.ca/cfod/finance/Pages/Property-Tax/Tax-Bill-and-Tax-Rate-Calculation/Current-Property-Tax-Rates.aspx
        municipal_tax = 0.0042108
        provincial_tax = 0.0024432
        if assessed_value == None:
            assessed_value = self.mtg
        return assessed_value*municipal_tax + assessed_value*provincial_tax

    def total_monthly_expenses(self):
        return self.monthly_expenses.sum().values[0]
    
    def monthly_surplus(self):
        return self.monthly_income.loc['net'].values[0]-self.total_monthly_expenses()

    def total_debt_expenses(self):
        return self.monthly_expenses.loc[['housing','loans']].sum().values[0]

    def total_mortgage_housing(self):
        return self.calculate_monthly_payments()+self.calculate_property_taxes()/12

    def total_mortgage_debt(self):
        return self.monthly_expenses.loc['loans'].values[0]+self.calculate_monthly_payments()+self.calculate_property_taxes()/12

    def gross_debt_service_ratio(self, ratio=0.32):
        return dict(housing=self.monthly_expenses.loc['housing'].values[0],
                    GDS=self.monthly_income.loc['gross'].values[0]*ratio,
                    ratio=ratio)

    def total_debt_to_service_ratio(self, ratio=0.40):
        return dict(debt_load=self.total_debt_expenses(),
                    TDS=self.monthly_income.loc['gross'].values[0]*ratio,
                    ratio=ratio)
    
    def projected_monthly_expenses(self):
        return self.total_monthly_expenses()-self.monthly_expenses.loc['housing'].values[0]+self.total_mortgage_housing()

        # ax = df[['interest','principal']].plot(marker='o',linestyle='-',color=['r','b'])
        # ax.set_ylim(top=20000)
        # ax.set_ylabel('Annual Payments ($)')
        # ax.grid()

        # ax2 = ax.twinx()
        # df['remainder'].plot(ax=ax2, marker='^',linestyle='--',color='darkgreen', label='remaider')
        # ax2.set_ylabel("Outstanding ammount ($)", color='darkgreen')
        # ax2.yaxis.label.set_color('darkgreen')
        # ax2.tick_params(axis='y', color='darkgreen')
        # ax2.set_ylim(top=mtg*1.1)

        # ax.legend(loc=3)
        # plt.title(f'Mortgage Payments for a ${mtg-dp:,.2f} mortgage @ {i*100:.2f}% interest rate')

        # return ax

# plt.show()

model = Model(400000,60000,0.0265)

print(f"Current Total Monthly Payments: ${model.total_monthly_expenses():,.2f}",
    f"Current GDS: \n{model.gross_debt_service_ratio()}",
    f"Current TDS: \n{model.total_debt_to_service_ratio()}",
    '',
    f'{repr(model)}',
    f"Mortage Debt + Loans + Property Taxes: ${model.total_mortgage_debt():,.2f}",
    f"\tMortgage Housing: ${model.total_mortgage_housing():,.2f}",
    f'\tMonthly Mortgage: ${model.calculate_monthly_payments():,.2f}',
    f'\tMonthly Property Taxes: ${model.calculate_property_taxes()/12:,.2f}',
    f"\tMortgage GDS: {model.total_mortgage_housing()/model.monthly_income.loc['gross'].values[0]:.3f}",
    f"\tMortage TDS: {model.total_mortgage_debt()/model.monthly_income.loc['gross'].values[0]:.3f}"
    '',
    f'Projected Monthly Costs: ${model.projected_monthly_expenses():,.2f}', sep='\n')
