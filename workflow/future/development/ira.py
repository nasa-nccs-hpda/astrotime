from argparse import Namespace
import argparse

parser = argparse.ArgumentParser( prog='ira', usage='python ira.py --help', description='Computes the time extent of an ira.')
parser.add_argument('-sb',  '--start_balance',        type=float, default=1300.0)
parser.add_argument('-gr', '--growth_rate_pct',       type=float, default=4.0)
parser.add_argument('-mw', '--monthly_withdraw',      type=float, default=8.0)
args: Namespace = parser.parse_args()

growth_rate: float = args.growth_rate_pct / 100.0
monthly_growth_rate = growth_rate/12.0
balance: float = args.start_balance
print(f"Start balance: {balance:7.2f}, Monthly withdraw: {args.monthly_withdraw:7.2f}, Growth rate: {growth_rate:6.2f}%, Monthly Growth rate: {monthly_growth_rate:6.4f}%")
for imonth in range(1000):
	interest = balance * (growth_rate/12)
	avail_balance = balance * (1 + growth_rate/12)
	balance = avail_balance - args.monthly_withdraw
	if imonth % 12 == 0:
		print(f"{imonth:3d}: interest={interest:6.2f}, avail balance={avail_balance:8.2f}, new balance={balance:8.2f}")
	if balance <= 0.0:
		print(f"Funds fully allocated in {imonth/12.0} years.")
		break
