# Test code
from datetime import datetime, timedelta
from pysy.utils import Montre

if __name__ == "__main__":
	montre = Montre()
	cur_date = datetime.strptime("2016-1-30", r"%Y-%m-%d")
	
	new_date = montre.manage_time(cur_date, years = 3, months = 13, weeks = 2, days = 30, hours = 25, minutes = 72, seconds = 104)
	# new_date = montre.change_time(cur_date, years=3, months= 1)
	print(cur_date)
	print(new_date)
	print(timedelta(microseconds = 1000))
	print(timedelta(milliseconds = 100))
	print("ok")