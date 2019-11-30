from datetime import datetime, timedelta, date # date type is not datetime, it only accepts year, month and day.
from time import gmtime, strftime, ctime

class Montre(object):
	def __init__(self):
		super()

	def to_date(self, date_str, format = r"%Y-%m-%d"):
		return datetime.strptime(date_str, format)

	def to_str(self, cur_date, format = r"%Y-%m-%d"):
		return cur_date.strftime(format)

	# Check if the int given year is a leap year
	# return true if leap year or false otherwise
	def is_leap_year(self, year):
		if(year % 4) == 0:
			if(year % 100) == 0:
				if(year % 400) == 0:
					return True
				else:
					return False
			else:
				return True
		else:
			return False

	def manage_time(self, cur_date, years = 0, months = 0, weeks = 0, days = 0, hours = 0, minutes = 0, seconds = 0):
		# the finest scale is second.
		# input time must be datetime type
		if not isinstance(cur_date, datetime):
			raise(ValueError)
		# set output format
		format = r"%Y-%m-%d %H:%M:%S.%f"
		# disintegrate input time into subitems
		cur_year = cur_date.year
		cur_month = cur_date.month
		cur_day = cur_date.day
		cur_hour = cur_date.hour
		cur_minute = cur_date.minute
		cur_second = cur_date.second
		cur_ms = cur_date.microsecond
		# manage year add/substract
		if years != 0:
			cur_year = cur_year + years
		# mange month add/substract
		cur_month = cur_month + months
		if cur_month > 12:
			cur_year = int(cur_year + cur_month // 12)
			cur_month = int(cur_month % 12)

		if (cur_month == 2) and (cur_day > 28):
			if self.is_leap_year(cur_year):
				cur_day = 29
			else:
				cur_day = 28
		str_date = f"{cur_year}-{cur_month}-{cur_day} {cur_hour}:{cur_minute}:{cur_second}.{cur_ms}" 
		cur_date = datetime.strptime(str_date, format)
		# manage the rest
		delta_seconds = 0
		if weeks != 0:
			delta_seconds = delta_seconds +  weeks * 7 * 24 * 60 * 60
		if days != 0:
			delta_seconds = delta_seconds + days * 24 * 60 * 60
		if hours != 0:
			delta_seconds = delta_seconds + hours * 60 * 60
		if minutes != 0:
			delta_seconds = delta_seconds + minutes * 60
		if seconds != 0:
			delta_seconds = delta_seconds + seconds
		cur_date = cur_date + timedelta(seconds = delta_seconds)
		return cur_date


# # Test code
# if __name__ == "__main__":
# 	montre = Montre()
# 	cur_date = datetime.strptime("2016-1-30", r"%Y-%m-%d")
	
# 	new_date = montre.manage_time(cur_date, years = 3, months = 13, weeks = 2, days = 30, hours = 25, minutes = 72, seconds = 104)
# 	# new_date = montre.change_time(cur_date, years=3, months= 1)
# 	print(cur_date)
# 	print(new_date)
# 	print(timedelta(microseconds = 1000))
# 	print(timedelta(milliseconds = 100))
# 	print("ok")