# coding: utf-8
import numpy as np

def isleap(year):
    if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
        return "leap"
    else:
        return "non-leap"

def day2date(day, leap):
    if leap == "leap":
        if 0 < day <=31:
            if day < 10:
                return "010" + str(day)
            else:
                return "01" + str(day)
        elif 31 < day <= 60:
            if day - 31 < 10:
                return "020" + str(day - 31)
            else:
                return "02" + str(day - 31)
        elif 60 < day <= 91:
            if day - 60 < 10:
                return "030" + str(day - 60)
            else:
                return "03" + str(day - 60)
        elif 91 < day <= 121:
            if day - 91 < 10:
                return "040" + str(day - 91)
            else:
                return "04" + str(day - 91)
        elif 121 < day <= 152:
            if day - 121 < 10:
                return "050" + str(day - 121)
            else:
                return "05" + str(day - 121)
        elif 152 < day <= 182:
            if day - 152 < 10:
                return "060" + str(day - 152)
            else:
                return "06" + str(day - 152)
        elif 182 < day <= 213:
            if day - 182 < 10:
                return "070" + str(day - 182)
            else:
                return "07" + str(day - 182)
        elif 213 < day <= 244:
            if day - 213 < 10:
                return "080" + str(day - 213)
            else:
                return "08" + str(day - 213)
        elif 244 < day <= 274:
            if day - 244 < 10:
                return "090" + str(day - 244)
            else:
                return "09" + str(day - 244)
        elif 274 < day <= 305:
            if day - 274 < 10:
                return "100" + str(day - 274)
            else:
                return "10" + str(day - 274)
        elif 305 < day <= 335:
            if day - 305 < 10:
                return "110" + str(day - 305)
            else:
                return "11" + str(day - 305)
        elif 335 < day <= 366:
            if day - 335 < 10:
                return "120" + str(day - 335)
            else:
                return "12" + str(day - 335)


    if leap == "non-leap":
        if 0 < day <=31:
            if day < 10:
                return "010" + str(day)
            else:
                return "01" + str(day)
        elif 31 < day <= 59:
            if day - 31 < 10:
                return "020" + str(day - 31)
            else:
                return "02" + str(day - 31)
        elif 59 < day <= 90:
            if day - 59 < 10:
                return "030" + str(day - 59)
            else:
                return "03" + str(day - 59)
        elif 90 < day <= 120:
            if day - 90 < 10:
                return "040" + str(day - 90)
            else:
                return "04" + str(day - 90)
        elif 120 < day <= 151:
            if day - 120 < 10:
                return "050" + str(day - 120)
            else:
                return "05" + str(day - 120)
        elif 151 < day <= 181:
            if day - 151 < 10:
                return "060" + str(day - 151)
            else:
                return "06" + str(day - 151)
        elif 181 < day <= 212:
            if day - 181 < 10:
                return "070" + str(day - 181)
            else:
                return "07" + str(day - 181)
        elif 212 < day <= 243:
            if day - 212 < 10:
                return "080" + str(day - 212)
            else:
                return "08" + str(day - 212)
        elif 243 < day <= 273:
            if day - 243 < 10:
                return "090" + str(day - 243)
            else:
                return "09" + str(day - 243)
        elif 273 < day <= 304:
            if day - 273 < 10:
                return "100" + str(day - 273)
            else:
                return "10" + str(day - 273)
        elif 304 < day <= 334:
            if day - 304 < 10:
                return "110" + str(day - 304)
            else:
                return "11" + str(day - 304)
        elif 334 < day <= 365:
            if day - 334 < 10:
                return "120" + str(day - 334)
            else:
                return "12" + str(day - 334)

def date2day(date, leap):
    date = str(date)
    mon = date[-4: -2]
    if leap == "leap":
        if mon == "01":
            return int(date[-2::])
        if mon == "02":
            return 31 + int(date[-2::])
        if mon == "03":
            return 60 + int(date[-2::])
        if mon == "04":
            return 91 + int(date[-2::])
        if mon == "05":
            return 121 + int(date[-2::])
        if mon == "06":
            return 152 + int(date[-2::])
        if mon == "07":
            return 182 + int(date[-2::])
        if mon == "08":
            return 213 + int(date[-2::])
        if mon == "09":
            return 244 + int(date[-2::])
        if mon == "10":
            return 274 + int(date[-2::])
        if mon == "11":
            return 305 + int(date[-2::])
        if mon == "12":
            return 335 + int(date[-2::])

    if leap == "non-leap":
        if mon == "01":
            return int(date[-2::])
        if mon == "02":
            return 31 + int(date[-2::])
        if mon == "03":
            return 59 + int(date[-2::])
        if mon == "04":
            return 90 + int(date[-2::])
        if mon == "05":
            return 120 + int(date[-2::])
        if mon == "06":
            return 151 + int(date[-2::])
        if mon == "07":
            return 181 + int(date[-2::])
        if mon == "08":
            return 212 + int(date[-2::])
        if mon == "09":
            return 243 + int(date[-2::])
        if mon == "10":
            return 273 + int(date[-2::])
        if mon == "11":
            return 304 + int(date[-2::])
        if mon == "12":
            return 334 + int(date[-2::])
