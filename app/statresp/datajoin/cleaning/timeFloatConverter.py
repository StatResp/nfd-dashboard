import pandas as pd


def convertTime(time: float):
    """
    If a time is invalid, return None
    TODO: add case for string data type due to potential of "--" (didn't see any though)
    """
    # try:
        # assert isinstance(time, float)
        # assert isinstance(time, int)
    # except:
        # print("time: " + str(time) + " is not float")
        # return None


    if time is None:
        return None
    elif time < 0:
        return None

    hour = None
    minute = None
    if time < 100:
        hour = "00"
        if time < 10:
            minute = "0" + str(time)[0]
        elif 10 <= time < 60:
            minute = str(time)[:2]
    elif time < 1000:
        hour = "0" + str(time)[0]
        if float(str(time)[1:]) < 60:
            minute = str(time)[1:3]
    elif time >= 1000 and time < 2400:
        # 1000.0
        hour = str(time)[:2]
        minute = str(time)[2:4]

    if hour != None and minute != None:
        return (hour + ":" + minute)
    else:
        return None