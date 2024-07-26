import os
from datetime import datetime, timezone, timedelta


def convert_to_kst(timestamp):
    utc_time = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
    return utc_time + timedelta(hours=9)


def get_ms():
    now = datetime.now()
    return int(now.timestamp() * 1000) + int(now.microsecond / 1000)


def get_prev_month_ms():
    now = datetime.now()
    one_month_ago = now - timedelta(weeks=4)
    return int(one_month_ago.timestamp() * 1000) + int(one_month_ago.microsecond / 1000)


def get_prev_week_ms():
    now = datetime.now()
    one_week_ago = now - timedelta(weeks=1)
    return int(one_week_ago.timestamp() * 1000) + int(one_week_ago.microsecond / 1000)


def get_prev_day_ms():
    now = datetime.now()
    one_day_ago = now - timedelta(days=1)
    return int(one_day_ago.timestamp() * 1000) + int(one_day_ago.microsecond / 1000)


def get_prev_hour_ms():
    now = datetime.now()
    one_hour_ago = now - timedelta(hours=1)
    return int(one_hour_ago.timestamp() * 1000) + int(one_hour_ago.microsecond / 1000)


def find_files_by_name(directory, name_substring):
    files = []
    for file in os.listdir(directory):
        if name_substring in file:
            files.append(os.path.join(directory, file))
    return files
