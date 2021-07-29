import os
import re
from datetime import datetime, timedelta
from six.moves.urllib import parse

import numpy as np
import pandas as pd

audioTypes = ["wav", "WAV", "flac", "aif", "aiff", "mp3"]
FILENAME_FORMATS = [
    "site_yyyymmdd_HHMMSS",
    "site_x_yyyymmdd_HHMMSS",
    "site_x_x_yyyymmdd_HHMMSS",
    "site__x__yyyymmdd_HHMMSS",
    "x_x_x_site_x_yyyymmdd_HHMMSS",
    "site__yyyymmdd_HHMMSS",
]

SURVEY_SCHEMA = {
    "latitude": "Latitude",
    "longitude": "Longitude",
    "location": "Sensor_Name",
    "deployment_date": "Deployment_Date",
    "deployment_time": "Deployment_Time",
    "retrieval_date": "Retrieval_Date",
    "retrieval_time": "Retrieval_Time"
}

def strfind(text, matcher):
    return [m.start() for m in re.finditer(re.escape(matcher), text)]


def strmatch(matcher, textList):
    regex = re.compile("^{}".format(matcher))
    matches = [re.match(regex, text) for text in textList]
    return np.where(matches)[0]


def strcmpi(matcher, textList):
    matcher = matcher.lower()
    matches = [matcher == text.lower() for text in textList]
    return [idx for idx, m in enumerate(matches) if m]


def delim_locs(text, delims):
    # Find the location of the delimiters in fileNameFmt
    delimLoc = np.array([], dtype=int)
    for j in range(len(delims)):
        delimLoc = np.append(delimLoc, strfind(text, delims[j]))
    delimLoc.sort()
    # Remove consecutive delimiters
    delimLoc = np.delete(delimLoc, [np.where(np.diff(delimLoc) == 1)[0] + 1])

    return delimLoc


def parse_filename_metadata(filename, additional_filename_formats=[]):
    try:
        theDay, theTime, location, dateCell = parse_all_fmt(
            filename, FILENAME_FORMATS + additional_filename_formats
        )
        dt = "{} {}".format(theDay, theTime)
        hour = dateCell[3]
        return dt, hour, location
    except:
        return None, None, None

def parse_one_fmt(fileName, fileNameFmt, elapsedWithin=0):

    # Remove preceding path, if any
    parsedPath = parse.urlparse(fileName)
    fileName = os.path.splitext(os.path.basename(parsedPath.path))[0]
    delims = ""

    # parse fileNameFmt
    stuff = ["site", "yy", "mm", "dd", "HH", "MM", "SS", "x"]

    # Define delimiters as the unique set of everything not in 'stuff'
    delims = fileNameFmt
    for j in range(len(stuff)):
        delims = delims.replace(stuff[j], "")
    delims = list(set(delims))

    # Now find the location of the delimiters in fileNameFmt
    delimLoc = delim_locs(fileNameFmt, delims)

    # For the valid 'stuff' (aside from x) find the locations
    slot = np.zeros((len(stuff) - 1,), dtype=int)
    pos = np.zeros((len(stuff) - 1,), dtype=int)
    for j in range(len(stuff) - 1):
        idx = strfind(fileNameFmt, stuff[j])[-1]
        thisDelim = np.where(delimLoc < idx)[0]
        if len(thisDelim) == 0:
            slot[j] = 0
            pos[j] = idx
        else:
            # Which delim slot is this found
            slot[j] = thisDelim[-1] + 1
            pos[j] = idx - delimLoc[thisDelim[-1]]

    delim = np.concatenate(([0], delim_locs(fileName, delims)))

    # Site
    location = fileName[delim[slot[0]] : delim[slot[0] + 1]]

    # Date
    date = ""
    for j in [1, 2, 3]:
        startPos = delim[slot[j]] + pos[j]
        date = date + fileName[startPos : startPos + len(stuff[j])]

    # Time
    time = ""
    for j in [4, 5, 6]:
        startPos = delim[slot[j]] + pos[j]
        time = time + fileName[startPos : startPos + len(stuff[j])]

    try:
        int(date)
        int(time)
    except:
        raise ValueError("fileNameFmt does not match file name")

    date_event = datetime.strptime(date + time, "%y%m%d%H%M%S") + timedelta(
        seconds=elapsedWithin
    )
    date_str_event = date_event.strftime("%d %m %Y %H %M %S")
    date_cell = date_str_event.split(" ")

    dateField = "-".join(date_cell[2:None:-1])
    timeField = ":".join(date_cell[3:7])

    return dateField, timeField, location, date_cell


def parse_all_fmt(fileName, fileNameFmt, elapsedWithin=0):

    if isinstance(fileNameFmt, str):
        fileNameFmt = [fileNameFmt]

    for fmt in fileNameFmt:
        try:
            dateField, timeField, location, date_cell = parse_one_fmt(
                fileName, fmt, elapsedWithin=elapsedWithin
            )
            break
        except Exception as e:
            print(e)
            location = None
            dateField = None
            timeField = None
            date_cell = [None] * 6

    if location is None:
        print(
            "Warning: Could not parse site/date/time from {} using formats {}".format(
                fileName, fileNameFmt
            )
        )

    return dateField, timeField, location, date_cell

def metadata(survey_db, filename, additional_filename_formats=[], survey_schema=SURVEY_SCHEMA):
    dt, hour, location = parse_filename_metadata(filename, additional_filename_formats)

    # if failed to parse, return empty metadata
    if dt is None or location is None:
        return {}
    elif survey_db is None:
        return {
            "location": location,
            "deployment_datetime": pd.to_datetime(dt)
        }

    survey_for_location = survey_db[survey_db[survey_schema['location']] == location]
    if len(survey_for_location) == 0:
        return {
            "location": location,
            "deployment_datetime": pd.to_datetime(dt)
        }
    else:
       survey_for_location = survey_for_location.iloc[0]

    if (pd.to_datetime(dt) > survey_for_location['deployment_datetime'] and
        pd.to_datetime(dt) < survey_for_location['retrieval_datetime']):
        return {
            "location": location,
            "latitude": survey_for_location[survey_schema['latitude']],
            "longitude": survey_for_location[survey_schema['longitude']],
            "deployment_datetime": pd.to_datetime(dt)
        }
    else:
        # if outside of the retrieval/deployment datetimes, return empty metadata
        return {}

def survey_db(filename, survey_schema=SURVEY_SCHEMA):
    db = pd.read_csv(filename)
    db['retrieval_datetime'] = pd.to_datetime(db[survey_schema['retrieval_date']] + " " + db[survey_schema['retrieval_time']])
    db['deployment_datetime'] = pd.to_datetime(db[survey_schema['deployment_date']] + " " + db[survey_schema['deployment_time']])
    return db
