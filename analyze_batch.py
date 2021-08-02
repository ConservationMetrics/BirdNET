import os
import argparse
import time
import numpy as np
from dateutil import parser
import pathlib
import sys

import json
import config as cfg
from metadata import grid
from utils import audio
from model import model
from utils import log

import warnings
warnings.filterwarnings('ignore')

import cmi_metadata
import uuid

RAVEN_HEADER = 'Selection\tView\tChannel\tBegin File\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tSpecies Code\tCommon Name\tConfidence\tRank\n'

##################### LOAD MODEL #######################
def loadModel(path_to_model):

    # Load trained net
    snapshot = model.loadSnapshot(path_to_model)

    # Build simple model
    net = model.buildNet()

    # Load params
    net = model.loadParams(net, snapshot['params'])

    # Compile test function
    test_function = model.test_function(net, layer_index=-2)

    return test_function

######################### EBIRD #########################
def loadGridData():
    grid.load()

def setSpeciesList(lat, lon, week):

    if not week in range(1, 49):
        week = -1

    if cfg.USE_EBIRD_CHECKLIST:
        cfg.WHITE_LIST, cfg.BLACK_LIST = grid.getSpeciesLists(lat, lon, week, cfg.EBIRD_THRESHOLD)
    else:
        cfg.WHITE_LIST = cfg.CLASSES

    log.p(('SPECIES:', len(cfg.WHITE_LIST)), new_line=False)

######################  EXPORT ##########################
def getTimestamp(start, end):

    m_s, s_s = divmod(start, 60)
    h_s, m_s = divmod(m_s, 60)
    start = str(h_s).zfill(2) + ":" + str(m_s).zfill(2) + ":" + str(s_s).zfill(2)

    m_e, s_e = divmod(end, 60)
    h_e, m_e = divmod(m_e, 60)
    end = str(h_e).zfill(2) + ":" + str(m_e).zfill(2) + ":" + str(s_e).zfill(2)

    return start + '-' + end

def decodeTimestamp(t):

    start = t.split('-')[0].split(':')
    end = t.split('-')[1].split(':')

    start_seconds = float(start[0]) * 3600 + float(start[1]) * 60 + float(start[2])
    end_seconds = float(end[0]) * 3600 + float(end[1]) * 60 + float(end[2])

    return start_seconds, end_seconds

def getCode(label):

    codes = grid.CODES

    for c in codes:
        if codes[c] == label:
            return c

    return '????'

def getRavenSelectionTable(p, path):

    # Selection table
    stable = ''

    # Raven selection header
    selection_id = 0

    # Extract valid predictions for every timestamp
    for timestamp in sorted(p):
        rstring = ''
        start, end = decodeTimestamp(timestamp)
        min_conf = 0
        rank = 1
        for c in p[timestamp]:
            if c[1] > cfg.MIN_CONFIDENCE + min_conf and c[0] in cfg.WHITE_LIST:
                selection_id += 1
                rstring += str(selection_id) + '\tSpectrogram 1\t1\t' + path + '\t'
                rstring += str(start) + '\t' + str(end) + '\t' + str(cfg.SPEC_FMIN) + '\t' + str(cfg.SPEC_FMAX) + '\t'
                rstring += getCode(c[0]) + '\t' + c[0].split('_')[1] + '\t' + str(c[1]) + '\t' + str(rank) + '\n'
                rank += 1
            if rank > 3:
                break

        # Write result string to file
        if len(rstring) > 0:
            stable += rstring

    return stable, selection_id

def getAudacityLabels(p, path):

    # Selection table
    stext = ''

    # Extract valid predictions for every timestamp
    for timestamp in sorted(p):
        rstring = ''
        start, end = decodeTimestamp(timestamp)
        min_conf = 0
        for c in p[timestamp]:
            if c[1] > cfg.MIN_CONFIDENCE + min_conf and c[0] in cfg.WHITE_LIST:
                rstring += str(start) + '\t' + str(end) + '\t' + c[0].split('_')[1] + ';' + str(int(c[1] * 100) / 100.0) + '\n'

        # Write result string to file
        if len(rstring) > 0:
            stext += rstring

    return stext

###################### ANALYSIS #########################
def analyzeFile(soundscape, test_function):

    ncnt = 0

    # Store analysis here
    analysis = {}

    # Keep track of timestamps
    pred_start = 0

    # Set species list accordingly
    setSpeciesList(cfg.DEPLOYMENT_LOCATION[0], cfg.DEPLOYMENT_LOCATION[1], cfg.DEPLOYMENT_WEEK)

    # Get specs for file
    spec_batch = []
    for spec in audio.specsFromFile(soundscape,
                                    rate=cfg.SAMPLE_RATE,
                                    seconds=cfg.SPEC_LENGTH,
                                    overlap=cfg.SPEC_OVERLAP,
                                    minlen=cfg.SPEC_MINLEN,
                                    fmin=cfg.SPEC_FMIN,
                                    fmax=cfg.SPEC_FMAX,
                                    win_len=cfg.WIN_LEN,
                                    spec_type=cfg.SPEC_TYPE,
                                    magnitude_scale=cfg.MAGNITUDE_SCALE,
                                    bandpass=True,
                                    shape=(cfg.IM_SIZE[1], cfg.IM_SIZE[0]),
                                    offset=0,
                                    duration=None):

        # Prepare as input
        spec = model.prepareInput(spec)

        # Add to batch
        if len(spec_batch) > 0:
            spec_batch = np.vstack((spec_batch, spec))
        else:
            spec_batch = spec

        # Do we have enough specs for a prediction?
        if len(spec_batch) >= cfg.SPECS_PER_PREDICTION:

            # Make prediction
            p, _ = model.predict(spec_batch, test_function)

            # Calculate next timestamp
            pred_end = pred_start + cfg.SPEC_LENGTH + ((len(spec_batch) - 1) * (cfg.SPEC_LENGTH - cfg.SPEC_OVERLAP))

            # Store prediction
            analysis[getTimestamp(pred_start, pred_end)] = p

            # Advance to next timestamp
            pred_start = pred_end - cfg.SPEC_OVERLAP
            spec_batch = []

    return analysis

######################## MAIN ###########################
def process(soundscape, out_type, test_function):

    # Time
    start = time.time()
    log.p(('PROCESSING:', soundscape.split(os.sep)[-1]), new_line=False)

    # Analyze file
    p = analyzeFile(soundscape, test_function)

    if out_type == 'raven':
        results, _ = getRavenSelectionTable(p, soundscape)
    else:
        results = getAudacityLabels(p, soundscape)

    # Time
    t = time.time() - start

    # Stats
    log.p(('TIME:', int(t)))
    return results

def arguments():
    parser = argparse.ArgumentParser(
        description="Compute model predictions on a single .wav file."
    )
    parser.add_argument(
        '--survey-csv',
        default=None,
        help='Path to the CSV file with deployment/survey data'
    )
    parser.add_argument(
        '--survey-schema',
        help='Survey CSV schema',
        type=lambda s: json.loads(s),
        default=cmi_metadata.SURVEY_SCHEMA
    )
    parser.add_argument(
        '--output-type',
        default='raven',
        help='Output format of analysis results. Values in [\'audacity\', \'raven\']. Defaults to \'raven\'.'
    )
    parser.add_argument(
        "--source-files", help="Path to the files to be processed.", nargs="+"
    )
    parser.add_argument(
        "--source-dir", default="", type=str, help="Path to the working/root directory."
    )
    parser.add_argument("--source-host", default="", type=str)
    parser.add_argument(
        "--dataset-path", help="Path to the output", type=str, required=True
    )

    parser.add_argument(
        "--model-dir-path", help="Path to the model directory.", type=str
    )
    parser.add_argument(
        "--model-tag-path", help="Path to the model mat file.", type=str
    )
    parser.add_argument(
        "--parameters", help="Dict of parameters.", default={}, type=lambda s: json.loads(s))
    parser.add_argument("--processed-path", default=None)
    parser.add_argument(
        "--filename-formats",
        help="Expected filename formats, if non-default",
        nargs="+",
        default=[]
    )
    return parser.parse_args()

def week_from_datetime(dt):
    return dt.isocalendar()[1]

def ensure_directory(dir_path):
    return pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

def resolve_cfg(spp, overlap, sensitivity, min_conf,
                latitude, longitude, deployment_datetime=None, week=None,
                **kwargs):
    if deployment_datetime is not None:
        week = week_from_datetime(deployment_datetime)
    cfg.DEPLOYMENT_LOCATION = (latitude, longitude)
    cfg.DEPLOYMENT_WEEK = week or 25
    cfg.SPEC_OVERLAP = min(2.9, max(0.0, overlap))
    cfg.SPECS_PER_PREDICTION = max(1, spp)
    cfg.SENSITIVITY = max(min(-0.25, sensitivity * -1), -2.0)
    cfg.MIN_CONFIDENCE = min(0.99, max(0.01, min_conf))

def persist_results(dataset_path, output_type, output):
    if len(output) == 0:
        return

    ensure_directory(dataset_path)
    with open(os.path.join(dataset_path, str(uuid.uuid4()) + ".tsv"), "w") as f:
        if output_type == 'raven':
            f.write(RAVEN_HEADER)
        f.write(output)

def main(survey_csv,
         survey_schema,
         source_dir,
         source_files,
         model_tag_path,
         filename_formats,
         dataset_path,
         output_type,
         parameters):
    survey_db = None
    if survey_csv is not None:
        try:
            survey_db = cmi_metadata.survey_db(survey_csv)
        except Exception as e:
            print("Couldn't parse survey CSV because [%s]" % e)

    # Parse dataset
    dataset = [os.path.join(source_dir, source_file)
               for source_file in source_files]
    if not (len(dataset) > 0):
        print("No source files provided!")
        sys.exit(0)

    # Load model
    model_function = loadModel(model_tag_path)

    # Load eBird grid data
    loadGridData()

    results = ""
    results_without_metadata = ""

    for s in dataset:
        file_metadata = cmi_metadata.metadata(survey_db, s, filename_formats, survey_schema=survey_schema)

        configuration = {**parameters, **file_metadata}
        resolve_cfg(**configuration)

        file_results = process(s, output_type, model_function)
        # determine location, determine week, set those, run model
        if len(file_metadata) > 0:
            results += file_results
        else:
            results_without_metadata += file_results

    persist_results(dataset_path, output_type, results)
    persist_results(os.path.join(dataset_path, "without_metadata"), output_type, results_without_metadata)

if __name__ == '__main__':
    options = arguments()
    main(options.survey_csv,
         options.survey_schema,
         options.source_dir,
         options.source_files,
         options.model_tag_path,
         options.filename_formats,
         options.dataset_path,
         options.output_type,
         options.parameters)
