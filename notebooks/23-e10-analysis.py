# exported and formatted 2024-07-27T12:00 from notebook for profiling with py-spy
# run with `rye run py-spy record --subprocesses --format speedscope -o e10-analysis-run3.speedscope -- python 23-e10-analysis.py`

#!/usr/bin/env python
# coding: utf-8

# get_ipython().run_line_magic("run", '"06-notebook-header.ipynb"')
# manually replaced by:
import numpy as np
from scipy import fft, stats
from matplotlib.pyplot import *

from src.utils import mygrid

import src
from src.utils import energy2db, copy_attributes, smoothed_ir_level
from src.SdmIsmAnalysis import (
    Plane,
    SdmIsmAnalysis,
    estimate_mixing_time_95,
    matlab_with_SDMtools,
    reflection_matrix_from_plane,
)
from src.SimpleSynthesis import SimpleSynthesis
from pathlib import Path
import sofar
import pyfar as pf
import sounddevice
import scipy
import pandas as pd
import haggis
import polars as pl

import logging

file_handler = logging.FileHandler("23-e10-analysis-run3.log")
stream_handler = logging.StreamHandler()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[file_handler, stream_handler],
)

# manually added
logging.info("starting")

eng = matlab_with_SDMtools()

dpm = 2072 / 8.43  # dots per meter, measured in GIMP

approx_volume = 7 * 6 * 2.8
approx_mixing_time = estimate_mixing_time_95(approx_volume)


a = SdmIsmAnalysis(
    eng,
    planes=[Plane(0, np.array([1, 0, 0]))],  # dummy, set later
    mixing_time=approx_mixing_time,
    HRTF_path="data/processed/RWTH-2020-11307/Kemar_HRTF_sofa-onax_normalized.sofa",
)

# manually added
logging.info("reading source fr")

a.read_source_fr("../data/external/RL906-spatial-FR.sofa")


# Levels of loop:
# - room (create new SdmIsmAnalysis with new planes,  etc?)
# - listener (read new listener sofa, run all steps afterwards)
# - source (do new ISM, prune, extract, save, filter reflections, extract, save)

rooms = ["HL05W", "HL06W"]


room_planes = [
    [  # going clockwise from SiL, looking from above
        Plane(4.59 - 1.04, np.array([0.0, 1.0, 0.0])),  # next to SiL
        Plane(4.07 - 0.3, np.array([1.0, 0.0, 0.0])),  # behind FC
        Plane(6.93 + 1.04 - 4.59, np.array([0.0, -1.0, 0.0])),  # behind SiR
        Plane(5.94 + 0.3 - 4.07, np.array([-1.0, 0.0, 0.0])),  # next to BC
        Plane(
            499 / dpm, np.array([-1.0, 1.0, 0.0]) / np.sqrt(2)
        ),  # diagonal, approx. 499 px from origin as measured in GIMP
        Plane(0, np.array([0.0, 0.0, -1.0])),  # floor
        Plane(
            2.8, np.array([0.0, 0.0, 1.0])
        ),  # ceiling, height in paper on page 3 ("Example Measurement")
    ],
    [  # going clockwise from SiL, looking from above
        Plane(4.59 - 1.04, np.array([0.0, 1.0, 0.0])),  # next to SiL
        Plane(4.07 - 0.3, np.array([1.0, 0.0, 0.0])),  # behind FC
        Plane(6.93 + 1.04 - 4.59, np.array([0.0, -1.0, 0.0])),  # behind SiR
        Plane(5.94 + 0.3 - 4.07, np.array([-1.0, 0.0, 0.0])),  # next to BC
        Plane(4.59 - 1.04 - 3 * 0.99, np.array([0.0, 1.0, 0.0])),  # behind BC
        Plane(-(4.07 - 0.3 - 3 * 0.99), np.array([-1.0, 0, 0])),  # behind SiL
        Plane(0, np.array([0.0, 0.0, -1.0])),  # floor
        Plane(
            2.8, np.array([0.0, 0.0, 1.0])
        ),  # ceiling, height in paper on page 3 ("Example Measurement")
    ],
]


listeners = [
    "0.0X_0.0Y",
    "3.0X_2.0Y",
    "1.75X_-2.0Y",
    "-1.75X_-3.0Y",  # lower right corner in floor plan
    "-1.25X_0.0Y",
]


sources = range(5)


for room, planes in zip(rooms, room_planes):
    a.planes = planes
    logging.info(f"set planes for {room}")
    for listener in listeners:
        a.read_sofa(
            Path(f"../data/external/zenodo.10450779/{room}/{room}_{listener}.sofa")
        )
        logging.info(f"read sofa {a.input_sofa_path}")
        for source in sources:
            logging.info(f"calculating source {source}")
            a.calculate_doa(source_index=source)
            a.calculate_ism()
            a.quantize_doa()
            a.smooth_quantized_doa()
            a.calculate_extraction_metrics()
            a.extract_selected()
            a.render_reference_BRIRs()
            a.render_diffuse_BRIRs()
            a.save_result("../data/processed/analysis_e10_full_run3")
            # commented out for run2
            # logging.info(f"filtering reflections")
            # a.filter_reflections()
            # if a.runs.shape[0] > 0:
            #     a.extract_selected()
            #     a.render_diffuse_BRIRs()
            #     a.save_result("../data/processed/analysis_e10_filtered")
