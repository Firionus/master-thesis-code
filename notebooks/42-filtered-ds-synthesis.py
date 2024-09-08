# run with `rye run py-spy record --subprocesses --format speedscope -o 42-filtered-ds-synthesis-run1.speedscope -- python 42-filtered-ds-synthesis.py`

import logging
from pathlib import Path
import os

from src.BrirComparison import brir_comparison
from src.SimpleSynthesis import SimpleSynthesis

import polars as pl

file_handler = logging.FileHandler("42-filtered-ds-synthesis-run1.log")
stream_handler = logging.StreamHandler()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[file_handler, stream_handler],
)

# manually added
logging.info("starting")

rooms = ["HL05W", "HL06W"]

listeners = [
    "0.0X_0.0Y",
    "3.0X_2.0Y",
    "1.75X_-2.0Y",
    "-1.75X_-3.0Y",  # lower right corner in floor plan
    "-1.25X_0.0Y",
]

sources = range(5)

# synthesize from this one
a = (
    SimpleSynthesis()
    .read_HRTFs(
        "../data/processed/RWTH-2020-11307/Kemar_HRTF_sofa-onax_normalized.sofa"
    )
    .read_source_fr("../data/external/RL906-spatial-FR.sofa")
)

# try to match this one
b = SimpleSynthesis()

main_path = Path("../data/processed/analysis_e10_full_run2")

dummy_run = False

# can be used to exclude froms that were already calculated
exclude_froms = []

for room in rooms:
    logging.info(f"room {room}")
    for from_listener in listeners:
        for from_source in sources:
            from_name = f"{room}_{from_listener}_{from_source}SRC"
            if from_name in exclude_froms:
                continue

            from_path = main_path / from_name
            dummy_run or a.read_input_sofas(from_path)
            logging.info(f"read input sofas in {from_path}")

            runs = pl.read_parquet(from_path / "runs.parquet").with_row_index()
            filtered_imsrcs = runs.filter(pl.col("filter_energy") >= 5.5)[
                "index"
            ].to_list()
            logging.info(
                f"filtering imsrcs indices: {filtered_imsrcs} | src: {runs[filtered_imsrcs]['src'].to_list()}"
            )

            save_path = from_path / "extrapolation_quality2"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for to_listener in listeners:
                for to_source in sources:
                    to_name = f"{room}_{to_listener}_{to_source}SRC"
                    to_path = main_path / to_name
                    dummy_run or b.read_input_sofas(to_path)
                    logging.info(f"extrapolating to {to_name}")

                    dropout_data = []
                    if not dummy_run:
                        logging.info("extrapolating with filtered reflections")
                        dropout_brirs = a.extrapolating_synthesis(
                            b.source_position,
                            b.source_orientation,
                            b.listener_position,
                            ignore_imsrcs=[],
                            imsrcs_make_unextrapolated=filtered_imsrcs,
                        )
                        data = brir_comparison(b.reference_brirs, dropout_brirs)
                        data["condition"] = "filtered"
                        dropout_data.append(data)

                        logging.info("extrapolating DS only")
                        dropout_brirs = a.extrapolating_synthesis(
                            b.source_position,
                            b.source_orientation,
                            b.listener_position,
                            ignore_imsrcs=[],
                            imsrcs_make_unextrapolated=list(
                                range(1, len(a.ism_histories))
                            ),  # all except 0
                        )
                        data = brir_comparison(b.reference_brirs, dropout_brirs)
                        data["condition"] = "ds_only"
                        dropout_data.append(data)

                    df_dropout = pl.DataFrame(dropout_data)
                    df_dropout.write_parquet(save_path / to_name)
