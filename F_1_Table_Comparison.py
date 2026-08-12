# =============================================================================
# CELL 1A — DEPENDENCY INSTALL
# =============================================================================
# EXPLANATION:
# This cell installs the Python packages required for the SAM3 production
# notebook to run correctly in Databricks.
#
# This setup cell prepares the Python environment by:
#   1) installing core SAM3 runtime dependencies
#   2) installing image / video / mask-processing utilities
#   3) pinning OpenCV and NumPy to a compatible version combination
#
# IMPORTANT:
#   - Run this only when setting up a new Databricks cluster/session.
#   - Do not run this inside the production batch-inference loop.
#   - After this cell completes, run CELL 1B immediately.
#   - Later cells assume these packages are already installed and available.
# =============================================================================


# -----------------------------------------------------------------------------
# 1.1 Install SAM3 runtime dependencies
# -----------------------------------------------------------------------------
# EXPLANATION:
# These packages support:
#   - model utilities
#   - image / video decoding
#   - mask and detection utilities
#   - text-processing helpers used by the SAM3 codebase
# -----------------------------------------------------------------------------
%pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    iopath timm decord pycocotools ftfy scikit-image==0.24.0


# -----------------------------------------------------------------------------
# 1.2 Install compatible OpenCV + NumPy versions
# -----------------------------------------------------------------------------
# EXPLANATION:
# NumPy and OpenCV version mismatches can cause ABI/import errors.
#
# NumPy is pinned to 1.26.4 because this is the known stable version used by the
# current SAM3 Databricks workflow.
# -----------------------------------------------------------------------------
%pip install opencv-python numpy==1.26.4 --force-reinstall


# =============================================================================
# CELL 1B — HARD RESTART PYTHON
# =============================================================================
# EXPLANATION:
# Databricks must restart the Python process so the newly installed packages are
# loaded cleanly into the notebook session.
#
# IMPORTANT:
#   - Run this immediately after CELL 1A.
#   - After restart, continue from CELL 2.
#   - Do not place production inference code in this cell.
# =============================================================================


# -----------------------------------------------------------------------------
# 1B.1 Restart Python
# -----------------------------------------------------------------------------
# EXPLANATION:
# This clears the current Python process and reloads the environment with the
# packages installed in CELL 1A.
# -----------------------------------------------------------------------------
dbutils.library.restartPython()


# =============================================================================
# CELL 2 — ENVIRONMENT + HUGGING FACE CACHE
# =============================================================================
# EXPLANATION:
# This cell configures the Hugging Face cache locations used by the SAM3
# production notebook.
#
# It does three things:
#   1) points HF_HOME to the shared Databricks Volume cache
#   2) points HF_HUB_CACHE to the Hugging Face model hub cache location
#   3) defines a reusable CACHE_DIR variable for later cells
#
# IMPORTANT:
#   - Run this before any Hugging Face / SAM3 model-loading logic.
#   - This cell does not install packages.
#   - This cell keeps model artifacts in persistent Databricks Volume storage.
#   - For production clusters, confirm that the Volume path exists and is
#     accessible by the user / job cluster.
# =============================================================================


# -----------------------------------------------------------------------------
# 2.1 Import required standard library modules
# -----------------------------------------------------------------------------
# EXPLANATION:
# os is used to set environment variables and create cache directories.
# -----------------------------------------------------------------------------
import os


# -----------------------------------------------------------------------------
# 2.2 Configure Hugging Face cache paths
# -----------------------------------------------------------------------------
# EXPLANATION:
# These environment variables tell Hugging Face where to store and read cached
# model artifacts, configs, tokenizers, and downloaded files.
#
# HF_HOME:
#   Main Hugging Face cache root.
#
# HF_HUB_CACHE:
#   Specific cache path used by huggingface_hub for downloaded model files.
# -----------------------------------------------------------------------------
os.environ["HF_HOME"] = (
    "/Volumes/models/hf_cache"
)

os.environ["HF_HUB_CACHE"] = (
    "/Volumes/models/hf_cache/hub"
)


# -----------------------------------------------------------------------------
# 2.3 Ensure cache directories exist
# -----------------------------------------------------------------------------
# EXPLANATION:
# Create the cache folders if they do not already exist so later model-download
# and model-loading steps can safely use them.
# -----------------------------------------------------------------------------
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)


# -----------------------------------------------------------------------------
# 2.4 Define reusable cache directory variable
# -----------------------------------------------------------------------------
# EXPLANATION:
# CACHE_DIR is a convenience variable used by later cells when loading models,
# checkpoints, or related Hugging Face artifacts.
# -----------------------------------------------------------------------------
CACHE_DIR = os.environ["HF_HOME"]


# -----------------------------------------------------------------------------
# 2.5 Print configured cache paths
# -----------------------------------------------------------------------------
# EXPLANATION:
# This is a lightweight sanity check so the notebook run clearly records which
# persistent Databricks Volume paths are being used.
#
# NOTE:
# For a fully silent production job, this print block can later be wrapped with
# a VERBOSE flag.
# -----------------------------------------------------------------------------
print("HF_HOME     :", os.environ["HF_HOME"])
print("HF_HUB_CACHE:", os.environ["HF_HUB_CACHE"])
print("CACHE_DIR   :", CACHE_DIR)



# =============================================================================
# CELL 3A — CORE IMPORTS + SYSTEM CHECKS
# =============================================================================
# EXPLANATION:
# This cell loads the core Python libraries needed for the SAM3 production
# notebook and performs an early runtime sanity check.
#
# It does four things:
#   1) imports standard library, data, image, plotting, and ML packages
#   2) applies controlled global warning behaviour
#   3) records installed runtime versions
#   4) verifies that CUDA / GPU is available for SAM3 inference
#
# IMPORTANT:
#   - Run this after CELL 2.
#   - Run this before any SAM3 model setup or batch inference logic.
#   - This cell does not load the SAM3 model.
#   - SAM3-specific imports such as build_sam3_image_model, Sam3Processor,
#     and plot_results should stay in the later SAM3 import cell.
#   - This cell should fail early if the notebook is not attached to a GPU
#     cluster, because SAM3 inference is expected to run on CUDA.
# =============================================================================


# -----------------------------------------------------------------------------
# 3A.1 Standard library imports
# -----------------------------------------------------------------------------
# EXPLANATION:
# These modules support:
#   - file and path handling
#   - Python path setup for local SAM3 repo imports
#   - JSON / metadata output
#   - runtime timing
#   - deterministic sampling / QA selection
#   - batch error handling
#   - cleanup / garbage collection
# -----------------------------------------------------------------------------
import os
import gc
import re
import io
import sys
import json
import time
import glob
import math
import shutil
import random
import zipfile
import warnings
import traceback

from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# 3A.2 Data handling imports
# -----------------------------------------------------------------------------
# EXPLANATION:
# pandas is used for image inventories, detection tables, trace tables, and
# final production outputs.
# -----------------------------------------------------------------------------
import pandas as pd


# -----------------------------------------------------------------------------
# 3A.3 Image, geometry, and visualisation imports
# -----------------------------------------------------------------------------
# EXPLANATION:
# These packages support:
#   - image loading and conversion
#   - OpenCV mask morphology / geometry operations
#   - NumPy array operations
#   - optional debug overlays and plots
#   - optional final QA / presentation image rendering
# -----------------------------------------------------------------------------
import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image, ImageOps, ImageDraw


# -----------------------------------------------------------------------------
# 3A.4 Machine learning imports
# -----------------------------------------------------------------------------
# EXPLANATION:
# torch / torchvision are required for CUDA checks and SAM3 runtime execution.
#
# NOTE:
# torchvision is not directly required by CELL 16, but keeping it here is useful
# for runtime version checks and compatibility diagnostics.
# -----------------------------------------------------------------------------
import torch
import torchvision


# -----------------------------------------------------------------------------
# 3A.5 Notebook display import
# -----------------------------------------------------------------------------
# EXPLANATION:
# display is used by helper functions such as _safe_display in debug/inspection
# paths.
#
# In production batch jobs, display output should eventually be disabled or
# gated behind debug flags, but importing display here keeps the helper layer
# compatible with the cleaned CELL 16 workflow.
# -----------------------------------------------------------------------------
try:
    from IPython.display import display
except Exception:
    display = None


# -----------------------------------------------------------------------------
# 3A.6 Global warning behaviour
# -----------------------------------------------------------------------------
# EXPLANATION:
# Keep this True for a quieter notebook. Set to False while debugging package,
# CUDA, or model-loading issues.
# -----------------------------------------------------------------------------
SUPPRESS_WARNINGS = False

if SUPPRESS_WARNINGS:
    warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# 3A.7 Print runtime version information
# -----------------------------------------------------------------------------
# EXPLANATION:
# This records the active package versions in the notebook output. This is useful
# when comparing runs across clusters or debugging environment issues.
# -----------------------------------------------------------------------------
print("Runtime package versions:")
print(f"  PyTorch version     : {torch.__version__}")
print(f"  Torch CUDA build    : {torch.version.cuda}")
print(f"  TorchVision version : {torchvision.__version__}")
print(f"  NumPy version       : {np.__version__}")
print(f"  OpenCV version      : {cv2.__version__}")
print(f"  Pillow version      : {PIL.__version__}")


# -----------------------------------------------------------------------------
# 3A.8 Verify GPU / CUDA availability
# -----------------------------------------------------------------------------
# EXPLANATION:
# SAM3 production inference is expected to run on GPU.
#
# If CUDA is unavailable, fail early here instead of failing later during model
# setup or batch inference.
# -----------------------------------------------------------------------------
CUDA_AVAILABLE = bool(torch.cuda.is_available())

print("\nCUDA runtime check:")
print(f"  CUDA available      : {CUDA_AVAILABLE}")

if not CUDA_AVAILABLE:
    raise RuntimeError(
        "CUDA/GPU is not available.\n"
        "Please attach this notebook to a GPU cluster before running SAM3."
    )


# -----------------------------------------------------------------------------
# 3A.9 Define reusable runtime device
# -----------------------------------------------------------------------------
# EXPLANATION:
# DEVICE is used by later model-loading and inference cells.
#
# Keep this as a string because the cleaned CELL 16 checks for DEVICE and uses it
# as a simple runtime setting.
# -----------------------------------------------------------------------------
DEVICE = "cuda"

print(f"  CUDA device count   : {torch.cuda.device_count()}")
print(f"  Current CUDA device : {torch.cuda.current_device()}")
print(f"  CUDA device name    : {torch.cuda.get_device_name(0)}")
print(f"  DEVICE              : {DEVICE}")


# -----------------------------------------------------------------------------
# 3A.10 Optional GPU memory sanity check
# -----------------------------------------------------------------------------
# EXPLANATION:
# These values help confirm the notebook is attached to the expected GPU type
# before model loading.
# -----------------------------------------------------------------------------
gpu_props = torch.cuda.get_device_properties(0)

print("\nGPU memory:")
print(f"  Total memory GB     : {gpu_props.total_memory / (1024 ** 3):.2f}")
print(f"  Allocated memory GB : {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f}")
print(f"  Reserved memory GB  : {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f}")


# =============================================================================
# CELL 3B — GLOBAL CONSTANTS + NOTEBOOK CONFIG
# =============================================================================
# EXPLANATION:
# This cell defines notebook-wide constants used across the Databricks SAM3
# production workflow.
#
# WHAT THIS CELL DOES:
#   1) defines runtime constants
#   2) defines shared SAM3 thresholds
#   3) defines shared file / naming / overwrite controls
#   4) defines CELL 13 pole-detection constants
#   5) defines CELL 13 selected-pole overlay constants
#   6) defines CELL 14 fixed-canvas pole-top ROI constants
#   7) defines CELL 16 crossarm-detection and post-processing constants
#   8) defines CELL 17 insulator-detection and QA constants
#   9) creates a shared SAM3_TASK_CONFIG dictionary
#
# IMPORTANT:
#   - Run this after CELL 3A.
#   - Later cells should read these values from globals().
#   - Path definitions remain in their path-specific cells.
#   - Single-row debug controls should stay inside debug/prototype cells.
#   - Production CELL 16 should use CROSSARM_PROMPT_TEXT, not PROMPT_TEXT.
# =============================================================================

# =============================================================================
# 3B.1 SHARED SAM3 THRESHOLDS
# =============================================================================
# EXPLANATION:
# Keep common SAM3 thresholds in one place.
#
# TEXT threshold:
#   Filters prompt-level detections.
#
# MASK threshold:
#   Filters mask pixels during post-processing.
# =============================================================================

GLOBAL_TEXT_SCORE_THRESHOLD = 0.30
MASK_THRESHOLD = 0.50


# =============================================================================
# 3B.2 SHARED FILE / NAMING CONSTANTS
# =============================================================================
# EXPLANATION:
# These constants are shared by image discovery, ingestion, output naming, and
# downstream production tables.
# =============================================================================

VALID_IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
)

IMAGE_ID_PREFIX = "img"


# =============================================================================
# 3B.3 SHARED RUN CONTROLS
# =============================================================================
# EXPLANATION:
# Keep destructive rebuild controls in one visible location.
#
# IMPORTANT:
#   Check these before running a full production batch.
# =============================================================================

OVERWRITE_BRONZE = True
OVERWRITE_POLE_ROIS = True


# =============================================================================
# 3B.4 CONFIG SUMMARY CONTROL
# =============================================================================
# EXPLANATION:
# Controls whether this config cell prints a summary.
#
# NOTE:
#   For scheduled production jobs, this can be set to False for quieter logs.
# =============================================================================

PRINT_CONFIG_SUMMARY = True


# =============================================================================
# 3B.5 CELL 13 — POLE DETECTION PROMPTS + THRESHOLDS
# =============================================================================
# EXPLANATION:
# These constants are used by the selected-pole detection stage.
# =============================================================================

POLE_PROMPT_TEXT = ["utility pole"]
POLE_TEXT_THRESHOLD = GLOBAL_TEXT_SCORE_THRESHOLD


# =============================================================================
# 3B.6 CELL 13 — POLE POST-PROCESSING CONSTANTS
# =============================================================================
# EXPLANATION:
# These constants control selected-pole candidate filtering and ranking.
# =============================================================================

POLE_MIN_SCORE = 0.25
POLE_MIN_AREA_FRAC = 0.005
POLE_MIN_HEIGHT_FRAC = 0.15
POLE_MIN_ASPECT = 1.80
POLE_MAX_WIDTH_FRAC = 0.10
POLE_MAX_BOX_W_PX = 500

SHAFT_WIDTH_FRAC_THRESHOLD = 0.12
SHAFT_PENALTY_FACTOR = 0.40

W_X_CENTER = 0.45
W_HEIGHT = 0.30
W_AREA = 0.10
W_CONF = 0.10
W_EDGE = 0.05


# =============================================================================
# 3B.7 CELL 13 — SELECTED-POLE OVERLAY STYLING
# =============================================================================
# EXPLANATION:
# These constants control saved selected-pole QA overlays only.
#
# IMPORTANT:
#   Full-resolution coordinates and masks remain the source of truth.
# =============================================================================

POLE_SELECTED_MASK_RGB = (1.0, 0.0, 0.0)
POLE_SELECTED_MASK_ALPHA = 0.28
POLE_SELECTED_BOX_COLOR = "red"
POLE_SELECTED_BOX_LINEWIDTH = 3.0
POLE_SELECTED_TEXT_COLOR = "white"
POLE_SELECTED_LABEL_FONTSIZE = 14
POLE_SELECTED_LABEL_BG_ALPHA = 0.85
POLE_SELECTED_LABEL_BBOX_PAD = 4
POLE_SELECTED_LABEL_Y_OFFSET = 12
POLE_OVERLAY_MAX_WIDTH = 1600
NO_RELIABLE_POLE_LABEL_TEXT = "NO RELIABLE POLE FOUND"


# =============================================================================
# 3B.8 CELL 14 — FIXED-CANVAS POLE-TOP ROI SETTINGS
# =============================================================================
# EXPLANATION:
# These constants define the fixed-size saved ROI and padding behaviour for the
# Silver pole-top ROI generation stage.
# =============================================================================

FIXED_ROI_WIDTH = 2600
FIXED_ROI_HEIGHT = 3500
POLE_TOP_BUFFER_ABOVE = 500
PAD_RGB = (0, 0, 0)


# =============================================================================
# 3B.9 CELL 16 — CROSSARM SAM3 PROMPT SETTINGS
# =============================================================================
# EXPLANATION:
# These constants control the crossarm prompt sent to SAM3.
#
# IMPORTANT:
#   Use CROSSARM_PROMPT_TEXT in production instead of a generic PROMPT_TEXT so
#   pole prompts and crossarm prompts do not get confused.
# =============================================================================

CROSSARM_PROMPT_TEXT = "utility pole crossarm"
CROSSARM_TEXT_THRESHOLD = GLOBAL_TEXT_SCORE_THRESHOLD


# =============================================================================
# 3B.10A CELL 16 — RAW SCORE PREFILTER
# =============================================================================
# EXPLANATION:
# Removes weak SAM3 crossarm detections before more expensive geometric checks.
# =============================================================================

CROSSARM_RAW_SCORE_REMOVE_MAX = 0.39


# =============================================================================
# 3B.10b CELL 16 - FULL POLE CONTAINER VETO
# =============================================================================
# -----------------------------------------------------------------------------
# Full-pole container veto before containment suppression
# -----------------------------------------------------------------------------
POLE_TRUNK_CONTAINER_VETO_ENABLED = True

# A false container box that swallows almost the entire selected-pole mask should
# not be allowed into containment suppression.
POLE_TRUNK_CONTAINER_VETO_MIN_POLE_CONTAINMENT = 0.90

# The box must also span a large fraction of the visible pole height.
# This prevents normal crossarm boxes that touch the pole top from being removed.
POLE_TRUNK_CONTAINER_VETO_MIN_VERTICAL_SPAN_RATIO = 0.65


# =============================================================================
# 3B.11 CELL 16 — CONTAINMENT SUPPRESSION SETTINGS
# =============================================================================
# EXPLANATION:
# These constants remove duplicate or fragment detections while protecting real
# overlapping crossarms using mask-containment evidence.
# =============================================================================

CONTAINMENT_THRESHOLD = 0.80
MIN_AREA_RATIO = 1.20
MIN_SCORE_ADVANTAGE = 0.0

MASK_CONTAINMENT_FILTER_ENABLED = True
MASK_CONTAINMENT_VETO_THRESHOLD = 0.30

NEAR_TOTAL_BOX_CONTAINMENT_THRESHOLD = 0.95
MASK_CONTAINMENT_HIGH = 0.80

PAIR_DEBUG_MIN_BOX_CONTAINMENT = 0.50


# =============================================================================
# 3B.12 CELL 16 — MAIN-CLUSTER FILTER SETTINGS
# =============================================================================
# EXPLANATION:
# Keeps the main spatial group of crossarm detections and removes isolated false
# positives.
# =============================================================================

CENTER_DIST_FACTOR = 2.75


# =============================================================================
# 3B.13 CELL 16 — POLE-OVERLAP / POLE-CORRIDOR FILTER SETTINGS
# =============================================================================
# EXPLANATION:
# These constants remove detections that are far away from the selected pole or
# detections that are mostly pole rather than crossarm.
# =============================================================================

POLE_MASK_FILTER_ENABLED = True

POLE_OVERLAP_MIN_FRACTION = 0.001
POLE_OVERLAP_REJECT_FRACTION = 0.70
POLE_OVERLAP_MAX_FRACTION = POLE_OVERLAP_REJECT_FRACTION

TOP_BAND_ABOVE = 80
TOP_BAND_BELOW = 250

MIN_RELATIVE_WIDTH_TO_MAX = 0.35
POLE_ATTACH_MARGIN_PX = 120

# -----------------------------------------------------------------------------
# One-sided lower thin-arm veto
# -----------------------------------------------------------------------------
# EXPLANATION:
# Removes streetlight-arm / cantilever-arm false positives that SAM3 detects as
# crossarms.
#
# This rule is intentionally inside the pole-overlap / pole-corridor stage
# because it needs selected-pole geometry and should run before merge/split/PCA.
#
# Removal signature:
#   - another upper/main crossarm candidate exists
#   - candidate is lower than the primary/top candidate
#   - candidate is thin / rod-like
#   - candidate extends mostly on one side of the pole centreline
#   - candidate touches the expanded pole attachment corridor
# -----------------------------------------------------------------------------
ONE_SIDED_LOWER_ARM_VETO_ENABLED = True

ONE_SIDED_LOWER_ARM_SIDE_BALANCE_MAX = 0.15
ONE_SIDED_LOWER_ARM_SHORT_SIDE_FRAC_MAX = 0.15

ONE_SIDED_LOWER_ARM_ASPECT_MIN = 6.0
ONE_SIDED_LOWER_ARM_HEIGHT_TO_MEDIAN_MAX = 0.80

ONE_SIDED_LOWER_ARM_MIN_Y_GAP_PX = 120
ONE_SIDED_LOWER_ARM_MIN_Y_GAP_FACTOR = 1.25


# =============================================================================
# 3B.14 CELL 16 — SAME-CROSSARM CONTINUITY MERGE SETTINGS
# =============================================================================
# EXPLANATION:
# These constants merge two detections when they are likely left/right fragments
# of the same physical crossarm.
# =============================================================================

SAME_XARM_MERGE_ENABLED = True
SHOW_SAME_XARM_MERGE_DEBUG = False

SAME_XARM_MERGE_MIN_MASK_PIXELS = 40
SAME_XARM_MERGE_MAX_ANGLE_DIFF_DEG = 20.0
SAME_XARM_MERGE_MAX_PERP_DIST_PX = 90.0
SAME_XARM_MERGE_MAX_GAP_PX = 450.0

SAME_XARM_MERGE_REQUIRE_POLE_BRIDGE = True

SAME_XARM_MERGE_BOX_PAD_PX = 4
SAME_XARM_MERGE_SCORE_MODE = "max"


# =============================================================================
# 3B.15 CELL 16 — RESERVED LEVEL-DEDUPE SETTINGS
# =============================================================================
# EXPLANATION:
# These constants are currently reserved. The current CELL 16 logic does not run
# a level-band dedupe stage.
# =============================================================================

CROSSARM_LEVEL_FILTER_ENABLED = True
CROSSARM_LEVEL_BAND_FACTOR = 0.60
MAX_BOX_H_TO_MEDIAN_RATIO = 1.80
KEEP_PER_LEVEL = 4


# =============================================================================
# 3B.16 CELL 16 — PCA CLEANUP SETTINGS
# =============================================================================
# EXPLANATION:
# These constants identify suspicious broad/messy crossarm masks that should
# have one dominant axis.
# =============================================================================

CROSSARM_PCA_FILTER_ENABLED = True

PCA_SUSPICIOUS_ASPECT_MAX = 2.20
PCA_SUSPICIOUS_HEIGHT_TO_MEDIAN_MIN = 1.15
PCA_SUSPICIOUS_REL_WIDTH_MAX = 0.85

PCA_MIN_MASK_PIXELS = 80
PCA_MIN_PC1_RATIO = 0.85
PCA_MIN_ANISOTROPY = 4.00


# =============================================================================
# 3B.17 CELL 16 — SINGLE-BOX X-SPLIT SETTINGS
# =============================================================================
# EXPLANATION:
# These constants control splitting one broad SAM3 detection into two crossing
# crossarms when the mask/box looks like an X-shaped crossarm.
# =============================================================================

SINGLE_XSPLIT_ENABLED = True
SHOW_SINGLE_XSPLIT_DEBUG = False

XSPLIT_HEIGHT_TO_MEDIAN_TRIGGER = 1.25
XSPLIT_AREA_TO_MEDIAN_TRIGGER = 1.40
XSPLIT_MAX_ASPECT_FOR_SUSPICIOUS = 1.80
XSPLIT_PC1_RATIO_MAX_FOR_XLIKE = 0.92

XSPLIT_HOUGH_THRESHOLD = 8
XSPLIT_HOUGH_MIN_LINE_LENGTH_FRAC = 0.18
XSPLIT_HOUGH_MAX_LINE_GAP = 18
XSPLIT_MIN_ANGLE_DIFF_DEG = 35.0
XSPLIT_MAX_ANGLE_DIFF_DEG = 145.0
XSPLIT_MIN_GROUP_LENGTH_FRAC = 0.20

XSPLIT_MIN_PARENT_MASK_PIXELS = 180
XSPLIT_MIN_CHILD_MASK_PIXELS = 60
XSPLIT_MIN_CHILD_FRAC_OF_PARENT = 0.10
XSPLIT_MIN_CHILD_BALANCE_RATIO = 0.18
XSPLIT_CHILD_BOX_PAD_PX = 2

XSPLIT_EDGE_CANNY_LOW = 60
XSPLIT_EDGE_CANNY_HIGH = 160
XSPLIT_EDGE_MASK_DILATE_ITER = 2


# =============================================================================
# 3B.18 CELL 16 — PCA AXIS CLEANUP SETTINGS
# =============================================================================
# EXPLANATION:
# These constants control tight mask cleanup around a dominant PCA axis.
#
# IMPORTANT:
#   This cleanup should skip X-like detections because an X has two valid axes.
# =============================================================================

AXIS_CLEANUP_ENABLED = True
SHOW_AXIS_CLEANUP_DEBUG = False

AXIS_CLEANUP_MIN_MASK_PIXELS = 120
AXIS_CLEANUP_HEIGHT_TO_MEDIAN_TRIGGER = 1.35
AXIS_CLEANUP_BOX_OVERLAP_FRAC_TRIGGER = 0.20
AXIS_CLEANUP_PERP_STD_TRIGGER_PX = 45.0

AXIS_CLEANUP_HALF_WIDTH_EXTRA_PX = 8.0
AXIS_CLEANUP_MIN_HALF_WIDTH_PX = 10.0
AXIS_CLEANUP_MAX_HALF_WIDTH_PX = 55.0
AXIS_CLEANUP_MIN_RETAINED_FRAC = 0.35
AXIS_CLEANUP_BOX_PAD_PX = 2


# =============================================================================
# 3B.19 CELL 16 — TARGETED TWO-BOX X-OWNERSHIP SETTINGS
# =============================================================================
# EXPLANATION:
# These constants control targeted pixel ownership only when two detections form
# a likely X-shaped crossing pair.
# =============================================================================

TWO_BOX_XOWNERSHIP_ENABLED = True
SHOW_XOWNERSHIP_DEBUG = False

XOWN_MIN_SHARED_PIXELS = 40
XOWN_MIN_SHARED_FRAC_OF_SMALLER = 0.04
XOWN_MIN_ANGLE_DIFF_DEG = 25.0
XOWN_MIN_CHILD_PIXELS_AFTER = 60
XOWN_MIN_RETAINED_FRAC_AFTER = 0.35
XOWN_BOX_PAD_PX = 2


# =============================================================================
# 3B.20 CELL 16 — FINAL DEDUPE / REVIEW SETTINGS
# =============================================================================
# EXPLANATION:
# These constants control the final cleanup and review-flag stage.
# =============================================================================

FINAL_DEDUPE_ENABLED = True
SHOW_FINAL_DEBUG = False

EXPECTED_MAX_CROSSARMS_FOR_DEBUG = 6


# =============================================================================
# 3B.21 CELL 16 — CROSSARM VISUALISATION / DEBUG SETTINGS
# =============================================================================
# EXPLANATION:
# These constants control optional QA/review overlays.
#
# IMPORTANT:
#   Stage grids should remain disabled for production batch runs.
# =============================================================================

CROSSARM_MASK_ALPHA = 0.40
POLE_MASK_ALPHA = 0.30
LABEL_BG = "#1E90FF"

SHOW_STAGE_GRID = False
GRID_FIGSIZE = (20, 10)

# =============================================================================
# 3B.22 CELL 17 — INSULATOR SAM3 PROMPT SETTINGS
# =============================================================================
# EXPLANATION:
# These constants control the generic insulator prompt sent to SAM3.
#
# IMPORTANT:
#   Start with one broad insulator prompt during raw candidate discovery.
#   Do not add glass, porcelain, polymer, or other subtype prompts yet.
#
#   INSULATOR_TEXT_THRESHOLD initially inherits the shared text threshold.
#   Tune this named constant here after reviewing the insulator pilot results.
# =============================================================================

INSULATOR_PROMPT_TEXT = "electrical insulator"
INSULATOR_TEXT_THRESHOLD = GLOBAL_TEXT_SCORE_THRESHOLD


# =============================================================================
# 3B.23 CELL 17 — INSULATOR QA OVERLAY SETTINGS
# =============================================================================
# EXPLANATION:
# These constants control raw insulator QA overlays only.
#
# IMPORTANT:
#   QA overlays are review artifacts. They must not alter detection masks,
#   bounding boxes, scores, or raw candidate state.
# =============================================================================

INSULATOR_MASK_ALPHA = 0.42

INSULATOR_BOX_RGB = (255, 215, 0)
INSULATOR_BOX_LINEWIDTH = 4

INSULATOR_MASK_RGB = (255, 80, 80)

INSULATOR_LABEL_RGB = (255, 255, 255)
INSULATOR_LABEL_BACKGROUND_RGB = (20, 20, 20)
INSULATOR_LABEL_PADDING_PX = 4



# =============================================================================
# 3B.24 SHARED TASK CONFIG DICTIONARY
# =============================================================================
# EXPLANATION:
# SAM3_TASK_CONFIG gives later cells a structured config object while preserving
# the plain global constants expected by the current notebook.
# =============================================================================

SAM3_TASK_CONFIG = {
    "runtime": {
        "device": DEVICE,
    },

    "thresholds": {
        "text_score_threshold": GLOBAL_TEXT_SCORE_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
    },

    "files": {
        "valid_image_extensions": VALID_IMAGE_EXTENSIONS,
    },

    "naming": {
        "image_id_prefix": IMAGE_ID_PREFIX,
    },

    "run_controls": {
        "overwrite_bronze": OVERWRITE_BRONZE,
        "overwrite_pole_rois": OVERWRITE_POLE_ROIS,
    },

    "pole_detection": {
        "prompts": POLE_PROMPT_TEXT,
        "text_score_threshold": POLE_TEXT_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
    },

    "pole_postprocess": {
        "min_score": POLE_MIN_SCORE,
        "min_area_frac": POLE_MIN_AREA_FRAC,
        "min_height_frac": POLE_MIN_HEIGHT_FRAC,
        "min_aspect": POLE_MIN_ASPECT,
        "max_width_frac": POLE_MAX_WIDTH_FRAC,
        "max_box_w_px": POLE_MAX_BOX_W_PX,
        "shaft_width_frac_threshold": SHAFT_WIDTH_FRAC_THRESHOLD,
        "shaft_penalty_factor": SHAFT_PENALTY_FACTOR,
        "weights": {
            "x_center": W_X_CENTER,
            "height": W_HEIGHT,
            "area": W_AREA,
            "conf": W_CONF,
            "edge": W_EDGE,
        },
    },

    "pole_overlay_selected": {
        "mask_rgb": POLE_SELECTED_MASK_RGB,
        "mask_alpha": POLE_SELECTED_MASK_ALPHA,
        "box_color": POLE_SELECTED_BOX_COLOR,
        "box_linewidth": POLE_SELECTED_BOX_LINEWIDTH,
        "text_color": POLE_SELECTED_TEXT_COLOR,
        "label_fontsize": POLE_SELECTED_LABEL_FONTSIZE,
        "label_bg_alpha": POLE_SELECTED_LABEL_BG_ALPHA,
        "label_bbox_pad": POLE_SELECTED_LABEL_BBOX_PAD,
        "label_y_offset": POLE_SELECTED_LABEL_Y_OFFSET,
        "overlay_max_width": POLE_OVERLAY_MAX_WIDTH,
        "no_reliable_label_text": NO_RELIABLE_POLE_LABEL_TEXT,
    },

    "pole_roi_fixed_canvas": {
        "fixed_roi_width": FIXED_ROI_WIDTH,
        "fixed_roi_height": FIXED_ROI_HEIGHT,
        "pole_top_buffer_above": POLE_TOP_BUFFER_ABOVE,
        "pad_rgb": PAD_RGB,
    },

    "crossarm_detection": {
        "prompt_text": CROSSARM_PROMPT_TEXT,
        "text_score_threshold": CROSSARM_TEXT_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
    },

    "crossarm_postprocess": {
        "raw_score_remove_max": CROSSARM_RAW_SCORE_REMOVE_MAX,

        "containment": {
            "containment_threshold": CONTAINMENT_THRESHOLD,
            "min_area_ratio": MIN_AREA_RATIO,
            "min_score_advantage": MIN_SCORE_ADVANTAGE,
            "mask_containment_filter_enabled": MASK_CONTAINMENT_FILTER_ENABLED,
            "mask_containment_veto_threshold": MASK_CONTAINMENT_VETO_THRESHOLD,
            "near_total_box_containment_threshold": NEAR_TOTAL_BOX_CONTAINMENT_THRESHOLD,
            "mask_containment_high": MASK_CONTAINMENT_HIGH,
            "pair_debug_min_box_containment": PAIR_DEBUG_MIN_BOX_CONTAINMENT,
        },

        "main_cluster": {
            "center_dist_factor": CENTER_DIST_FACTOR,
        },

        "pole_overlap": {
            "enabled": POLE_MASK_FILTER_ENABLED,
            "min_fraction": POLE_OVERLAP_MIN_FRACTION,
            "reject_fraction": POLE_OVERLAP_REJECT_FRACTION,
            "max_fraction": POLE_OVERLAP_MAX_FRACTION,
            "top_band_above": TOP_BAND_ABOVE,
            "top_band_below": TOP_BAND_BELOW,
            "min_relative_width_to_max": MIN_RELATIVE_WIDTH_TO_MAX,
            "pole_attach_margin_px": POLE_ATTACH_MARGIN_PX,
            "one_sided_lower_arm_veto_enabled": ONE_SIDED_LOWER_ARM_VETO_ENABLED,
            "one_sided_lower_arm_side_balance_max": ONE_SIDED_LOWER_ARM_SIDE_BALANCE_MAX,
            "one_sided_lower_arm_short_side_frac_max": ONE_SIDED_LOWER_ARM_SHORT_SIDE_FRAC_MAX,
            "one_sided_lower_arm_aspect_min": ONE_SIDED_LOWER_ARM_ASPECT_MIN,
            "one_sided_lower_arm_height_to_median_max": ONE_SIDED_LOWER_ARM_HEIGHT_TO_MEDIAN_MAX,
            "one_sided_lower_arm_min_y_gap_px": ONE_SIDED_LOWER_ARM_MIN_Y_GAP_PX,
            "one_sided_lower_arm_min_y_gap_factor": ONE_SIDED_LOWER_ARM_MIN_Y_GAP_FACTOR,
        },

        "same_xarm_merge": {
            "enabled": SAME_XARM_MERGE_ENABLED,
            "show_debug": SHOW_SAME_XARM_MERGE_DEBUG,
            "min_mask_pixels": SAME_XARM_MERGE_MIN_MASK_PIXELS,
            "max_angle_diff_deg": SAME_XARM_MERGE_MAX_ANGLE_DIFF_DEG,
            "max_perp_dist_px": SAME_XARM_MERGE_MAX_PERP_DIST_PX,
            "max_gap_px": SAME_XARM_MERGE_MAX_GAP_PX,
            "require_pole_bridge": SAME_XARM_MERGE_REQUIRE_POLE_BRIDGE,
            "box_pad_px": SAME_XARM_MERGE_BOX_PAD_PX,
            "score_mode": SAME_XARM_MERGE_SCORE_MODE,
        },

        "reserved_level_dedupe": {
            "enabled": CROSSARM_LEVEL_FILTER_ENABLED,
            "level_band_factor": CROSSARM_LEVEL_BAND_FACTOR,
            "max_box_h_to_median_ratio": MAX_BOX_H_TO_MEDIAN_RATIO,
            "keep_per_level": KEEP_PER_LEVEL,
        },

        "pca_cleanup": {
            "enabled": CROSSARM_PCA_FILTER_ENABLED,
            "suspicious_aspect_max": PCA_SUSPICIOUS_ASPECT_MAX,
            "suspicious_height_to_median_min": PCA_SUSPICIOUS_HEIGHT_TO_MEDIAN_MIN,
            "suspicious_rel_width_max": PCA_SUSPICIOUS_REL_WIDTH_MAX,
            "min_mask_pixels": PCA_MIN_MASK_PIXELS,
            "min_pc1_ratio": PCA_MIN_PC1_RATIO,
            "min_anisotropy": PCA_MIN_ANISOTROPY,
        },

        "single_xsplit": {
            "enabled": SINGLE_XSPLIT_ENABLED,
            "show_debug": SHOW_SINGLE_XSPLIT_DEBUG,
            "height_to_median_trigger": XSPLIT_HEIGHT_TO_MEDIAN_TRIGGER,
            "area_to_median_trigger": XSPLIT_AREA_TO_MEDIAN_TRIGGER,
            "max_aspect_for_suspicious": XSPLIT_MAX_ASPECT_FOR_SUSPICIOUS,
            "pc1_ratio_max_for_xlike": XSPLIT_PC1_RATIO_MAX_FOR_XLIKE,
            "hough_threshold": XSPLIT_HOUGH_THRESHOLD,
            "hough_min_line_length_frac": XSPLIT_HOUGH_MIN_LINE_LENGTH_FRAC,
            "hough_max_line_gap": XSPLIT_HOUGH_MAX_LINE_GAP,
            "min_angle_diff_deg": XSPLIT_MIN_ANGLE_DIFF_DEG,
            "max_angle_diff_deg": XSPLIT_MAX_ANGLE_DIFF_DEG,
            "min_group_length_frac": XSPLIT_MIN_GROUP_LENGTH_FRAC,
            "min_parent_mask_pixels": XSPLIT_MIN_PARENT_MASK_PIXELS,
            "min_child_mask_pixels": XSPLIT_MIN_CHILD_MASK_PIXELS,
            "min_child_frac_of_parent": XSPLIT_MIN_CHILD_FRAC_OF_PARENT,
            "min_child_balance_ratio": XSPLIT_MIN_CHILD_BALANCE_RATIO,
            "child_box_pad_px": XSPLIT_CHILD_BOX_PAD_PX,
            "edge_canny_low": XSPLIT_EDGE_CANNY_LOW,
            "edge_canny_high": XSPLIT_EDGE_CANNY_HIGH,
            "edge_mask_dilate_iter": XSPLIT_EDGE_MASK_DILATE_ITER,
        },

        "axis_cleanup": {
            "enabled": AXIS_CLEANUP_ENABLED,
            "show_debug": SHOW_AXIS_CLEANUP_DEBUG,
            "min_mask_pixels": AXIS_CLEANUP_MIN_MASK_PIXELS,
            "height_to_median_trigger": AXIS_CLEANUP_HEIGHT_TO_MEDIAN_TRIGGER,
            "box_overlap_frac_trigger": AXIS_CLEANUP_BOX_OVERLAP_FRAC_TRIGGER,
            "perp_std_trigger_px": AXIS_CLEANUP_PERP_STD_TRIGGER_PX,
            "half_width_extra_px": AXIS_CLEANUP_HALF_WIDTH_EXTRA_PX,
            "min_half_width_px": AXIS_CLEANUP_MIN_HALF_WIDTH_PX,
            "max_half_width_px": AXIS_CLEANUP_MAX_HALF_WIDTH_PX,
            "min_retained_frac": AXIS_CLEANUP_MIN_RETAINED_FRAC,
            "box_pad_px": AXIS_CLEANUP_BOX_PAD_PX,
        },

        "two_box_xownership": {
            "enabled": TWO_BOX_XOWNERSHIP_ENABLED,
            "show_debug": SHOW_XOWNERSHIP_DEBUG,
            "min_shared_pixels": XOWN_MIN_SHARED_PIXELS,
            "min_shared_frac_of_smaller": XOWN_MIN_SHARED_FRAC_OF_SMALLER,
            "min_angle_diff_deg": XOWN_MIN_ANGLE_DIFF_DEG,
            "min_child_pixels_after": XOWN_MIN_CHILD_PIXELS_AFTER,
            "min_retained_frac_after": XOWN_MIN_RETAINED_FRAC_AFTER,
            "box_pad_px": XOWN_BOX_PAD_PX,
        },

        "final_dedupe": {
            "enabled": FINAL_DEDUPE_ENABLED,
            "show_debug": SHOW_FINAL_DEBUG,
            "expected_max_crossarms_for_debug": EXPECTED_MAX_CROSSARMS_FOR_DEBUG,
        },
    },

    "crossarm_visualisation": {
        "crossarm_mask_alpha": CROSSARM_MASK_ALPHA,
        "pole_mask_alpha": POLE_MASK_ALPHA,
        "label_bg": LABEL_BG,
        "show_stage_grid": SHOW_STAGE_GRID,
        "grid_figsize": GRID_FIGSIZE,
    },

    "insulator_detection": {
        "prompt_text": INSULATOR_PROMPT_TEXT,
        "text_score_threshold": INSULATOR_TEXT_THRESHOLD,
    },

    "insulator_visualisation": {
        "mask_alpha": INSULATOR_MASK_ALPHA,
        "box_rgb": INSULATOR_BOX_RGB,
        "box_linewidth": INSULATOR_BOX_LINEWIDTH,
        "mask_rgb": INSULATOR_MASK_RGB,
        "label_rgb": INSULATOR_LABEL_RGB,
        "label_background_rgb": INSULATOR_LABEL_BACKGROUND_RGB,
        "label_padding_px": INSULATOR_LABEL_PADDING_PX,
    },
}

# =============================================================================
# 3B.25 CONFIG SUMMARY
# =============================================================================
# EXPLANATION:
# Print a compact summary so each notebook run records the active production
# settings.
# =============================================================================

if PRINT_CONFIG_SUMMARY:
    print("Global constants loaded.\n")

    print("=" * 90)
    print("RUNTIME / GLOBAL")
    print("=" * 90)
    print(f"DEVICE                              : {DEVICE}")
    print(f"GLOBAL_TEXT_SCORE_THRESHOLD         : {GLOBAL_TEXT_SCORE_THRESHOLD}")
    print(f"MASK_THRESHOLD                      : {MASK_THRESHOLD}")

    print("\n" + "=" * 90)
    print("FILES / NAMING / RUN CONTROLS")
    print("=" * 90)
    print(f"VALID_IMAGE_EXTENSIONS              : {VALID_IMAGE_EXTENSIONS}")
    print(f"IMAGE_ID_PREFIX                     : {IMAGE_ID_PREFIX}")
    print(f"OVERWRITE_BRONZE                    : {OVERWRITE_BRONZE}")
    print(f"OVERWRITE_POLE_ROIS                 : {OVERWRITE_POLE_ROIS}")

    print("\n" + "=" * 90)
    print("CELL 13 — POLE DETECTION / POST-PROCESS")
    print("=" * 90)
    print(f"POLE_PROMPT_TEXT                    : {POLE_PROMPT_TEXT}")
    print(f"POLE_TEXT_THRESHOLD                 : {POLE_TEXT_THRESHOLD}")
    print(f"POLE_MIN_SCORE                      : {POLE_MIN_SCORE}")
    print(f"POLE_MIN_AREA_FRAC                  : {POLE_MIN_AREA_FRAC}")
    print(f"POLE_MIN_HEIGHT_FRAC                : {POLE_MIN_HEIGHT_FRAC}")
    print(f"POLE_MIN_ASPECT                     : {POLE_MIN_ASPECT}")
    print(f"POLE_MAX_WIDTH_FRAC                 : {POLE_MAX_WIDTH_FRAC}")
    print(f"POLE_MAX_BOX_W_PX                   : {POLE_MAX_BOX_W_PX}")

    print("\n" + "=" * 90)
    print("CELL 14 — FIXED POLE-TOP ROI")
    print("=" * 90)
    print(f"FIXED_ROI_WIDTH                     : {FIXED_ROI_WIDTH}")
    print(f"FIXED_ROI_HEIGHT                    : {FIXED_ROI_HEIGHT}")
    print(f"POLE_TOP_BUFFER_ABOVE               : {POLE_TOP_BUFFER_ABOVE}")
    print(f"PAD_RGB                             : {PAD_RGB}")

    print("\n" + "=" * 90)
    print("CELL 16 — CROSSARM DETECTION")
    print("=" * 90)
    print(f"CROSSARM_PROMPT_TEXT                : {CROSSARM_PROMPT_TEXT}")
    print(f"CROSSARM_TEXT_THRESHOLD             : {CROSSARM_TEXT_THRESHOLD}")
    print(f"CROSSARM_RAW_SCORE_REMOVE_MAX       : {CROSSARM_RAW_SCORE_REMOVE_MAX}")
    print(f"CENTER_DIST_FACTOR                  : {CENTER_DIST_FACTOR}")

    print("\n" + "=" * 90)
    print("CELL 16 — CROSSARM POST-PROCESS")
    print("=" * 90)
    print(f"CONTAINMENT_THRESHOLD               : {CONTAINMENT_THRESHOLD}")
    print(f"MASK_CONTAINMENT_VETO_THRESHOLD     : {MASK_CONTAINMENT_VETO_THRESHOLD}")
    print(f"POLE_OVERLAP_MIN_FRACTION           : {POLE_OVERLAP_MIN_FRACTION}")
    print(f"POLE_OVERLAP_REJECT_FRACTION        : {POLE_OVERLAP_REJECT_FRACTION}")
    print(f"SAME_XARM_MERGE_ENABLED             : {SAME_XARM_MERGE_ENABLED}")
    print(f"SINGLE_XSPLIT_ENABLED               : {SINGLE_XSPLIT_ENABLED}")
    print(f"AXIS_CLEANUP_ENABLED                : {AXIS_CLEANUP_ENABLED}")
    print(f"TWO_BOX_XOWNERSHIP_ENABLED          : {TWO_BOX_XOWNERSHIP_ENABLED}")
    print(f"FINAL_DEDUPE_ENABLED                : {FINAL_DEDUPE_ENABLED}")
    print(f"EXPECTED_MAX_CROSSARMS_FOR_DEBUG    : {EXPECTED_MAX_CROSSARMS_FOR_DEBUG}")
    print(f"ONE_SIDED_LOWER_ARM_VETO_ENABLED    : {ONE_SIDED_LOWER_ARM_VETO_ENABLED}")
    print(f"ONE_SIDED_LOWER_ARM_ASPECT_MIN      : {ONE_SIDED_LOWER_ARM_ASPECT_MIN}")
    print(f"ONE_SIDED_LOWER_ARM_SIDE_BALANCE_MAX: {ONE_SIDED_LOWER_ARM_SIDE_BALANCE_MAX}")

    print("\n" + "=" * 90)
    print("CELL 16 — DEBUG / VISUAL FLAGS")
    print("=" * 90)
    print(f"SHOW_SAME_XARM_MERGE_DEBUG          : {SHOW_SAME_XARM_MERGE_DEBUG}")
    print(f"SHOW_SINGLE_XSPLIT_DEBUG            : {SHOW_SINGLE_XSPLIT_DEBUG}")
    print(f"SHOW_AXIS_CLEANUP_DEBUG             : {SHOW_AXIS_CLEANUP_DEBUG}")
    print(f"SHOW_XOWNERSHIP_DEBUG               : {SHOW_XOWNERSHIP_DEBUG}")
    print(f"SHOW_FINAL_DEBUG                    : {SHOW_FINAL_DEBUG}")
    print(f"SHOW_STAGE_GRID                     : {SHOW_STAGE_GRID}")
    print(f"CROSSARM_MASK_ALPHA                 : {CROSSARM_MASK_ALPHA}")
    print(f"POLE_MASK_ALPHA                     : {POLE_MASK_ALPHA}")
    print(f"LABEL_BG                            : {LABEL_BG}")
    print(f"GRID_FIGSIZE                        : {GRID_FIGSIZE}")
    print("\n" + "=" * 90)
    print("CELL 17 — INSULATOR DETECTION")
    print("=" * 90)
    print(f"INSULATOR_PROMPT_TEXT                : {INSULATOR_PROMPT_TEXT}")
    print(f"INSULATOR_TEXT_THRESHOLD             : {INSULATOR_TEXT_THRESHOLD}")
    print(f"INSULATOR_MASK_ALPHA                 : {INSULATOR_MASK_ALPHA}")
    print(f"INSULATOR_BOX_RGB                    : {INSULATOR_BOX_RGB}")
    print(f"INSULATOR_BOX_LINEWIDTH              : {INSULATOR_BOX_LINEWIDTH}")
    print(f"INSULATOR_MASK_RGB                   : {INSULATOR_MASK_RGB}")
    print(f"INSULATOR_LABEL_RGB                  : {INSULATOR_LABEL_RGB}")
    print(
        "INSULATOR_LABEL_BACKGROUND_RGB       : "
        f"{INSULATOR_LABEL_BACKGROUND_RGB}"
    )
    print(f"INSULATOR_LABEL_PADDING_PX           : {INSULATOR_LABEL_PADDING_PX}")
    
    
# =============================================================================
# CELL 4 — ADD LOCAL SAM3 REPO TO PYTHONPATH
# =============================================================================
# OVERVIEW:
# This cell makes the local SAM3 codebase importable inside the Databricks
# notebook.
#
# The SAM3 repo is stored in a Databricks Volume instead of being installed as a
# normal pip package, so the repository root must be added to Python's import
# path before later cells can import SAM3 modules.
#
# STRUCTURE:
#   A. PATH SETUP
#        Section 4.1. Imports
#        Section 4.2. Define SAM3 repository root
#
#   B. VALIDATION AND PATH INJECTION
#        Section 4.3. Validate repository structure
#        Section 4.4. Add repository root to sys.path
#        Section 4.5. Import and validate sam3 package
#
#   C. OUTPUT SUMMARY
#        Section 4.6. Config summary
#
# IMPORTANT:
#   - Run this after CELL 3B.
#   - This cell does not load model weights.
#   - SAM3_REPO_ROOT must be the folder that contains the sam3 package:
#       SAM3_REPO_ROOT/sam3/__init__.py
#   - Later SAM3-specific imports happen in CELL 5.
# =============================================================================


# =============================================================================
# A. PATH SETUP
# =============================================================================


# =============================================================================
# 4.1 IMPORTS
# =============================================================================
# EXPLANATION:
# os is used to validate the repository path.
# sys is used to modify Python's import path.
# importlib is used to confirm that the sam3 package can be imported.
# =============================================================================

import os
import sys
import importlib


# =============================================================================
# 4.2 DEFINE SAM3 REPOSITORY ROOT
# =============================================================================
# EXPLANATION:
# SAM3_REPO_ROOT must point to the repository folder that contains the sam3
# package directory.
#
# Expected layout:
#   /Volumes/repos/sam3/
#       sam3/
#           __init__.py
# =============================================================================

SAM3_REPO_ROOT = str(
    globals().get(
        "SAM3_REPO_ROOT",
        "/Volumes/repos/sam3",
    )
)

SAM3_REPO_ROOT = os.path.abspath(SAM3_REPO_ROOT)


# =============================================================================
# B. VALIDATION AND PATH INJECTION
# =============================================================================


# =============================================================================
# 4.3 VALIDATE REPOSITORY STRUCTURE
# =============================================================================
# EXPLANATION:
# Fail early if the configured path is missing or does not contain an importable
# sam3 package.
# =============================================================================

if not os.path.isdir(SAM3_REPO_ROOT):
    raise FileNotFoundError(
        "SAM3 repository root was not found.\n"
        f"Expected path: {SAM3_REPO_ROOT}\n"
        "Please confirm the SAM3 repo is mounted in the Databricks Volume."
    )

SAM3_PACKAGE_DIR = os.path.join(SAM3_REPO_ROOT, "sam3")
SAM3_PACKAGE_INIT = os.path.join(SAM3_PACKAGE_DIR, "__init__.py")

if not os.path.isfile(SAM3_PACKAGE_INIT):
    raise FileNotFoundError(
        "Could not find an importable sam3 package inside SAM3_REPO_ROOT.\n\n"
        "Expected file:\n"
        f"  {SAM3_PACKAGE_INIT}\n\n"
        "Please confirm SAM3_REPO_ROOT points to the folder that contains the "
        "sam3 package directory."
    )


# =============================================================================
# 4.4 ADD REPOSITORY ROOT TO sys.path
# =============================================================================
# EXPLANATION:
# Insert SAM3_REPO_ROOT at the front of sys.path so the local Databricks repo
# copy takes priority over any other sam3 package in the environment.
# =============================================================================

if SAM3_REPO_ROOT in sys.path:
    sys.path.remove(SAM3_REPO_ROOT)

sys.path.insert(0, SAM3_REPO_ROOT)


# =============================================================================
# 4.5 IMPORT AND VALIDATE sam3 PACKAGE
# =============================================================================
# EXPLANATION:
# Import sam3 immediately so path issues are caught in this setup cell instead
# of failing later during model setup.
# =============================================================================

sam3 = importlib.import_module("sam3")

SAM3_PACKAGE_PATH = getattr(
    sam3,
    "__file__",
    None,
)

if SAM3_PACKAGE_PATH is None:
    raise ImportError(
        "sam3 was imported, but its package path could not be resolved."
    )


# =============================================================================
# C. OUTPUT SUMMARY
# =============================================================================


# =============================================================================
# 4.6 CONFIG SUMMARY
# =============================================================================
# EXPLANATION:
# Print a compact summary when PRINT_CONFIG_SUMMARY is enabled.
# =============================================================================

if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print("Local SAM3 repo added to Python path.\n")

    print("=" * 90)
    print("SAM3 LOCAL REPO IMPORT")
    print("=" * 90)
    print(f"SAM3_REPO_ROOT                    : {SAM3_REPO_ROOT}")
    print(f"SAM3_PACKAGE_DIR                  : {SAM3_PACKAGE_DIR}")
    print(f"SAM3_PACKAGE_PATH                 : {SAM3_PACKAGE_PATH}")
    print(f"sys.path[0]                       : {sys.path[0]}")



# =============================================================================
# CELL 5 — SAM3 IMPORTS
# =============================================================================
# EXPLANATION:
# This cell imports the SAM3-specific classes and helper functions used later
# for model creation, image processing, and visualisation.
#
# WHAT THIS CELL DOES:
#   1) validates that the SAM3 repo path setup has already run
#   2) imports the SAM3 model builder
#   3) imports the SAM3 processor used for image + prompt inference
#   4) imports helper functions for box conversion and visualisation
#
# IMPORTANT:
# - this cell imports SAM3 code only; it does not yet build the model
# =============================================================================

# -----------------------------------------------------------------------------
# 5.0 Safety checks
# -----------------------------------------------------------------------------
required_cell5_globals = [
    "SAM3_REPO_ROOT",
    "SAM3_PACKAGE_PATH",
]

missing_cell5_globals = [
    name for name in required_cell5_globals
    if name not in globals()
]

if missing_cell5_globals:
    raise NameError(
        "CELL 5 requires CELL 4 to run successfully first.\n"
        f"Missing globals: {missing_cell5_globals}"
    )

# -----------------------------------------------------------------------------
# 5.1 Import SAM3 model builder and processor
# -----------------------------------------------------------------------------
# EXPLANATION:
# These are the main SAM3 components used later to:
# - build the image model
# - prepare images for inference
# - apply text prompts
# -----------------------------------------------------------------------------
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# -----------------------------------------------------------------------------
# 5.2 Import SAM3 helper utilities
# -----------------------------------------------------------------------------
# EXPLANATION:
# These helper utilities are useful later for bounding box conversion,
# coordinate normalisation, and visualisation.
# -----------------------------------------------------------------------------
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import (
    draw_box_on_image,
    normalize_bbox,
    plot_results,
)

# -----------------------------------------------------------------------------
# 5.3 Print import confirmation
# -----------------------------------------------------------------------------
# EXPLANATION:
# This provides a simple sanity check that the key SAM3 components are available.
# -----------------------------------------------------------------------------
print("SAM3 imports ready.")
print("build_sam3_image_model loaded")
print("Sam3Processor loaded")
print("SAM3 utility functions loaded")



# =============================================================================
# CELL 6 — GPU SETTINGS + RUNTIME BEHAVIOUR
# =============================================================================
# EXPLANATION:
# This cell applies the GPU runtime settings used by the notebook.
#
# WHAT THIS CELL DOES:
#   1) validates that PyTorch and CUDA are available
#   2) enables TF32 for supported NVIDIA GPUs
#   3) keeps inference in standard float32 mode
#
# IMPORTANT:
#   - Run this after CELL 5.
#   - This cell does not load model weights.
# =============================================================================


# -----------------------------------------------------------------------------
# 6.1 Safety checks
# -----------------------------------------------------------------------------
if "torch" not in globals():
    raise NameError(
        "torch is not available.\n"
        "Please run CELL 3A before CELL 6."
    )

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA/GPU is not available.\n"
        "Please attach this notebook to a GPU cluster before running SAM3."
    )


# -----------------------------------------------------------------------------
# 6.2 Enable TF32 on supported GPUs
# -----------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# -----------------------------------------------------------------------------
# 6.3 Config summary
# -----------------------------------------------------------------------------
if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print("GPU runtime settings applied.\n")

    print("=" * 90)
    print("GPU RUNTIME SETTINGS")
    print("=" * 90)
    print(f"TF32 matmul enabled              : {torch.backends.cuda.matmul.allow_tf32}")
    print(f"TF32 cuDNN enabled               : {torch.backends.cudnn.allow_tf32}")



# =============================================================================
# CELL 7 — SAM3 PATH DEFINITIONS + FILE CHECKS
# =============================================================================
# EXPLANATION:
# This cell defines the key filesystem paths needed to build the SAM3 model.
#
# WHAT THIS CELL DOES:
#   1) validates that the SAM3 repo root from CELL 4 exists
#   2) derives the inner SAM3 code root from that repo root
#   3) defines the BPE tokenizer vocab path
#   4) defines the checkpoint / weight file path
#   5) checks that the required files actually exist
#
# IMPORTANT:
# - later cells assume these path variables are already defined
# =============================================================================

# -----------------------------------------------------------------------------
# 7.1 Define SAM3 code root
# -----------------------------------------------------------------------------
# EXPLANATION:
# This is the inner SAM3 package folder that contains assets such as the BPE
# vocabulary file used for text prompts.
# Deriving it from SAM3_REPO_ROOT keeps the notebook path setup consistent.
# -----------------------------------------------------------------------------
SAM3_CODE_ROOT = os.path.join(SAM3_REPO_ROOT, "sam3")

# -----------------------------------------------------------------------------
# 7.2 Define BPE vocab path
# -----------------------------------------------------------------------------
# EXPLANATION:
# This file is required because you are using text prompts with SAM3.
# -----------------------------------------------------------------------------
BPE_PATH = os.path.join(
    SAM3_CODE_ROOT,
    "assets",
    "bpe_simple_vocab_16e6.txt.gz",
)

# -----------------------------------------------------------------------------
# 7.3 Define model checkpoint path
# -----------------------------------------------------------------------------
# EXPLANATION:
# This is the SAM3 weight file stored in the shared Databricks model cache.
#
# It is derived from HF_HUB_CACHE configured in CELL 2 so the notebook does not
# hardcode the cache root in multiple places.
# -----------------------------------------------------------------------------
CHECKPOINT_PATH = os.path.join(
    os.environ["HF_HUB_CACHE"],
    "sam3",
    "sam3.pt",
)

# -----------------------------------------------------------------------------
# 7.4 Validate required files
# -----------------------------------------------------------------------------
# EXPLANATION:
# This is a quick sanity check before model creation so file path issues are
# caught early and clearly.
# -----------------------------------------------------------------------------
if not os.path.isdir(SAM3_CODE_ROOT):
    raise FileNotFoundError(f"SAM3 code root folder not found: {SAM3_CODE_ROOT}")

if not os.path.exists(BPE_PATH):
    raise FileNotFoundError(f"BPE file not found: {BPE_PATH}")

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint file not found: {CHECKPOINT_PATH}")


# -----------------------------------------------------------------------------
# 7.5 Config summary
# -----------------------------------------------------------------------------
if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print("SAM3 path definitions ready.\n")

    print("=" * 90)
    print("SAM3 MODEL PATHS")
    print("=" * 90)
    print(f"SAM3_REPO_ROOT                  : {SAM3_REPO_ROOT}")
    print(f"SAM3_CODE_ROOT                  : {SAM3_CODE_ROOT}")
    print(f"BPE_PATH                        : {BPE_PATH}")
    print(f"CHECKPOINT_PATH                 : {CHECKPOINT_PATH}")
    


# =============================================================================
# CELL 8 — BUILD SAM3 MODEL
# =============================================================================
# EXPLANATION:
# This cell builds the SAM3 image model using the previously defined
# tokenizer vocabulary path and checkpoint path.
#
# WHAT THIS CELL DOES:
#   1) validates that model-build inputs already exist
#   2) clears any stale CUDA memory
#   3) builds the SAM3 model from local code + local weights
#   4) moves the model to the configured device
#   5) switches the model into evaluation mode
#
# IMPORTANT:
# - this cell builds the model only once per notebook session
# =============================================================================

# -----------------------------------------------------------------------------
# 8.1 Clear any stale CUDA cache
# -----------------------------------------------------------------------------
# EXPLANATION:
# This helps reduce the chance of memory fragmentation before model creation.
# -----------------------------------------------------------------------------
torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# 8.2 Build SAM3 image model
# -----------------------------------------------------------------------------
# EXPLANATION:
# The SAM3 model is created from:
# - local SAM3 Python code
# - local BPE vocab file
# - local checkpoint weights
# -----------------------------------------------------------------------------
model = build_sam3_image_model(
    bpe_path=BPE_PATH,
    checkpoint_path=CHECKPOINT_PATH
)

# -----------------------------------------------------------------------------
# 8.3 Move model to configured device
# -----------------------------------------------------------------------------
# EXPLANATION:
# This makes device placement explicit so the notebook does not rely on the
# builder function to place the model on GPU implicitly.
# -----------------------------------------------------------------------------
model = model.to(DEVICE)

# -----------------------------------------------------------------------------
# 8.4 Set model to evaluation mode
# -----------------------------------------------------------------------------
# EXPLANATION:
# Evaluation mode disables training-specific behaviours and is the correct mode
# for inference.
# -----------------------------------------------------------------------------
model.eval()

# -----------------------------------------------------------------------------
# 8.5 Print model build confirmation
# -----------------------------------------------------------------------------
# EXPLANATION:
# This is a quick sanity check so we know the model is ready before moving on
# to image processing and prompts.
# -----------------------------------------------------------------------------
print("SAM3 model built successfully.")
print("Model type           :", type(model))
print("Model eval mode      :", not model.training)
print("Model parameter dtype:", next(model.parameters()).dtype)
print("Model device         :", next(model.parameters()).device)



# =============================================================================
# CELL 9 — BUILD SAM3 PROCESSOR
# =============================================================================
# EXPLANATION:
# This cell creates the SAM3 processor object used to prepare images and text
# prompts for inference.
#
# WHAT THIS CELL DOES:
#   1) creates the SAM3 processor from the built model
#   2) prints a small confirmation
#
# IMPORTANT:
#   - Run this after CELL 8 (the model must already be built).
#   - Sam3Processor must already be imported from CELL 5.
# =============================================================================

# -----------------------------------------------------------------------------
# 9.1 Build processor
# -----------------------------------------------------------------------------
processor = Sam3Processor(model)

# -----------------------------------------------------------------------------
# 9.2 Print confirmation
# -----------------------------------------------------------------------------
print("SAM3 processor created successfully.")
print("Processor type:", type(processor))


# =============================================================================
# CELL 10 — WORKSPACE + DIRECTORY STRUCTURE
# =============================================================================
# OVERVIEW:
# This cell defines the Databricks project workspace and creates the major
# directory structure used by the SAM3 production notebook.
#
# The folder structure follows a Bronze / Silver / Gold layout:
#
#   Bronze:
#       Raw source images copied into the project workspace.
#
#   Silver:
#       Production processing outputs such as pole ROIs, selected-pole overlays,
#       asset candidates, crossarm candidates, crossarm processing tables, and
#       review artifacts.
#
#   Gold:
#       Final cleaned outputs ready for downstream use, export, QA, or reporting.
#
# STRUCTURE:
#   A. SAFETY CHECKS
#        Section 10.1. Required setup checks
#
#   B. WORKSPACE PATHS
#        Section 10.2. Project root
#        Section 10.3. State and artifact folders
#        Section 10.4. Bronze folders
#        Section 10.5. Silver folders
#        Section 10.6. Gold folders
#
#   C. DIRECTORY CREATION
#        Section 10.7. Collect directories
#        Section 10.8. Create directory tree
#        Section 10.9. Verify directory creation
#
#   D. OUTPUT SUMMARY
#        Section 10.10. Workspace summary
#
# IMPORTANT:
#   - Run this after CELL 9.
#   - This cell does not ingest images.
#   - This cell does not run SAM3 inference.
#   - Path definitions remain centralized here rather than in CELL 3B.
#   - Run-scoped output folders are created later by CELL 10B / CELL 16 using
#     RUN_ID.
# =============================================================================

# =============================================================================
# A. SAFETY CHECKS
# =============================================================================

# =============================================================================
# 10.1 REQUIRED SETUP CHECKS
# =============================================================================
# EXPLANATION:
# Fail early if core setup cells have not been run.
#
# Required earlier cells:
#   - CELL 2  : os and cache setup
#   - CELL 8  : model
#   - CELL 9  : processor
# =============================================================================

required_cell10_globals = [
    "os",
    "model",
    "processor",
]

missing_cell10_globals = [
    name for name in required_cell10_globals
    if name not in globals()
]

if missing_cell10_globals:
    raise NameError(
        "CELL 10 requires earlier setup cells to run successfully first.\n"
        f"Missing globals: {missing_cell10_globals}"
    )

# =============================================================================
# B. WORKSPACE PATHS
# =============================================================================

# =============================================================================
# 10.2 DEFINE PROJECT WORKSPACE ROOT
# =============================================================================
# EXPLANATION:
# WORK_DIR is the main root folder for the SAM3 project inside the Databricks
# Volume.
#
# IMPORTANT:
#   Keep this path unchanged for now.
# =============================================================================

WORK_DIR = (
    "/Volumes/"
    "/sharyn_volume/"
    "sam3_project"
)

# =============================================================================
# 10.3 DEFINE STATE AND ARTIFACT FOLDERS
# =============================================================================
# EXPLANATION:
# These folders hold notebook state, saved tabular manifests, and general output
# artifacts.
# =============================================================================

STATE_DIR = os.path.join(WORK_DIR, "state")
DF_DIR = os.path.join(STATE_DIR, "dataframes")
ART_DIR = os.path.join(WORK_DIR, "artifacts")

# =============================================================================
# 10.4 DEFINE BRONZE LAYER FOLDERS
# =============================================================================
# EXPLANATION:
# Bronze holds raw input images before processing, cropping, or inference.
# =============================================================================

BRONZE_ROOT = os.path.join(WORK_DIR, "bronze")
BRONZE_SOURCE_IMAGES = os.path.join(BRONZE_ROOT, "source_images")

# =============================================================================
# 10.5 DEFINE SILVER LAYER FOLDERS
# =============================================================================
# EXPLANATION:
# Silver holds production processing outputs from the pole, asset, and crossarm
# branches.
#
# Crossarm folder roles:
#   - candidates  : raw / candidate crossarm detection outputs
#   - processing  : lightweight production audit/stage tables
#   - review      : review images and QA artifacts
# =============================================================================

SILVER_ROOT = os.path.join(WORK_DIR, "silver")

# Pole ROI and selected-pole outputs.
SILVER_POLE_ROIS = os.path.join(SILVER_ROOT, "pole_rois")
SILVER_POLE_SELECTION = os.path.join(SILVER_ROOT, "pole_selection")
SILVER_POLE_SELECTION_OVERLAYS = os.path.join(
    SILVER_POLE_SELECTION,
    "overlays",
)

# Silver branch 1: asset detection candidates.
SILVER_ASSET_DETECTION_CANDIDATES = os.path.join(
    SILVER_ROOT,
    "asset_detection_candidates",
)

SILVER_ASSET_PROMPT_RUNS = os.path.join(
    SILVER_ASSET_DETECTION_CANDIDATES,
    "prompt_runs",
)

SILVER_ASSET_OVERLAYS = os.path.join(
    SILVER_ASSET_DETECTION_CANDIDATES,
    "overlays",
)

SILVER_ASSET_MASKS = os.path.join(
    SILVER_ASSET_DETECTION_CANDIDATES,
    "masks",
)

# Silver branch 2: crossarm detection.
SILVER_CROSSARM_DETECTION = os.path.join(
    SILVER_ROOT,
    "crossarm_detection",
)

SILVER_CROSSARM_CANDIDATES = os.path.join(
    SILVER_CROSSARM_DETECTION,
    "candidates",
)

SILVER_CROSSARM_PROCESSING = os.path.join(
    SILVER_CROSSARM_DETECTION,
    "processing",
)

SILVER_CROSSARM_REVIEW = os.path.join(
    SILVER_CROSSARM_DETECTION,
    "review",
)

# =============================================================================
# 10.6 DEFINE GOLD LAYER FOLDERS
# =============================================================================
# EXPLANATION:
# Gold holds final cleaned outputs ready for downstream analysis, export, QA, or
# reporting.
# =============================================================================

GOLD_ROOT = os.path.join(WORK_DIR, "gold")

GOLD_ASSET_DETECTIONS = os.path.join(
    GOLD_ROOT,
    "asset_detections",
)

GOLD_CROSSARM_DETECTIONS = os.path.join(
    GOLD_ROOT,
    "crossarm_detections",
)

# =============================================================================
# C. DIRECTORY CREATION
# =============================================================================

# =============================================================================
# 10.7 COLLECT DIRECTORIES TO CREATE
# =============================================================================
# EXPLANATION:
# Keeping the directory list in one place makes this cell easy to inspect,
# extend, and validate.
# =============================================================================

DIRECTORIES_TO_CREATE = [
    WORK_DIR,

    STATE_DIR,
    DF_DIR,
    ART_DIR,

    BRONZE_ROOT,
    BRONZE_SOURCE_IMAGES,

    SILVER_ROOT,

    SILVER_POLE_ROIS,
    SILVER_POLE_SELECTION,
    SILVER_POLE_SELECTION_OVERLAYS,

    SILVER_ASSET_DETECTION_CANDIDATES,
    SILVER_ASSET_PROMPT_RUNS,
    SILVER_ASSET_OVERLAYS,
    SILVER_ASSET_MASKS,

    SILVER_CROSSARM_DETECTION,
    SILVER_CROSSARM_CANDIDATES,
    SILVER_CROSSARM_PROCESSING,
    SILVER_CROSSARM_REVIEW,

    GOLD_ROOT,
    GOLD_ASSET_DETECTIONS,
    GOLD_CROSSARM_DETECTIONS,
]

# =============================================================================
# 10.8 CREATE DIRECTORY TREE
# =============================================================================
# EXPLANATION:
# Create all project folders before later cells try to write files into them.
# =============================================================================

for directory_path in DIRECTORIES_TO_CREATE:
    os.makedirs(directory_path, exist_ok=True)

# =============================================================================
# 10.9 VERIFY DIRECTORY CREATION
# =============================================================================
# EXPLANATION:
# Fail early if any expected directory still does not exist after creation.
# =============================================================================

missing_dirs_after_create = [
    directory_path
    for directory_path in DIRECTORIES_TO_CREATE
    if not os.path.isdir(directory_path)
]

if missing_dirs_after_create:
    raise RuntimeError(
        "Some project directories were not created successfully.\n"
        f"Missing directories: {missing_dirs_after_create}"
    )

# =============================================================================
# D. OUTPUT SUMMARY
# =============================================================================

# =============================================================================
# 10.10 WORKSPACE SUMMARY
# =============================================================================
# EXPLANATION:
# Print a compact workspace summary when PRINT_CONFIG_SUMMARY is enabled.
# =============================================================================

if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print("Databricks SAM3 project workspace ready.\n")

    print("=" * 90)
    print("PROJECT ROOT")
    print("=" * 90)
    print(f"WORK_DIR                         : {WORK_DIR}")

    print("\n" + "=" * 90)
    print("STATE / ARTIFACTS")
    print("=" * 90)
    print(f"STATE_DIR                        : {STATE_DIR}")
    print(f"DF_DIR                           : {DF_DIR}")
    print(f"ART_DIR                          : {ART_DIR}")

    print("\n" + "=" * 90)
    print("BRONZE")
    print("=" * 90)
    print(f"BRONZE_ROOT                      : {BRONZE_ROOT}")
    print(f"BRONZE_SOURCE_IMAGES             : {BRONZE_SOURCE_IMAGES}")

    print("\n" + "=" * 90)
    print("SILVER — POLE")
    print("=" * 90)
    print(f"SILVER_POLE_ROIS                 : {SILVER_POLE_ROIS}")
    print(f"SILVER_POLE_SELECTION            : {SILVER_POLE_SELECTION}")
    print(f"SILVER_POLE_SELECTION_OVERLAYS   : {SILVER_POLE_SELECTION_OVERLAYS}")

    print("\n" + "=" * 90)
    print("SILVER — ASSET CANDIDATES")
    print("=" * 90)
    print(f"SILVER_ASSET_PROMPT_RUNS         : {SILVER_ASSET_PROMPT_RUNS}")
    print(f"SILVER_ASSET_OVERLAYS            : {SILVER_ASSET_OVERLAYS}")
    print(f"SILVER_ASSET_MASKS               : {SILVER_ASSET_MASKS}")

    print("\n" + "=" * 90)
    print("SILVER — CROSSARM")
    print("=" * 90)
    print(f"SILVER_CROSSARM_DETECTION        : {SILVER_CROSSARM_DETECTION}")
    print(f"SILVER_CROSSARM_CANDIDATES       : {SILVER_CROSSARM_CANDIDATES}")
    print(f"SILVER_CROSSARM_PROCESSING       : {SILVER_CROSSARM_PROCESSING}")
    print(f"SILVER_CROSSARM_REVIEW           : {SILVER_CROSSARM_REVIEW}")

    print("\n" + "=" * 90)
    print("GOLD")
    print("=" * 90)
    print(f"GOLD_ASSET_DETECTIONS            : {GOLD_ASSET_DETECTIONS}")
    print(f"GOLD_CROSSARM_DETECTIONS         : {GOLD_CROSSARM_DETECTIONS}")
    
    
# =============================================================================
# CELL 10B — PRODUCTION OUTPUT CONFIG + SAVE HELPERS
# =============================================================================
# OVERVIEW:
# This cell defines production run identity, output-save flags, run-scoped output
# paths, and reusable save helpers for crossarm and asset-candidate processing.
#
# SAVE POLICY:
#   Bronze:
#       Overwrite each run. Bronze is a transient source-image landing zone.
#
#   Silver:
#       Save lightweight production audit/stage tables only.
#       Do not save intermediate masks.
#       Do not save stage-grid images.
#
#   Gold:
#       Save final production tables as Parquet.
#       Save final production review images as PNG.
#
#   RUN_ID:
#       Every saved table and image path is stamped with RUN_ID so runs never
#       overwrite each other.
#
# STRUCTURE:
#   A. SAFETY CHECKS
#        Section 10B.1. Required setup checks
#        Section 10B.2. Parquet engine check
#
#   B. RUN CONFIG
#        Section 10B.3. Run identity
#        Section 10B.4. Save flags
#        Section 10B.5. Run-scoped output paths
#
#   C. SAVE HELPERS
#        Section 10B.6. Safe path-name helper
#        Section 10B.7. Parquet-safe DataFrame helper
#        Section 10B.8. Save run table helper
#        Section 10B.9. Final-image save decision helper
#        Section 10B.10. Save final image helper
#
#   D. OUTPUT SUMMARY
#        Section 10B.11. Production output summary
#
# IMPORTANT:
#   - Run this after CELL 10.
#   - This cell does not run SAM3 inference.
#   - This cell does not save anything by itself.
#   - Batch CELL 16 and CELL 17 accumulate rows in memory and then call these
#     helpers.
# =============================================================================

# =============================================================================
# A. SAFETY CHECKS
# =============================================================================

# =============================================================================
# 10B.1 REQUIRED SETUP CHECKS
# =============================================================================
# EXPLANATION:
# Fail early if required workspace variables from CELL 10 are missing.
# =============================================================================

required_cell10b_globals = [
    "os",
    "pd",
    "json",
    "datetime",
    "timezone",
    "plt",
    "GOLD_CROSSARM_DETECTIONS",
    "SILVER_CROSSARM_PROCESSING",
]

missing_cell10b_globals = [
    name for name in required_cell10b_globals
    if name not in globals()
]

if missing_cell10b_globals:
    raise NameError(
        "CELL 10B requires earlier setup cells to run successfully first.\n"
        "Please run CELL 10 before CELL 10B.\n"
        f"Missing globals: {missing_cell10b_globals}"
    )

# =============================================================================
# 10B.2 PARQUET ENGINE CHECK
# =============================================================================
# EXPLANATION:
# pandas requires either pyarrow or fastparquet to write Parquet files.
#
# Databricks usually has pyarrow available. This check makes the failure clear
# if the runtime does not.
# =============================================================================

PARQUET_ENGINE = None

try:
    import pyarrow  # noqa: F401
    PARQUET_ENGINE = "pyarrow"
except Exception:
    try:
        import fastparquet  # noqa: F401
        PARQUET_ENGINE = "fastparquet"
    except Exception:
        PARQUET_ENGINE = None

if PARQUET_ENGINE is None:
    raise ImportError(
        "No Parquet engine is available for pandas.to_parquet().\n"
        "Install pyarrow or fastparquet before running production saves.\n"
        "Recommended option: %pip install pyarrow"
    )

# =============================================================================
# B. RUN CONFIG
# =============================================================================

# =============================================================================
# 10B.3 RUN IDENTITY
# =============================================================================
# EXPLANATION:
# RUN_ID identifies this production batch run.
#
# Timestamp format:
#   run_YYYYMMDD_HHMMSS
#
# RUN_TIMESTAMP is stored as a column in saved tables for auditability.
# =============================================================================

RUN_START_UTC = datetime.now(timezone.utc)

RUN_ID = RUN_START_UTC.strftime("run_%Y%m%d_%H%M%S")
RUN_TIMESTAMP = RUN_START_UTC.isoformat()

# =============================================================================
# 10B.4 SAVE FLAGS
# =============================================================================
# EXPLANATION:
# These flags control production save behaviour for the 975-image validation
# run.
#
# Production defaults:
#   - save lightweight Silver crossarm stage tables
#   - do not save intermediate crossarm masks
#   - do not save crossarm stage-grid images
#   - save final Gold crossarm tables
#   - save final Gold crossarm images
#   - save raw Silver insulator candidate tables
#   - save raw Silver insulator QA overlays
# =============================================================================

OVERWRITE_BRONZE = bool(globals().get("OVERWRITE_BRONZE", True))

SAVE_SILVER_STAGE_TABLES = True
SAVE_SILVER_MASKS = False
SAVE_SILVER_STAGE_IMAGES = False

SAVE_GOLD_TABLES = True
SAVE_GOLD_FINAL_IMAGES = True
SAVE_GOLD_FINAL_IMAGES_REVIEW_ONLY = False

# Insulator raw-candidate discovery outputs.
SAVE_INSULATOR_RAW_TABLES = True
SAVE_INSULATOR_QA_OVERLAYS = True

# =============================================================================
# 10B.5 RUN-SCOPED OUTPUT PATHS
# =============================================================================
# EXPLANATION:
# Root output folders are stable, while each run writes into a RUN_ID subfolder.
#
# This gives cumulative outputs without Delta and without overwriting prior runs.
# =============================================================================

GOLD_CROSSARM_TABLES_ROOT = os.path.join(
    GOLD_CROSSARM_DETECTIONS,
    "tables",
)

GOLD_CROSSARM_IMAGES_ROOT = os.path.join(
    GOLD_CROSSARM_DETECTIONS,
    "images",
)

SILVER_CROSSARM_STAGE_TABLES_ROOT = os.path.join(
    SILVER_CROSSARM_PROCESSING,
    "stage_tables",
)

RUN_GOLD_TABLES_DIR = os.path.join(
    GOLD_CROSSARM_TABLES_ROOT,
    RUN_ID,
)

RUN_GOLD_IMAGES_DIR = os.path.join(
    GOLD_CROSSARM_IMAGES_ROOT,
    RUN_ID,
)

RUN_SILVER_STAGE_TABLES_DIR = os.path.join(
    SILVER_CROSSARM_STAGE_TABLES_ROOT,
    RUN_ID,
)

# =============================================================================
# C. SAVE HELPERS
# =============================================================================

# =============================================================================
# 10B.6 SAFE PATH-NAME HELPER
# =============================================================================
# EXPLANATION:
# Convert image IDs, ROI names, and output names into safe folder/file parts.
# =============================================================================

def make_safe_path_part(value, fallback="unknown"):
    """
    Convert a value into a safe path component.

    Args:
        value:
            Value to convert into a path-safe string.

        fallback:
            Value to use when input is missing or empty.

    Returns:
        str:
            Path-safe string.
    """
    if value is None:
        text = fallback
    else:
        text = str(value).strip()

    if len(text) == 0 or text.lower() == "nan":
        text = fallback

    text = text.replace(os.sep, "_")
    text = text.replace(" ", "_")

    safe_chars = []
    for ch in text:
        if ch.isalnum() or ch in ["_", "-", ".", "="]:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")

    return "".join(safe_chars)

# =============================================================================
# 10B.7 PARQUET-SAFE DATAFRAME HELPER
# =============================================================================
# EXPLANATION:
# Prepare a pandas DataFrame for Parquet output.
#
# This helper:
#   1) adds run_id and run_timestamp if missing
#   2) drops raw list columns when a matching *_text column exists
#   3) converts unsupported object values to JSON strings or plain strings
# =============================================================================

def prepare_dataframe_for_parquet(df):
    """
    Prepare a DataFrame for stable Parquet output.

    Args:
        df:
            pandas DataFrame.

    Returns:
        pandas.DataFrame:
            Parquet-safe copy of the input DataFrame.
    """
    df_out = df.copy()

    if "run_id" not in df_out.columns:
        df_out["run_id"] = RUN_ID

    if "run_timestamp" not in df_out.columns:
        df_out["run_timestamp"] = RUN_TIMESTAMP

    # Prefer existing string lineage columns over raw list columns.
    for list_col in ["source_orig_det_idxs"]:
        text_col = f"{list_col}_text"

        if list_col in df_out.columns and text_col in df_out.columns:
            df_out = df_out.drop(columns=[list_col])

    # Convert remaining complex object values into stable strings.
    for col_name in df_out.columns:
        if df_out[col_name].dtype != object:
            continue

        def _convert_object_value(value):
            if isinstance(value, (list, tuple, dict, set)):
                try:
                    return json.dumps(value, default=str)
                except Exception:
                    return str(value)

            return value

        df_out[col_name] = df_out[col_name].map(_convert_object_value)

    return df_out

# =============================================================================
# 10B.8 SAVE RUN TABLE HELPER
# =============================================================================
# EXPLANATION:
# Save one accumulated DataFrame as one Parquet file.
#
# IMPORTANT:
#   Batch cells should accumulate rows across all ROIs first, then call this
#   helper once per table at the end of the run.
# =============================================================================

def save_run_table(df, out_dir, table_name):
    """
    Save one accumulated DataFrame as a single Parquet file.

    Args:
        df:
            pandas DataFrame containing all rows for this run/table.

        out_dir:
            Run-scoped output directory.

        table_name:
            Output file stem, without extension.

    Returns:
        str or None:
            Written Parquet path, or None if nothing was written.
    """
    if df is None or len(df) == 0:
        if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
            print(f"  [save] {table_name}: empty, skipped")

        return None

    df_out = prepare_dataframe_for_parquet(df)

    os.makedirs(out_dir, exist_ok=True)

    safe_table_name = make_safe_path_part(table_name, fallback="table")
    out_path = os.path.join(out_dir, f"{safe_table_name}.parquet")

    try:
        df_out.to_parquet(
            out_path,
            index=False,
            engine=PARQUET_ENGINE,
        )

        if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
            print(f"  [save] {safe_table_name}: {len(df_out)} rows -> {out_path}")

        return out_path

    except Exception as exc:
        # Final fallback for difficult object columns.
        for col_name in df_out.columns:
            if df_out[col_name].dtype == object:
                df_out[col_name] = df_out[col_name].astype(str)

        df_out.to_parquet(
            out_path,
            index=False,
            engine=PARQUET_ENGINE,
        )

        if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
            print(
                f"  [save] {safe_table_name}: {len(df_out)} rows -> {out_path} "
                f"(object columns stringified after: {exc})"
            )

        return out_path


# =============================================================================
# 10B.9 FINAL-IMAGE SAVE DECISION HELPER
# =============================================================================
# EXPLANATION:
# Decide whether final images should be saved for a given ROI.
#
# If SAVE_GOLD_FINAL_IMAGES_REVIEW_ONLY is True, final images are saved only
# when the ROI review reason is not "ok".
# =============================================================================

def should_save_gold_final_images(final_roi_review_reason=None):
    """
    Decide whether final Gold images should be saved for one ROI.

    Args:
        final_roi_review_reason:
            Review reason from final output, usually "ok" or a review flag.

    Returns:
        bool:
            True if final images should be saved.
    """
    if not SAVE_GOLD_FINAL_IMAGES:
        return False

    if not SAVE_GOLD_FINAL_IMAGES_REVIEW_ONLY:
        return True

    reason = (
        str(final_roi_review_reason).strip().lower()
        if final_roi_review_reason is not None
        else "unknown"
    )

    return reason != "ok"


# =============================================================================
# 10B.10 SAVE FINAL IMAGE HELPER
# =============================================================================
# EXPLANATION:
# Save one matplotlib figure under:
#
#   GOLD_CROSSARM_DETECTIONS/images/<RUN_ID>/<image_id>/
#
# The figure is closed after saving to prevent memory buildup in batch runs.
# =============================================================================

def save_final_image(fig, image_id, roi_file_name, image_name, dpi=120):
    """
    Save one final production image figure.

    Args:
        fig:
            matplotlib Figure object.

        image_id:
            Source image identifier.

        roi_file_name:
            ROI filename or identifier.

        image_name:
            Output image type, e.g. "Final_Image_Real_Mask".

        dpi:
            Output image DPI.

    Returns:
        str:
            Written PNG path.
    """
    safe_image_id = make_safe_path_part(
        image_id,
        fallback="unknown_image",
    )

    safe_roi_name = make_safe_path_part(
        roi_file_name,
        fallback="unknown_roi",
    )

    safe_image_name = make_safe_path_part(
        image_name,
        fallback="final_image",
    )

    out_dir = os.path.join(
        RUN_GOLD_IMAGES_DIR,
        safe_image_id,
    )

    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(
        out_dir,
        f"{safe_roi_name}__{safe_image_name}.png",
    )

    fig.savefig(
        out_path,
        bbox_inches="tight",
        dpi=dpi,
    )

    try:
        plt.close(fig)
    except Exception:
        pass

    return out_path


# =============================================================================
# D. OUTPUT SUMMARY
# =============================================================================

# =============================================================================
# 10B.11 PRODUCTION OUTPUT SUMMARY
# =============================================================================
# EXPLANATION:
# Print the active run identity, save flags, and run-scoped output paths.
# =============================================================================

if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print("Production output config ready.\n")

    print("=" * 90)
    print("RUN IDENTITY")
    print("=" * 90)
    print(f"RUN_ID                              : {RUN_ID}")
    print(f"RUN_TIMESTAMP                       : {RUN_TIMESTAMP}")
    print(f"PARQUET_ENGINE                      : {PARQUET_ENGINE}")

    print("\n" + "=" * 90)
    print("SAVE FLAGS")
    print("=" * 90)
    print(f"OVERWRITE_BRONZE                    : {OVERWRITE_BRONZE}")
    print(f"SAVE_SILVER_STAGE_TABLES            : {SAVE_SILVER_STAGE_TABLES}")
    print(f"SAVE_SILVER_MASKS                   : {SAVE_SILVER_MASKS}")
    print(f"SAVE_SILVER_STAGE_IMAGES            : {SAVE_SILVER_STAGE_IMAGES}")
    print(f"SAVE_GOLD_TABLES                    : {SAVE_GOLD_TABLES}")
    print(f"SAVE_GOLD_FINAL_IMAGES              : {SAVE_GOLD_FINAL_IMAGES}")
    print(f"SAVE_GOLD_FINAL_IMAGES_REVIEW_ONLY  : {SAVE_GOLD_FINAL_IMAGES_REVIEW_ONLY}")
    print(f"SAVE_INSULATOR_RAW_TABLES           : {SAVE_INSULATOR_RAW_TABLES}")
    print(f"SAVE_INSULATOR_QA_OVERLAYS          : {SAVE_INSULATOR_QA_OVERLAYS}")

    print("\n" + "=" * 90)
    print("RUN-SCOPED OUTPUT PATHS")
    print("=" * 90)
    print(f"RUN_GOLD_TABLES_DIR                 : {RUN_GOLD_TABLES_DIR}")
    print(f"RUN_GOLD_IMAGES_DIR                 : {RUN_GOLD_IMAGES_DIR}")
    print(f"RUN_SILVER_STAGE_TABLES_DIR         : {RUN_SILVER_STAGE_TABLES_DIR}")
    
    
# =============================================================================
# CELL 11 — DATA INGESTION: SCAN SOURCE VOLUME INTO BRONZE
# =============================================================================
# OVERVIEW:
# This cell performs production image ingestion from a source Databricks Volume
# into the Bronze layer.
#
# Bronze is treated as a transient working copy:
#   - source images are discovered from SOURCE_IMAGE_ROOT
#   - images are copied into BRONZE_SOURCE_IMAGES
#   - relative folder structure is preserved
#   - the Bronze working copy is rebuilt when OVERWRITE_BRONZE is True
#
# STRUCTURE:
#   A. SAFETY CHECKS
#        Section 11.1. Required setup checks
#
#   B. SOURCE CONFIG
#        Section 11.2. Define source image folder
#        Section 11.3. Resolve supported image extensions
#        Section 11.4. Resolve overwrite/run controls
#
#   C. SOURCE DISCOVERY
#        Section 11.5. Discover source image files
#        Section 11.6. Validate discovered files
#
#   D. BRONZE INGESTION
#        Section 11.7. Prepare temporary Bronze ingest folder
#        Section 11.8. Copy images into temporary Bronze folder
#        Section 11.9. Verify copied files
#        Section 11.10. Promote temporary folder to Bronze
#
#   E. MANIFEST OUTPUT
#        Section 11.11. Build images_df manifest
#        Section 11.12. Save images_df manifest
#        Section 11.13. Ingestion summary
#
# IMPORTANT:
#   - Run this after CELL 10B.
#   - This cell does not run SAM3 inference.
#   - Bronze is transient and may be overwritten when OVERWRITE_BRONZE is True.
#   - The source folder is not modified.
# =============================================================================

# =============================================================================
# A. SAFETY CHECKS
# =============================================================================

# =============================================================================
# 11.1 REQUIRED SETUP CHECKS
# =============================================================================
# EXPLANATION:
# Fail early if required variables from earlier setup cells are missing.
#
# Required earlier cells:
#   - CELL 3A  : os, pd, np, shutil, datetime, timezone
#   - CELL 3B  : VALID_IMAGE_EXTENSIONS
#   - CELL 10  : BRONZE_ROOT, BRONZE_SOURCE_IMAGES, DF_DIR
#   - CELL 10B : RUN_ID, RUN_TIMESTAMP, OVERWRITE_BRONZE, save_run_table
# =============================================================================

required_cell11_globals = [
    "os",
    "pd",
    "np",
    "shutil",
    "datetime",
    "timezone",
    "BRONZE_ROOT",
    "BRONZE_SOURCE_IMAGES",
    "DF_DIR",
    "VALID_IMAGE_EXTENSIONS",
]

missing_cell11_globals = [
    name for name in required_cell11_globals
    if name not in globals()
]

if missing_cell11_globals:
    raise NameError(
        "CELL 11 requires earlier setup cells to run successfully first.\n"
        "Please run CELL 10 and CELL 10B before CELL 11.\n"
        f"Missing globals: {missing_cell11_globals}"
    )


# =============================================================================
# B. SOURCE CONFIG
# =============================================================================

# =============================================================================
# 11.2 DEFINE SOURCE IMAGE FOLDER
# =============================================================================
# EXPLANATION:
# SOURCE_IMAGE_ROOT is the external source folder containing raw images to ingest.
#
# IMPORTANT:
#   Keep this path unchanged for now.
# =============================================================================

SOURCE_IMAGE_ROOT = (
    "/Volumes/"
    "/sharyn_volume/sam3_project/"
    "test_images"
)

SOURCE_IMAGE_ROOT = os.path.abspath(SOURCE_IMAGE_ROOT)

if not os.path.isdir(SOURCE_IMAGE_ROOT):
    raise FileNotFoundError(
        "Source image folder does not exist.\n"
        f"SOURCE_IMAGE_ROOT: {SOURCE_IMAGE_ROOT}"
    )

# =============================================================================
# 11.3 RESOLVE SUPPORTED IMAGE EXTENSIONS
# =============================================================================
# EXPLANATION:
# Use the shared extension list from CELL 3B so file-type support has one source
# of truth.
# =============================================================================

IMAGE_EXTS_LOWER = tuple(
    sorted({
        str(ext).lower()
        for ext in VALID_IMAGE_EXTENSIONS
    })
)

if len(IMAGE_EXTS_LOWER) == 0:
    raise ValueError(
        "VALID_IMAGE_EXTENSIONS is empty. Please check CELL 3B."
    )

# =============================================================================
# 11.4 RESOLVE OVERWRITE AND RUN CONTROLS
# =============================================================================
# EXPLANATION:
# OVERWRITE_BRONZE should normally come from CELL 10B. This fallback keeps CELL
# 11 safe if it is run independently during development.
#
# RUN_ID and RUN_TIMESTAMP should normally come from CELL 10B. If missing, this
# cell creates fallback values so images_df remains auditable.
# =============================================================================

OVERWRITE_BRONZE = bool(
    globals().get(
        "OVERWRITE_BRONZE",
        True,
    )
)

if "RUN_ID" not in globals() or "RUN_TIMESTAMP" not in globals():
    _fallback_ingest_time = datetime.now(timezone.utc)
    RUN_ID = _fallback_ingest_time.strftime("run_%Y%m%d_%H%M%S")
    RUN_TIMESTAMP = _fallback_ingest_time.isoformat()


# Guard against accidentally using Bronze itself as the source.
source_abs = os.path.abspath(SOURCE_IMAGE_ROOT)
bronze_abs = os.path.abspath(BRONZE_SOURCE_IMAGES)

if source_abs == bronze_abs:
    raise ValueError(
        "SOURCE_IMAGE_ROOT cannot be the same as BRONZE_SOURCE_IMAGES.\n"
        f"SOURCE_IMAGE_ROOT       : {source_abs}\n"
        f"BRONZE_SOURCE_IMAGES    : {bronze_abs}"
    )

# =============================================================================
# C. SOURCE DISCOVERY
# =============================================================================

# =============================================================================
# 11.5 DISCOVER SOURCE IMAGE FILES
# =============================================================================
# EXPLANATION:
# Recursively scan SOURCE_IMAGE_ROOT and collect all supported image files.
# =============================================================================

source_image_files = []

for root, _, files in os.walk(SOURCE_IMAGE_ROOT):
    for file_name in files:
        if file_name.lower().endswith(IMAGE_EXTS_LOWER):
            source_image_files.append(
                os.path.join(
                    root,
                    file_name,
                )
            )

source_image_files = sorted(source_image_files)

# =============================================================================
# 11.6 VALIDATE DISCOVERED FILES
# =============================================================================
# EXPLANATION:
# Fail early if no valid source images were discovered.
# =============================================================================

if len(source_image_files) == 0:
    raise ValueError(
        "No supported image files were found in SOURCE_IMAGE_ROOT.\n"
        f"SOURCE_IMAGE_ROOT          : {SOURCE_IMAGE_ROOT}\n"
        f"VALID_IMAGE_EXTENSIONS    : {VALID_IMAGE_EXTENSIONS}"
    )


# Build relative paths before copying so path preservation is explicit.
source_relative_paths = [
    os.path.relpath(
        src_path,
        SOURCE_IMAGE_ROOT,
    )
    for src_path in source_image_files
]

duplicate_relative_paths = (
    pd.Series(source_relative_paths)
    .value_counts()
    .loc[lambda series: series > 1]
)

if len(duplicate_relative_paths) > 0:
    raise ValueError(
        "Duplicate relative source paths were found. This should not happen "
        "inside one source root.\n"
        f"Examples: {duplicate_relative_paths.head(10).to_dict()}"
    )

# =============================================================================
# D. BRONZE INGESTION
# =============================================================================

# =============================================================================
# 11.7 PREPARE TEMPORARY BRONZE INGEST FOLDER
# =============================================================================
# EXPLANATION:
# Copy files into a run-scoped temporary folder first. Only after all copies are
# verified do we replace BRONZE_SOURCE_IMAGES.
#
# This avoids leaving Bronze empty or half-copied if ingestion fails midway.
# =============================================================================

temp_bronze_ingest_dir = os.path.join(
    BRONZE_ROOT,
    f"_tmp_ingest_{RUN_ID}",
)

if os.path.isdir(temp_bronze_ingest_dir):
    shutil.rmtree(temp_bronze_ingest_dir)

os.makedirs(
    temp_bronze_ingest_dir,
    exist_ok=True,
)

# =============================================================================
# 11.8 COPY IMAGES INTO TEMPORARY BRONZE FOLDER
# =============================================================================
# EXPLANATION:
# Preserve relative subfolder structure to avoid basename collisions.
#
# IMPORTANT:
#   If copying fails midway, remove the temporary ingest folder so failed runs do
#   not leave orphan folders under BRONZE_ROOT.
# =============================================================================

temp_bronze_image_paths = []

try:
    for src_path, rel_path in zip(source_image_files, source_relative_paths):
        dst_path = os.path.join(
            temp_bronze_ingest_dir,
            rel_path,
        )

        os.makedirs(
            os.path.dirname(dst_path),
            exist_ok=True,
        )

        shutil.copy2(
            src_path,
            dst_path,
        )

        temp_bronze_image_paths.append(dst_path)

except Exception:
    if os.path.isdir(temp_bronze_ingest_dir):
        shutil.rmtree(temp_bronze_ingest_dir)

    raise

# =============================================================================
# 11.9 VERIFY COPIED FILES
# =============================================================================
# EXPLANATION:
# Confirm every expected temporary Bronze file exists before promoting the temp
# folder into the official Bronze location.
# =============================================================================

missing_temp_bronze_files = [
    path
    for path in temp_bronze_image_paths
    if not os.path.exists(path)
]

if missing_temp_bronze_files:
    raise RuntimeError(
        "Some temporary Bronze image files were not copied successfully.\n"
        f"Missing examples: {missing_temp_bronze_files[:10]}"
    )

# =============================================================================
# 11.10 PROMOTE TEMPORARY FOLDER TO BRONZE
# =============================================================================
# EXPLANATION:
# Replace BRONZE_SOURCE_IMAGES only after the temporary folder is fully copied
# and verified.
#
# IMPORTANT:
#   BRONZE_SOURCE_IMAGES may already exist as an empty folder because CELL 10
#   creates the directory tree. Therefore, remove the existing destination folder
#   before shutil.move(...). Otherwise shutil.move() may place the temp folder
#   inside BRONZE_SOURCE_IMAGES instead of replacing it.
# =============================================================================

if os.path.isdir(BRONZE_SOURCE_IMAGES):
    existing_bronze_items = os.listdir(BRONZE_SOURCE_IMAGES)

    if existing_bronze_items and not OVERWRITE_BRONZE:
        shutil.rmtree(temp_bronze_ingest_dir)

        raise RuntimeError(
            "BRONZE_SOURCE_IMAGES already contains files.\n"
            "Re-running CELL 11 would replace the Bronze working copy.\n"
            "To rebuild Bronze, set:\n"
            "OVERWRITE_BRONZE = True\n"
            "and then run CELL 11 again."
        )

    # Remove the destination folder whether it is empty or populated.
    # This prevents shutil.move() from nesting the temp folder inside it.
    shutil.rmtree(BRONZE_SOURCE_IMAGES)

os.makedirs(
    BRONZE_ROOT,
    exist_ok=True,
)

shutil.move(
    temp_bronze_ingest_dir,
    BRONZE_SOURCE_IMAGES,
)


# Build final Bronze image paths after promotion.
bronze_image_paths = [
    os.path.join(
        BRONZE_SOURCE_IMAGES,
        rel_path,
    )
    for rel_path in source_relative_paths
]

# =============================================================================
# E. MANIFEST OUTPUT
# =============================================================================

# =============================================================================
# 11.11 BUILD images_df MANIFEST
# =============================================================================
# EXPLANATION:
# images_df becomes the raw image tracking table for downstream cells.
#
# It contains one row per Bronze image and records:
#   - run identity
#   - original source path
#   - Bronze working-copy path
#   - relative image path
#   - file metadata useful for audits
# =============================================================================

images_df = pd.DataFrame({
    "run_id": RUN_ID,
    "run_timestamp": RUN_TIMESTAMP,
    "source_image_path": source_image_files,
    "image_path": bronze_image_paths,
    "relative_image_path": source_relative_paths,
})

images_df["file_name"] = images_df["image_path"].map(os.path.basename)

images_df["stem"] = images_df["file_name"].map(
    lambda value: os.path.splitext(value)[0]
)

images_df["ext"] = images_df["file_name"].map(
    lambda value: os.path.splitext(value)[1]
)

images_df["ext_lower"] = images_df["ext"].str.lower()

images_df["source_layer"] = "bronze"
images_df["source_root"] = SOURCE_IMAGE_ROOT
images_df["bronze_root"] = BRONZE_SOURCE_IMAGES

images_df["source_file_size_bytes"] = images_df["source_image_path"].map(
    lambda path: os.path.getsize(path) if os.path.exists(path) else np.nan
)

images_df["bronze_file_size_bytes"] = images_df["image_path"].map(
    lambda path: os.path.getsize(path) if os.path.exists(path) else np.nan
)

images_df["source_modified_time_utc"] = images_df["source_image_path"].map(
    lambda path: (
        datetime.fromtimestamp(
            os.path.getmtime(path),
            timezone.utc,
        ).isoformat()
        if os.path.exists(path)
        else None
    )
)

images_df["bronze_modified_time_utc"] = images_df["image_path"].map(
    lambda path: (
        datetime.fromtimestamp(
            os.path.getmtime(path),
            timezone.utc,
        ).isoformat()
        if os.path.exists(path)
        else None
    )
)


# Validate final Bronze files.
missing_final_bronze_files = [
    path
    for path in images_df["image_path"].tolist()
    if not os.path.exists(path)
]

if missing_final_bronze_files:
    raise RuntimeError(
        "Some final Bronze image files are missing after promotion.\n"
        f"Missing examples: {missing_final_bronze_files[:10]}"
    )


# =============================================================================
# 11.12 SAVE images_df MANIFEST
# =============================================================================
# EXPLANATION:
# Save the ingestion manifest as one Parquet file for this run.
#
# If CELL 10B save_run_table is available, use it. Otherwise, write directly.
# =============================================================================

RUN_BRONZE_MANIFEST_DIR = os.path.join(
    DF_DIR,
    "bronze_ingestion",
    RUN_ID,
)

if "save_run_table" in globals() and callable(globals().get("save_run_table")):
    images_manifest_path = save_run_table(
        images_df,
        RUN_BRONZE_MANIFEST_DIR,
        "images_manifest",
    )
else:
    os.makedirs(
        RUN_BRONZE_MANIFEST_DIR,
        exist_ok=True,
    )

    images_manifest_path = os.path.join(
        RUN_BRONZE_MANIFEST_DIR,
        "images_manifest.parquet",
    )

    images_df.to_parquet(
        images_manifest_path,
        index=False,
    )

# =============================================================================
# 11.13 INGESTION SUMMARY
# =============================================================================
# EXPLANATION:
# Print a compact production ingestion summary when PRINT_CONFIG_SUMMARY is
# enabled.
# =============================================================================

if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print("Bronze ingestion complete.\n")

    print("=" * 90)
    print("BRONZE INGESTION SUMMARY")
    print("=" * 90)
    print(f"RUN_ID                         : {RUN_ID}")
    print(f"RUN_TIMESTAMP                  : {RUN_TIMESTAMP}")
    print(f"SOURCE_IMAGE_ROOT              : {SOURCE_IMAGE_ROOT}")
    print(f"BRONZE_SOURCE_IMAGES           : {BRONZE_SOURCE_IMAGES}")
    print(f"OVERWRITE_BRONZE               : {OVERWRITE_BRONZE}")
    print(f"Discovered source image count  : {len(source_image_files)}")
    print(f"Copied Bronze image count      : {len(bronze_image_paths)}")
    print(f"images_df shape                : {images_df.shape}")
    print(f"images_manifest_path           : {images_manifest_path}")

    preview_cols = [
        "run_id",
        "relative_image_path",
        "image_path",
        "file_name",
        "source_file_size_bytes",
        "bronze_file_size_bytes",
    ]

    print("\nPreview:")
    print(images_df[preview_cols].head())
    
    
# =============================================================================
# CELL 12 — PREPARE IMAGE DATAFRAME FOR PRODUCTION PIPELINE
# =============================================================================
# OVERVIEW:
# This cell prepares the Bronze image manifest for downstream SAM3 production
# processing.
#
# It does not run SAM3 inference.
#
# This cell starts from images_df created in CELL 11 and creates run_images_df,
# the clean production working table used by later cells.
#
# STRUCTURE:
#   A. SAFETY CHECKS
#        Section 12.1. Imports
#        Section 12.2. Required setup checks
#        Section 12.3. Validate images_df
#        Section 12.4. Validate required columns
#
#   B. MANIFEST PREPARATION
#        Section 12.5. Create working copy
#        Section 12.6. Validate run identity columns
#        Section 12.7. Sort into stable production order
#        Section 12.8. Add processing order
#        Section 12.9. Create stable image_id values
#        Section 12.10. Validate Bronze image paths
#        Section 12.11. Reorder key columns
#
#   C. OUTPUT
#        Section 12.12. Final production checks
#        Section 12.13. Save run_images_df manifest
#        Section 12.14. Manifest summary
#
# IMPORTANT:
#   - Run this after CELL 11.
#   - This cell does not mutate images_df.
#   - This cell does not run SAM3 inference.
#   - image_id values are deterministic from relative_image_path, not row index.
#   - This cell reuses make_safe_path_part from CELL 10B so path/name sanitising
#     has one source of truth.
# =============================================================================

# =============================================================================
# A. SAFETY CHECKS
# =============================================================================

# =============================================================================
# 12.1 IMPORTS
# =============================================================================
# EXPLANATION:
# hashlib is used to create deterministic short hashes from relative image paths.
#
# Keeping the import here is acceptable because it is only used in this cell.
# =============================================================================

import hashlib

# =============================================================================
# 12.2 REQUIRED SETUP CHECKS
# =============================================================================
# EXPLANATION:
# Fail early if required variables from earlier cells are missing.
#
# Required earlier cells:
#   - CELL 3A  : os, pd
#   - CELL 3B  : IMAGE_ID_PREFIX
#   - CELL 10  : DF_DIR
#   - CELL 10B : RUN_ID, RUN_TIMESTAMP, make_safe_path_part, save_run_table
#   - CELL 11  : images_df
# =============================================================================

required_cell12_globals = [
    "os",
    "pd",
    "images_df",
    "IMAGE_ID_PREFIX",
    "DF_DIR",
    "RUN_ID",
    "RUN_TIMESTAMP",
    "make_safe_path_part",
    "save_run_table",
]

missing_cell12_globals = [
    name for name in required_cell12_globals
    if name not in globals()
]

if missing_cell12_globals:
    raise NameError(
        "CELL 12 requires earlier setup cells to run successfully first.\n"
        "Please run CELL 10B and CELL 11 before CELL 12.\n"
        f"Missing globals: {missing_cell12_globals}"
    )

# =============================================================================
# 12.3 VALIDATE images_df
# =============================================================================
# EXPLANATION:
# images_df should be the Bronze ingestion manifest produced by CELL 11.
# =============================================================================

if not isinstance(images_df, pd.DataFrame):
    raise TypeError(
        "images_df exists but is not a pandas DataFrame."
    )

if images_df.empty:
    raise ValueError(
        "images_df exists but is empty.\n"
        "Please check CELL 11."
    )

# =============================================================================
# 12.4 VALIDATE REQUIRED COLUMNS
# =============================================================================
# EXPLANATION:
# These columns are required by downstream image loading, tracking, and output
# naming logic.
# =============================================================================

required_images_df_cols = [
    "image_path",
    "relative_image_path",
    "file_name",
    "stem",
    "ext",
]

missing_images_df_cols = [
    col_name
    for col_name in required_images_df_cols
    if col_name not in images_df.columns
]

if missing_images_df_cols:
    raise ValueError(
        "images_df is missing required columns.\n"
        f"Missing columns: {missing_images_df_cols}"
    )

# =============================================================================
# B. MANIFEST PREPARATION
# =============================================================================

# =============================================================================
# 12.5 CREATE WORKING COPY
# =============================================================================
# EXPLANATION:
# Do not mutate the raw Bronze manifest directly. Later cells should use
# run_images_df as the production working image table.
# =============================================================================

run_images_df = images_df.copy()

# =============================================================================
# 12.6 VALIDATE RUN IDENTITY COLUMNS
# =============================================================================
# EXPLANATION:
# run_id and run_timestamp should come from CELL 11's manifest. If missing, add
# them from CELL 10B. If present but inconsistent with CELL 10B, fail early.
#
# This prevents writing a run_images manifest into one RUN_ID folder while the
# table rows carry a different run_id.
# =============================================================================

if "run_id" not in run_images_df.columns:
    run_images_df["run_id"] = RUN_ID

if "run_timestamp" not in run_images_df.columns:
    run_images_df["run_timestamp"] = RUN_TIMESTAMP

unique_manifest_run_ids = (
    run_images_df["run_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

if len(unique_manifest_run_ids) != 1:
    raise RuntimeError(
        "run_images_df must contain exactly one run_id.\n"
        f"Found run_id values: {unique_manifest_run_ids[:10]}"
    )

if str(unique_manifest_run_ids[0]) != str(RUN_ID):
    raise RuntimeError(
        "run_images_df run_id does not match active RUN_ID.\n"
        f"run_images_df run_id : {unique_manifest_run_ids[0]}\n"
        f"active RUN_ID        : {RUN_ID}\n"
        "Please rerun CELL 10B, CELL 11, and CELL 12 in order for a clean run."
    )

# =============================================================================
# 12.7 SORT INTO STABLE PRODUCTION ORDER
# =============================================================================
# EXPLANATION:
# Sorting by relative_image_path makes production runs easier to compare across
# reruns.
# =============================================================================

run_images_df = (
    run_images_df
    .sort_values(
        [
            "relative_image_path",
            "file_name",
            "image_path",
        ]
    )
    .reset_index(drop=True)
)

# =============================================================================
# 12.8 ADD PROCESSING ORDER
# =============================================================================
# EXPLANATION:
# processing_order records the exact row order used by downstream batch cells.
# =============================================================================

run_images_df["processing_order"] = range(len(run_images_df))

# =============================================================================
# 12.9 CREATE STABLE image_id VALUES
# =============================================================================
# EXPLANATION:
# image_id should be:
#   - deterministic
#   - path-safe
#   - unique
#   - stable even if another image is inserted earlier in processing order
#
# This production version uses:
#   img_<safe_stem>_<hash(relative_image_path)>
#
# The hash is based on relative_image_path rather than row index, so IDs are less
# likely to shift between runs when the source image set changes.
# =============================================================================

def _cell12_short_hash(value, length=12):
    """
    Create a deterministic short hash for one value.

    Args:
        value:
            Value to hash.

        length:
            Number of hexadecimal characters to keep.

    Returns:
        str:
            Short deterministic hash.
    """
    text = "" if value is None else str(value)

    return hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:length]


safe_stems = (
    run_images_df["stem"]
    .fillna("image")
    .astype(str)
    .map(lambda value: make_safe_path_part(value, fallback="image"))
)

relative_path_hashes = (
    run_images_df["relative_image_path"]
    .fillna("")
    .astype(str)
    .map(lambda value: _cell12_short_hash(value, length=12))
)

run_images_df["image_id"] = [
    f"{IMAGE_ID_PREFIX}_{safe_stem}_{path_hash}"
    for safe_stem, path_hash in zip(safe_stems, relative_path_hashes)
]

# =============================================================================
# 12.10 VALIDATE BRONZE IMAGE PATHS
# =============================================================================
# EXPLANATION:
# Downstream inference cells load images from image_path. Fail now if any Bronze
# image file is missing.
# =============================================================================

run_images_df["image_exists"] = run_images_df["image_path"].map(
    lambda path: os.path.exists(path)
)

missing_image_paths = (
    run_images_df
    .loc[~run_images_df["image_exists"], "image_path"]
    .tolist()
)

if missing_image_paths:
    raise FileNotFoundError(
        "Some Bronze image files referenced by run_images_df are missing.\n"
        f"Missing examples: {missing_image_paths[:10]}"
    )

# =============================================================================
# 12.11 REORDER KEY COLUMNS
# =============================================================================
# EXPLANATION:
# Put the most commonly used production tracking columns first.
# =============================================================================

preferred_front_cols = [
    "run_id",
    "run_timestamp",
    "processing_order",
    "image_id",
    "file_name",
    "stem",
    "ext",
    "ext_lower",
    "relative_image_path",
    "image_path",
    "image_exists",
    "source_image_path",
    "source_layer",
    "source_root",
    "bronze_root",
    "source_file_size_bytes",
    "bronze_file_size_bytes",
    "source_modified_time_utc",
    "bronze_modified_time_utc",
]

existing_front_cols = [
    col_name
    for col_name in preferred_front_cols
    if col_name in run_images_df.columns
]

remaining_cols = [
    col_name
    for col_name in run_images_df.columns
    if col_name not in existing_front_cols
]

run_images_df = run_images_df[
    existing_front_cols + remaining_cols
]

# =============================================================================
# C. OUTPUT
# =============================================================================

# =============================================================================
# 12.12 FINAL PRODUCTION CHECKS
# =============================================================================
# EXPLANATION:
# Enforce expected production output shape before later inference cells run.
# =============================================================================

if run_images_df["image_id"].duplicated().any():
    duplicate_image_ids = (
        run_images_df
        .loc[run_images_df["image_id"].duplicated(), "image_id"]
        .tolist()
    )

    raise RuntimeError(
        "run_images_df contains duplicate image_id values.\n"
        f"Duplicate examples: {duplicate_image_ids[:10]}"
    )

if run_images_df["processing_order"].duplicated().any():
    raise RuntimeError(
        "run_images_df contains duplicate processing_order values."
    )

if not run_images_df["processing_order"].is_monotonic_increasing:
    raise RuntimeError(
        "run_images_df processing_order is not sorted."
    )


# =============================================================================
# 12.13 SAVE run_images_df MANIFEST
# =============================================================================
# EXPLANATION:
# Save the prepared production image manifest as one Parquet file for this run.
#
# This manifest is useful for:
#   - audit
#   - rerun tracking
#   - joining image-level outputs back to the source Bronze manifest
# =============================================================================

run_images_manifest_dir = os.path.join(
    DF_DIR,
    "run_images",
    RUN_ID,
)

run_images_manifest_path = save_run_table(
    run_images_df,
    run_images_manifest_dir,
    "run_images_manifest",
)

# =============================================================================
# 12.14 MANIFEST SUMMARY
# =============================================================================
# EXPLANATION:
# Print a compact production summary when PRINT_CONFIG_SUMMARY is enabled.
# =============================================================================

if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print("Production image manifest preparation complete.\n")

    print("=" * 90)
    print("RUN IMAGE MANIFEST SUMMARY")
    print("=" * 90)
    print(f"RUN_ID                         : {RUN_ID}")
    print(f"run_images_df rows             : {len(run_images_df)}")
    print(f"IMAGE_ID_PREFIX                : {IMAGE_ID_PREFIX}")
    print(f"run_images_manifest_path       : {run_images_manifest_path}")

    preview_cols = [
        "processing_order",
        "image_id",
        "relative_image_path",
        "image_path",
    ]

    print("\nPreview:")
    print(run_images_df[preview_cols].head())
    
    
# =============================================================================
# CELL 13 — PRODUCTION POLE SELECTION + QA OVERLAY SAVING
# =============================================================================
# OVERVIEW:
# This cell runs production pole selection across all images in run_images_df.
#
# It uses SAM3 text prompting to detect utility-pole candidates, applies
# production filtering and ranking rules, selects one best pole per image when
# possible, saves optional human-review QA overlays, and prepares full-resolution
# selected-pole outputs for downstream ROI generation.
#
# IMPORTANT DESIGN RULES:
#   - Full-resolution image coordinates remain the source of truth.
#   - Full-resolution selected-pole masks are kept in pole_mask_lookup.
#   - Saved overlay PNGs are QA artifacts only.
#   - Downstream cells must use pole_selection_df and pole_mask_lookup, not the
#     saved overlay PNGs.
#
# PRODUCTION BEHAVIOUR:
#   - Processes every image in run_images_df for the active RUN_ID.
#   - Validates run_images_df lineage, RUN_ID consistency, image_id uniqueness,
#     and source image-path existence before inference.
#   - Captures per-image failures in pole_failures_df so one bad image does not
#     stop the full production batch.
#   - Produces exactly one outcome per input image:
#
#       either:
#           one row in pole_selection_df
#
#       or:
#           one row in pole_failures_df
#
#   - pole_selection_df may contain:
#
#       selection_status == "selected"
#           A reliable pole was selected.
#
#       selection_status == "no_reliable_pole_found"
#           The image processed successfully, but no reliable pole was selected.
#
#   - pole_mask_lookup is pruned after selection so it contains selected-pole
#     masks only.
#
# STRUCTURE:
#   A. SAFETY CHECKS + OUTPUT SETUP
#        Section 13.1.  Required setup checks
#        Section 13.2.  Validate run_images_df
#        Section 13.3.  Validate required run_images_df columns, RUN_ID,
#                       image_id uniqueness, and image paths
#        Section 13.4.  Validate pole prompt/config values
#        Section 13.5.  Define production save flags
#        Section 13.6.  Define run-scoped output paths
#        Section 13.7.  Create output directories
#        Section 13.8.  Initialise production accumulators
#        Section 13.9.  Set processor device
#        Section 13.10. Setup summary
#
#   B. HELPER FUNCTIONS
#        Section 13.11. Row value helper
#        Section 13.12. Lineage field helper
#        Section 13.13. NumPy conversion helper
#        Section 13.14. Detection key helper
#        Section 13.15. Detection count helper
#        Section 13.16. Box normalisation helper
#        Section 13.17. Score normalisation helper
#        Section 13.18. Mask normalisation helper
#        Section 13.19. Box clipping helper
#        Section 13.20. Pole overlay label helper
#        Section 13.21. Empty overlay metadata helper
#        Section 13.22. Pole overlay output-path helper
#        Section 13.23. Pole overlay save helper
#        Section 13.24. No-reliable-pole row helper
#        Section 13.25. Pole failure row helper
#
#   C. PRODUCTION POLE SELECTION LOOP
#        Section 13.26. Run SAM3 pole selection across all images
#
#   D. OUTPUT ASSEMBLY + FINAL CHECKS
#        Section 13.27. Combine accumulated output rows
#        Section 13.28. Prune pole_mask_lookup to selected poles only
#        Section 13.29. Reorder output columns
#        Section 13.30. Final production consistency checks
#        Section 13.31. Output count summary object
#
#   E. SAVE OUTPUT TABLES + FINAL SUMMARY
#        Section 13.32. Required save checks
#        Section 13.33. Save production pole tables
#        Section 13.34. Validate required saved outputs
#        Section 13.35. Final CELL 13 summary
#
# OUTPUTS:
#   pole_candidates_df:
#       All scored pole candidates across all successfully processed images.
#       This is an audit/debug table, not the downstream geometry source.
#
#   pole_selection_df:
#       One successful outcome row per processed image.
#       Contains either a selected pole or a no-reliable-pole outcome.
#       This is the critical downstream table for CELL 14 ROI generation.
#
#   pole_failures_df:
#       One row per image that failed during pole-selection processing.
#       Used for production auditability and batch robustness.
#
#   pole_mask_lookup:
#       Full-resolution selected-pole mask lookup keyed by:
#
#           (image_id, prompt, det_idx)
#
#       This is used by downstream ROI/crossarm cells when pole mask geometry is
#       needed.
#
#   cell13_output_counts:
#       Compact count summary used by the final save/summary section.
#
#   cell13_saved_paths:
#       Paths returned by save_run_table() for saved Cell 13 Parquet outputs.
#
# SAVED OUTPUTS:
#   Tables:
#       SILVER_POLE_SELECTION/tables/<RUN_ID>/
#           pole_candidates.parquet
#           pole_selection.parquet
#           pole_failures.parquet   # may be skipped if empty
#
#   QA overlays:
#       SILVER_POLE_SELECTION/overlays/<RUN_ID>/
#
# IMPORTANT:
#   - Run this after CELL 12.
#   - This cell uses RUN_ID and RUN_TIMESTAMP from CELL 10B.
#   - Tables are saved using save_run_table(), not save_state().
#   - pole_selection.parquet is the required saved table for downstream CELL 14.
#   - Candidate and failure tables are audit outputs.
#   - Overlay PNGs are optional review artifacts and must not be used as crop or
#     geometry inputs.
# =============================================================================

# =============================================================================
# A. SAFETY CHECKS + OUTPUT SETUP
# =============================================================================

# =============================================================================
# 13.1 REQUIRED SETUP CHECKS
# =============================================================================
# EXPLANATION:
# Fail early if required variables from earlier production cells are missing.
#
# Required earlier cells:
#   - CELL 3A  : os, pd, np, torch, cv2, plt, patches, Image
#   - CELL 3B  : pole thresholds and overlay constants
#   - CELL 9   : processor
#   - CELL 10  : Silver pole output folders
#   - CELL 10B : RUN_ID, RUN_TIMESTAMP, save_run_table, make_safe_path_part
#   - CELL 12  : run_images_df
# =============================================================================

required_cell13_globals = [
    # Core libraries.
    "os",
    "pd",
    "np",
    "torch",
    "cv2",
    "plt",
    "patches",
    "Image",

    # Production inputs.
    "run_images_df",
    "processor",
    "DEVICE",
    "RUN_ID",
    "RUN_TIMESTAMP",

    # Save helpers from CELL 10B.
    "save_run_table",
    "make_safe_path_part",

    # Output folders from CELL 10.
    "SILVER_POLE_SELECTION",
    "SILVER_POLE_SELECTION_OVERLAYS",

    # Pole prompt and filter constants from CELL 3B.
    "POLE_PROMPT_TEXT",
    "POLE_OVERLAY_MAX_WIDTH",
    "POLE_MIN_SCORE",
    "POLE_MIN_AREA_FRAC",
    "POLE_MIN_HEIGHT_FRAC",
    "POLE_MIN_ASPECT",
    "POLE_MAX_WIDTH_FRAC",
    "POLE_MAX_BOX_W_PX",
    "SHAFT_WIDTH_FRAC_THRESHOLD",
    "SHAFT_PENALTY_FACTOR",

    # Pole ranking weights from CELL 3B.
    "W_X_CENTER",
    "W_HEIGHT",
    "W_AREA",
    "W_CONF",
    "W_EDGE",

    # Pole overlay styling from CELL 3B.
    "POLE_SELECTED_MASK_RGB",
    "POLE_SELECTED_MASK_ALPHA",
    "POLE_SELECTED_BOX_COLOR",
    "POLE_SELECTED_BOX_LINEWIDTH",
    "POLE_SELECTED_TEXT_COLOR",
    "POLE_SELECTED_LABEL_FONTSIZE",
    "POLE_SELECTED_LABEL_BG_ALPHA",
    "POLE_SELECTED_LABEL_BBOX_PAD",
    "POLE_SELECTED_LABEL_Y_OFFSET",
    "NO_RELIABLE_POLE_LABEL_TEXT",
]

missing_cell13_globals = [
    name for name in required_cell13_globals
    if name not in globals()
]

if missing_cell13_globals:
    raise NameError(
        "CELL 13 requires earlier production setup cells to run successfully.\n"
        "Please run CELL 10B, CELL 11, and CELL 12 before CELL 13.\n"
        f"Missing globals: {missing_cell13_globals}"
    )

# =============================================================================
# 13.2 VALIDATE run_images_df
# =============================================================================
# EXPLANATION:
# run_images_df is the production image manifest created by CELL 12.
# =============================================================================

if not isinstance(run_images_df, pd.DataFrame):
    raise TypeError(
        "run_images_df exists but is not a pandas DataFrame."
    )

if run_images_df.empty:
    raise ValueError(
        "run_images_df is empty.\n"
        "Please check CELL 12."
    )

# =============================================================================
# 13.3 VALIDATE REQUIRED run_images_df COLUMNS
# =============================================================================
# EXPLANATION:
# These columns are required for image loading, output naming, lineage tracking,
# and run consistency.
# =============================================================================

required_run_images_cols = [
    "run_id",
    "run_timestamp",
    "processing_order",
    "image_id",
    "file_name",
    "image_path",
    "relative_image_path",
]

missing_run_images_cols = [
    col_name
    for col_name in required_run_images_cols
    if col_name not in run_images_df.columns
]

if missing_run_images_cols:
    raise ValueError(
        "run_images_df is missing required production columns.\n"
        f"Missing columns: {missing_run_images_cols}"
    )


# Validate that the manifest belongs to the active RUN_ID.
unique_run_ids = (
    run_images_df["run_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

if len(unique_run_ids) != 1:
    raise RuntimeError(
        "run_images_df must contain exactly one run_id.\n"
        f"Found run_id values: {unique_run_ids[:10]}"
    )

if str(unique_run_ids[0]) != str(RUN_ID):
    raise RuntimeError(
        "run_images_df run_id does not match active RUN_ID.\n"
        f"run_images_df run_id : {unique_run_ids[0]}\n"
        f"active RUN_ID        : {RUN_ID}\n"
        "Please rerun CELL 10B, CELL 11, CELL 12, and CELL 13 in order."
    )


# Validate image_id uniqueness before building mask lookup keys.
if run_images_df["image_id"].duplicated().any():
    duplicate_image_ids = (
        run_images_df
        .loc[run_images_df["image_id"].duplicated(), "image_id"]
        .tolist()
    )

    raise RuntimeError(
        "run_images_df contains duplicate image_id values.\n"
        f"Duplicate examples: {duplicate_image_ids[:10]}"
    )


# Validate that image paths still exist.
missing_input_images = (
    run_images_df
    .loc[
        ~run_images_df["image_path"].map(lambda path: isinstance(path, str) and os.path.exists(path)),
        "image_path",
    ]
    .tolist()
)

if missing_input_images:
    raise FileNotFoundError(
        "Some image_path values in run_images_df do not exist.\n"
        f"Missing examples: {missing_input_images[:10]}"
    )

# =============================================================================
# 13.4 VALIDATE POLE PROMPT AND CONFIG VALUES
# =============================================================================
# EXPLANATION:
# Check production pole-selection constants before running SAM3 across the full
# image batch.
# =============================================================================

if not isinstance(POLE_PROMPT_TEXT, (list, tuple)) or len(POLE_PROMPT_TEXT) == 0:
    raise ValueError(
        "POLE_PROMPT_TEXT must be a non-empty list or tuple."
    )

POLE_PROMPT_TEXT = [
    str(prompt).strip()
    for prompt in POLE_PROMPT_TEXT
    if str(prompt).strip()
]

if len(POLE_PROMPT_TEXT) == 0:
    raise ValueError(
        "POLE_PROMPT_TEXT contains no usable prompt strings."
    )


numeric_pole_config = {
    "POLE_OVERLAY_MAX_WIDTH": POLE_OVERLAY_MAX_WIDTH,
    "POLE_MIN_SCORE": POLE_MIN_SCORE,
    "POLE_MIN_AREA_FRAC": POLE_MIN_AREA_FRAC,
    "POLE_MIN_HEIGHT_FRAC": POLE_MIN_HEIGHT_FRAC,
    "POLE_MIN_ASPECT": POLE_MIN_ASPECT,
    "POLE_MAX_WIDTH_FRAC": POLE_MAX_WIDTH_FRAC,
    "POLE_MAX_BOX_W_PX": POLE_MAX_BOX_W_PX,
    "SHAFT_WIDTH_FRAC_THRESHOLD": SHAFT_WIDTH_FRAC_THRESHOLD,
    "SHAFT_PENALTY_FACTOR": SHAFT_PENALTY_FACTOR,
    "W_X_CENTER": W_X_CENTER,
    "W_HEIGHT": W_HEIGHT,
    "W_AREA": W_AREA,
    "W_CONF": W_CONF,
    "W_EDGE": W_EDGE,
}

invalid_numeric_config = {
    name: value
    for name, value in numeric_pole_config.items()
    if not pd.notna(value)
}

if invalid_numeric_config:
    raise ValueError(
        "Some numeric pole-selection config values are missing or NaN.\n"
        f"Invalid values: {invalid_numeric_config}"
    )

if float(POLE_OVERLAY_MAX_WIDTH) <= 0:
    raise ValueError(
        "POLE_OVERLAY_MAX_WIDTH must be positive."
    )

# =============================================================================
# 13.5 DEFINE PRODUCTION SAVE FLAGS
# =============================================================================
# EXPLANATION:
# These flags control Cell 13 output behaviour.
#
# Defaults:
#   - save pole candidates table
#   - save selected-pole table
#   - save pole failure table
#   - save one QA overlay image per input image
# =============================================================================

SAVE_POLE_CANDIDATES_TABLE = bool(
    globals().get(
        "SAVE_POLE_CANDIDATES_TABLE",
        True,
    )
)

SAVE_POLE_SELECTION_TABLE = bool(
    globals().get(
        "SAVE_POLE_SELECTION_TABLE",
        True,
    )
)

SAVE_POLE_FAILURES_TABLE = bool(
    globals().get(
        "SAVE_POLE_FAILURES_TABLE",
        True,
    )
)

SAVE_POLE_SELECTION_OVERLAYS = bool(
    globals().get(
        "SAVE_POLE_SELECTION_OVERLAYS",
        True,
    )
)

# =============================================================================
# 13.6 DEFINE RUN-SCOPED OUTPUT PATHS
# =============================================================================
# EXPLANATION:
# Pole outputs are written under RUN_ID folders so production runs do not
# overwrite each other.
#
# Tables:
#   SILVER_POLE_SELECTION/tables/<RUN_ID>/
#
# Overlays:
#   SILVER_POLE_SELECTION/overlays/<RUN_ID>/
# =============================================================================

SILVER_POLE_SELECTION_TABLES_ROOT = os.path.join(
    SILVER_POLE_SELECTION,
    "tables",
)

RUN_SILVER_POLE_SELECTION_TABLES_DIR = os.path.join(
    SILVER_POLE_SELECTION_TABLES_ROOT,
    RUN_ID,
)

RUN_SILVER_POLE_SELECTION_OVERLAYS_DIR = os.path.join(
    SILVER_POLE_SELECTION_OVERLAYS,
    RUN_ID,
)


# =============================================================================
# 13.7 CREATE OUTPUT DIRECTORIES
# =============================================================================
# EXPLANATION:
# Create run-scoped output folders before helper functions begin writing files.
# =============================================================================

if SAVE_POLE_CANDIDATES_TABLE or SAVE_POLE_SELECTION_TABLE or SAVE_POLE_FAILURES_TABLE:
    os.makedirs(
        RUN_SILVER_POLE_SELECTION_TABLES_DIR,
        exist_ok=True,
    )

if SAVE_POLE_SELECTION_OVERLAYS:
    os.makedirs(
        RUN_SILVER_POLE_SELECTION_OVERLAYS_DIR,
        exist_ok=True,
    )

# =============================================================================
# 13.8 INITIALISE PRODUCTION ACCUMULATORS
# =============================================================================
# EXPLANATION:
# These accumulators are filled by the later pole-selection loop.
#
# candidate_frames:
#   Stores per-image candidate DataFrames before final concatenation.
#
# selection_rows:
#   Stores one selected-pole output row per image.
#
# failure_rows:
#   Stores image-level failures if the production loop catches an exception.
#
# pole_mask_lookup:
#   Stores full-resolution selected-pole masks for downstream ROI generation.
# =============================================================================

candidate_frames = []
selection_rows = []
failure_rows = []
pole_mask_lookup = {}

# =============================================================================
# 13.9 SET PROCESSOR DEVICE
# =============================================================================
# EXPLANATION:
# Keep processor device aligned with the configured production DEVICE.
# =============================================================================

if hasattr(processor, "device"):
    processor.device = DEVICE

# =============================================================================
# 13.10 SETUP SUMMARY
# =============================================================================
# EXPLANATION:
# Print a compact setup summary when PRINT_CONFIG_SUMMARY is enabled.
# =============================================================================

if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print("CELL 13 pole-selection setup ready.\n")

    print("=" * 90)
    print("POLE SELECTION RUN CONFIG")
    print("=" * 90)
    print(f"RUN_ID                              : {RUN_ID}")
    print(f"RUN_TIMESTAMP                       : {RUN_TIMESTAMP}")
    print(f"Input image count                   : {len(run_images_df)}")
    print(f"POLE_PROMPT_TEXT                    : {POLE_PROMPT_TEXT}")

    print("\n" + "=" * 90)
    print("POLE OUTPUT PATHS")
    print("=" * 90)
    print(f"RUN_SILVER_POLE_SELECTION_TABLES_DIR   : {RUN_SILVER_POLE_SELECTION_TABLES_DIR}")
    print(f"RUN_SILVER_POLE_SELECTION_OVERLAYS_DIR : {RUN_SILVER_POLE_SELECTION_OVERLAYS_DIR}")

    print("\n" + "=" * 90)
    print("POLE SAVE FLAGS")
    print("=" * 90)
    print(f"SAVE_POLE_CANDIDATES_TABLE          : {SAVE_POLE_CANDIDATES_TABLE}")
    print(f"SAVE_POLE_SELECTION_TABLE           : {SAVE_POLE_SELECTION_TABLE}")
    print(f"SAVE_POLE_FAILURES_TABLE            : {SAVE_POLE_FAILURES_TABLE}")
    print(f"SAVE_POLE_SELECTION_OVERLAYS        : {SAVE_POLE_SELECTION_OVERLAYS}")
    
    
# =============================================================================
# B. HELPER FUNCTIONS
# =============================================================================

# =============================================================================
# 13.11 ROW VALUE HELPER
# =============================================================================
# EXPLANATION:
# Safely read values from a pandas Series or dictionary-like row.
#
# This keeps helper functions robust when they receive either:
#   - a row from run_images_df
#   - a plain dictionary
#   - None
# =============================================================================

def _cell13_get_row_value(row, column_name, default=None):
    """
    Safely get a value from a row-like object.

    Args:
        row:
            pandas Series, dictionary-like object, or None.

        column_name:
            Column/key name to read.

        default:
            Value returned when the row or column/key is missing.

    Returns:
        Any:
            Row value or default.
    """
    if row is None:
        return default

    if isinstance(row, pd.Series):
        if column_name in row.index:
            return row[column_name]

        return default

    if isinstance(row, dict):
        return row.get(column_name, default)

    try:
        return row[column_name]
    except Exception:
        return default


# =============================================================================
# 13.12 LINEAGE FIELD HELPER
# =============================================================================
# EXPLANATION:
# Build common production lineage fields that should appear in output tables.
#
# These fields make it easier to join pole outputs back to:
#   - run_images_df
#   - images_df
#   - later ROI and crossarm tables
# =============================================================================

def _cell13_build_lineage_fields(source_row=None):
    """
    Build common production lineage fields from the current image row.

    Args:
        source_row:
            Row from run_images_df.

    Returns:
        dict:
            Common lineage fields for output rows.
    """
    lineage = {
        "run_id": RUN_ID,
        "run_timestamp": RUN_TIMESTAMP,
    }

    optional_cols = [
        "processing_order",
        "image_id",
        "file_name",
        "stem",
        "ext",
        "ext_lower",
        "relative_image_path",
        "image_path",
        "source_image_path",
        "source_layer",
        "source_root",
        "bronze_root",
    ]

    for col_name in optional_cols:
        value = _cell13_get_row_value(
            source_row,
            col_name,
            default=None,
        )

        if value is not None:
            lineage[col_name] = value

    return lineage


# =============================================================================
# 13.13 NUMPY CONVERSION HELPER
# =============================================================================
# EXPLANATION:
# SAM3 outputs may be torch tensors, numpy arrays, or array-like objects.
# This helper normalises them into CPU numpy arrays where possible.
# =============================================================================

def _to_numpy_safe(x):
    """
    Convert tensors / arrays safely to numpy.

    Args:
        x:
            Tensor, numpy array, array-like object, or None.

    Returns:
        numpy.ndarray or None:
            Converted numpy array, or None when conversion is not possible.
    """
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    try:
        return np.asarray(x)
    except Exception:
        return None


# =============================================================================
# 13.14 DETECTION KEY HELPER
# =============================================================================
# EXPLANATION:
# Full-resolution pole masks are stored in pole_mask_lookup using a stable key.
# Downstream cells use the same key pattern to retrieve selected-pole masks.
# =============================================================================

def _candidate_key(image_id, prompt, det_idx):
    """
    Build a stable lookup key for one detection.

    Args:
        image_id:
            Stable image identifier.

        prompt:
            Prompt string used for the detection.

        det_idx:
            Detection index within the prompt run.

    Returns:
        tuple:
            Key suitable for mask lookup dictionaries.
    """
    return (
        str(image_id),
        str(prompt),
        int(det_idx),
    )


# =============================================================================
# 13.15 DETECTION COUNT HELPER
# =============================================================================
# EXPLANATION:
# Infer the number of detections returned by SAM3 from boxes, scores, or masks.
# This keeps the downstream normalisation code stable even when one output type
# is missing.
# =============================================================================

def _infer_num_detections(raw_boxes, raw_scores, raw_masks):
    """
    Infer how many detections are present from boxes, scores, or masks.

    Args:
        raw_boxes:
            Raw boxes object from processor state.

        raw_scores:
            Raw scores object from processor state.

        raw_masks:
            Raw masks object from processor state.

    Returns:
        int:
            Number of detections inferred from available outputs.
    """
    boxes_arr = _to_numpy_safe(raw_boxes)
    scores_arr = _to_numpy_safe(raw_scores)

    if boxes_arr is not None:
        if boxes_arr.ndim == 2 and boxes_arr.shape[1] == 4:
            return int(boxes_arr.shape[0])

        if boxes_arr.ndim == 1 and boxes_arr.size == 4:
            return 1

    if scores_arr is not None:
        scores_arr = scores_arr.reshape(-1)

        if scores_arr.size > 0:
            return int(scores_arr.size)

    if raw_masks is not None:
        if isinstance(raw_masks, (list, tuple)):
            return int(len(raw_masks))

        masks_arr = _to_numpy_safe(raw_masks)

        if masks_arr is not None:
            if masks_arr.ndim == 2:
                return 1

            if masks_arr.ndim >= 3:
                return int(masks_arr.shape[0])

    return 0


# =============================================================================
# 13.16 BOX NORMALISATION HELPER
# =============================================================================
# EXPLANATION:
# Convert SAM3 raw boxes into a stable N x 4 float32 array.
# =============================================================================

def _normalize_boxes_local(raw_boxes, num_detections):
    """
    Normalize raw boxes into a stable (N, 4) float32 array.

    Args:
        raw_boxes:
            Raw boxes object from processor state.

        num_detections:
            Expected number of detections.

    Returns:
        numpy.ndarray:
            Float32 array of shape (num_detections, 4).
    """
    if num_detections <= 0:
        return np.zeros(
            (0, 4),
            dtype=np.float32,
        )

    if raw_boxes is None:
        return np.zeros(
            (num_detections, 4),
            dtype=np.float32,
        )

    try:
        arr = _to_numpy_safe(raw_boxes).astype(np.float32)
    except Exception:
        return np.zeros(
            (num_detections, 4),
            dtype=np.float32,
        )

    if arr.ndim == 1 and arr.size == 4:
        arr = arr.reshape(1, 4)

    if arr.ndim != 2 or arr.shape[1] != 4:
        return np.zeros(
            (num_detections, 4),
            dtype=np.float32,
        )

    if arr.shape[0] < num_detections:
        pad = np.zeros(
            (num_detections - arr.shape[0], 4),
            dtype=np.float32,
        )

        arr = np.vstack(
            [
                arr,
                pad,
            ]
        )

    return arr[:num_detections]


# =============================================================================
# 13.17 SCORE NORMALISATION HELPER
# =============================================================================
# EXPLANATION:
# Convert SAM3 raw scores into a stable one-dimensional float32 array.
# =============================================================================

def _normalize_scores_local(raw_scores, num_detections):
    """
    Normalize raw scores into a stable (N,) float32 array.

    Args:
        raw_scores:
            Raw scores object from processor state.

        num_detections:
            Expected number of detections.

    Returns:
        numpy.ndarray:
            Float32 array of shape (num_detections,).
    """
    if num_detections <= 0:
        return np.zeros(
            (0,),
            dtype=np.float32,
        )

    if raw_scores is None:
        return np.zeros(
            (num_detections,),
            dtype=np.float32,
        )

    try:
        arr = _to_numpy_safe(raw_scores).astype(np.float32).reshape(-1)
    except Exception:
        return np.zeros(
            (num_detections,),
            dtype=np.float32,
        )

    if arr.size < num_detections:
        pad = np.zeros(
            (num_detections - arr.size,),
            dtype=np.float32,
        )

        arr = np.concatenate(
            [
                arr,
                pad,
            ]
        )

    return arr[:num_detections]


# =============================================================================
# 13.18 MASK NORMALISATION HELPER
# =============================================================================
# EXPLANATION:
# Convert SAM3 raw masks into a list of full-resolution boolean masks.
#
# IMPORTANT:
#   Masks are kept in memory for downstream geometry. This helper does not save
#   masks to disk.
# =============================================================================

def _normalize_masks_local(raw_masks, num_detections, image_h, image_w):
    """
    Normalize raw masks into a list of 2D boolean masks.

    Args:
        raw_masks:
            Raw masks object from processor state.

        num_detections:
            Expected number of detections.

        image_h:
            Source image height.

        image_w:
            Source image width.

    Returns:
        list:
            List of length num_detections containing 2D boolean masks or None.
    """
    if num_detections <= 0:
        return []

    if raw_masks is None:
        return [None] * num_detections

    if isinstance(raw_masks, (list, tuple)):
        mask_items = list(raw_masks)
    else:
        arr = _to_numpy_safe(raw_masks)

        if arr is None:
            return [None] * num_detections

        if arr.ndim == 2:
            mask_items = [arr]

        elif arr.ndim == 3:
            mask_items = [
                arr[i]
                for i in range(
                    min(
                        arr.shape[0],
                        num_detections,
                    )
                )
            ]

        elif arr.ndim == 4:
            mask_items = [
                arr[i]
                for i in range(
                    min(
                        arr.shape[0],
                        num_detections,
                    )
                )
            ]

        else:
            return [None] * num_detections

    norm_masks = []

    for det_idx in range(num_detections):
        if det_idx >= len(mask_items):
            norm_masks.append(None)
            continue

        mask_arr = _to_numpy_safe(mask_items[det_idx])

        if mask_arr is None:
            norm_masks.append(None)
            continue

        mask_arr = np.squeeze(mask_arr)

        if mask_arr.ndim != 2:
            norm_masks.append(None)
            continue

        if mask_arr.shape != (image_h, image_w):
            norm_masks.append(None)
            continue

        mask_bool = (
            mask_arr.copy()
            if mask_arr.dtype == bool
            else (mask_arr > 0)
        )

        if mask_bool.sum() == 0:
            norm_masks.append(None)
        else:
            norm_masks.append(mask_bool)

    if len(norm_masks) < num_detections:
        norm_masks.extend(
            [None] * (num_detections - len(norm_masks))
        )

    return norm_masks[:num_detections]


# =============================================================================
# 13.19 BOX CLIPPING HELPER
# =============================================================================
# EXPLANATION:
# Clip box coordinates to the image boundary and ensure x1 <= x2 and y1 <= y2.
# =============================================================================

def _clip_box_to_image(x1, y1, x2, y2, image_w, image_h):
    """
    Clip and sort box coordinates to stay inside image bounds.

    Args:
        x1, y1, x2, y2:
            Raw box coordinates.

        image_w:
            Source image width.

        image_h:
            Source image height.

    Returns:
        tuple:
            Clipped and ordered coordinates.
    """
    def _safe_float(value, fallback=0.0):
        try:
            value_float = float(value)

            if not np.isfinite(value_float):
                return fallback

            return value_float

        except Exception:
            return fallback

    x1 = _safe_float(x1)
    y1 = _safe_float(y1)
    x2 = _safe_float(x2)
    y2 = _safe_float(y2)

    x1 = float(np.clip(x1, 0, max(image_w - 1, 0)))
    y1 = float(np.clip(y1, 0, max(image_h - 1, 0)))
    x2 = float(np.clip(x2, 0, max(image_w - 1, 0)))
    y2 = float(np.clip(y2, 0, max(image_h - 1, 0)))

    x1, x2 = sorted(
        [
            x1,
            x2,
        ]
    )

    y1, y2 = sorted(
        [
            y1,
            y2,
        ]
    )

    return x1, y1, x2, y2


# =============================================================================
# 13.20 POLE OVERLAY LABEL HELPER
# =============================================================================
# EXPLANATION:
# Build the text label rendered on the saved pole QA overlay.
# =============================================================================

def _build_pole_overlay_label(prompt, score, final_score):
    """
    Build the label text rendered on the saved QA overlay.

    Args:
        prompt:
            Selected prompt string.

        score:
            Raw SAM score.

        final_score:
            Final pole ranking score.

    Returns:
        str:
            Human-readable label string.
    """
    label_bits = ["POLE"]

    if prompt is not None and str(prompt).strip():
        label_bits.append(
            str(prompt).strip()
        )

    if pd.notna(score):
        label_bits.append(
            f"score={float(score):.3f}"
        )

    if pd.notna(final_score):
        label_bits.append(
            f"final={float(final_score):.3f}"
        )

    return " | ".join(label_bits)


# =============================================================================
# 13.21 EMPTY OVERLAY METADATA HELPER
# =============================================================================
# EXPLANATION:
# Return a consistent overlay metadata object when overlay saving is disabled or
# skipped.
# =============================================================================

def _empty_pole_overlay_meta():
    """
    Build empty overlay metadata.

    Returns:
        dict:
            Empty overlay metadata with stable keys.
    """
    return {
        "overlay_image_path": None,
        "overlay_resize_ratio": np.nan,
        "overlay_image_w": np.nan,
        "overlay_image_h": np.nan,
    }


# =============================================================================
# 13.22 POLE OVERLAY OUTPUT PATH HELPER
# =============================================================================
# EXPLANATION:
# Build a run-scoped output path for the saved pole QA overlay image.
#
# Output pattern:
#   SILVER_POLE_SELECTION/overlays/<RUN_ID>/<relative_folder>/
#       <safe_stem>__<safe_image_id>__pole_overlay.png
# =============================================================================

def _build_overlay_output_path(row, image_id):
    """
    Build the output path for the saved pole QA overlay image.

    Args:
        row:
            Input row from run_images_df.

        image_id:
            Stable image identifier.

    Returns:
        str or None:
            Absolute output path for the saved overlay PNG, or None when overlay
            saving is disabled.
    """
    if not bool(globals().get("SAVE_POLE_SELECTION_OVERLAYS", True)):
        return None

    relative_image_path = _cell13_get_row_value(
        row,
        "relative_image_path",
        default=None,
    )

    if not isinstance(relative_image_path, str) or len(relative_image_path.strip()) == 0:
        relative_image_path = _cell13_get_row_value(
            row,
            "file_name",
            default="image",
        )

    relative_text = str(relative_image_path).replace("\\", "/")

    relative_dir = os.path.dirname(relative_text)
    base_stem = os.path.splitext(
        os.path.basename(relative_text)
    )[0]

    safe_dir_parts = []

    if relative_dir not in ["", "."]:
        for part in relative_dir.split("/"):
            if part in ["", ".", ".."]:
                continue

            safe_dir_parts.append(
                make_safe_path_part(
                    part,
                    fallback="folder",
                )
            )

    safe_base_stem = make_safe_path_part(
        base_stem,
        fallback="image",
    )

    safe_image_id = make_safe_path_part(
        image_id,
        fallback="unknown_image",
    )

    target_dir = os.path.join(
        RUN_SILVER_POLE_SELECTION_OVERLAYS_DIR,
        *safe_dir_parts,
    )

    os.makedirs(
        target_dir,
        exist_ok=True,
    )

    overlay_file_name = (
        f"{safe_base_stem}__{safe_image_id}__pole_overlay.png"
    )

    return os.path.join(
        target_dir,
        overlay_file_name,
    )


# =============================================================================
# 13.23 POLE OVERLAY SAVE HELPER
# =============================================================================
# EXPLANATION:
# Save a smaller human-QA pole overlay image.
#
# IMPORTANT:
#   - This function never calls plt.show().
#   - This function closes the figure after saving.
#   - This function returns empty metadata if overlay saving is disabled.
#   - The saved PNG is a review artifact only, not a geometry source.
#   - The fig.savefig(...) call intentionally matches the original working
#     CELL 13 behaviour: no pad_inches=0 and no bbox_inches="tight".
# =============================================================================

def _save_selected_overlay_matplotlib(
    image_rgb,
    output_path,
    selected_row=None,
    mask_2d=None,
    label_text=None,
):
    """
    Save a smaller human-QA pole overlay image.

    Args:
        image_rgb:
            Full-resolution RGB image as a numpy array.

        output_path:
            Path where the QA overlay PNG should be saved.

        selected_row:
            Selected-pole row with full-resolution coordinates, or None.

        mask_2d:
            Full-resolution boolean mask for the selected pole, or None.

        label_text:
            Text to render on the overlay.

    Returns:
        dict:
            Overlay metadata:
              - overlay_image_path
              - overlay_resize_ratio
              - overlay_image_w
              - overlay_image_h
    """
    if not bool(globals().get("SAVE_POLE_SELECTION_OVERLAYS", True)):
        return _empty_pole_overlay_meta()

    if output_path is None or len(str(output_path).strip()) == 0:
        return _empty_pole_overlay_meta()

    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3:
        raise ValueError(
            "image_rgb must be a 3D numpy array."
        )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    full_h, full_w = image_rgb.shape[:2]

    overlay_resize_ratio = min(
        1.0,
        float(POLE_OVERLAY_MAX_WIDTH) / float(max(full_w, 1)),
    )

    if overlay_resize_ratio < 1.0:
        vis_w = int(
            round(
                full_w * overlay_resize_ratio,
            )
        )

        vis_h = int(
            round(
                full_h * overlay_resize_ratio,
            )
        )

        image_rgb_vis = cv2.resize(
            image_rgb,
            (vis_w, vis_h),
            interpolation=cv2.INTER_AREA,
        )

        if selected_row is not None:
            selected_row_vis = selected_row.copy()

            for col in [
                "x1",
                "y1",
                "x2",
                "y2",
            ]:
                selected_row_vis[col] = (
                    float(selected_row_vis[col]) * overlay_resize_ratio
                )
        else:
            selected_row_vis = None

        if isinstance(mask_2d, np.ndarray) and mask_2d.ndim == 2:
            mask_2d_vis = cv2.resize(
                mask_2d.astype(np.uint8),
                (vis_w, vis_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            mask_2d_vis = None

    else:
        image_rgb_vis = image_rgb
        selected_row_vis = (
            selected_row.copy()
            if selected_row is not None
            else None
        )
        mask_2d_vis = mask_2d
        vis_h, vis_w = image_rgb_vis.shape[:2]

    dpi = 100
    fig = None

    try:
        fig = plt.figure(
            figsize=(
                vis_w / dpi,
                vis_h / dpi,
            ),
            dpi=dpi,
        )

        ax = fig.add_axes(
            [
                0.0,
                0.0,
                1.0,
                1.0,
            ]
        )

        ax.imshow(image_rgb_vis)
        ax.axis("off")

        style_scale = float(
            np.clip(
                vis_w / 1200.0,
                1.0,
                1.5,
            )
        )

        label_fontsize = max(
            POLE_SELECTED_LABEL_FONTSIZE * style_scale,
            12.0,
        )

        box_linewidth = max(
            POLE_SELECTED_BOX_LINEWIDTH * style_scale,
            1.5,
        )

        label_pad = max(
            POLE_SELECTED_LABEL_BBOX_PAD * style_scale,
            2.5,
        )

        label_y_offset = max(
            POLE_SELECTED_LABEL_Y_OFFSET * style_scale,
            label_fontsize * 1.2,
        )

        if (
            isinstance(mask_2d_vis, np.ndarray)
            and mask_2d_vis.ndim == 2
            and mask_2d_vis.shape == image_rgb_vis.shape[:2]
            and mask_2d_vis.sum() > 0
        ):
            overlay = np.zeros(
                (
                    mask_2d_vis.shape[0],
                    mask_2d_vis.shape[1],
                    4,
                ),
                dtype=np.float32,
            )

            overlay[..., 0] = POLE_SELECTED_MASK_RGB[0]
            overlay[..., 1] = POLE_SELECTED_MASK_RGB[1]
            overlay[..., 2] = POLE_SELECTED_MASK_RGB[2]
            overlay[..., 3] = (
                mask_2d_vis.astype(np.float32)
                * POLE_SELECTED_MASK_ALPHA
            )

            ax.imshow(overlay)

        if selected_row_vis is not None:
            x1 = float(selected_row_vis["x1"])
            y1 = float(selected_row_vis["y1"])
            x2 = float(selected_row_vis["x2"])
            y2 = float(selected_row_vis["y2"])

            rect = patches.Rectangle(
                (x1, y1),
                max(
                    0.0,
                    x2 - x1,
                ),
                max(
                    0.0,
                    y2 - y1,
                ),
                linewidth=box_linewidth,
                edgecolor=POLE_SELECTED_BOX_COLOR,
                facecolor="none",
            )

            ax.add_patch(rect)

            if label_text is not None and str(label_text).strip():
                ax.text(
                    x1,
                    max(
                        y1 - label_y_offset,
                        8 * style_scale,
                    ),
                    str(label_text).strip(),
                    fontsize=label_fontsize,
                    color=POLE_SELECTED_TEXT_COLOR,
                    bbox=dict(
                        facecolor=POLE_SELECTED_BOX_COLOR,
                        alpha=POLE_SELECTED_LABEL_BG_ALPHA,
                        edgecolor="none",
                        pad=label_pad,
                    ),
                )

        else:
            if label_text is not None and str(label_text).strip():
                ax.text(
                    8 * style_scale,
                    18 * style_scale,
                    str(label_text).strip(),
                    fontsize=label_fontsize,
                    color=POLE_SELECTED_TEXT_COLOR,
                    bbox=dict(
                        facecolor=POLE_SELECTED_BOX_COLOR,
                        alpha=POLE_SELECTED_LABEL_BG_ALPHA,
                        edgecolor="none",
                        pad=label_pad,
                    ),
                )

        # ---------------------------------------------------------------------
        # Save using the same savefig behaviour as the original working CELL 13.
        #
        # IMPORTANT:
        #   Do not add pad_inches=0 or bbox_inches="tight" here because that
        #   changes the visual output compared with the original working
        #   overlay-saving logic.
        # ---------------------------------------------------------------------
        fig.savefig(
            output_path,
            dpi=dpi,
            edgecolor="none",
        )

    finally:
        if fig is not None:
            plt.close(fig)

    return {
        "overlay_image_path": output_path,
        "overlay_resize_ratio": float(overlay_resize_ratio),
        "overlay_image_w": int(vis_w),
        "overlay_image_h": int(vis_h),
    }


# =============================================================================
# 13.24 NO-RELIABLE-POLE ROW HELPER
# =============================================================================
# EXPLANATION:
# Build one consistent output row when an image has no reliable selected pole.
# =============================================================================

def _make_no_reliable_pole_row(
    image_id,
    file_name,
    image_path,
    image_w,
    image_h,
    n_raw_candidates,
    n_kept_candidates,
    fallback_triggered,
    overlay_label_text,
    overlay_meta=None,
    source_row=None,
):
    """
    Build one consistent no-reliable-pole output row.

    Args:
        image_id:
            Stable image identifier.

        file_name:
            Source image filename.

        image_path:
            Full path to the source image.

        image_w:
            Image width.

        image_h:
            Image height.

        n_raw_candidates:
            Number of raw pole candidates found.

        n_kept_candidates:
            Number of candidates kept after prefiltering.

        fallback_triggered:
            Whether fallback scoring was triggered.

        overlay_label_text:
            Label text intended for the QA overlay.

        overlay_meta:
            Overlay metadata dictionary.

        source_row:
            Original row from run_images_df.

    Returns:
        dict:
            One pole_selection_df-compatible row.
    """
    if overlay_meta is None:
        overlay_meta = _empty_pole_overlay_meta()

    row_out = _cell13_build_lineage_fields(
        source_row=source_row,
    )

    row_out.update({
        "image_id": image_id,
        "file_name": file_name,
        "image_path": image_path,
        "image_w": int(image_w),
        "image_h": int(image_h),
        "selection_status": "no_reliable_pole_found",
        "selection_mode": "no_reliable_pole_found",
        "is_selected_pole": False,
        "n_raw_candidates": int(n_raw_candidates),
        "n_kept_candidates": int(n_kept_candidates),
        "prompt": None,
        "det_idx": None,
        "score": np.nan,
        "x1": np.nan,
        "y1": np.nan,
        "x2": np.nan,
        "y2": np.nan,
        "box_w": np.nan,
        "box_h": np.nan,
        "box_area": np.nan,
        "pole_cx": np.nan,
        "pole_cy": np.nan,
        "x_center_dist_norm": np.nan,
        "width_frac": np.nan,
        "height_frac": np.nan,
        "area_frac": np.nan,
        "aspect_ratio": np.nan,
        "shaft_penalty": np.nan,
        "final_score": np.nan,
        "has_mask": False,
        "fallback_triggered": bool(fallback_triggered),
        "overlay_label_text": overlay_label_text,
        "overlay_image_path": overlay_meta.get("overlay_image_path", None),
        "overlay_resize_ratio": overlay_meta.get("overlay_resize_ratio", np.nan),
        "overlay_image_w": overlay_meta.get("overlay_image_w", np.nan),
        "overlay_image_h": overlay_meta.get("overlay_image_h", np.nan),
    })

    return row_out


# =============================================================================
# 13.25 POLE FAILURE ROW HELPER
# =============================================================================
# EXPLANATION:
# Build one failure row when a single image fails during pole selection.
#
# This enables the production loop to continue processing the remaining images
# instead of failing the whole batch.
# =============================================================================

def _make_pole_failure_row(source_row, error, stage="pole_selection"):
    """
    Build one pole failure row.

    Args:
        source_row:
            Input row from run_images_df.

        error:
            Exception raised while processing the image.

        stage:
            Logical processing stage where the error occurred.

    Returns:
        dict:
            One failure row.
    """
    row_out = _cell13_build_lineage_fields(
        source_row=source_row,
    )

    row_out.update({
        "failure_stage": str(stage),
        "error_type": type(error).__name__,
        "error_message": str(error),
    })

    if "traceback" in globals():
        try:
            row_out["error_traceback"] = traceback.format_exc()
        except Exception:
            row_out["error_traceback"] = None
    else:
        row_out["error_traceback"] = None

    return row_out


# =============================================================================
# C. PRODUCTION POLE SELECTION LOOP
# =============================================================================

# =============================================================================
# 13.26 RUN POLE SELECTION ACROSS ALL IMAGES
# =============================================================================
# EXPLANATION:
# This section runs SAM3 pole detection across every image in run_images_df.
#
# Production behaviour:
#   - one image failure does not stop the full batch
#   - failures are captured in failure_rows
#   - candidate rows are stamped with run/image lineage
#   - selected-pole rows are stamped with run/image lineage
#   - selected-pole masks remain full resolution in pole_mask_lookup
#   - overlay saving is handled through the 13B helper and respects
#     SAVE_POLE_SELECTION_OVERLAYS
#
# IMPORTANT:
#   This section only fills the accumulators:
#     - candidate_frames
#     - selection_rows
#     - failure_rows
#     - pole_mask_lookup
#
#   Final DataFrame assembly happens in Piece 13D.
# =============================================================================

for row_idx in range(len(run_images_df)):
    row = run_images_df.iloc[row_idx]

    image_id_for_failure_cleanup = _cell13_get_row_value(
        row,
        "image_id",
        default=None,
    )

    try:
        # ---------------------------------------------------------------------
        # 13.26.1 Read core image lineage values
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # These values come from run_images_df created in CELL 12.
        # ---------------------------------------------------------------------
        source_lineage = _cell13_build_lineage_fields(
            source_row=row,
        )

        image_id = row["image_id"]
        file_name = row["file_name"]
        image_path = row["image_path"]

        if not isinstance(image_path, str) or len(image_path.strip()) == 0:
            raise ValueError(
                f"Invalid image_path for image_id={image_id}: {image_path}"
            )

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image file not found for image_id={image_id}: {image_path}"
            )

        # ---------------------------------------------------------------------
        # 13.26.2 Load full-resolution image
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # Keep the original image resolution for pole boxes, masks, and
        # downstream ROI creation.
        # ---------------------------------------------------------------------
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                image = img.convert("RGB")
            else:
                image = img.copy()

            image.load()

        image_rgb = np.array(image)
        image_w, image_h = image.size
        image_cx = image_w / 2.0

        # ---------------------------------------------------------------------
        # 13.26.3 Initialise SAM3 image state
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # Use one fresh SAM3 image state per source image. Prompt state is reset
        # before each prompt.
        # ---------------------------------------------------------------------
        state = {}
        state = processor.set_image(
            image,
            state=state,
        )

        raw_rows = []

        # ---------------------------------------------------------------------
        # 13.26.4 Run all configured pole prompts
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # Each prompt produces candidate boxes/scores/masks. Candidate masks are
        # kept in memory so the final selected-pole mask can be retained.
        # ---------------------------------------------------------------------
        for prompt in POLE_PROMPT_TEXT:
            reset_result = processor.reset_all_prompts(state)

            if reset_result is not None:
                state = reset_result

            state = processor.set_text_prompt(
                prompt,
                state,
            )

            raw_boxes = state.get("boxes", None)
            raw_scores = state.get("scores", None)
            raw_masks = state.get("masks", None)

            num_detections = _infer_num_detections(
                raw_boxes,
                raw_scores,
                raw_masks,
            )

            boxes = _normalize_boxes_local(
                raw_boxes,
                num_detections,
            )

            scores = _normalize_scores_local(
                raw_scores,
                num_detections,
            )

            masks_2d = _normalize_masks_local(
                raw_masks,
                num_detections,
                image_h,
                image_w,
            )

            for det_idx in range(num_detections):
                x1, y1, x2, y2 = [
                    float(value)
                    for value in boxes[det_idx]
                ]

                x1, y1, x2, y2 = _clip_box_to_image(
                    x1,
                    y1,
                    x2,
                    y2,
                    image_w,
                    image_h,
                )

                box_w = max(
                    1.0,
                    x2 - x1,
                )

                box_h = max(
                    1.0,
                    y2 - y1,
                )

                box_area = box_w * box_h

                mask_2d = (
                    masks_2d[det_idx]
                    if det_idx < len(masks_2d)
                    else None
                )

                has_mask = (
                    isinstance(mask_2d, np.ndarray)
                    and mask_2d.ndim == 2
                    and mask_2d.sum() > 0
                )

                key = _candidate_key(
                    image_id,
                    prompt,
                    det_idx,
                )

                if has_mask:
                    pole_mask_lookup[key] = mask_2d

                raw_row = source_lineage.copy()

                raw_row.update({
                    "image_id": image_id,
                    "file_name": file_name,
                    "image_path": image_path,
                    "image_w": int(image_w),
                    "image_h": int(image_h),
                    "prompt": prompt,
                    "det_idx": int(det_idx),
                    "score": float(scores[det_idx]),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "box_w": box_w,
                    "box_h": box_h,
                    "box_area": box_area,
                    "has_mask": bool(has_mask),
                })

                raw_rows.append(raw_row)

        raw_df = pd.DataFrame(raw_rows)
        n_raw_candidates = int(len(raw_df))

        # ---------------------------------------------------------------------
        # 13.26.5 No-detection case
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # If SAM3 returns no pole candidates for this image, emit one
        # no-reliable-pole row and continue to the next image.
        # ---------------------------------------------------------------------
        if raw_df.empty:
            overlay_label_text = NO_RELIABLE_POLE_LABEL_TEXT

            overlay_output_path = _build_overlay_output_path(
                row=row,
                image_id=image_id,
            )

            overlay_meta = _save_selected_overlay_matplotlib(
                image_rgb=image_rgb,
                output_path=overlay_output_path,
                selected_row=None,
                mask_2d=None,
                label_text=overlay_label_text,
            )

            selection_rows.append(
                _make_no_reliable_pole_row(
                    image_id=image_id,
                    file_name=file_name,
                    image_path=image_path,
                    image_w=image_w,
                    image_h=image_h,
                    n_raw_candidates=0,
                    n_kept_candidates=0,
                    fallback_triggered=False,
                    overlay_label_text=overlay_label_text,
                    overlay_meta=overlay_meta,
                    source_row=row,
                )
            )

            continue

        # ---------------------------------------------------------------------
        # 13.26.6 Add full-resolution candidate features
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # Preserve the original scoring behaviour from the working CELL 13.
        # ---------------------------------------------------------------------
        scored_df = raw_df.copy()

        scored_df["pole_cx"] = (
            scored_df["x1"] + scored_df["x2"]
        ) / 2.0

        scored_df["pole_cy"] = (
            scored_df["y1"] + scored_df["y2"]
        ) / 2.0

        scored_df["image_area"] = (
            scored_df["image_w"] * scored_df["image_h"]
        )

        scored_df["area_frac"] = (
            scored_df["box_area"] /
            scored_df["image_area"].clip(lower=1.0)
        )

        scored_df["height_frac"] = (
            scored_df["box_h"] /
            scored_df["image_h"].clip(lower=1.0)
        )

        scored_df["width_frac"] = (
            scored_df["box_w"] /
            scored_df["image_w"].clip(lower=1.0)
        )

        scored_df["aspect_ratio"] = (
            scored_df["box_h"] /
            scored_df["box_w"].clip(lower=1.0)
        )

        scored_df["x_center_dist_norm"] = (
            np.abs(scored_df["pole_cx"] - image_cx) /
            max(image_w / 2.0, 1.0)
        )

        scored_df["x_center_score"] = (
            1.0 -
            np.clip(
                scored_df["x_center_dist_norm"],
                0.0,
                1.0,
            )
        )

        max_h = max(
            float(scored_df["box_h"].max()),
            1.0,
        )

        max_a = max(
            float(scored_df["box_area"].max()),
            1.0,
        )

        scored_df["height_score"] = scored_df["box_h"] / max_h
        scored_df["area_score"] = scored_df["box_area"] / max_a
        scored_df["conf_score"] = scored_df["score"]

        edge_margin = np.minimum.reduce([
            scored_df["x1"].values,
            scored_df["y1"].values,
            (scored_df["image_w"] - scored_df["x2"]).values,
            (scored_df["image_h"] - scored_df["y2"]).values,
        ])

        edge_norm_denom = 0.05 * np.minimum(
            scored_df["image_w"],
            scored_df["image_h"],
        )

        edge_norm_denom = edge_norm_denom.clip(
            lower=1.0,
        )

        scored_df["edge_margin"] = edge_margin

        scored_df["edge_score"] = np.clip(
            scored_df["edge_margin"] / edge_norm_denom,
            0.0,
            1.0,
        )

        # ---------------------------------------------------------------------
        # 13.26.7 Prefilter candidates
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # Preserve the original production pole prefilter.
        # ---------------------------------------------------------------------
        scored_df["keep_score"] = (
            scored_df["score"] >= POLE_MIN_SCORE
        )

        scored_df["keep_area"] = (
            scored_df["area_frac"] >= POLE_MIN_AREA_FRAC
        )

        scored_df["keep_height"] = (
            scored_df["height_frac"] >= POLE_MIN_HEIGHT_FRAC
        )

        scored_df["keep_aspect"] = (
            scored_df["aspect_ratio"] >= POLE_MIN_ASPECT
        )

        scored_df["keep_width_frac"] = (
            scored_df["width_frac"] <= POLE_MAX_WIDTH_FRAC
        )

        scored_df["keep_width_px"] = (
            scored_df["box_w"] <= POLE_MAX_BOX_W_PX
        )

        scored_df["is_kept_after_prefilter"] = (
            scored_df["keep_score"] &
            scored_df["keep_area"] &
            scored_df["keep_height"] &
            scored_df["keep_aspect"] &
            scored_df["keep_width_frac"] &
            scored_df["keep_width_px"]
        )

        n_kept_candidates = int(
            scored_df["is_kept_after_prefilter"].sum()
        )

        scored_df["selection_mode"] = "not_kept"
        scored_df["final_score"] = np.nan
        scored_df["is_selected_pole"] = False

        kept_df = scored_df[
            scored_df["is_kept_after_prefilter"] == True
        ].copy()

        fallback_triggered = kept_df.empty

        if fallback_triggered:
            kept_df = scored_df.copy()
            kept_df["selection_mode"] = "fallback_all_candidates"
        else:
            kept_df["selection_mode"] = "prefilter_kept"

        # ---------------------------------------------------------------------
        # 13.26.8 Score and select best pole
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # Preserve the original weighted scoring and fallback behaviour.
        # ---------------------------------------------------------------------
        kept_df["shaft_penalty"] = np.where(
            kept_df["width_frac"] > SHAFT_WIDTH_FRAC_THRESHOLD,
            SHAFT_PENALTY_FACTOR,
            1.0,
        )

        kept_df["final_score"] = (
            (
                W_X_CENTER * kept_df["x_center_score"] +
                W_HEIGHT   * kept_df["height_score"] +
                W_AREA     * kept_df["area_score"] +
                W_CONF     * kept_df["conf_score"] +
                W_EDGE     * kept_df["edge_score"]
            )
            * kept_df["shaft_penalty"]
        )

        kept_df = (
            kept_df
            .sort_values(
                by=[
                    "final_score",
                    "score",
                    "box_h",
                    "x_center_score",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

        if len(kept_df) > 0:
            kept_df.loc[0, "is_selected_pole"] = True

        score_cols = [
            "image_id",
            "prompt",
            "det_idx",
            "selection_mode",
            "final_score",
            "is_selected_pole",
            "shaft_penalty",
        ]

        scored_df = scored_df.drop(
            columns=[
                "selection_mode",
                "final_score",
                "is_selected_pole",
                "shaft_penalty",
            ],
            errors="ignore",
        )

        scored_df = scored_df.merge(
            kept_df[score_cols],
            on=[
                "image_id",
                "prompt",
                "det_idx",
            ],
            how="left",
        )

        scored_df["selection_mode"] = (
            scored_df["selection_mode"].fillna("not_kept")
        )

        scored_df["is_selected_pole"] = (
            scored_df["is_selected_pole"].fillna(False)
        )

        scored_df["shaft_penalty"] = (
            scored_df["shaft_penalty"].fillna(np.nan)
        )

        scored_df["fallback_triggered"] = bool(fallback_triggered)

        scored_df["selection_status"] = np.where(
            scored_df["is_selected_pole"].astype(bool),
            "selected",
            "not_selected",
        )

        candidate_frames.append(scored_df)

        selected_df = scored_df[
            scored_df["is_selected_pole"] == True
        ].copy()

        if len(selected_df) > 1:
            raise RuntimeError(
                f"Multiple poles selected for image_id={image_id}"
            )

        # ---------------------------------------------------------------------
        # 13.26.9 Exactly one selected pole
        # ---------------------------------------------------------------------
        if len(selected_df) == 1:
            best_row = selected_df.iloc[0]

            overlay_label_text = _build_pole_overlay_label(
                prompt=best_row["prompt"],
                score=best_row["score"],
                final_score=best_row["final_score"],
            )

            selected_mask_key = _candidate_key(
                image_id=image_id,
                prompt=best_row["prompt"],
                det_idx=int(best_row["det_idx"]),
            )

            selected_mask = pole_mask_lookup.get(
                selected_mask_key,
                None,
            )

            overlay_output_path = _build_overlay_output_path(
                row=row,
                image_id=image_id,
            )

            overlay_meta = _save_selected_overlay_matplotlib(
                image_rgb=image_rgb,
                output_path=overlay_output_path,
                selected_row=best_row,
                mask_2d=selected_mask,
                label_text=overlay_label_text,
            )

            selected_row_out = source_lineage.copy()

            selected_row_out.update({
                "image_id": image_id,
                "file_name": file_name,
                "image_path": image_path,
                "image_w": int(image_w),
                "image_h": int(image_h),
                "selection_status": "selected",
                "selection_mode": str(best_row["selection_mode"]),
                "is_selected_pole": True,
                "n_raw_candidates": n_raw_candidates,
                "n_kept_candidates": n_kept_candidates,
                "prompt": str(best_row["prompt"]),
                "det_idx": int(best_row["det_idx"]),
                "score": float(best_row["score"]),
                "x1": float(best_row["x1"]),
                "y1": float(best_row["y1"]),
                "x2": float(best_row["x2"]),
                "y2": float(best_row["y2"]),
                "box_w": float(best_row["box_w"]),
                "box_h": float(best_row["box_h"]),
                "box_area": float(best_row["box_area"]),
                "pole_cx": float(best_row["pole_cx"]),
                "pole_cy": float(best_row["pole_cy"]),
                "x_center_dist_norm": float(best_row["x_center_dist_norm"]),
                "width_frac": float(best_row["width_frac"]),
                "height_frac": float(best_row["height_frac"]),
                "area_frac": float(best_row["area_frac"]),
                "aspect_ratio": float(best_row["aspect_ratio"]),
                "shaft_penalty": (
                    float(best_row["shaft_penalty"])
                    if pd.notna(best_row["shaft_penalty"])
                    else np.nan
                ),
                "final_score": (
                    float(best_row["final_score"])
                    if pd.notna(best_row["final_score"])
                    else np.nan
                ),
                "has_mask": bool(best_row["has_mask"]),
                "fallback_triggered": bool(fallback_triggered),
                "overlay_label_text": overlay_label_text,
                "overlay_image_path": overlay_meta["overlay_image_path"],
                "overlay_resize_ratio": overlay_meta["overlay_resize_ratio"],
                "overlay_image_w": overlay_meta["overlay_image_w"],
                "overlay_image_h": overlay_meta["overlay_image_h"],
            })

            selection_rows.append(selected_row_out)

        # ---------------------------------------------------------------------
        # 13.26.10 No selected pole after scoring / fallback
        # ---------------------------------------------------------------------
        else:
            overlay_label_text = NO_RELIABLE_POLE_LABEL_TEXT

            overlay_output_path = _build_overlay_output_path(
                row=row,
                image_id=image_id,
            )

            overlay_meta = _save_selected_overlay_matplotlib(
                image_rgb=image_rgb,
                output_path=overlay_output_path,
                selected_row=None,
                mask_2d=None,
                label_text=overlay_label_text,
            )

            selection_rows.append(
                _make_no_reliable_pole_row(
                    image_id=image_id,
                    file_name=file_name,
                    image_path=image_path,
                    image_w=image_w,
                    image_h=image_h,
                    n_raw_candidates=n_raw_candidates,
                    n_kept_candidates=n_kept_candidates,
                    fallback_triggered=fallback_triggered,
                    overlay_label_text=overlay_label_text,
                    overlay_meta=overlay_meta,
                    source_row=row,
                )
            )

    except Exception as exc:
        # ---------------------------------------------------------------------
        # 13.26.11 Per-image failure capture
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # A single image failure should not stop the full production batch.
        # Store the failure row and remove any masks already accumulated for this
        # image so downstream cells only see masks for successfully processed
        # images.
        # ---------------------------------------------------------------------
        failure_rows.append(
            _make_pole_failure_row(
                source_row=row,
                error=exc,
                stage="pole_selection",
            )
        )

        if image_id_for_failure_cleanup is not None:
            pole_mask_lookup = {
                key: value
                for key, value in pole_mask_lookup.items()
                if key[0] != str(image_id_for_failure_cleanup)
            }

        continue
    
    
# =============================================================================
# D. OUTPUT ASSEMBLY + FINAL CHECKS
# =============================================================================

# =============================================================================
# 13.27 COMBINE ACCUMULATED OUTPUT ROWS
# =============================================================================
# EXPLANATION:
# The production loop fills lightweight Python accumulators:
#
#   candidate_frames:
#       Per-image candidate DataFrames.
#
#   selection_rows:
#       One selected/no-reliable-pole row for each successfully processed image.
#
#   failure_rows:
#       One failure row for each image that failed during pole selection.
#
# This section converts those accumulators into the final DataFrames expected by
# downstream cells.
# =============================================================================

candidate_front_cols = [
    "run_id",
    "run_timestamp",
    "processing_order",
    "image_id",
    "file_name",
    "stem",
    "ext",
    "ext_lower",
    "relative_image_path",
    "image_path",
    "source_image_path",
    "source_layer",
    "source_root",
    "bronze_root",

    "image_w",
    "image_h",
    "image_area",

    "prompt",
    "det_idx",
    "score",

    "selection_status",
    "selection_mode",
    "is_selected_pole",
    "final_score",
    "fallback_triggered",

    "x1",
    "y1",
    "x2",
    "y2",
    "box_w",
    "box_h",
    "box_area",
    "has_mask",

    "pole_cx",
    "pole_cy",
    "x_center_dist_norm",
    "x_center_score",

    "width_frac",
    "height_frac",
    "area_frac",
    "aspect_ratio",

    "height_score",
    "area_score",
    "conf_score",
    "edge_margin",
    "edge_score",

    "keep_score",
    "keep_area",
    "keep_height",
    "keep_aspect",
    "keep_width_frac",
    "keep_width_px",
    "is_kept_after_prefilter",
    "shaft_penalty",
]

selection_front_cols = [
    "run_id",
    "run_timestamp",
    "processing_order",
    "image_id",
    "file_name",
    "stem",
    "ext",
    "ext_lower",
    "relative_image_path",
    "image_path",
    "source_image_path",
    "source_layer",
    "source_root",
    "bronze_root",

    "image_w",
    "image_h",

    "selection_status",
    "selection_mode",
    "is_selected_pole",
    "n_raw_candidates",
    "n_kept_candidates",

    "prompt",
    "det_idx",
    "score",

    "x1",
    "y1",
    "x2",
    "y2",
    "box_w",
    "box_h",
    "box_area",
    "has_mask",

    "pole_cx",
    "pole_cy",
    "x_center_dist_norm",

    "width_frac",
    "height_frac",
    "area_frac",
    "aspect_ratio",

    "shaft_penalty",
    "final_score",
    "fallback_triggered",

    "overlay_label_text",
    "overlay_image_path",
    "overlay_resize_ratio",
    "overlay_image_w",
    "overlay_image_h",
]

failure_front_cols = [
    "run_id",
    "run_timestamp",
    "processing_order",
    "image_id",
    "file_name",
    "stem",
    "ext",
    "ext_lower",
    "relative_image_path",
    "image_path",
    "source_image_path",
    "source_layer",
    "source_root",
    "bronze_root",

    "failure_stage",
    "error_type",
    "error_message",
    "error_traceback",
]


pole_candidates_df = (
    pd.concat(
        candidate_frames,
        ignore_index=True,
    )
    if len(candidate_frames) > 0
    else pd.DataFrame(columns=candidate_front_cols)
)

pole_selection_df = (
    pd.DataFrame(selection_rows)
    if len(selection_rows) > 0
    else pd.DataFrame(columns=selection_front_cols)
)

pole_failures_df = (
    pd.DataFrame(failure_rows)
    if len(failure_rows) > 0
    else pd.DataFrame(columns=failure_front_cols)
)

# =============================================================================
# 13.28 PRUNE pole_mask_lookup TO SELECTED POLES ONLY
# =============================================================================
# EXPLANATION:
# During selection, pole_mask_lookup may temporarily contain masks for many pole
# candidates.
#
# Downstream cells only need masks for final selected poles. Pruning here keeps
# memory smaller before ROI generation.
#
# Failed images were already cleaned inside 13C. This section keeps only masks
# whose keys match selected rows in pole_selection_df.
# =============================================================================

selected_keys = set()

if (
    not pole_selection_df.empty
    and "selection_status" in pole_selection_df.columns
    and "image_id" in pole_selection_df.columns
    and "prompt" in pole_selection_df.columns
    and "det_idx" in pole_selection_df.columns
):
    for _, selected_row in pole_selection_df.iterrows():
        selection_status = selected_row.get(
            "selection_status",
            None,
        )

        prompt_value = selected_row.get(
            "prompt",
            None,
        )

        det_idx_value = selected_row.get(
            "det_idx",
            None,
        )

        if (
            selection_status == "selected"
            and pd.notna(prompt_value)
            and pd.notna(det_idx_value)
        ):
            selected_keys.add(
                _candidate_key(
                    selected_row["image_id"],
                    prompt_value,
                    int(det_idx_value),
                )
            )


pole_mask_lookup = {
    key: value
    for key, value in pole_mask_lookup.items()
    if key in selected_keys
}

# =============================================================================
# 13.29 REORDER OUTPUT COLUMNS
# =============================================================================
# EXPLANATION:
# Put common production lineage and decision columns first while preserving any
# extra columns added by the loop.
#
# IMPORTANT:
#   This uses the existing + remaining pattern so the new lineage columns do not
#   break compatibility with the old candidate/scoring columns.
# =============================================================================

def _cell13_reorder_columns(df, front_cols):
    """
    Reorder a DataFrame by placing preferred columns first.

    Args:
        df:
            pandas DataFrame to reorder.

        front_cols:
            Preferred column order.

    Returns:
        pandas.DataFrame:
            Reordered DataFrame.
    """
    if df is None:
        return pd.DataFrame(columns=front_cols)

    existing_front_cols = [
        col_name
        for col_name in front_cols
        if col_name in df.columns
    ]

    remaining_cols = [
        col_name
        for col_name in df.columns
        if col_name not in existing_front_cols
    ]

    return df[
        existing_front_cols + remaining_cols
    ]


pole_candidates_df = _cell13_reorder_columns(
    pole_candidates_df,
    candidate_front_cols,
)

pole_selection_df = _cell13_reorder_columns(
    pole_selection_df,
    selection_front_cols,
)

pole_failures_df = _cell13_reorder_columns(
    pole_failures_df,
    failure_front_cols,
)


# =============================================================================
# 13.30 FINAL PRODUCTION CONSISTENCY CHECKS
# =============================================================================
# EXPLANATION:
# Production Cell 13 now allows per-image failures.
#
# Therefore, the correct reconciliation is:
#
#   successful selection/no-reliable rows + failure rows == input image rows
#
# not:
#
#   pole_selection_df rows == input image rows
#
# This preserves batch robustness while still making sure every input image has
# an outcome.
# =============================================================================

input_image_count = int(len(run_images_df))
selection_row_count = int(len(pole_selection_df))
failure_row_count = int(len(pole_failures_df))

processed_outcome_count = (
    selection_row_count + failure_row_count
)

if processed_outcome_count != input_image_count:
    raise RuntimeError(
        "CELL 13 output rows do not reconcile with input images.\n"
        "Each input image must have either one pole_selection_df row or one "
        "pole_failures_df row.\n"
        f"run_images_df rows       : {input_image_count}\n"
        f"pole_selection_df rows   : {selection_row_count}\n"
        f"pole_failures_df rows    : {failure_row_count}\n"
        f"combined outcome rows    : {processed_outcome_count}"
    )


# -----------------------------------------------------------------------------
# Validate image_id coverage.
# -----------------------------------------------------------------------------
run_image_ids = set(
    run_images_df["image_id"]
    .astype(str)
    .tolist()
)

selection_image_ids = (
    set(
        pole_selection_df["image_id"]
        .dropna()
        .astype(str)
        .tolist()
    )
    if "image_id" in pole_selection_df.columns
    else set()
)

failure_image_ids = (
    set(
        pole_failures_df["image_id"]
        .dropna()
        .astype(str)
        .tolist()
    )
    if "image_id" in pole_failures_df.columns
    else set()
)

processed_image_ids = (
    selection_image_ids | failure_image_ids
)

missing_outcome_image_ids = sorted(
    run_image_ids - processed_image_ids
)

unexpected_outcome_image_ids = sorted(
    processed_image_ids - run_image_ids
)

if missing_outcome_image_ids:
    raise RuntimeError(
        "Some input images have no Cell 13 outcome row.\n"
        f"Missing image_id examples: {missing_outcome_image_ids[:10]}"
    )

if unexpected_outcome_image_ids:
    raise RuntimeError(
        "Some Cell 13 outcome rows do not match run_images_df.\n"
        f"Unexpected image_id examples: {unexpected_outcome_image_ids[:10]}"
    )


# -----------------------------------------------------------------------------
# Validate duplicate outcomes.
# -----------------------------------------------------------------------------
if (
    not pole_selection_df.empty
    and "image_id" in pole_selection_df.columns
    and pole_selection_df["image_id"].duplicated().any()
):
    duplicate_selection_ids = (
        pole_selection_df
        .loc[pole_selection_df["image_id"].duplicated(), "image_id"]
        .astype(str)
        .tolist()
    )

    raise RuntimeError(
        "pole_selection_df contains duplicate image_id values.\n"
        f"Duplicate examples: {duplicate_selection_ids[:10]}"
    )

if (
    not pole_failures_df.empty
    and "image_id" in pole_failures_df.columns
    and pole_failures_df["image_id"].duplicated().any()
):
    duplicate_failure_ids = (
        pole_failures_df
        .loc[pole_failures_df["image_id"].duplicated(), "image_id"]
        .astype(str)
        .tolist()
    )

    raise RuntimeError(
        "pole_failures_df contains duplicate image_id values.\n"
        f"Duplicate examples: {duplicate_failure_ids[:10]}"
    )

overlap_image_ids = sorted(
    selection_image_ids & failure_image_ids
)

if overlap_image_ids:
    raise RuntimeError(
        "Some images appear in both pole_selection_df and pole_failures_df.\n"
        f"Overlapping image_id examples: {overlap_image_ids[:10]}"
    )


# -----------------------------------------------------------------------------
# Validate selected-pole uniqueness.
# -----------------------------------------------------------------------------
selected_count = int(
    (
        pole_selection_df["selection_status"] == "selected"
    ).sum()
) if "selection_status" in pole_selection_df.columns else 0

no_reliable_count = int(
    (
        pole_selection_df["selection_status"] == "no_reliable_pole_found"
    ).sum()
) if "selection_status" in pole_selection_df.columns else 0

known_selection_status_count = (
    selected_count + no_reliable_count
)

if known_selection_status_count != selection_row_count:
    unknown_status_rows = (
        pole_selection_df
        .loc[
            ~pole_selection_df["selection_status"].isin(
                [
                    "selected",
                    "no_reliable_pole_found",
                ]
            ),
            [
                "image_id",
                "selection_status",
            ],
        ]
        .head(10)
        .to_dict("records")
        if (
            not pole_selection_df.empty
            and "selection_status" in pole_selection_df.columns
        )
        else []
    )

    raise RuntimeError(
        "pole_selection_df contains unexpected selection_status values.\n"
        f"Examples: {unknown_status_rows}"
    )


# -----------------------------------------------------------------------------
# Validate selected-pole mask lookup consistency.
# -----------------------------------------------------------------------------
missing_selected_mask_keys = []

if (
    not pole_selection_df.empty
    and "selection_status" in pole_selection_df.columns
    and "has_mask" in pole_selection_df.columns
):
    for _, selected_row in pole_selection_df.iterrows():
        if selected_row.get("selection_status") != "selected":
            continue

        has_mask = bool(
            selected_row.get(
                "has_mask",
                False,
            )
        )

        if not has_mask:
            continue

        selected_key = _candidate_key(
            selected_row["image_id"],
            selected_row["prompt"],
            int(selected_row["det_idx"]),
        )

        if selected_key not in pole_mask_lookup:
            missing_selected_mask_keys.append(selected_key)

if missing_selected_mask_keys:
    raise RuntimeError(
        "Some selected-pole rows indicate has_mask=True, but their masks are "
        "missing from pole_mask_lookup after pruning.\n"
        f"Missing key examples: {missing_selected_mask_keys[:10]}"
    )


# -----------------------------------------------------------------------------
# Validate overlay files only when overlay saving is enabled.
# -----------------------------------------------------------------------------
missing_overlay_files = []

if (
    SAVE_POLE_SELECTION_OVERLAYS
    and not pole_selection_df.empty
    and "overlay_image_path" in pole_selection_df.columns
):
    null_overlay_rows = (
        pole_selection_df
        .loc[
            pole_selection_df["overlay_image_path"].isna(),
            [
                "image_id",
                "selection_status",
            ],
        ]
        .head(10)
        .to_dict("records")
    )

    if null_overlay_rows:
        raise RuntimeError(
            "SAVE_POLE_SELECTION_OVERLAYS=True, but some successful Cell 13 "
            "outcome rows have no overlay_image_path.\n"
            f"Examples: {null_overlay_rows}"
        )

    missing_overlay_files = [
        path
        for path in pole_selection_df["overlay_image_path"].dropna().tolist()
        if not os.path.exists(path)
    ]

    if missing_overlay_files:
        raise RuntimeError(
            "Some pole overlay images were expected but were not found on disk.\n"
            f"Missing file examples: {missing_overlay_files[:10]}"
        )


# =============================================================================
# 13.31 OUTPUT COUNT SUMMARY OBJECT
# =============================================================================
# EXPLANATION:
# Store compact counts for 13E so the final save/summary section can report the
# run outcome without recomputing everything.
# =============================================================================

cell13_output_counts = {
    "input_image_count": input_image_count,
    "pole_candidate_row_count": int(len(pole_candidates_df)),
    "pole_selection_row_count": selection_row_count,
    "pole_failure_row_count": failure_row_count,
    "selected_pole_count": selected_count,
    "no_reliable_pole_count": no_reliable_count,
    "selected_mask_count": int(len(pole_mask_lookup)),
    "missing_overlay_file_count": int(len(missing_overlay_files)),
}

if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print("CELL 13 output assembly complete.\n")

    print("=" * 90)
    print("CELL 13 OUTPUT COUNTS")
    print("=" * 90)
    print(f"Input images                         : {cell13_output_counts['input_image_count']}")
    print(f"Pole candidate rows                  : {cell13_output_counts['pole_candidate_row_count']}")
    print(f"Pole selection rows                  : {cell13_output_counts['pole_selection_row_count']}")
    print(f"Pole failure rows                    : {cell13_output_counts['pole_failure_row_count']}")
    print(f"Selected poles                       : {cell13_output_counts['selected_pole_count']}")
    print(f"No reliable pole rows                : {cell13_output_counts['no_reliable_pole_count']}")
    print(f"Selected masks retained              : {cell13_output_counts['selected_mask_count']}")
    
    
# =============================================================================
# E. SAVE OUTPUT TABLES + FINAL SUMMARY
# =============================================================================

# =============================================================================
# 13.32 REQUIRED SAVE CHECKS
# =============================================================================
# EXPLANATION:
# Confirm that 13D has already created the final output DataFrames and summary
# object before this save section runs.
# =============================================================================

required_cell13e_globals = [
    "pole_candidates_df",
    "pole_selection_df",
    "pole_failures_df",
    "pole_mask_lookup",
    "cell13_output_counts",
    "RUN_SILVER_POLE_SELECTION_TABLES_DIR",
    "save_run_table",
]

missing_cell13e_globals = [
    name for name in required_cell13e_globals
    if name not in globals()
]

if missing_cell13e_globals:
    raise NameError(
        "CELL 13E requires CELL 13D to run successfully first.\n"
        f"Missing globals: {missing_cell13e_globals}"
    )

# =============================================================================
# 13.33 SAVE PRODUCTION POLE TABLES
# =============================================================================
# EXPLANATION:
# Save final Cell 13 production tables as Parquet using save_run_table().
#
# IMPORTANT:
#   - This replaces the old save_state tail.
#   - Tables are saved under:
#
#       SILVER_POLE_SELECTION/tables/<RUN_ID>/
#
#   - save_run_table() skips empty DataFrames and returns None.
#     Therefore, if there are zero failures, pole_failures.parquet may not be
#     written. That is expected.
# =============================================================================

cell13_saved_paths = {
    "pole_candidates": None,
    "pole_selection": None,
    "pole_failures": None,
}

if SAVE_POLE_CANDIDATES_TABLE:
    cell13_saved_paths["pole_candidates"] = save_run_table(
        pole_candidates_df,
        RUN_SILVER_POLE_SELECTION_TABLES_DIR,
        "pole_candidates",
    )

if SAVE_POLE_SELECTION_TABLE:
    cell13_saved_paths["pole_selection"] = save_run_table(
        pole_selection_df,
        RUN_SILVER_POLE_SELECTION_TABLES_DIR,
        "pole_selection",
    )

if SAVE_POLE_FAILURES_TABLE:
    cell13_saved_paths["pole_failures"] = save_run_table(
        pole_failures_df,
        RUN_SILVER_POLE_SELECTION_TABLES_DIR,
        "pole_failures",
    )

# =============================================================================
# 13.34 VALIDATE REQUIRED SAVED OUTPUTS
# =============================================================================
# EXPLANATION:
# The pole selection table is required for downstream ROI generation. Candidate
# and failure tables are useful audit outputs, but the selected-pole table is the
# critical production output.
# =============================================================================

if SAVE_POLE_SELECTION_TABLE and cell13_saved_paths["pole_selection"] is None:
    raise RuntimeError(
        "SAVE_POLE_SELECTION_TABLE=True, but pole_selection_df was not saved.\n"
        "This usually means pole_selection_df was unexpectedly empty."
    )

if (
    SAVE_POLE_SELECTION_TABLE
    and cell13_saved_paths["pole_selection"] is not None
    and not os.path.exists(cell13_saved_paths["pole_selection"])
):
    raise RuntimeError(
        "pole_selection parquet path was returned but does not exist on disk.\n"
        f"Path: {cell13_saved_paths['pole_selection']}"
    )

if (
    SAVE_POLE_CANDIDATES_TABLE
    and cell13_saved_paths["pole_candidates"] is not None
    and not os.path.exists(cell13_saved_paths["pole_candidates"])
):
    raise RuntimeError(
        "pole_candidates parquet path was returned but does not exist on disk.\n"
        f"Path: {cell13_saved_paths['pole_candidates']}"
    )

if (
    SAVE_POLE_FAILURES_TABLE
    and cell13_saved_paths["pole_failures"] is not None
    and not os.path.exists(cell13_saved_paths["pole_failures"])
):
    raise RuntimeError(
        "pole_failures parquet path was returned but does not exist on disk.\n"
        f"Path: {cell13_saved_paths['pole_failures']}"
    )

# =============================================================================
# 13.35 FINAL CELL 13 SUMMARY
# =============================================================================
# EXPLANATION:
# Print a compact final summary when PRINT_CONFIG_SUMMARY is enabled.
#
# Downstream cells should use:
#   - pole_selection_df
#   - pole_mask_lookup
#
# Saved overlay PNGs are QA artifacts only.
# =============================================================================

if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print("CELL 13 production pole selection completed.\n")

    print("=" * 90)
    print("CELL 13 FINAL OUTPUT COUNTS")
    print("=" * 90)
    print(f"RUN_ID                              : {RUN_ID}")
    print(f"Input images                        : {cell13_output_counts['input_image_count']}")
    print(f"Pole candidate rows                 : {cell13_output_counts['pole_candidate_row_count']}")
    print(f"Pole selection rows                 : {cell13_output_counts['pole_selection_row_count']}")
    print(f"Pole failure rows                   : {cell13_output_counts['pole_failure_row_count']}")
    print(f"Selected poles                      : {cell13_output_counts['selected_pole_count']}")
    print(f"No reliable pole rows               : {cell13_output_counts['no_reliable_pole_count']}")
    print(f"Selected masks retained             : {cell13_output_counts['selected_mask_count']}")
    print(f"Missing overlay files               : {cell13_output_counts['missing_overlay_file_count']}")

    print("\n" + "=" * 90)
    print("CELL 13 SAVE FLAGS")
    print("=" * 90)
    print(f"SAVE_POLE_CANDIDATES_TABLE          : {SAVE_POLE_CANDIDATES_TABLE}")
    print(f"SAVE_POLE_SELECTION_TABLE           : {SAVE_POLE_SELECTION_TABLE}")
    print(f"SAVE_POLE_FAILURES_TABLE            : {SAVE_POLE_FAILURES_TABLE}")
    print(f"SAVE_POLE_SELECTION_OVERLAYS        : {SAVE_POLE_SELECTION_OVERLAYS}")

    print("\n" + "=" * 90)
    print("CELL 13 SAVED TABLE PATHS")
    print("=" * 90)
    print(f"RUN_SILVER_POLE_SELECTION_TABLES_DIR: {RUN_SILVER_POLE_SELECTION_TABLES_DIR}")
    print(f"pole_candidates                     : {cell13_saved_paths['pole_candidates']}")
    print(f"pole_selection                      : {cell13_saved_paths['pole_selection']}")
    print(f"pole_failures                       : {cell13_saved_paths['pole_failures']}")

    print("\n" + "=" * 90)
    print("CELL 13 DOWNSTREAM OBJECTS READY")
    print("=" * 90)
    print(f"pole_selection_df shape             : {pole_selection_df.shape}")
    print(f"pole_candidates_df shape            : {pole_candidates_df.shape}")
    print(f"pole_failures_df shape              : {pole_failures_df.shape}")
    print(f"pole_mask_lookup entries            : {len(pole_mask_lookup)}")
    print(f"Pole overlay folder                 : {RUN_SILVER_POLE_SELECTION_OVERLAYS_DIR}")
    
    
# =============================================================================
# CELL 14 — PRODUCTION POLE-TOP FIXED CANVAS ROI GENERATION
# =============================================================================
# OVERVIEW:
# This cell builds one clean, fixed-size pole-top ROI crop for every selected
# pole from CELL 13.
#
# It uses the original full-resolution source image and the selected-pole
# coordinates stored in pole_selection_df. It does not use saved QA overlay PNGs
# as crop inputs.
#
# The saved ROI images are clean RGB crops. They do not contain pole masks,
# boxes, labels, or QA overlay graphics.
#
# IMPORTANT DESIGN RULES:
#   - Do NOT crop from saved selected-pole overlay PNGs.
#   - Use pole_selection_df["image_path"] as the source image.
#   - Use selected-pole coordinates from pole_selection_df.
#   - Keep ROI crops clean for downstream SAM3 crossarm detection.
#   - Use run-scoped output folders so previous runs are not overwritten.
#   - Use image_id-based ROI filenames to avoid stem/suffix collisions.
#   - Capture row-level crop failures so one bad image does not stop the batch.
#
# STRUCTURE:
#   A. SAFETY CHECKS + OUTPUT SETUP
#        Section 14.1.  Required setup checks
#        Section 14.2.  Validate pole_selection_df
#        Section 14.3.  Validate required selected-pole columns
#        Section 14.4.  Validate run identity
#        Section 14.5.  Keep selected poles only
#        Section 14.6.  Resolve fixed-canvas ROI config
#        Section 14.7.  Define run-scoped output paths
#        Section 14.8.  Prepare run-scoped output folders
#        Section 14.9.  Define row-failure handling behaviour
#        Section 14.10. Initialise accumulators
#        Section 14.11. Config summary
#
#   B. HELPER FUNCTIONS
#        Section 14.12. Row value helper
#        Section 14.13. Lineage field helper
#        Section 14.14. Selected-pole mask lookup key helper
#        Section 14.15. ROI output path helper
#        Section 14.16. Shift fixed box inside image helper
#        Section 14.17. Build fixed pole-top ROI request helper
#        Section 14.18. Render fixed-size canvas helper
#        Section 14.19. Pole ROI failure row helper
#
#   C. ROI GENERATION LOOP
#        Section 14.20. Run fixed-canvas ROI generation
#        Section 14.21. Loop cleanup + summary
#
#   D. OUTPUT MANIFEST + SAVE
#        Section 14.22. Assemble ROI output DataFrames
#        Section 14.23. Reorder output columns
#        Section 14.24. Final production consistency checks
#        Section 14.25. Verify saved ROI image files exist
#        Section 14.26. Build output count summary
#        Section 14.27. Save CELL 14 output tables
#        Section 14.28. Validate saved table outputs
#        Section 14.29. Final CELL 14 summary
#        Section 14.30. Fail loudly if no ROI crops were created
#
# OUTPUTS:
#   pole_rois_df:
#       One row per successfully created selected-pole ROI.
#       This is the primary downstream ROI manifest for CELL 16.
#
#   pole_roi_failures_df:
#       One row per selected-pole ROI that failed during crop creation.
#
#   cell14_output_counts:
#       Compact count summary for audit/debug checks.
#
#   cell14_saved_paths:
#       Saved Parquet paths returned by save_run_table().
#
# SAVED OUTPUTS:
#   Clean ROI PNGs:
#       SILVER_POLE_ROIS/images/<RUN_ID>/
#
#   ROI tables:
#       SILVER_POLE_ROIS/tables/<RUN_ID>/
#
# DOWNSTREAM HANDOFF:
#   - CELL 16 should use pole_rois_df as the ROI input table.
#   - CELL 16 may also use pole_mask_lookup from CELL 13 for pole-mask /
#     pole-corridor filtering.
#
# IMPORTANT:
#   - Run this after CELL 13.
#   - This cell uses RUN_ID and RUN_TIMESTAMP from CELL 10B.
#   - ROI filenames use image_id, not trailing numeric suffixes.
#   - Tables are saved using save_run_table(), not save_state().
# =============================================================================

# =============================================================================
# A. SAFETY CHECKS + OUTPUT SETUP
# =============================================================================


# =============================================================================
# 14.1 REQUIRED SETUP CHECKS
# =============================================================================
# EXPLANATION:
# Fail early if required variables from earlier production cells are missing.
#
# Required earlier cells:
#   - CELL 3A  : os, gc, shutil, traceback, pd, np, Image
#   - CELL 3B  : fixed ROI constants
#   - CELL 10  : SILVER_POLE_ROIS
#   - CELL 10B : RUN_ID, RUN_TIMESTAMP, save_run_table, make_safe_path_part
#   - CELL 13  : pole_selection_df, pole_mask_lookup
# =============================================================================

required_cell14_globals = [
    # Core libraries.
    "os",
    "gc",
    "shutil",
    "traceback",
    "pd",
    "np",
    "Image",

    # Production run identity.
    "RUN_ID",
    "RUN_TIMESTAMP",

    # Save helpers from CELL 10B.
    "save_run_table",
    "make_safe_path_part",

    # Output folder from CELL 10.
    "SILVER_POLE_ROIS",

    # Fixed ROI config from CELL 3B.
    "FIXED_ROI_WIDTH",
    "FIXED_ROI_HEIGHT",
    "POLE_TOP_BUFFER_ABOVE",
    "PAD_RGB",
    "OVERWRITE_POLE_ROIS",

    # CELL 13 outputs.
    "pole_selection_df",
    "pole_mask_lookup",
]

missing_cell14_globals = [
    name for name in required_cell14_globals
    if name not in globals()
]

if missing_cell14_globals:
    raise NameError(
        "CELL 14 requires earlier production setup cells to run successfully.\n"
        "Please run CELL 10B and CELL 13 before CELL 14.\n"
        f"Missing globals: {missing_cell14_globals}"
    )

if not isinstance(pole_mask_lookup, dict):
    raise TypeError(
        "pole_mask_lookup exists but is not a dictionary.\n"
        "Please check CELL 13."
    )


# =============================================================================
# 14.2 VALIDATE pole_selection_df
# =============================================================================
# EXPLANATION:
# pole_selection_df is the production selected-pole table created by CELL 13.
#
# CELL 13 should produce one row per input image, including rows where no
# reliable pole was selected. CELL 14 keeps only selected rows.
# =============================================================================

if not isinstance(pole_selection_df, pd.DataFrame):
    raise TypeError(
        "pole_selection_df exists but is not a pandas DataFrame."
    )

if pole_selection_df.empty:
    raise ValueError(
        "pole_selection_df is empty.\n"
        "Please check CELL 13 before running CELL 14."
    )


# =============================================================================
# 14.3 VALIDATE REQUIRED SELECTED-POLE COLUMNS
# =============================================================================
# EXPLANATION:
# CELL 14 keeps the same lineage block needed by downstream production cells.
#
# The required columns include:
#   - run identity
#   - source-image lineage
#   - processing order
#   - selected-pole identity
#   - selected-pole geometry
#   - selected-pole scoring / mask flags
#
# IMPORTANT:
#   Coordinates must be full-resolution original-image coordinates, not overlay
#   PNG coordinates.
# =============================================================================

required_pole_roi_input_cols = [
    # Run lineage.
    "run_id",
    "run_timestamp",
    "processing_order",

    # Source image lineage.
    "image_id",
    "relative_image_path",
    "file_name",
    "stem",
    "ext",
    "ext_lower",
    "image_path",
    "image_w",
    "image_h",

    # Selected-pole identity.
    "selection_status",
    "selection_mode",
    "prompt",
    "det_idx",

    # Selected-pole scores / mask status.
    "score",
    "final_score",
    "has_mask",

    # Selected-pole full-resolution geometry.
    "x1",
    "y1",
    "x2",
    "y2",
    "box_w",
    "box_h",
    "pole_cx",
    "pole_cy",
]

missing_pole_roi_input_cols = [
    col_name
    for col_name in required_pole_roi_input_cols
    if col_name not in pole_selection_df.columns
]

if missing_pole_roi_input_cols:
    raise ValueError(
        "pole_selection_df is missing columns required by CELL 14.\n"
        "This usually means CELL 13 did not finish with the expected production "
        "schema.\n"
        f"Missing columns: {missing_pole_roi_input_cols}"
    )


# =============================================================================
# 14.4 VALIDATE RUN IDENTITY
# =============================================================================
# EXPLANATION:
# pole_selection_df should belong to the active RUN_ID.
#
# This prevents a common production mistake:
#   - CELL 14 writes files under the current RUN_ID folder
#   - but the rows came from an older CELL 13 run
# =============================================================================

unique_pole_selection_run_ids = (
    pole_selection_df["run_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

if len(unique_pole_selection_run_ids) != 1:
    raise RuntimeError(
        "pole_selection_df must contain exactly one run_id.\n"
        f"Found run_ids: {unique_pole_selection_run_ids}"
    )

if unique_pole_selection_run_ids[0] != str(RUN_ID):
    raise RuntimeError(
        "pole_selection_df run_id does not match the active RUN_ID.\n\n"
        f"pole_selection_df run_id : {unique_pole_selection_run_ids[0]}\n"
        f"active RUN_ID            : {RUN_ID}\n\n"
        "Please rerun CELL 13 for this active run, or reset RUN_ID intentionally."
    )

if pole_selection_df["run_timestamp"].dropna().empty:
    raise RuntimeError(
        "pole_selection_df contains no non-null run_timestamp values.\n"
        "Please check CELL 13 output."
    )


# =============================================================================
# 14.5 KEEP SELECTED POLES ONLY
# =============================================================================
# EXPLANATION:
# CELL 13 keeps one row per image.
#
# CELL 14 only builds ROIs for rows where a reliable pole was selected.
# No-reliable-pole rows are intentionally excluded from ROI creation.
# =============================================================================

selected_poles_df = pole_selection_df[
    pole_selection_df["selection_status"].astype(str).str.lower() == "selected"
].copy()

if selected_poles_df.empty:
    raise ValueError(
        "No selected poles were found in pole_selection_df.\n"
        "CELL 14 cannot create pole-top ROIs without selected poles."
    )

selected_poles_df = selected_poles_df.sort_values(
    by=["processing_order", "image_id"],
    kind="mergesort",
).reset_index(drop=True)


# =============================================================================
# 14.6 RESOLVE FIXED-CANVAS ROI CONFIG
# =============================================================================
# EXPLANATION:
# These values come from CELL 3B.
#
# The chosen defaults preserve the fixed-size crop behaviour already tested in
# the development pipeline:
#   - width  : 2600 px
#   - height : 3500 px
#   - top buffer above selected pole box : 500 px
#
# IMPORTANT:
#   Do not change the crop geometry here unless intentionally retuning the
#   production ROI strategy. CELL 14B preserves the tested shift/pad/paste math.
# =============================================================================

FIXED_ROI_WIDTH = int(FIXED_ROI_WIDTH)
FIXED_ROI_HEIGHT = int(FIXED_ROI_HEIGHT)
POLE_TOP_BUFFER_ABOVE = int(POLE_TOP_BUFFER_ABOVE)
OVERWRITE_POLE_ROIS = bool(OVERWRITE_POLE_ROIS)

if FIXED_ROI_WIDTH <= 0:
    raise ValueError(
        f"FIXED_ROI_WIDTH must be positive. Got: {FIXED_ROI_WIDTH}"
    )

if FIXED_ROI_HEIGHT <= 0:
    raise ValueError(
        f"FIXED_ROI_HEIGHT must be positive. Got: {FIXED_ROI_HEIGHT}"
    )

if POLE_TOP_BUFFER_ABOVE < 0:
    raise ValueError(
        "POLE_TOP_BUFFER_ABOVE must be zero or positive. "
        f"Got: {POLE_TOP_BUFFER_ABOVE}"
    )

try:
    PAD_RGB = tuple(int(v) for v in PAD_RGB)
except Exception as exc:
    raise ValueError(
        "PAD_RGB must be convertible to a 3-value RGB tuple."
    ) from exc

if len(PAD_RGB) != 3:
    raise ValueError(
        "PAD_RGB must contain exactly three RGB values. "
        f"Got: {PAD_RGB}"
    )

for channel_value in PAD_RGB:
    if channel_value < 0 or channel_value > 255:
        raise ValueError(
            "PAD_RGB values must be between 0 and 255. "
            f"Got: {PAD_RGB}"
        )
        

# =============================================================================
# 14.7 DEFINE RUN-SCOPED OUTPUT PATHS
# =============================================================================
# EXPLANATION:
# Do not write directly into the root SILVER_POLE_ROIS folder.
#
# Instead, write each production run into RUN_ID-specific folders:
#
#   SILVER_POLE_ROIS/images/<RUN_ID>/
#   SILVER_POLE_ROIS/tables/<RUN_ID>/
#
# This avoids overwriting previous run outputs.
# =============================================================================

RUN_SILVER_POLE_ROIS_IMAGES_DIR = os.path.join(
    SILVER_POLE_ROIS,
    "images",
    RUN_ID,
)

RUN_SILVER_POLE_ROIS_TABLES_DIR = os.path.join(
    SILVER_POLE_ROIS,
    "tables",
    RUN_ID,
)

# IMPORTANT:
# ROI filenames are intentionally based on image_id.
#
# Reason:
#   CELL 12 creates deterministic unique image_id values. Using image_id avoids
#   collisions that can happen with stem-only or trailing-suffix filename logic.
#
# CELL 14B should not treat this as swappable config. The production filename
# contract is image_id-based.


# =============================================================================
# 14.8 PREPARE RUN-SCOPED OUTPUT FOLDERS
# =============================================================================
# EXPLANATION:
# Only the current RUN_ID output folders are reset.
#
# IMPORTANT:
#   The root SILVER_POLE_ROIS folder is not deleted. This protects prior run
#   outputs and keeps production runs auditable.
# =============================================================================

cell14_output_dirs = [
    RUN_SILVER_POLE_ROIS_IMAGES_DIR,
    RUN_SILVER_POLE_ROIS_TABLES_DIR,
]

for output_dir in cell14_output_dirs:
    if os.path.isdir(output_dir):
        existing_items = os.listdir(output_dir)

        if existing_items and not OVERWRITE_POLE_ROIS:
            raise RuntimeError(
                "CELL 14 output folder already contains files for this RUN_ID.\n"
                f"Existing folder: {output_dir}\n\n"
                "To rebuild this run's ROI outputs, set:\n"
                "OVERWRITE_POLE_ROIS = True\n"
                "and rerun CELL 14."
            )

        if existing_items and OVERWRITE_POLE_ROIS:
            shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

missing_cell14_output_dirs = [
    output_dir
    for output_dir in cell14_output_dirs
    if not os.path.isdir(output_dir)
]

if missing_cell14_output_dirs:
    raise RuntimeError(
        "Some CELL 14 output directories were not created successfully.\n"
        f"Missing directories: {missing_cell14_output_dirs}"
    )
    
    
# =============================================================================
# 14.9 DEFINE ROW-FAILURE HANDLING BEHAVIOUR
# =============================================================================
# EXPLANATION:
# CELL 14 defaults to production-safe batch behaviour:
#   - capture row-level ROI crop failures
#   - continue processing remaining selected poles
#   - reconcile success + failure counts in CELL 14D
#
# Set CELL14_STOP_ON_ROW_FAILURE=True only while debugging when you want the
# first failed ROI row to raise immediately.
# =============================================================================

CELL14_STOP_ON_ROW_FAILURE = bool(
    globals().get("CELL14_STOP_ON_ROW_FAILURE", False)
)


# =============================================================================
# 14.10 INITIALISE ACCUMULATORS
# =============================================================================
# EXPLANATION:
# roi_rows accumulates one output row per successfully created ROI.
#
# roi_failure_rows accumulates per-image failures so a single bad image does not
# stop the full 975-image production run.
# =============================================================================

roi_rows = []
roi_failure_rows = []


# =============================================================================
# 14.11 CONFIG SUMMARY
# =============================================================================
# EXPLANATION:
# Print a compact summary when PRINT_CONFIG_SUMMARY is enabled.
# =============================================================================

if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print("CELL 14 setup complete.\n")

    print("=" * 90)
    print("CELL 14 — FIXED POLE-TOP ROI SETUP")
    print("=" * 90)
    print(f"RUN_ID                              : {RUN_ID}")
    print(f"RUN_TIMESTAMP                       : {RUN_TIMESTAMP}")
    print(f"pole_selection_df rows              : {len(pole_selection_df)}")
    print(f"Selected pole rows                  : {len(selected_poles_df)}")
    print(f"FIXED_ROI_WIDTH                     : {FIXED_ROI_WIDTH}")
    print(f"FIXED_ROI_HEIGHT                    : {FIXED_ROI_HEIGHT}")
    print(f"POLE_TOP_BUFFER_ABOVE               : {POLE_TOP_BUFFER_ABOVE}")
    print(f"PAD_RGB                             : {PAD_RGB}")
    print(f"OVERWRITE_POLE_ROIS                 : {OVERWRITE_POLE_ROIS}")
    print("ROI filename scheme                 : image_id-based")
    print(f"CELL14_STOP_ON_ROW_FAILURE          : {CELL14_STOP_ON_ROW_FAILURE}")
    print(f"RUN_SILVER_POLE_ROIS_IMAGES_DIR     : {RUN_SILVER_POLE_ROIS_IMAGES_DIR}")
    print(f"RUN_SILVER_POLE_ROIS_TABLES_DIR     : {RUN_SILVER_POLE_ROIS_TABLES_DIR}")
    
    
# =============================================================================
# B. HELPER FUNCTIONS
# =============================================================================


# =============================================================================
# 14.12 ROW VALUE HELPER
# =============================================================================
# EXPLANATION:
# Safely read values from a pandas Series or dictionary-like row.
#
# This keeps the helper layer robust when receiving:
#   - a row from selected_poles_df
#   - a plain dictionary
#   - None
# =============================================================================

def _cell14_get_row_value(row, column_name, default=None):
    """
    Safely get a value from a row-like object.

    Args:
        row:
            pandas Series, dictionary-like object, or None.

        column_name:
            Column/key name to read.

        default:
            Value returned when the row or column/key is missing.

    Returns:
        Any:
            Row value or default.
    """
    if row is None:
        return default

    if isinstance(row, pd.Series):
        if column_name in row.index:
            return row[column_name]

        return default

    if isinstance(row, dict):
        return row.get(column_name, default)

    try:
        return row[column_name]
    except Exception:
        return default


# =============================================================================
# 14.13 LINEAGE FIELD HELPER
# =============================================================================
# EXPLANATION:
# Build common production lineage fields that should appear in CELL 14 output
# tables.
#
# These fields make it easier to join pole ROI outputs back to:
#   - run_images_df
#   - pole_selection_df
#   - later crossarm detection outputs
# =============================================================================

def _cell14_build_lineage_fields(source_row=None):
    """
    Build common production lineage fields from a selected-pole row.

    Args:
        source_row:
            Row from selected_poles_df.

    Returns:
        dict:
            Common lineage fields for CELL 14 output rows.
    """
    lineage = {
        "run_id": RUN_ID,
        "run_timestamp": RUN_TIMESTAMP,
    }

    optional_cols = [
        "processing_order",
        "image_id",
        "file_name",
        "stem",
        "ext",
        "ext_lower",
        "relative_image_path",
        "image_path",
        "source_image_path",
        "source_root",
        "bronze_root",
    ]

    for col_name in optional_cols:
        value = _cell14_get_row_value(
            source_row,
            col_name,
            default=None,
        )

        if value is not None:
            lineage[col_name] = value

    return lineage


# =============================================================================
# 14.14 SELECTED-POLE MASK LOOKUP KEY HELPER
# =============================================================================
# EXPLANATION:
# CELL 13 stores selected-pole masks in pole_mask_lookup using:
#
#   (image_id, prompt, det_idx)
#
# CELL 14 uses the same key format to record whether the selected-pole mask is
# available for downstream cells.
# =============================================================================

def _cell14_candidate_key(image_id, prompt, det_idx):
    """
    Build the selected-pole mask lookup key used by CELL 13.

    Args:
        image_id:
            Stable image identifier.

        prompt:
            Pole prompt used for the selected detection.

        det_idx:
            Detection index within the prompt run.

    Returns:
        tuple:
            (image_id, prompt, det_idx)
    """
    return (
        str(image_id),
        str(prompt),
        int(det_idx),
    )


# =============================================================================
# 14.15 ROI OUTPUT PATH HELPER
# =============================================================================
# EXPLANATION:
# Build the saved ROI output path under the run-scoped CELL 14 image folder.
#
# Output pattern:
#
#   SILVER_POLE_ROIS/images/<RUN_ID>/<relative_folder>/
#       <safe_original_stem>__<safe_image_id>__pole_roi.png
#
# IMPORTANT:
#   ROI filenames are intentionally image_id-based.
#
#   The original stem is kept only for readability. The collision protection
#   comes from image_id, which CELL 12 already made deterministic and unique.
# =============================================================================

def build_roi_output_path(row):
    """
    Build the saved ROI output path.

    Args:
        row:
            Selected pole row from selected_poles_df.

    Returns:
        dict:
            Output path metadata containing:
              - roi_file_name
              - roi_relative_dir
              - roi_image_path
    """
    image_id = _cell14_get_row_value(
        row,
        "image_id",
        default=None,
    )

    if image_id is None or len(str(image_id).strip()) == 0:
        raise ValueError(
            "Selected pole row is missing image_id. "
            "CELL 14 requires image_id-based ROI filenames."
        )

    relative_image_path = _cell14_get_row_value(
        row,
        "relative_image_path",
        default=None,
    )

    fallback_file_name = _cell14_get_row_value(
        row,
        "file_name",
        default="image",
    )

    if not isinstance(relative_image_path, str) or len(relative_image_path.strip()) == 0:
        relative_image_path = fallback_file_name

    relative_text = str(relative_image_path).replace("\\", "/")

    relative_dir = os.path.dirname(relative_text)
    base_stem = os.path.splitext(
        os.path.basename(relative_text)
    )[0]

    safe_dir_parts = []

    if relative_dir not in ["", "."]:
        for part in relative_dir.split("/"):
            if part in ["", ".", ".."]:
                continue

            safe_dir_parts.append(
                make_safe_path_part(
                    part,
                    fallback="folder",
                )
            )

    safe_base_stem = make_safe_path_part(
        base_stem,
        fallback="image",
    )

    safe_image_id = make_safe_path_part(
        image_id,
        fallback="unknown_image",
    )

    target_dir = os.path.join(
        RUN_SILVER_POLE_ROIS_IMAGES_DIR,
        *safe_dir_parts,
    )

    os.makedirs(
        target_dir,
        exist_ok=True,
    )

    roi_file_name = (
        f"{safe_base_stem}__{safe_image_id}__pole_roi.png"
    )

    roi_image_path = os.path.join(
        target_dir,
        roi_file_name,
    )

    roi_relative_dir = (
        "/".join(safe_dir_parts)
        if len(safe_dir_parts) > 0
        else ""
    )

    return {
        "roi_file_name": roi_file_name,
        "roi_relative_dir": roi_relative_dir,
        "roi_image_path": roi_image_path,
    }


# =============================================================================
# 14.16 SHIFT A FIXED BOX INSIDE THE IMAGE WHEN POSSIBLE
# =============================================================================
# EXPLANATION:
# This preserves the tested development crop geometry.
#
# IMPORTANT:
#   Do not change this shift logic unless intentionally retuning the ROI crop
#   strategy. This is the working shift-first / pad-second behaviour.
# =============================================================================

def shift_box_inside_image(x1, y1, box_w, box_h, image_w, image_h):
    """
    Shift a fixed-size box inside the image when possible, while keeping the
    box size unchanged.

    Args:
        x1, y1:
            Requested top-left corner.

        box_w, box_h:
            Fixed box dimensions.

        image_w, image_h:
            Image dimensions.

    Returns:
        dict:
            Shifted fixed-size box coordinates.
    """
    x1 = int(round(x1))
    y1 = int(round(y1))
    box_w = int(box_w)
    box_h = int(box_h)

    x2 = x1 + box_w
    y2 = y1 + box_h

    if image_w >= box_w:
        if x1 < 0:
            x2 += (-x1)
            x1 = 0
        if x2 > image_w:
            x1 -= (x2 - image_w)
            x2 = image_w

    if image_h >= box_h:
        if y1 < 0:
            y2 += (-y1)
            y1 = 0
        if y2 > image_h:
            y1 -= (y2 - image_h)
            y2 = image_h

    # Recompute the far corner from the final fixed-size origin.
    x2 = x1 + box_w
    y2 = y1 + box_h

    return {
        "req_x1": int(x1),
        "req_y1": int(y1),
        "req_x2": int(x2),
        "req_y2": int(y2),
        "req_w": int(box_w),
        "req_h": int(box_h),
    }


# =============================================================================
# 14.17 BUILD THE FIXED POLE-TOP ROI REQUEST
# =============================================================================
# EXPLANATION:
# This preserves the tested development crop geometry.
#
# The ROI is centred horizontally on the selected pole centre and starts above
# the selected pole box by POLE_TOP_BUFFER_ABOVE pixels.
#
# IMPORTANT:
#   This intentionally reads image_w / image_h from the selected-pole row,
#   matching the old working crop behaviour.
#
#   The validation here does not change the crop math. It only ensures bad
#   selected-pole rows fail cleanly inside the CELL 14C per-row try/except.
# =============================================================================

def build_pole_top_roi_request(row):
    """
    Build a fixed-size pole-top ROI request from the selected pole row.

    Args:
        row:
            Selected pole row from selected_poles_df.

    Returns:
        dict:
            Requested fixed-size ROI geometry after shift-to-fit.
    """

    def _required_finite_float(column_name):
        value = _cell14_get_row_value(
            row,
            column_name,
            default=None,
        )

        try:
            value_float = float(value)
        except Exception as exc:
            raise ValueError(
                f"Selected pole row has invalid numeric value for {column_name}: "
                f"{value}"
            ) from exc

        if not np.isfinite(value_float):
            raise ValueError(
                f"Selected pole row has non-finite value for {column_name}: "
                f"{value}"
            )

        return value_float

    def _required_positive_int(column_name):
        value = _cell14_get_row_value(
            row,
            column_name,
            default=None,
        )

        try:
            value_int = int(value)
        except Exception as exc:
            raise ValueError(
                f"Selected pole row has invalid integer value for {column_name}: "
                f"{value}"
            ) from exc

        if value_int <= 0:
            raise ValueError(
                f"Selected pole row has non-positive value for {column_name}: "
                f"{value}"
            )

        return value_int

    x1 = _required_finite_float("x1")
    y1 = _required_finite_float("y1")
    x2 = _required_finite_float("x2")
    y2 = _required_finite_float("y2")

    image_w = _required_positive_int("image_w")
    image_h = _required_positive_int("image_h")

    pole_w = max(x2 - x1, 1.0)
    pole_h = max(y2 - y1, 1.0)

    pole_cx_value = _cell14_get_row_value(
        row,
        "pole_cx",
        default=None,
    )

    if pole_cx_value is not None and pd.notna(pole_cx_value):
        pole_cx = float(pole_cx_value)

        if not np.isfinite(pole_cx):
            pole_cx = (x1 + x2) / 2.0
    else:
        pole_cx = (x1 + x2) / 2.0

    req_x1 = pole_cx - (FIXED_ROI_WIDTH / 2.0)
    req_y1 = y1 - POLE_TOP_BUFFER_ABOVE

    shifted = shift_box_inside_image(
        x1=req_x1,
        y1=req_y1,
        box_w=FIXED_ROI_WIDTH,
        box_h=FIXED_ROI_HEIGHT,
        image_w=image_w,
        image_h=image_h,
    )

    return {
        "pole_w": float(pole_w),
        "pole_h": float(pole_h),
        "pole_cx_used": float(pole_cx),
        "req_x1": int(shifted["req_x1"]),
        "req_y1": int(shifted["req_y1"]),
        "req_x2": int(shifted["req_x2"]),
        "req_y2": int(shifted["req_y2"]),
        "req_w": int(shifted["req_w"]),
        "req_h": int(shifted["req_h"]),
    }
    

# =============================================================================
# 14.18 RENDER A FIXED-SIZE CANVAS FROM THE REQUESTED ROI
# =============================================================================
# EXPLANATION:
# This preserves the tested development crop geometry.
#
# The function:
#   1) crops the overlap between the requested ROI and the source image
#   2) pastes that overlap onto a fixed-size RGB canvas
#   3) pads only where the requested ROI extends outside the source image
#
# IMPORTANT:
#   This function creates a clean crop only.
#   It does NOT draw the pole mask, boxes, labels, or overlays.
# =============================================================================

def render_fixed_canvas_roi(image_pil, roi_request):
    """
    Render the final fixed-size ROI canvas by cropping the overlapping source
    region and pasting it onto a fixed-size canvas.

    Args:
        image_pil:
            Source PIL RGB image.

        roi_request:
            Dict from build_pole_top_roi_request.

    Returns:
        dict:
            Final crop details and the fixed-size PIL canvas.
    """
    image_w, image_h = image_pil.size

    req_x1 = int(roi_request["req_x1"])
    req_y1 = int(roi_request["req_y1"])
    req_x2 = int(roi_request["req_x2"])
    req_y2 = int(roi_request["req_y2"])

    src_x1 = max(0, req_x1)
    src_y1 = max(0, req_y1)
    src_x2 = min(image_w, req_x2)
    src_y2 = min(image_h, req_y2)

    overlap_w = max(0, src_x2 - src_x1)
    overlap_h = max(0, src_y2 - src_y1)

    dst_x1 = max(0, src_x1 - req_x1)
    dst_y1 = max(0, src_y1 - req_y1)

    roi_canvas = Image.new("RGB", (FIXED_ROI_WIDTH, FIXED_ROI_HEIGHT), PAD_RGB)

    if overlap_w > 0 and overlap_h > 0:
        src_crop = image_pil.crop((src_x1, src_y1, src_x2, src_y2))
        roi_canvas.paste(src_crop, (dst_x1, dst_y1))

    pad_left = int(max(0, -req_x1))
    pad_top = int(max(0, -req_y1))
    pad_right = int(max(0, req_x2 - image_w))
    pad_bottom = int(max(0, req_y2 - image_h))

    return {
        "roi_canvas": roi_canvas,
        "src_x1": int(src_x1),
        "src_y1": int(src_y1),
        "src_x2": int(src_x2),
        "src_y2": int(src_y2),
        "src_w": int(overlap_w),
        "src_h": int(overlap_h),
        "dst_x1": int(dst_x1),
        "dst_y1": int(dst_y1),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "pad_right": int(pad_right),
        "pad_bottom": int(pad_bottom),
        "was_padded": bool(
            (pad_left > 0) or (pad_top > 0) or
            (pad_right > 0) or (pad_bottom > 0)
        ),
    }


# =============================================================================
# 14.19 POLE ROI FAILURE ROW HELPER
# =============================================================================
# EXPLANATION:
# Build one failure row when a selected pole fails during ROI generation.
#
# This allows CELL 14C to continue processing the remaining selected poles
# instead of failing the whole batch.
# =============================================================================

def _make_pole_roi_failure_row(source_row, error, stage="pole_roi_generation"):
    """
    Build one pole ROI failure row.

    Args:
        source_row:
            Selected-pole row from selected_poles_df.

        error:
            Exception raised while creating the ROI.

        stage:
            Logical processing stage where the error occurred.

    Returns:
        dict:
            One failure row.
    """
    row_out = _cell14_build_lineage_fields(
        source_row=source_row,
    )

    row_out.update({
        "selection_status": _cell14_get_row_value(
            source_row,
            "selection_status",
            default=None,
        ),
        "selection_mode": _cell14_get_row_value(
            source_row,
            "selection_mode",
            default=None,
        ),
        "prompt": _cell14_get_row_value(
            source_row,
            "prompt",
            default=None,
        ),
        "det_idx": _cell14_get_row_value(
            source_row,
            "det_idx",
            default=None,
        ),
        "failure_stage": str(stage),
        "error_type": type(error).__name__,
        "error_message": str(error),
    })

    if "traceback" in globals():
        try:
            row_out["error_traceback"] = traceback.format_exc()
        except Exception:
            row_out["error_traceback"] = None
    else:
        row_out["error_traceback"] = None

    return row_out


# =============================================================================
# C. ROI GENERATION LOOP
# =============================================================================


# =============================================================================
# 14.20 RUN FIXED-CANVAS ROI GENERATION
# =============================================================================
# EXPLANATION:
# This section creates one clean fixed-size pole-top ROI image for each selected
# pole row from CELL 13.
#
# Production behaviour:
#   - source image comes from pole_selection_df["image_path"]
#   - crop geometry comes from full-resolution selected-pole coordinates
#   - saved ROI images are clean RGB crops only
#   - row-level failures are captured in roi_failure_rows
#   - one bad selected-pole row does not stop the full batch unless
#     CELL14_STOP_ON_ROW_FAILURE=True
#
# IMPORTANT:
#   This section only fills accumulators:
#
#       roi_rows
#       roi_failure_rows
#
#   DataFrame assembly, reconciliation, file verification, and saving happen in
#   CELL 14D.
# =============================================================================

if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print(
        f"\nCreating fixed pole-top ROI crops for "
        f"{len(selected_poles_df)} selected pole row(s)..."
    )

for row_idx in range(len(selected_poles_df)):
    row = selected_poles_df.iloc[row_idx]

    try:
        # ---------------------------------------------------------------------
        # 14.20.1 Read core selected-pole lineage values
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # These values come from pole_selection_df created by CELL 13.
        # ---------------------------------------------------------------------
        source_lineage = _cell14_build_lineage_fields(
            source_row=row,
        )

        image_id = _cell14_get_row_value(
            row,
            "image_id",
            default=None,
        )

        file_name = _cell14_get_row_value(
            row,
            "file_name",
            default=None,
        )

        image_path = _cell14_get_row_value(
            row,
            "image_path",
            default=None,
        )

        relative_image_path = _cell14_get_row_value(
            row,
            "relative_image_path",
            default=None,
        )

        if image_id is None or len(str(image_id).strip()) == 0:
            raise ValueError(
                "Selected pole row is missing a valid image_id."
            )

        if (
            image_path is None
            or pd.isna(image_path)
            or not isinstance(image_path, str)
            or len(image_path.strip()) == 0
        ):
            raise ValueError(
                f"Selected pole row is missing a valid image_path. "
                f"image_id={image_id}"
            )

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Original source image not found for image_id={image_id}: "
                f"{image_path}"
            )

        if (
            file_name is None
            or pd.isna(file_name)
            or not isinstance(file_name, str)
            or len(file_name.strip()) == 0
        ):
            file_name = os.path.basename(image_path)

        # ---------------------------------------------------------------------
        # 14.20.2 Read selected-pole identity and score fields
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # These fields allow CELL 14 outputs to join back to CELL 13 selected
        # poles and pole_mask_lookup.
        # ---------------------------------------------------------------------
        selection_mode = _cell14_get_row_value(
            row,
            "selection_mode",
            default=None,
        )

        prompt = _cell14_get_row_value(
            row,
            "prompt",
            default=None,
        )

        det_idx_value = _cell14_get_row_value(
            row,
            "det_idx",
            default=None,
        )

        if det_idx_value is None or pd.isna(det_idx_value):
            raise ValueError(
                f"Selected pole row is missing det_idx. image_id={image_id}"
            )

        det_idx = int(det_idx_value)

        score_value = _cell14_get_row_value(
            row,
            "score",
            default=np.nan,
        )

        final_score_value = _cell14_get_row_value(
            row,
            "final_score",
            default=np.nan,
        )

        has_mask_value = _cell14_get_row_value(
            row,
            "has_mask",
            default=False,
        )

        raw_score = (
            float(score_value)
            if pd.notna(score_value)
            else np.nan
        )

        final_score = (
            float(final_score_value)
            if pd.notna(final_score_value)
            else np.nan
        )

        has_mask = (
            bool(has_mask_value)
            if pd.notna(has_mask_value)
            else False
        )

        mask_lookup_hit = False

        if (
            isinstance(pole_mask_lookup, dict)
            and prompt is not None
            and pd.notna(prompt)
        ):
            mask_lookup_hit = (
                _cell14_candidate_key(
                    image_id,
                    prompt,
                    det_idx,
                )
                in pole_mask_lookup
            )

        # ---------------------------------------------------------------------
        # 14.20.3 Build ROI request and output path
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # build_pole_top_roi_request preserves the tested fixed-canvas crop
        # geometry.
        #
        # build_roi_output_path uses the production image_id-based filename
        # scheme and writes under the run-scoped ROI image folder.
        # ---------------------------------------------------------------------
        roi_request = build_pole_top_roi_request(
            row,
        )

        roi_path_meta = build_roi_output_path(
            row,
        )

        roi_file_name = roi_path_meta["roi_file_name"]
        roi_relative_dir = roi_path_meta["roi_relative_dir"]
        roi_image_path = roi_path_meta["roi_image_path"]

        # ---------------------------------------------------------------------
        # 14.20.4 Render and save clean fixed-canvas ROI crop
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # The saved ROI is a clean RGB crop only.
        # It has no pole mask, no box, no label, and no QA overlay graphics.
        #
        # IMPORTANT:
        #   Image.open(...) closes the source image automatically because it is
        #   inside a with-block.
        #
        #   img_source.convert("RGB") creates a separate PIL image object.
        #   We close that converted RGB image explicitly in a finally block so
        #   failed ROI rows do not slowly leak decoded image buffers.
        # ---------------------------------------------------------------------
        img_rgb = None
        roi_canvas = None
        roi_render = None

        with Image.open(image_path) as img_source:
            try:
                img_rgb = img_source.convert("RGB")

                roi_render = render_fixed_canvas_roi(
                    image_pil=img_rgb,
                    roi_request=roi_request,
                )

                roi_canvas = roi_render["roi_canvas"]

                os.makedirs(
                    os.path.dirname(roi_image_path),
                    exist_ok=True,
                )

                roi_canvas.save(
                    roi_image_path,
                    format="PNG",
                )

            finally:
                if roi_canvas is not None:
                    try:
                        roi_canvas.close()
                    except Exception:
                        pass

                if img_rgb is not None:
                    try:
                        img_rgb.close()
                    except Exception:
                        pass

        # ---------------------------------------------------------------------
        # 14.20.5 Build one successful ROI output row
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # The row carries:
        #   - CELL 13 lineage
        #   - selected-pole identity and geometry
        #   - fixed ROI request geometry
        #   - source overlap / padding geometry
        #   - saved ROI path metadata
        # ---------------------------------------------------------------------
        roi_row = source_lineage.copy()

        roi_row.update({
            # -----------------------------------------------------------------
            # CELL 13 selected-pole identity
            # -----------------------------------------------------------------
            "image_id": image_id,
            "file_name": file_name,
            "relative_image_path": relative_image_path,
            "image_path": image_path,
            "selection_status": "selected",
            "selection_mode": selection_mode,
            "prompt": prompt,
            "det_idx": det_idx,
            "score": raw_score,
            "final_score": final_score,
            "has_mask": has_mask,
            "mask_lookup_hit": bool(mask_lookup_hit),

            # -----------------------------------------------------------------
            # Full source image geometry
            # -----------------------------------------------------------------
            "image_w": int(_cell14_get_row_value(row, "image_w")),
            "image_h": int(_cell14_get_row_value(row, "image_h")),

            # -----------------------------------------------------------------
            # Selected pole full-resolution geometry
            # -----------------------------------------------------------------
            "x1": float(_cell14_get_row_value(row, "x1")),
            "y1": float(_cell14_get_row_value(row, "y1")),
            "x2": float(_cell14_get_row_value(row, "x2")),
            "y2": float(_cell14_get_row_value(row, "y2")),
            "box_w": (
                float(_cell14_get_row_value(row, "box_w"))
                if pd.notna(_cell14_get_row_value(row, "box_w"))
                else float(roi_request["pole_w"])
            ),
            "box_h": (
                float(_cell14_get_row_value(row, "box_h"))
                if pd.notna(_cell14_get_row_value(row, "box_h"))
                else float(roi_request["pole_h"])
            ),
            "pole_cx": (
                float(_cell14_get_row_value(row, "pole_cx"))
                if pd.notna(_cell14_get_row_value(row, "pole_cx"))
                else np.nan
            ),
            "pole_cy": (
                float(_cell14_get_row_value(row, "pole_cy"))
                if pd.notna(_cell14_get_row_value(row, "pole_cy"))
                else np.nan
            ),
            "pole_w": float(roi_request["pole_w"]),
            "pole_h": float(roi_request["pole_h"]),
            "pole_cx_used": float(roi_request["pole_cx_used"]),

            # -----------------------------------------------------------------
            # Requested fixed ROI geometry
            # -----------------------------------------------------------------
            "req_x1": int(roi_request["req_x1"]),
            "req_y1": int(roi_request["req_y1"]),
            "req_x2": int(roi_request["req_x2"]),
            "req_y2": int(roi_request["req_y2"]),
            "req_w": int(roi_request["req_w"]),
            "req_h": int(roi_request["req_h"]),

            # -----------------------------------------------------------------
            # Actual source overlap / paste geometry
            # -----------------------------------------------------------------
            "src_x1": int(roi_render["src_x1"]),
            "src_y1": int(roi_render["src_y1"]),
            "src_x2": int(roi_render["src_x2"]),
            "src_y2": int(roi_render["src_y2"]),
            "src_w": int(roi_render["src_w"]),
            "src_h": int(roi_render["src_h"]),
            "dst_x1": int(roi_render["dst_x1"]),
            "dst_y1": int(roi_render["dst_y1"]),
            "pad_left": int(roi_render["pad_left"]),
            "pad_top": int(roi_render["pad_top"]),
            "pad_right": int(roi_render["pad_right"]),
            "pad_bottom": int(roi_render["pad_bottom"]),
            "was_padded": bool(roi_render["was_padded"]),

            # -----------------------------------------------------------------
            # Saved fixed-canvas ROI output
            # -----------------------------------------------------------------
            "roi_w": int(FIXED_ROI_WIDTH),
            "roi_h": int(FIXED_ROI_HEIGHT),
            "roi_file_name": roi_file_name,
            "roi_relative_dir": roi_relative_dir,
            "roi_image_path": roi_image_path,
            "roi_output_layer": "silver",
            "roi_images_dir": RUN_SILVER_POLE_ROIS_IMAGES_DIR,
        })

        roi_rows.append(
            roi_row,
        )

        # ---------------------------------------------------------------------
        # 14.20.6 Optional progress logging
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # Keep production logs compact. Print only the first row, every 20 rows,
        # and the final row when PRINT_CONFIG_SUMMARY is enabled.
        # ---------------------------------------------------------------------
        if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
            completed_count = len(roi_rows) + len(roi_failure_rows)

            if (
                (completed_count == 1)
                or (completed_count % 20 == 0)
                or (completed_count == len(selected_poles_df))
            ):
                print(
                    f"  [{completed_count}/{len(selected_poles_df)}] "
                    f"saved {roi_file_name}"
                )

        del roi_canvas
        del roi_render

    except Exception as exc:
        # ---------------------------------------------------------------------
        # 14.20.7 Per-row failure capture
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # A single ROI crop failure should not stop the full production batch.
        # The failure row keeps lineage so the failed image can be traced back
        # and reviewed later.
        # ---------------------------------------------------------------------
        roi_failure_rows.append(
            _make_pole_roi_failure_row(
                source_row=row,
                error=exc,
                stage="pole_roi_generation",
            )
        )

        if CELL14_STOP_ON_ROW_FAILURE:
            raise

        continue


# =============================================================================
# 14.21 LOOP CLEANUP + SUMMARY
# =============================================================================
# EXPLANATION:
# Free temporary objects after the crop loop.
#
# Final reconciliation and save checks happen in CELL 14D.
# =============================================================================

gc.collect()

if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print("\nCELL 14C ROI generation loop complete.")

    print("=" * 90)
    print("CELL 14C — ROI GENERATION LOOP SUMMARY")
    print("=" * 90)
    print(f"Selected pole rows processed         : {len(selected_poles_df)}")
    print(f"Successful ROI rows                  : {len(roi_rows)}")
    print(f"Failed ROI rows                      : {len(roi_failure_rows)}")
    print(f"Clean ROI image folder               : {RUN_SILVER_POLE_ROIS_IMAGES_DIR}")
    
    
# =============================================================================
# D. OUTPUT MANIFEST + SAVE
# =============================================================================


# =============================================================================
# 14.22 ASSEMBLE ROI OUTPUT DATAFRAMES
# =============================================================================
# EXPLANATION:
# CELL 14C filled lightweight Python accumulators:
#
#   roi_rows:
#       One row per successfully saved clean ROI crop.
#
#   roi_failure_rows:
#       One row per selected pole that failed during ROI crop generation.
#
# This section converts those accumulators into production DataFrames.
# =============================================================================

roi_front_cols = [
    # -------------------------------------------------------------------------
    # Run lineage
    # -------------------------------------------------------------------------
    "run_id",
    "run_timestamp",
    "processing_order",

    # -------------------------------------------------------------------------
    # Source image lineage
    # -------------------------------------------------------------------------
    "image_id",
    "file_name",
    "stem",
    "ext",
    "ext_lower",
    "relative_image_path",
    "image_path",
    "source_image_path",
    "source_root",
    "bronze_root",

    # -------------------------------------------------------------------------
    # CELL 13 selected-pole identity
    # -------------------------------------------------------------------------
    "selection_status",
    "selection_mode",
    "prompt",
    "det_idx",
    "score",
    "final_score",
    "has_mask",
    "mask_lookup_hit",

    # -------------------------------------------------------------------------
    # Full source image geometry
    # -------------------------------------------------------------------------
    "image_w",
    "image_h",

    # -------------------------------------------------------------------------
    # Selected pole full-resolution geometry
    # -------------------------------------------------------------------------
    "x1",
    "y1",
    "x2",
    "y2",
    "box_w",
    "box_h",
    "pole_cx",
    "pole_cy",
    "pole_w",
    "pole_h",
    "pole_cx_used",

    # -------------------------------------------------------------------------
    # Requested fixed ROI geometry
    # -------------------------------------------------------------------------
    "req_x1",
    "req_y1",
    "req_x2",
    "req_y2",
    "req_w",
    "req_h",

    # -------------------------------------------------------------------------
    # Actual source overlap / paste geometry
    # -------------------------------------------------------------------------
    "src_x1",
    "src_y1",
    "src_x2",
    "src_y2",
    "src_w",
    "src_h",
    "dst_x1",
    "dst_y1",
    "pad_left",
    "pad_top",
    "pad_right",
    "pad_bottom",
    "was_padded",

    # -------------------------------------------------------------------------
    # Saved fixed-canvas ROI output
    # -------------------------------------------------------------------------
    "roi_w",
    "roi_h",
    "roi_file_name",
    "roi_relative_dir",
    "roi_image_path",
    "roi_output_layer",
    "roi_images_dir",
]

roi_failure_front_cols = [
    # -------------------------------------------------------------------------
    # Run lineage
    # -------------------------------------------------------------------------
    "run_id",
    "run_timestamp",
    "processing_order",

    # -------------------------------------------------------------------------
    # Source image lineage
    # -------------------------------------------------------------------------
    "image_id",
    "file_name",
    "stem",
    "ext",
    "ext_lower",
    "relative_image_path",
    "image_path",
    "source_image_path",
    "source_root",
    "bronze_root",

    # -------------------------------------------------------------------------
    # Selected-pole identity
    # -------------------------------------------------------------------------
    "selection_status",
    "selection_mode",
    "prompt",
    "det_idx",

    # -------------------------------------------------------------------------
    # Failure details
    # -------------------------------------------------------------------------
    "failure_stage",
    "error_type",
    "error_message",
    "error_traceback",
]


pole_rois_df = (
    pd.DataFrame(roi_rows)
    if len(roi_rows) > 0
    else pd.DataFrame(columns=roi_front_cols)
)

pole_roi_failures_df = (
    pd.DataFrame(roi_failure_rows)
    if len(roi_failure_rows) > 0
    else pd.DataFrame(columns=roi_failure_front_cols)
)


# =============================================================================
# 14.23 REORDER OUTPUT COLUMNS
# =============================================================================
# EXPLANATION:
# Keep production lineage and decision fields at the front, while preserving any
# extra columns that may be added later.
# =============================================================================

def _cell14_reorder_columns(df, front_cols):
    """
    Reorder a DataFrame by placing preferred columns first.

    Args:
        df:
            pandas DataFrame to reorder.

        front_cols:
            Preferred column order.

    Returns:
        pandas.DataFrame:
            Reordered DataFrame.
    """
    if df is None:
        return pd.DataFrame(columns=front_cols)

    existing_front_cols = [
        col_name
        for col_name in front_cols
        if col_name in df.columns
    ]

    remaining_cols = [
        col_name
        for col_name in df.columns
        if col_name not in existing_front_cols
    ]

    return df[
        existing_front_cols + remaining_cols
    ]


pole_rois_df = _cell14_reorder_columns(
    pole_rois_df,
    roi_front_cols,
)

pole_roi_failures_df = _cell14_reorder_columns(
    pole_roi_failures_df,
    roi_failure_front_cols,
)


# =============================================================================
# 14.24 FINAL PRODUCTION CONSISTENCY CHECKS
# =============================================================================
# EXPLANATION:
# CELL 14 allows per-row ROI failures.
#
# Therefore, the correct reconciliation is:
#
#   successful ROI rows + ROI failure rows == selected pole rows
#
# not:
#
#   pole_rois_df rows == selected_poles_df rows
#
# This mirrors the production robustness pattern used in CELL 13.
# =============================================================================

selected_pole_row_count = int(len(selected_poles_df))
roi_success_row_count = int(len(pole_rois_df))
roi_failure_row_count = int(len(pole_roi_failures_df))

processed_roi_outcome_count = (
    roi_success_row_count + roi_failure_row_count
)

if processed_roi_outcome_count != selected_pole_row_count:
    raise RuntimeError(
        "CELL 14 output rows do not reconcile with selected poles.\n"
        "Each selected pole must have either one pole_rois_df row or one "
        "pole_roi_failures_df row.\n"
        f"selected_poles_df rows      : {selected_pole_row_count}\n"
        f"pole_rois_df rows           : {roi_success_row_count}\n"
        f"pole_roi_failures_df rows   : {roi_failure_row_count}\n"
        f"combined outcome rows       : {processed_roi_outcome_count}"
    )


# -----------------------------------------------------------------------------
# Validate selected image_id coverage.
# -----------------------------------------------------------------------------
selected_image_ids = set(
    selected_poles_df["image_id"]
    .dropna()
    .astype(str)
    .tolist()
)

roi_success_image_ids = (
    set(
        pole_rois_df["image_id"]
        .dropna()
        .astype(str)
        .tolist()
    )
    if "image_id" in pole_rois_df.columns
    else set()
)

roi_failure_image_ids = (
    set(
        pole_roi_failures_df["image_id"]
        .dropna()
        .astype(str)
        .tolist()
    )
    if "image_id" in pole_roi_failures_df.columns
    else set()
)

roi_outcome_image_ids = (
    roi_success_image_ids | roi_failure_image_ids
)

missing_roi_outcome_image_ids = sorted(
    selected_image_ids - roi_outcome_image_ids
)

unexpected_roi_outcome_image_ids = sorted(
    roi_outcome_image_ids - selected_image_ids
)

if missing_roi_outcome_image_ids:
    raise RuntimeError(
        "Some selected poles have no CELL 14 outcome row.\n"
        f"Missing image_id examples: {missing_roi_outcome_image_ids[:10]}"
    )

if unexpected_roi_outcome_image_ids:
    raise RuntimeError(
        "Some CELL 14 outcome rows do not match selected_poles_df.\n"
        f"Unexpected image_id examples: {unexpected_roi_outcome_image_ids[:10]}"
    )


# -----------------------------------------------------------------------------
# Validate duplicate successful ROI outputs.
# -----------------------------------------------------------------------------
if (
    not pole_rois_df.empty
    and "image_id" in pole_rois_df.columns
    and pole_rois_df["image_id"].duplicated().any()
):
    duplicate_roi_image_ids = (
        pole_rois_df
        .loc[pole_rois_df["image_id"].duplicated(), "image_id"]
        .astype(str)
        .tolist()
    )

    raise RuntimeError(
        "pole_rois_df contains duplicate image_id values.\n"
        f"Duplicate examples: {duplicate_roi_image_ids[:10]}"
    )

if (
    not pole_roi_failures_df.empty
    and "image_id" in pole_roi_failures_df.columns
    and pole_roi_failures_df["image_id"].duplicated().any()
):
    duplicate_roi_failure_ids = (
        pole_roi_failures_df
        .loc[pole_roi_failures_df["image_id"].duplicated(), "image_id"]
        .astype(str)
        .tolist()
    )

    raise RuntimeError(
        "pole_roi_failures_df contains duplicate image_id values.\n"
        f"Duplicate examples: {duplicate_roi_failure_ids[:10]}"
    )

overlap_roi_image_ids = sorted(
    roi_success_image_ids & roi_failure_image_ids
)

if overlap_roi_image_ids:
    raise RuntimeError(
        "Some images appear in both pole_rois_df and pole_roi_failures_df.\n"
        f"Overlapping image_id examples: {overlap_roi_image_ids[:10]}"
    )


# -----------------------------------------------------------------------------
# Validate duplicate ROI file paths.
# -----------------------------------------------------------------------------
if (
    not pole_rois_df.empty
    and "roi_image_path" in pole_rois_df.columns
    and pole_rois_df["roi_image_path"].duplicated().any()
):
    duplicate_roi_paths = (
        pole_rois_df
        .loc[pole_rois_df["roi_image_path"].duplicated(), "roi_image_path"]
        .astype(str)
        .tolist()
    )

    raise RuntimeError(
        "pole_rois_df contains duplicate roi_image_path values.\n"
        f"Duplicate examples: {duplicate_roi_paths[:10]}"
    )


# =============================================================================
# 14.25 VERIFY SAVED ROI IMAGE FILES EXIST
# =============================================================================
# EXPLANATION:
# Confirm that each successful ROI row points to a saved PNG on disk.
#
# This preserves the old dev-cell check, but now applies it only to successful
# ROI rows.
# =============================================================================

missing_roi_files = []

if not pole_rois_df.empty and "roi_image_path" in pole_rois_df.columns:
    missing_roi_files = [
        roi_path
        for roi_path in pole_rois_df["roi_image_path"].dropna().astype(str).tolist()
        if not os.path.exists(roi_path)
    ]

if len(missing_roi_files) > 0:
    raise RuntimeError(
        "Some ROI files were expected but were not found on disk.\n"
        f"Missing files: {missing_roi_files[:10]}"
    )


# =============================================================================
# 14.26 BUILD OUTPUT COUNT SUMMARY
# =============================================================================
# EXPLANATION:
# Keep a compact summary object for later audit/debug checks.
# =============================================================================

cell14_output_counts = {
    "selected_pole_rows": int(selected_pole_row_count),
    "successful_roi_rows": int(roi_success_row_count),
    "failed_roi_rows": int(roi_failure_row_count),
    "combined_roi_outcome_rows": int(processed_roi_outcome_count),
    "padded_roi_rows": (
        int(pole_rois_df["was_padded"].sum())
        if (
            not pole_rois_df.empty
            and "was_padded" in pole_rois_df.columns
        )
        else 0
    ),
    "mask_lookup_hit_rows": (
        int(pole_rois_df["mask_lookup_hit"].sum())
        if (
            not pole_rois_df.empty
            and "mask_lookup_hit" in pole_rois_df.columns
        )
        else 0
    ),
}


# =============================================================================
# 14.27 SAVE CELL 14 OUTPUT TABLES
# =============================================================================
# EXPLANATION:
# Save CELL 14 production tables using save_run_table() from CELL 10B.
#
# IMPORTANT:
#   - Do not use save_state().
#   - ROI image PNGs were already saved in CELL 14C.
#   - This section only saves Parquet tables.
# =============================================================================

SAVE_POLE_ROIS_TABLE = bool(
    globals().get(
        "SAVE_POLE_ROIS_TABLE",
        True,
    )
)

SAVE_POLE_ROI_FAILURES_TABLE = bool(
    globals().get(
        "SAVE_POLE_ROI_FAILURES_TABLE",
        True,
    )
)

cell14_saved_paths = {}

if SAVE_POLE_ROIS_TABLE:
    cell14_saved_paths["pole_rois"] = save_run_table(
        pole_rois_df,
        RUN_SILVER_POLE_ROIS_TABLES_DIR,
        "pole_rois",
    )
else:
    cell14_saved_paths["pole_rois"] = None

if SAVE_POLE_ROI_FAILURES_TABLE:
    cell14_saved_paths["pole_roi_failures"] = save_run_table(
        pole_roi_failures_df,
        RUN_SILVER_POLE_ROIS_TABLES_DIR,
        "pole_roi_failures",
    )
else:
    cell14_saved_paths["pole_roi_failures"] = None


# =============================================================================
# 14.28 VALIDATE SAVED TABLE OUTPUTS
# =============================================================================
# EXPLANATION:
# Validate required saved Parquet files when the corresponding table is non-empty
# and saving is enabled.
#
# save_run_table() intentionally skips empty tables, so empty failure tables are
# allowed to have no saved path.
# =============================================================================

if (
    SAVE_POLE_ROIS_TABLE
    and not pole_rois_df.empty
):
    pole_rois_saved_path = cell14_saved_paths.get(
        "pole_rois",
        None,
    )

    if (
        pole_rois_saved_path is None
        or not isinstance(pole_rois_saved_path, str)
        or not os.path.exists(pole_rois_saved_path)
    ):
        raise RuntimeError(
            "pole_rois_df was non-empty, but its Parquet output was not found.\n"
            f"Saved path: {pole_rois_saved_path}"
        )

if (
    SAVE_POLE_ROI_FAILURES_TABLE
    and not pole_roi_failures_df.empty
):
    failures_saved_path = cell14_saved_paths.get(
        "pole_roi_failures",
        None,
    )

    if (
        failures_saved_path is None
        or not isinstance(failures_saved_path, str)
        or not os.path.exists(failures_saved_path)
    ):
        raise RuntimeError(
            "pole_roi_failures_df was non-empty, but its Parquet output was not found.\n"
            f"Saved path: {failures_saved_path}"
        )


# =============================================================================
# 14.29 FINAL CELL 14 SUMMARY
# =============================================================================
# EXPLANATION:
# Print a compact final summary when PRINT_CONFIG_SUMMARY is enabled.
# =============================================================================

if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print("\n" + "=" * 100)
    print("CELL 14 — FIXED POLE-TOP ROI FINAL SUMMARY")
    print("=" * 100)

    print(f"RUN_ID                         : {RUN_ID}")
    print(f"Selected pole rows             : {cell14_output_counts['selected_pole_rows']}")
    print(f"Successful ROI rows            : {cell14_output_counts['successful_roi_rows']}")
    print(f"Failed ROI rows                : {cell14_output_counts['failed_roi_rows']}")
    print(f"Padded ROI rows                : {cell14_output_counts['padded_roi_rows']}")
    print(f"Mask lookup hit rows           : {cell14_output_counts['mask_lookup_hit_rows']}")

    print("\nROI IMAGE OUTPUTS")
    print("-" * 100)
    print(f"ROI image folder               : {RUN_SILVER_POLE_ROIS_IMAGES_DIR}")
    print(f"Fixed ROI size                 : {FIXED_ROI_WIDTH}x{FIXED_ROI_HEIGHT}")
    print("Saved ROI type                 : clean crops only; no pole mask/box/label overlay")

    print("\nTABLE OUTPUTS")
    print("-" * 100)
    print(f"ROI table folder               : {RUN_SILVER_POLE_ROIS_TABLES_DIR}")
    print(f"SAVE_POLE_ROIS_TABLE           : {SAVE_POLE_ROIS_TABLE}")
    print(f"SAVE_POLE_ROI_FAILURES_TABLE   : {SAVE_POLE_ROI_FAILURES_TABLE}")
    print(f"pole_rois saved path           : {cell14_saved_paths.get('pole_rois')}")
    print(f"pole_roi_failures saved path   : {cell14_saved_paths.get('pole_roi_failures')}")

    print("\nCELL 14 completed.")
    print("Saved outputs:")
    print("  - pole_rois_df")
    print("  - pole_roi_failures_df")
    print("  - cell14_output_counts")
    print("  - cell14_saved_paths")
    print("  - clean ROI PNG crops under RUN_SILVER_POLE_ROIS_IMAGES_DIR")

    print("\nDownstream handoff:")
    print("  - pole_rois_df is the primary ROI manifest for CELL 16")
    print("  - pole_mask_lookup remains the selected-pole mask source from CELL 13")


# =============================================================================
# 14.30 FAIL LOUDLY IF NO ROI CROPS WERE CREATED
# =============================================================================
# EXPLANATION:
# If every selected pole failed ROI generation, CELL 14 should fail before
# downstream CELL 16 tries to run crossarm detection on an empty ROI table.
#
# This check happens after table saving so failure details are still persisted.
# =============================================================================

if pole_rois_df.empty:
    raise RuntimeError(
        "CELL 14 completed with zero successful ROI crops.\n"
        "pole_roi_failures_df has been created and saved if saving is enabled.\n"
        "Please inspect pole_roi_failures_df before running downstream cells."
    )
    
    
    
# =============================================================================
# CELL 16 — PRODUCTION CROSSARM DETECTION + XARM LABELLING
# =============================================================================
# OVERVIEW:
# This cell runs production crossarm / xarm detection over every pole-top ROI
# created by CELL 14.
#
# It starts from:
#   - pole_rois_df      : clean fixed-canvas pole-top ROI crops from CELL 14
#   - pole_mask_lookup  : selected full-resolution pole masks from CELL 13
#
# It runs SAM3 with the production crossarm prompt, applies the locked
# crossarm post-processing pipeline from the tested development CELL 16, assigns
# final xarm labels, and saves production Gold/Silver outputs.
#
# IMPORTANT PRODUCTION DESIGN RULES:
#   - Batch input is pole_rois_df, not a single debug ROI row.
#   - One pole ROI per image_id is assumed and validated.
#   - Crossarm masks are per-ROI working state.
#   - crossarm_mask_lookup must be reset inside each ROI iteration.
#   - Do not initialise one global crossarm_mask_lookup for the full run.
#   - Debug/stage visual displays are forced off for batch production.
#   - No plt.show() calls should run inside the 975-ROI batch loop.
#   - Final image saving must use CELL 10B helpers:
#         should_save_gold_final_images(...)
#         save_final_image(...)
#
# LOCKED POST-PROCESSING ORDER:
#   1) Raw SAM3 crossarm detections
#   2) Raw score prefilter
#   3) Mask-veto containment suppression
#   4) Main-cluster filtering
#   5) Pole-overlap / pole-corridor filtering
#   6) Same-crossarm continuity merge
#   7) Single-box X-shaped crossarm split
#   8) PCA / axis cleanup for non-X detections only
#   9) Targeted two-box X ownership only if needed
#   10) Final dedupe / review flags / xarm labels
#
# STRUCTURE:
#   16A. Safety checks + setup
#   16B. Helper functions, one block, namespaced where needed
#   16C. Per-ROI batch loop with full locked post-processing pipeline
#   16D. Assemble run-level output tables
#   16E. Reconciliation + consistency checks
#   16F. Save Gold/Silver outputs + final summary
#
# OUTPUTS CREATED LATER IN THIS CELL:
#   crossarm_image_results_df:
#       One row per ROI/image outcome.
#
#   crossarm_final_detections_df:
#       Final labelled xarm detections.
#
#   crossarm_trace_df:
#       Trace/audit rows explaining post-processing outcomes.
#
#   crossarm_failures_df:
#       ROI rows that failed during crossarm processing.
#
#   crossarm_stage_summary_df:
#       Optional lightweight per-ROI/per-stage count summary.
#       Saved only if Silver stage-table saving is enabled.
#
#   crossarm_saved_images_df:
#       Optional manifest of final review images saved through save_final_image().
#
# IMPORTANT:
#   - Run this after CELL 14.
#   - This cell uses RUN_ID / RUN_TIMESTAMP from CELL 10B.
#   - This cell uses save_run_table(), should_save_gold_final_images(), and
#     save_final_image() from CELL 10B.
# =============================================================================

# =============================================================================
# 16A. SAFETY CHECKS + SETUP
# =============================================================================

# =============================================================================
# 16.1 REQUIRED SETUP CHECKS
# =============================================================================
# EXPLANATION:
# Fail early if required variables from earlier production cells are missing.
#
# Required earlier cells:
#   - CELL 3A  : core libraries
#   - CELL 3B  : crossarm constants and fixed ROI constants
#   - CELL 8   : model
#   - CELL 9   : processor
#   - CELL 10B : RUN_ID, RUN_TIMESTAMP, save helpers, run-scoped Gold/Silver dirs
#   - CELL 13  : pole_mask_lookup
#   - CELL 14  : pole_rois_df
#
# IMPORTANT:
#   Visual styling constants such as CROSSARM_MASK_ALPHA, POLE_MASK_ALPHA,
#   LABEL_BG, and GRID_FIGSIZE are intentionally not required here. Production
#   visual outputs are optional and should read those later with safe defaults.
#
#   MASK_THRESHOLD is required for configuration visibility, but CELL 16 should
#   preserve the tested development mask behaviour until raw SAM3 mask dtype/range
#   has been verified before finalising 16B.
# =============================================================================

required_cell16_globals = [
    # -------------------------------------------------------------------------
    # Core libraries used by helpers and batch loop.
    # -------------------------------------------------------------------------
    "os",
    "gc",
    "time",
    "traceback",
    "pd",
    "np",
    "torch",
    "cv2",
    "math",
    "plt",
    "patches",
    "Image",

    # -------------------------------------------------------------------------
    # Production model / processor.
    # -------------------------------------------------------------------------
    "model",
    "processor",
    "DEVICE",

    # -------------------------------------------------------------------------
    # Primary CELL 16 inputs.
    # -------------------------------------------------------------------------
    "pole_rois_df",
    "pole_mask_lookup",

    # -------------------------------------------------------------------------
    # Production run identity.
    # -------------------------------------------------------------------------
    "RUN_ID",
    "RUN_TIMESTAMP",

    # -------------------------------------------------------------------------
    # Save helpers and safe naming helper from CELL 10B.
    # -------------------------------------------------------------------------
    "save_run_table",
    "save_final_image",
    "should_save_gold_final_images",
    "make_safe_path_part",

    # -------------------------------------------------------------------------
    # Run-scoped output folders from CELL 10B.
    # -------------------------------------------------------------------------
    "RUN_GOLD_TABLES_DIR",
    "RUN_GOLD_IMAGES_DIR",
    "RUN_SILVER_STAGE_TABLES_DIR",

    # -------------------------------------------------------------------------
    # Fixed ROI config from CELL 3B / CELL 14 contract.
    # -------------------------------------------------------------------------
    "FIXED_ROI_WIDTH",
    "FIXED_ROI_HEIGHT",

    # -------------------------------------------------------------------------
    # SAM3 crossarm prompt settings from CELL 3B.
    # -------------------------------------------------------------------------
    "CROSSARM_PROMPT_TEXT",
    "CROSSARM_TEXT_THRESHOLD",
    "MASK_THRESHOLD",

    # -------------------------------------------------------------------------
    # Core crossarm post-processing constants from CELL 3B.
    # -------------------------------------------------------------------------
    "CROSSARM_RAW_SCORE_REMOVE_MAX",
    "CONTAINMENT_THRESHOLD",
    "MIN_AREA_RATIO",
    "MIN_SCORE_ADVANTAGE",
    "MASK_CONTAINMENT_FILTER_ENABLED",
    "MASK_CONTAINMENT_VETO_THRESHOLD",
    "NEAR_TOTAL_BOX_CONTAINMENT_THRESHOLD",
    "MASK_CONTAINMENT_HIGH",
    "PAIR_DEBUG_MIN_BOX_CONTAINMENT",
    "POLE_TRUNK_CONTAINER_VETO_ENABLED",
    "POLE_TRUNK_CONTAINER_VETO_MIN_POLE_CONTAINMENT",
    "POLE_TRUNK_CONTAINER_VETO_MIN_VERTICAL_SPAN_RATIO",
    "CENTER_DIST_FACTOR",
    "POLE_MASK_FILTER_ENABLED",
    "POLE_OVERLAP_MIN_FRACTION",
    "POLE_OVERLAP_REJECT_FRACTION",
    "POLE_OVERLAP_MAX_FRACTION",
    "TOP_BAND_ABOVE",
    "TOP_BAND_BELOW",
    "MIN_RELATIVE_WIDTH_TO_MAX",
    "POLE_ATTACH_MARGIN_PX",
    "ONE_SIDED_LOWER_ARM_VETO_ENABLED",
    "ONE_SIDED_LOWER_ARM_SIDE_BALANCE_MAX",
    "ONE_SIDED_LOWER_ARM_SHORT_SIDE_FRAC_MAX",
    "ONE_SIDED_LOWER_ARM_ASPECT_MIN",
    "ONE_SIDED_LOWER_ARM_HEIGHT_TO_MEDIAN_MAX",
    "ONE_SIDED_LOWER_ARM_MIN_Y_GAP_PX",
    "ONE_SIDED_LOWER_ARM_MIN_Y_GAP_FACTOR",

    # -------------------------------------------------------------------------
    # Same-crossarm merge constants from CELL 3B.
    # -------------------------------------------------------------------------
    "SAME_XARM_MERGE_ENABLED",
    "SAME_XARM_MERGE_MIN_MASK_PIXELS",
    "SAME_XARM_MERGE_MAX_ANGLE_DIFF_DEG",
    "SAME_XARM_MERGE_MAX_PERP_DIST_PX",
    "SAME_XARM_MERGE_MAX_GAP_PX",
    "SAME_XARM_MERGE_REQUIRE_POLE_BRIDGE",
    "SAME_XARM_MERGE_BOX_PAD_PX",
    "SAME_XARM_MERGE_SCORE_MODE",

    # -------------------------------------------------------------------------
    # Single-box X-split constants from CELL 3B.
    # -------------------------------------------------------------------------
    "SINGLE_XSPLIT_ENABLED",
    "XSPLIT_HEIGHT_TO_MEDIAN_TRIGGER",
    "XSPLIT_AREA_TO_MEDIAN_TRIGGER",
    "XSPLIT_MAX_ASPECT_FOR_SUSPICIOUS",
    "XSPLIT_PC1_RATIO_MAX_FOR_XLIKE",
    "XSPLIT_HOUGH_THRESHOLD",
    "XSPLIT_HOUGH_MIN_LINE_LENGTH_FRAC",
    "XSPLIT_HOUGH_MAX_LINE_GAP",
    "XSPLIT_MIN_ANGLE_DIFF_DEG",
    "XSPLIT_MAX_ANGLE_DIFF_DEG",
    "XSPLIT_MIN_GROUP_LENGTH_FRAC",
    "XSPLIT_MIN_PARENT_MASK_PIXELS",
    "XSPLIT_MIN_CHILD_MASK_PIXELS",
    "XSPLIT_MIN_CHILD_FRAC_OF_PARENT",
    "XSPLIT_MIN_CHILD_BALANCE_RATIO",
    "XSPLIT_CHILD_BOX_PAD_PX",
    "XSPLIT_EDGE_CANNY_LOW",
    "XSPLIT_EDGE_CANNY_HIGH",
    "XSPLIT_EDGE_MASK_DILATE_ITER",

    # -------------------------------------------------------------------------
    # PCA / axis cleanup constants from CELL 3B.
    # -------------------------------------------------------------------------
    "CROSSARM_PCA_FILTER_ENABLED",
    "PCA_SUSPICIOUS_ASPECT_MAX",
    "PCA_SUSPICIOUS_HEIGHT_TO_MEDIAN_MIN",
    "PCA_SUSPICIOUS_REL_WIDTH_MAX",
    "PCA_MIN_MASK_PIXELS",
    "PCA_MIN_PC1_RATIO",
    "PCA_MIN_ANISOTROPY",
    "AXIS_CLEANUP_ENABLED",
    "AXIS_CLEANUP_MIN_MASK_PIXELS",
    "AXIS_CLEANUP_HEIGHT_TO_MEDIAN_TRIGGER",
    "AXIS_CLEANUP_BOX_OVERLAP_FRAC_TRIGGER",
    "AXIS_CLEANUP_PERP_STD_TRIGGER_PX",
    "AXIS_CLEANUP_HALF_WIDTH_EXTRA_PX",
    "AXIS_CLEANUP_MIN_HALF_WIDTH_PX",
    "AXIS_CLEANUP_MAX_HALF_WIDTH_PX",
    "AXIS_CLEANUP_MIN_RETAINED_FRAC",
    "AXIS_CLEANUP_BOX_PAD_PX",

    # -------------------------------------------------------------------------
    # Targeted two-box X ownership constants from CELL 3B.
    # -------------------------------------------------------------------------
    "TWO_BOX_XOWNERSHIP_ENABLED",
    "XOWN_MIN_SHARED_PIXELS",
    "XOWN_MIN_SHARED_FRAC_OF_SMALLER",
    "XOWN_MIN_ANGLE_DIFF_DEG",
    "XOWN_MIN_CHILD_PIXELS_AFTER",
    "XOWN_MIN_RETAINED_FRAC_AFTER",
    "XOWN_BOX_PAD_PX",

    # -------------------------------------------------------------------------
    # Final dedupe / review constants from CELL 3B.
    # -------------------------------------------------------------------------
    "FINAL_DEDUPE_ENABLED",
    "EXPECTED_MAX_CROSSARMS_FOR_DEBUG",
]

missing_cell16_globals = [
    name for name in required_cell16_globals
    if name not in globals()
]

if missing_cell16_globals:
    raise NameError(
        "CELL 16 requires earlier production setup cells to run successfully.\n"
        "Please run CELL 10B, CELL 13, and CELL 14 before CELL 16.\n"
        f"Missing globals: {missing_cell16_globals}"
    )
    
# =============================================================================
# 16.2 VALIDATE PRODUCTION INPUT OBJECTS
# =============================================================================
# EXPLANATION:
# CELL 16 starts from the clean ROI crops created in CELL 14.
#
# pole_rois_df:
#   Primary production input table.
#
# pole_mask_lookup:
#   Selected-pole full-resolution mask lookup from CELL 13.
#   Keys are expected to be:
#       (str(image_id), str(prompt), int(det_idx))
# =============================================================================

cell16_setup_warnings = []

if not isinstance(pole_rois_df, pd.DataFrame):
    raise TypeError(
        "pole_rois_df exists but is not a pandas DataFrame."
    )

if pole_rois_df.empty:
    raise ValueError(
        "pole_rois_df is empty.\n"
        "Please check CELL 14 before running CELL 16."
    )

if not isinstance(pole_mask_lookup, dict):
    raise TypeError(
        "pole_mask_lookup exists but is not a dictionary.\n"
        "Please check CELL 13 before running CELL 16."
    )

CELL16_POLE_MASK_LOOKUP_AVAILABLE = bool(len(pole_mask_lookup) > 0)

if bool(POLE_MASK_FILTER_ENABLED) and not CELL16_POLE_MASK_LOOKUP_AVAILABLE:
    raise ValueError(
        "pole_mask_lookup is empty, but POLE_MASK_FILTER_ENABLED=True.\n"
        "CELL 16 production requires selected-pole masks from CELL 13 for the "
        "pole-overlap / pole-corridor filter."
    )

if not CELL16_POLE_MASK_LOOKUP_AVAILABLE:
    cell16_setup_warnings.append(
        "pole_mask_lookup is empty; pole-overlap / pole-corridor filtering "
        "will not have selected-pole masks."
    )
    

# =============================================================================
# 16.3 VALIDATE REQUIRED pole_rois_df COLUMNS
# =============================================================================
# EXPLANATION:
# CELL 16 needs:
#   - run lineage
#   - source image lineage
#   - selected-pole identity fields for pole_mask_lookup
#   - ROI crop paths
#   - source-to-ROI geometry for pole-mask projection
#
# IMPORTANT:
#   CELL 16 assumes one ROI row per image_id for the current production pipeline.
# =============================================================================

required_cell16_roi_cols = [
    # -------------------------------------------------------------------------
    # Run lineage.
    # -------------------------------------------------------------------------
    "run_id",
    "run_timestamp",
    "processing_order",

    # -------------------------------------------------------------------------
    # Source image lineage.
    # -------------------------------------------------------------------------
    "image_id",
    "file_name",
    "stem",
    "ext",
    "ext_lower",
    "relative_image_path",
    "image_path",

    # -------------------------------------------------------------------------
    # CELL 13 selected-pole identity.
    # -------------------------------------------------------------------------
    "selection_status",
    "selection_mode",
    "prompt",
    "det_idx",
    "score",
    "final_score",
    "has_mask",
    "mask_lookup_hit",

    # -------------------------------------------------------------------------
    # Source image and selected pole geometry.
    # -------------------------------------------------------------------------
    "image_w",
    "image_h",
    "x1",
    "y1",
    "x2",
    "y2",
    "box_w",
    "box_h",
    "pole_cx",
    "pole_cy",

    # -------------------------------------------------------------------------
    # ROI request and source-to-ROI paste geometry from CELL 14.
    # -------------------------------------------------------------------------
    "req_x1",
    "req_y1",
    "req_x2",
    "req_y2",
    "src_x1",
    "src_y1",
    "src_x2",
    "src_y2",
    "dst_x1",
    "dst_y1",

    # -------------------------------------------------------------------------
    # Saved ROI image metadata.
    # -------------------------------------------------------------------------
    "roi_w",
    "roi_h",
    "roi_file_name",
    "roi_relative_dir",
    "roi_image_path",
]

missing_cell16_roi_cols = [
    col_name
    for col_name in required_cell16_roi_cols
    if col_name not in pole_rois_df.columns
]

if missing_cell16_roi_cols:
    raise ValueError(
        "pole_rois_df is missing columns required by CELL 16.\n"
        "Please check CELL 14 output schema.\n"
        f"Missing columns: {missing_cell16_roi_cols}"
    )
    
    
# =============================================================================
# 16.4 VALIDATE RUN IDENTITY
# =============================================================================
# EXPLANATION:
# pole_rois_df should belong to the active RUN_ID.
#
# This prevents accidentally running CELL 16 on ROI rows from an older run while
# saving outputs into the current run folder.
#
# IMPORTANT:
#   Check missing run_id / run_timestamp values before using run lineage.
#   Otherwise missing run lineage could be hidden by dropna().
# =============================================================================

if pole_rois_df["run_id"].isna().any():
    raise RuntimeError(
        "pole_rois_df contains missing run_id values.\n"
        "Please check CELL 14 output."
    )

if pole_rois_df["run_timestamp"].isna().any():
    raise RuntimeError(
        "pole_rois_df contains missing run_timestamp values.\n"
        "Please check CELL 14 output."
    )

unique_cell16_roi_run_ids = (
    pole_rois_df["run_id"]
    .astype(str)
    .unique()
    .tolist()
)

if len(unique_cell16_roi_run_ids) != 1:
    raise RuntimeError(
        "pole_rois_df must contain exactly one run_id.\n"
        f"Found run_ids: {unique_cell16_roi_run_ids}"
    )

if unique_cell16_roi_run_ids[0] != str(RUN_ID):
    raise RuntimeError(
        "pole_rois_df run_id does not match the active RUN_ID.\n\n"
        f"pole_rois_df run_id : {unique_cell16_roi_run_ids[0]}\n"
        f"active RUN_ID       : {RUN_ID}\n\n"
        "Please rerun CELL 14 for this active run, or reset RUN_ID intentionally."
    )
    
    
# =============================================================================
# 16.5 PREPARE SAFE ROI INPUT TABLE
# =============================================================================
# EXPLANATION:
# Build roi_input_df from pole_rois_df.
#
# Production rules:
#   - keep only selected-pole ROI rows
#   - require non-null roi_image_path
#   - sort in deterministic processing order
#   - validate one ROI per image_id
# =============================================================================

roi_input_df = pole_rois_df.copy()

roi_input_df = roi_input_df[
    roi_input_df["roi_image_path"].notna()
].copy()

roi_input_df = roi_input_df[
    roi_input_df["selection_status"].astype(str).str.strip().str.lower() == "selected"
].copy()

if roi_input_df.empty:
    raise ValueError(
        "No usable selected-pole ROI rows were found in pole_rois_df.\n"
        "Please check CELL 14 output."
    )

roi_input_df = roi_input_df.sort_values(
    by=["processing_order", "image_id"],
    kind="mergesort",
).reset_index(drop=True)

if roi_input_df["image_id"].isna().any():
    raise ValueError(
        "roi_input_df contains missing image_id values."
    )

if roi_input_df["image_id"].astype(str).duplicated().any():
    duplicate_image_ids = (
        roi_input_df
        .loc[roi_input_df["image_id"].astype(str).duplicated(), "image_id"]
        .astype(str)
        .tolist()
    )

    raise RuntimeError(
        "CELL 16 currently assumes one pole ROI per image_id.\n"
        "roi_input_df contains duplicate image_id values.\n"
        f"Duplicate examples: {duplicate_image_ids[:10]}"
    )

if roi_input_df["roi_image_path"].astype(str).duplicated().any():
    duplicate_roi_paths = (
        roi_input_df
        .loc[roi_input_df["roi_image_path"].astype(str).duplicated(), "roi_image_path"]
        .astype(str)
        .tolist()
    )

    raise RuntimeError(
        "roi_input_df contains duplicate roi_image_path values.\n"
        f"Duplicate examples: {duplicate_roi_paths[:10]}"
    )
    
    
# =============================================================================
# 16.6 VALIDATE ROI IMAGE FILES AND CORE ROW VALUES
# =============================================================================
# EXPLANATION:
# Fail before running SAM3 if any saved ROI crop is missing from disk or if
# required selected-pole identity fields / ROI geometry fields are invalid.
#
# This section validates:
#   - 16.6A: ROI image paths and pole-mask key identity fields
#   - 16.6B: ROI geometry and fixed-canvas size
#   - 16.6C: selected-pole mask lookup key consistency
#
# IMPORTANT:
#   16C must build the selected-pole mask key as:
#       (str(image_id), str(prompt), int(det_idx))
#
#   This must match CELL 13 / CELL 14 exactly.
# =============================================================================


# =============================================================================
# 16.6A VALIDATE ROI IMAGE FILES + POLE MASK KEY IDENTITY FIELDS
# =============================================================================
# EXPLANATION:
# Validate and normalise the fields needed before ROI loading and pole-mask
# lookup key construction:
#   - roi_image_path
#   - prompt
#   - det_idx
#
# IMPORTANT:
#   prompt is stripped here because CELL 13 also strips prompt text before
#   creating pole_mask_lookup keys. Keeping prompt normalisation symmetric
#   prevents silent lookup misses.
# =============================================================================

# -----------------------------------------------------------------------------
# 16.6A.1 Validate ROI image paths
# -----------------------------------------------------------------------------
# EXPLANATION:
# 16.5 already keeps rows where roi_image_path is not null.
#
# This block additionally:
#   - strips whitespace
#   - rejects blank paths
#   - verifies files exist on disk
#
# The stripped path becomes the canonical path used later by 16C.
# -----------------------------------------------------------------------------
roi_input_df["roi_image_path"] = (
    roi_input_df["roi_image_path"]
    .astype(str)
    .str.strip()
)

if (roi_input_df["roi_image_path"] == "").any():
    raise ValueError(
        "roi_input_df contains blank roi_image_path values."
    )

missing_cell16_roi_files = [
    roi_path
    for roi_path in roi_input_df["roi_image_path"].tolist()
    if not os.path.exists(roi_path)
]

if missing_cell16_roi_files:
    raise FileNotFoundError(
        "Some ROI image files listed in pole_rois_df do not exist on disk.\n"
        f"Missing file examples: {missing_cell16_roi_files[:10]}"
    )


# -----------------------------------------------------------------------------
# 16.6A.2 Validate selected-pole prompt values
# -----------------------------------------------------------------------------
# EXPLANATION:
# prompt is part of the selected-pole mask lookup key:
#
#     (str(image_id), str(prompt), int(det_idx))
#
# Blank prompts would create invalid lookup keys and cause silent pole-mask
# misses later.
# -----------------------------------------------------------------------------
if roi_input_df["prompt"].isna().any():
    raise ValueError(
        "roi_input_df contains missing prompt values.\n"
        "CELL 16 needs prompt to build the pole mask lookup key."
    )

roi_input_df["prompt"] = (
    roi_input_df["prompt"]
    .astype(str)
    .str.strip()
)

if (roi_input_df["prompt"] == "").any():
    raise ValueError(
        "roi_input_df contains blank prompt values.\n"
        "CELL 16 needs prompt to build the pole mask lookup key."
    )


# -----------------------------------------------------------------------------
# 16.6A.3 Validate det_idx exists
# -----------------------------------------------------------------------------
# EXPLANATION:
# det_idx is the selected-pole detection index from CELL 13 / CELL 14.
# It is required to rebuild the selected-pole mask lookup key in 16C.
# -----------------------------------------------------------------------------
if roi_input_df["det_idx"].isna().any():
    raise ValueError(
        "roi_input_df contains missing det_idx values.\n"
        "CELL 16 needs det_idx to build the pole mask lookup key."
    )


# -----------------------------------------------------------------------------
# 16.6A.4 Validate det_idx can safely become int
# -----------------------------------------------------------------------------
# EXPLANATION:
# CELL 13 / CELL 14 use pole mask keys shaped as:
#
#     (str(image_id), str(prompt), int(det_idx))
#
# CELL 16C will rebuild the same key. Normalising det_idx here prevents silent
# lookup misses later.
# -----------------------------------------------------------------------------
det_idx_numeric = pd.to_numeric(
    roi_input_df["det_idx"],
    errors="coerce",
)

if det_idx_numeric.isna().any():
    raise ValueError(
        "roi_input_df contains det_idx values that cannot be converted to numbers."
    )

if not np.isclose(
    det_idx_numeric.to_numpy(dtype=float),
    np.round(det_idx_numeric.to_numpy(dtype=float)),
).all():
    raise ValueError(
        "roi_input_df contains non-integer det_idx values.\n"
        "CELL 16 requires det_idx values that can safely convert to int."
    )

roi_input_df["det_idx"] = det_idx_numeric.astype(int)


# =============================================================================
# 16.6B VALIDATE ROI GEOMETRY + FIXED-CANVAS SIZE
# =============================================================================
# EXPLANATION:
# Validate and normalise the ROI geometry columns needed for:
#   - ROI image dimension checks
#   - source-to-ROI pole-mask projection
#   - downstream overlay / final image rendering
#
# This section validates and persists:
#   - roi_w, roi_h
#   - src_x1, src_y1, src_x2, src_y2
#   - dst_x1, dst_y1
#
# IMPORTANT:
#   This section must not drop, filter, sort, or reset rows.
#
#   16.6C builds cell16_has_live_pole_mask by iterating over roi_input_df in the
#   current row order. Therefore, 16.6B should only cast values in place and
#   raise errors if invalid geometry is found.
#
#   Fixed ROI size is intentionally validated against FIXED_ROI_WIDTH and
#   FIXED_ROI_HEIGHT. If these values do not match, rerun CELL 14 so the saved
#   ROI crops and metadata match the active CELL 3B configuration.
# =============================================================================

# -----------------------------------------------------------------------------
# 16.6B.1 Validate and persist numeric ROI geometry columns
# -----------------------------------------------------------------------------
# EXPLANATION:
# Convert geometry columns to numeric values and write the converted integer
# values back into roi_input_df.
#
# This prevents 16C from later reading string/object versions of these columns.
# -----------------------------------------------------------------------------
for numeric_col in [
    "roi_w",
    "roi_h",
    "src_x1",
    "src_y1",
    "src_x2",
    "src_y2",
    "dst_x1",
    "dst_y1",
]:
    numeric_values = pd.to_numeric(
        roi_input_df[numeric_col],
        errors="coerce",
    )

    if numeric_values.isna().any():
        raise ValueError(
            f"roi_input_df contains non-numeric or missing values in {numeric_col}."
        )

    if not np.isclose(
        numeric_values.to_numpy(dtype=float),
        np.round(numeric_values.to_numpy(dtype=float)),
    ).all():
        raise ValueError(
            f"roi_input_df contains non-integer values in {numeric_col}."
        )

    roi_input_df[numeric_col] = numeric_values.astype(int)


# -----------------------------------------------------------------------------
# 16.6B.2 Validate fixed-canvas ROI size contract
# -----------------------------------------------------------------------------
# EXPLANATION:
# CELL 16 is built for the fixed-canvas ROI dimensions configured in CELL 3B.
#
# For the current production setup:
#   - roi_w should match FIXED_ROI_WIDTH
#   - roi_h should match FIXED_ROI_HEIGHT
#
# This protects the non-square ROI contract:
#   image size  = width x height
#   mask shape  = height x width
# -----------------------------------------------------------------------------
bad_roi_size_df = roi_input_df[
    (roi_input_df["roi_w"].astype(int) != int(FIXED_ROI_WIDTH))
    | (roi_input_df["roi_h"].astype(int) != int(FIXED_ROI_HEIGHT))
].copy()

if len(bad_roi_size_df) > 0:
    raise ValueError(
        "Some ROI rows do not match the expected fixed ROI size.\n"
        f"Expected ROI size: {FIXED_ROI_WIDTH} x {FIXED_ROI_HEIGHT}\n"
        "This usually means CELL 3B ROI constants changed after CELL 14 was run.\n"
        "Please rerun CELL 14 so ROI crops and metadata match the active config.\n"
        f"Examples:\n"
        f"{bad_roi_size_df[['image_id', 'roi_w', 'roi_h', 'roi_image_path']].head(10).to_string(index=False)}"
    )


# -----------------------------------------------------------------------------
# 16.6B.3 Validate source crop geometry
# -----------------------------------------------------------------------------
# EXPLANATION:
# Source geometry defines the crop region from the original full-resolution
# source image that was pasted into the fixed ROI canvas.
#
# These checks catch impossible or corrupted CELL 14 geometry before pole-mask
# projection.
# -----------------------------------------------------------------------------
if (roi_input_df["src_x2"] <= roi_input_df["src_x1"]).any():
    raise ValueError(
        "roi_input_df contains invalid source ROI geometry: src_x2 <= src_x1."
    )

if (roi_input_df["src_y2"] <= roi_input_df["src_y1"]).any():
    raise ValueError(
        "roi_input_df contains invalid source ROI geometry: src_y2 <= src_y1."
    )


# -----------------------------------------------------------------------------
# 16.6B.4 Validate destination paste geometry
# -----------------------------------------------------------------------------
# EXPLANATION:
# Destination geometry defines where the source crop was pasted inside the fixed
# ROI canvas.
#
# dst_x1 / dst_y1 should be inside the ROI canvas.
# -----------------------------------------------------------------------------
if (roi_input_df["dst_x1"] < 0).any() or (roi_input_df["dst_y1"] < 0).any():
    raise ValueError(
        "roi_input_df contains negative destination paste coordinates."
    )

if (roi_input_df["dst_x1"] >= roi_input_df["roi_w"]).any():
    raise ValueError(
        "roi_input_df contains dst_x1 values outside the ROI width."
    )

if (roi_input_df["dst_y1"] >= roi_input_df["roi_h"]).any():
    raise ValueError(
        "roi_input_df contains dst_y1 values outside the ROI height."
    )


# =============================================================================
# 16.6C VALIDATE SELECTED-POLE MASK LOOKUP KEYS
# =============================================================================
# EXPLANATION:
# Confirm selected-pole mask lookup consistency without crashing on legitimate
# selected poles that do not have masks.
#
# CELL 13 / CELL 14 contract:
#   - has_mask is an upstream mask-availability flag
#   - mask_lookup_hit is an upstream audit/check field
#   - the live pole_mask_lookup key is the authority used by CELL 16
#
# Expected key:
#     (str(image_id), str(prompt), int(det_idx))
#
# IMPORTANT:
#   This validation only raises when a row claims has_mask=True but the live
#   pole_mask_lookup does not contain the expected key.
#
#   CELL 13 already prunes and validates selected pole masks. Therefore, if this
#   hard-fail fires, the likely issue is upstream CELL 13 / CELL 14 lineage,
#   stale run state, or a run mismatch — not normal CELL 16 logic.
#
#   Rows without a live pole mask are still legal selected-pole rows. They should
#   be processed later with pole-mask-dependent stages skipped and a review
#   reason added.
# =============================================================================

def parse_bool(value):
    """
    Convert common boolean-like values into a safe bool.

    Args:
        value:
            Boolean, numeric, string, or missing value.

    Returns:
        bool:
            Parsed boolean value.
    """
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(int(value))

    text_value = str(value).strip().lower()

    if text_value in ["true", "1", "yes", "y", "t"]:
        return True

    if text_value in ["false", "0", "no", "n", "f", "", "none", "nan", "null"]:
        return False

    return bool(value)


# -----------------------------------------------------------------------------
# 16.6C.1 Define standard review reason for rows without live pole masks
# -----------------------------------------------------------------------------
# EXPLANATION:
# This constant must be defined outside the setup-summary print block because
# 16C will use it regardless of PRINT_CONFIG_SUMMARY.
# -----------------------------------------------------------------------------
CELL16_REVIEW_POLE_MASK_UNAVAILABLE = (
    "pole_mask_unavailable_pole_filter_skipped"
)


# -----------------------------------------------------------------------------
# 16.6C.2 Validate live pole_mask_lookup keys
# -----------------------------------------------------------------------------
# EXPLANATION:
# Use three values for each row:
#   - row_has_mask        : upstream has_mask flag
#   - row_mask_lookup_hit : upstream audit flag
#   - live_key_hit        : actual lookup result in current pole_mask_lookup
#
# The live lookup key is authoritative.
# -----------------------------------------------------------------------------
missing_required_pole_mask_key_rows = []
selected_without_live_mask_rows = []
mask_lookup_hit_mismatch_rows = []
cell16_has_live_pole_mask_values = []

for row_idx, roi_row in roi_input_df.iterrows():
    pole_key = (
        str(roi_row["image_id"]),
        str(roi_row["prompt"]),
        int(roi_row["det_idx"]),
    )

    row_has_mask = parse_bool(
        roi_row.get("has_mask", False)
    )

    row_mask_lookup_hit = parse_bool(
        roi_row.get("mask_lookup_hit", False)
    )

    live_key_hit = pole_key in pole_mask_lookup
    cell16_has_live_pole_mask_values.append(bool(live_key_hit))

    row_record = {
        "row_idx": int(row_idx),
        "image_id": str(roi_row["image_id"]),
        "prompt": str(roi_row["prompt"]),
        "det_idx": int(roi_row["det_idx"]),
        "has_mask": bool(row_has_mask),
        "mask_lookup_hit": bool(row_mask_lookup_hit),
        "live_key_hit": bool(live_key_hit),
        "expected_pole_key": str(pole_key),
    }

    if row_has_mask and not live_key_hit:
        missing_required_pole_mask_key_rows.append(row_record)

    if not live_key_hit:
        selected_without_live_mask_rows.append(row_record)

    if row_mask_lookup_hit != live_key_hit:
        mask_lookup_hit_mismatch_rows.append(row_record)

roi_input_df["cell16_has_live_pole_mask"] = cell16_has_live_pole_mask_values


# -----------------------------------------------------------------------------
# 16.6C.3 Hard-fail only for broken has_mask=True contract
# -----------------------------------------------------------------------------
if len(missing_required_pole_mask_key_rows) > 0:
    missing_required_pole_mask_key_df = pd.DataFrame(
        missing_required_pole_mask_key_rows
    )

    raise KeyError(
        "Some selected ROI rows have has_mask=True but no matching key in "
        "pole_mask_lookup.\n"
        "This violates the CELL 13/CELL 14 mask lookup contract.\n"
        "If this fires, check upstream CELL 13 pole_mask_lookup pruning, "
        "CELL 14 lineage, or stale RUN_ID state.\n"
        f"Missing key examples:\n"
        f"{missing_required_pole_mask_key_df.head(10).to_string(index=False)}"
    )


# -----------------------------------------------------------------------------
# 16.6C.4 Record setup warnings for legal no-mask rows and audit mismatches
# -----------------------------------------------------------------------------
selected_without_live_mask_df = pd.DataFrame(
    selected_without_live_mask_rows
)

mask_lookup_hit_mismatch_df = pd.DataFrame(
    mask_lookup_hit_mismatch_rows
)

if len(selected_without_live_mask_df) > 0:
    cell16_setup_warnings.append(
        f"{len(selected_without_live_mask_df)} selected ROI rows do not have a "
        "live selected-pole mask key in pole_mask_lookup. These rows are legal "
        "selected-pole rows, but 16C should skip pole-mask / pole-corridor "
        f"filtering for them and add review reason: "
        f"{CELL16_REVIEW_POLE_MASK_UNAVAILABLE}."
    )

if len(mask_lookup_hit_mismatch_df) > 0:
    cell16_setup_warnings.append(
        f"{len(mask_lookup_hit_mismatch_df)} ROI rows have mask_lookup_hit values "
        "that disagree with the live pole_mask_lookup check. The live lookup key "
        "will be treated as authoritative."
    )
    
    
# =============================================================================
# 16.6D CREATE STABLE PRODUCTION ROI INPUT ALIAS
# =============================================================================
# EXPLANATION:
# Keep a CELL 16-specific alias for the cleaned production ROI input table.
#
# IMPORTANT:
#   Use cell16_roi_input_df in the 16C batch loop instead of roi_input_df.
#
#   This alias is created only after:
#     - roi_image_path has been stripped and validated
#     - prompt has been stripped and validated
#     - det_idx has been converted to int
#     - ROI geometry has been validated and cast to int
#     - cell16_has_live_pole_mask has been created
#
#   Do not add tuple-valued pole-key columns unless a later saved table needs
#   them.
# =============================================================================

cell16_roi_input_df = roi_input_df.copy().reset_index(drop=True)


# =============================================================================
# 16.7 RESOLVE PRODUCTION CROSSARM CONFIG
# =============================================================================
# EXPLANATION:
# CELL 16 should read production constants from CELL 3B.
#
# IMPORTANT:
#   Do not locally retune thresholds in CELL 16 production code.
#   Tune in CELL 3B so the run configuration remains visible and auditable.
#
#   CELL16_MASK_THRESHOLD is recorded for configuration visibility only.
#   The locked development CELL 16 binarised SAM3 masks with m > 0.
#   Do not change production mask geometry to m > CELL16_MASK_THRESHOLD
#   until raw SAM3 mask dtype/range has been verified.
# =============================================================================

CELL16_RUN_DEVICE = str(DEVICE)

CELL16_PROMPT_TEXT = str(CROSSARM_PROMPT_TEXT).strip()
CELL16_TEXT_THRESHOLD = float(CROSSARM_TEXT_THRESHOLD)
CELL16_MASK_THRESHOLD = float(MASK_THRESHOLD)

if len(CELL16_PROMPT_TEXT) == 0:
    raise ValueError(
        "CROSSARM_PROMPT_TEXT is empty."
    )

if not np.isfinite(CELL16_TEXT_THRESHOLD):
    raise ValueError(
        f"CROSSARM_TEXT_THRESHOLD must be finite. Got: {CELL16_TEXT_THRESHOLD}"
    )

if not np.isfinite(CELL16_MASK_THRESHOLD):
    raise ValueError(
        f"MASK_THRESHOLD must be finite. Got: {CELL16_MASK_THRESHOLD}"
    )

if CELL16_TEXT_THRESHOLD < 0.0 or CELL16_TEXT_THRESHOLD > 1.0:
    raise ValueError(
        "CROSSARM_TEXT_THRESHOLD should be in [0, 1]. "
        f"Got: {CELL16_TEXT_THRESHOLD}"
    )

if CELL16_MASK_THRESHOLD < 0.0 or CELL16_MASK_THRESHOLD > 1.0:
    raise ValueError(
        "MASK_THRESHOLD should be in [0, 1]. "
        f"Got: {CELL16_MASK_THRESHOLD}"
    )
    
    
# =============================================================================
# 16.8 FORCE BATCH-SAFE DEBUG / VISUAL FLAGS
# =============================================================================
# EXPLANATION:
# The development CELL 16 used many visual checks and plt.show() calls.
#
# For a production batch, those must be disabled. Final review images may still
# be saved later through CELL 10B's save_final_image() helper.
#
# IMPORTANT:
#   Use CELL16_* runtime flags here instead of overwriting CELL 3B globals.
#   This keeps CELL 3B as the source of truth and avoids silently mutating
#   notebook-wide configuration.
# =============================================================================

CELL16_ALLOW_STAGE_VISUALS = False
CELL16_RUN_PLOT_RESULTS_DIAGNOSTIC = False

CELL16_SHOW_STAGE_GRID = False
CELL16_SHOW_SAME_XARM_MERGE_DEBUG = False
CELL16_SHOW_SINGLE_XSPLIT_DEBUG = False
CELL16_SHOW_AXIS_CLEANUP_DEBUG = False
CELL16_SHOW_XOWNERSHIP_DEBUG = False
CELL16_SHOW_FINAL_DEBUG = False

CELL16_SAVE_GOLD_TABLES = bool(
    globals().get(
        "SAVE_GOLD_TABLES",
        True,
    )
)

CELL16_SAVE_GOLD_FINAL_IMAGES = bool(
    globals().get(
        "SAVE_GOLD_FINAL_IMAGES",
        True,
    )
)

CELL16_SAVE_GOLD_FINAL_IMAGES_REVIEW_ONLY = bool(
    globals().get(
        "SAVE_GOLD_FINAL_IMAGES_REVIEW_ONLY",
        False,
    )
)

CELL16_SAVE_SILVER_STAGE_TABLES = bool(
    globals().get(
        "SAVE_SILVER_STAGE_TABLES",
        True,
    )
)

# Hard production override for bulky/intermediate artifacts.
CELL16_SAVE_SILVER_MASKS = False
CELL16_SAVE_SILVER_STAGE_IMAGES = False


# =============================================================================
# 16.9 RESOLVE skimage SKELETONIZE AVAILABILITY
# =============================================================================
# EXPLANATION:
# The tested development CELL 16 prefers skimage skeletonize for X-split, with
# an OpenCV fallback.
#
# Production keeps the same soft-fallback behaviour:
#   - use skimage when available
#   - fall back to OpenCV if unavailable
#   - do not stop the full batch only because skimage import failed
# =============================================================================

try:
    from skimage.morphology import skeletonize as _cell16_skimage_skeletonize

    CELL16_SKIMAGE_SKELETONIZE_AVAILABLE = True

except Exception:
    _cell16_skimage_skeletonize = None
    CELL16_SKIMAGE_SKELETONIZE_AVAILABLE = False

    cell16_setup_warnings.append(
        "scikit-image skeletonize is not available; CELL 16 will use the "
        "OpenCV fallback skeletonizer for X-split."
    )
    
    
# =============================================================================
# 16.10 PREPARE RUN-SCOPED OUTPUT DIRECTORIES
# =============================================================================
# EXPLANATION:
# CELL 10B already created run-scoped Gold/Silver paths.
#
# CELL 16 ensures these folders exist before the batch loop starts.
# =============================================================================

cell16_output_dirs = [
    RUN_GOLD_TABLES_DIR,
    RUN_GOLD_IMAGES_DIR,
    RUN_SILVER_STAGE_TABLES_DIR,
]

for output_dir in cell16_output_dirs:
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

missing_cell16_output_dirs = [
    output_dir
    for output_dir in cell16_output_dirs
    if not os.path.isdir(output_dir)
]

if missing_cell16_output_dirs:
    raise RuntimeError(
        "Some CELL 16 output directories were not created successfully.\n"
        f"Missing directories: {missing_cell16_output_dirs}"
    )
    
    
# =============================================================================
# 16.11 INITIALISE RUN-LEVEL ACCUMULATORS
# =============================================================================
# EXPLANATION:
# These lists accumulate lightweight production rows across all ROIs.
#
# IMPORTANT:
#   Do not initialise crossarm_mask_lookup here.
#
#   crossarm_mask_lookup is per-ROI mutable working state and must be reset
#   inside each ROI iteration in 16C.
# =============================================================================

crossarm_image_rows = []
crossarm_final_detection_rows = []
crossarm_trace_rows = []
crossarm_failure_rows = []
crossarm_stage_summary_rows = []
crossarm_saved_image_rows = []


CELL16_STOP_ON_ROW_FAILURE = bool(
    globals().get(
        "CELL16_STOP_ON_ROW_FAILURE",
        False,
    )
)

# =============================================================================
# 16.12 MODEL / DEVICE SANITY CHECK
# =============================================================================
# EXPLANATION:
# Make sure the SAM3 model is in eval mode before production inference.
# =============================================================================

if hasattr(model, "eval"):
    model.eval()

if CELL16_RUN_DEVICE != "cuda":
    raise RuntimeError(
        "CELL 16 expects DEVICE='cuda' for SAM3 production inference.\n"
        f"Current DEVICE: {CELL16_RUN_DEVICE}"
    )

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available before CELL 16 inference."
    )


# =============================================================================
# 16.13 SETUP SUMMARY
# =============================================================================
# EXPLANATION:
# Print a compact setup summary when PRINT_CONFIG_SUMMARY is enabled.
#
# IMPORTANT:
#   This is the only intended setup print block for 16A. The 16C batch loop
#   should not print per ROI unless an explicit production progress interval is
#   added later.
# =============================================================================

if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print("CELL 16 setup section complete.\n")

    print("=" * 100)
    print("CELL 16 — PRODUCTION CROSSARM / XARM SETUP")
    print("=" * 100)

    print("\nRUN")
    print("-" * 100)
    print(f"RUN_ID                              : {RUN_ID}")
    print(f"RUN_TIMESTAMP                       : {RUN_TIMESTAMP}")
    print(f"CELL16_RUN_DEVICE                   : {CELL16_RUN_DEVICE}")

    print("\nINPUTS")
    print("-" * 100)
    print(f"pole_rois_df rows                   : {len(pole_rois_df)}")
    print(f"cell16_roi_input_df rows            : {len(cell16_roi_input_df)}")
    print(f"Unique image_id count               : {cell16_roi_input_df['image_id'].astype(str).nunique()}")
    print(f"pole_mask_lookup entries            : {len(pole_mask_lookup)}")
    print(f"Pole mask lookup available          : {CELL16_POLE_MASK_LOOKUP_AVAILABLE}")
    print(
        f"Rows with live pole mask            : "
        f"{int(cell16_roi_input_df['cell16_has_live_pole_mask'].sum())}"
    )
    print(
        f"Rows without live pole mask         : "
        f"{int((~cell16_roi_input_df['cell16_has_live_pole_mask']).sum())}"
    )
    print(
        f"Expected fixed ROI size             : "
        f"{FIXED_ROI_WIDTH} x {FIXED_ROI_HEIGHT}"
    )

    print("\nSAM3 CROSSARM PROMPT")
    print("-" * 100)
    print(f"CELL16_PROMPT_TEXT                  : {CELL16_PROMPT_TEXT}")
    print(f"CELL16_TEXT_THRESHOLD               : {CELL16_TEXT_THRESHOLD}")
    print(f"CELL16_MASK_THRESHOLD               : {CELL16_MASK_THRESHOLD}")
    print("CELL16_MASK_THRESHOLD note          : config visibility only until raw mask dtype/range is verified")

    print("\nREVIEW REASONS")
    print("-" * 100)
    print(
        f"Pole-mask unavailable review reason : "
        f"{CELL16_REVIEW_POLE_MASK_UNAVAILABLE}"
    )

    print("\nBATCH-SAFE VISUAL FLAGS")
    print("-" * 100)
    print(f"CELL16_ALLOW_STAGE_VISUALS          : {CELL16_ALLOW_STAGE_VISUALS}")
    print(f"CELL16_RUN_PLOT_RESULTS_DIAGNOSTIC  : {CELL16_RUN_PLOT_RESULTS_DIAGNOSTIC}")
    print(f"CELL16_SHOW_STAGE_GRID              : {CELL16_SHOW_STAGE_GRID}")
    print(f"CELL16_SHOW_SAME_XARM_MERGE_DEBUG   : {CELL16_SHOW_SAME_XARM_MERGE_DEBUG}")
    print(f"CELL16_SHOW_SINGLE_XSPLIT_DEBUG     : {CELL16_SHOW_SINGLE_XSPLIT_DEBUG}")
    print(f"CELL16_SHOW_AXIS_CLEANUP_DEBUG      : {CELL16_SHOW_AXIS_CLEANUP_DEBUG}")
    print(f"CELL16_SHOW_XOWNERSHIP_DEBUG        : {CELL16_SHOW_XOWNERSHIP_DEBUG}")
    print(f"CELL16_SHOW_FINAL_DEBUG             : {CELL16_SHOW_FINAL_DEBUG}")

    print("\nX-SPLIT DEPENDENCY")
    print("-" * 100)
    print(f"CELL16_SKIMAGE_SKELETONIZE_AVAILABLE: {CELL16_SKIMAGE_SKELETONIZE_AVAILABLE}")

    print("\nSAVE SETTINGS")
    print("-" * 100)
    print(f"CELL16_SAVE_GOLD_TABLES             : {CELL16_SAVE_GOLD_TABLES}")
    print(f"CELL16_SAVE_GOLD_FINAL_IMAGES       : {CELL16_SAVE_GOLD_FINAL_IMAGES}")
    print(f"CELL16_SAVE_GOLD_FINAL_IMAGES_REVIEW_ONLY: {CELL16_SAVE_GOLD_FINAL_IMAGES_REVIEW_ONLY}")
    print(f"CELL16_SAVE_SILVER_STAGE_TABLES     : {CELL16_SAVE_SILVER_STAGE_TABLES}")
    print(f"CELL16_SAVE_SILVER_MASKS            : {CELL16_SAVE_SILVER_MASKS}")
    print(f"CELL16_SAVE_SILVER_STAGE_IMAGES     : {CELL16_SAVE_SILVER_STAGE_IMAGES}")
    print(f"CELL16_STOP_ON_ROW_FAILURE          : {CELL16_STOP_ON_ROW_FAILURE}")

    print("\nOUTPUT DIRS")
    print("-" * 100)
    print(f"RUN_GOLD_TABLES_DIR                 : {RUN_GOLD_TABLES_DIR}")
    print(f"RUN_GOLD_IMAGES_DIR                 : {RUN_GOLD_IMAGES_DIR}")
    print(f"RUN_SILVER_STAGE_TABLES_DIR         : {RUN_SILVER_STAGE_TABLES_DIR}")

    if len(cell16_setup_warnings) > 0:
        print("\nSETUP WARNINGS")
        print("-" * 100)

        for warning_text in cell16_setup_warnings:
            print(f"- {warning_text}")

    print("\nNEXT")
    print("-" * 100)
    print("Proceed to 16B helper functions.")
    
    
# =============================================================================
# 16B. HELPER FUNCTIONS
# =============================================================================
# EXPLANATION:
# All helper functions used by the production crossarm / xarm pipeline are
# grouped here by purpose. The execution sections later in CELL 16 call into
# these helpers.
#
# PRODUCTION RULE:
#   Helper functions should be quiet by default.
#   They should not display DataFrames, call display(), or call plt.show().
#
# IMPORTANT:
#   crossarm_mask_lookup is intentionally not created here.
#   It must be reset inside each ROI iteration in 16C.
# =============================================================================


# =============================================================================
# 16.14 SAM3 OUTPUT NORMALISATION HELPERS
# =============================================================================
# EXPLANATION:
# SAM3 outputs can arrive in slightly different shapes depending on the stateful
# processor path and whether there is one detection or many detections.
#
# These helpers standardise:
#   - boxes  -> NumPy array with shape (N, 4)
#   - scores -> NumPy array with shape (N,)
#   - masks  -> list of 2D boolean masks, one per detection
#
# IMPORTANT:
#   These are the CELL 16B production normalisation helpers. They are independent
#   of CELL 13's pole-selection helpers such as _normalize_masks_local.
#
#   normalize_masks is intentionally strict. It only accepts masks that already
#   match the ROI image size. If a mask has the wrong shape, it is returned as
#   None rather than resized, so production code does not silently change mask
#   geometry.
#
#   The mask shape contract is height-first:
#       mask.shape == (image_h, image_w)
#
#   For the current fixed ROI:
#       image_h = 3500
#       image_w = 2600
#
#   Mask binarisation currently preserves the tested development CELL 16
#   behaviour:
#       mask_bool = m > 0
#
#   Do not change this to CELL16_MASK_THRESHOLD until the raw SAM3 mask
#   dtype/range diagnostic confirms that thresholding is appropriate.
# =============================================================================

def to_numpy(x):
    """
    Convert torch tensors or array-like objects to NumPy arrays.

    Args:
        x:
            Torch tensor, NumPy array, list, tuple, scalar, or array-like object.

    Returns:
        np.ndarray:
            NumPy representation of x.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def infer_num_detections(raw_boxes, raw_scores, raw_masks):
    """
    Infer the number of SAM3 detections from boxes, scores, or masks.

    Args:
        raw_boxes:
            Raw SAM3 boxes output.

        raw_scores:
            Raw SAM3 scores output.

        raw_masks:
            Raw SAM3 masks output.

    Returns:
        int:
            Number of inferred detections.
    """
    boxes_arr = to_numpy(raw_boxes) if raw_boxes is not None else None
    scores_arr = to_numpy(raw_scores) if raw_scores is not None else None

    if boxes_arr is not None:
        if boxes_arr.ndim == 2 and boxes_arr.shape[1] == 4:
            return int(boxes_arr.shape[0])

        if boxes_arr.ndim == 1 and boxes_arr.size == 4:
            return 1

    if scores_arr is not None:
        scores_arr = scores_arr.reshape(-1)

        if scores_arr.size > 0:
            return int(scores_arr.size)

    if raw_masks is not None:
        if isinstance(raw_masks, (list, tuple)):
            return int(len(raw_masks))

        masks_arr = to_numpy(raw_masks)

        if masks_arr.ndim == 2:
            return 1

        if masks_arr.ndim >= 3:
            return int(masks_arr.shape[0])

    return 0


def normalize_boxes(boxes, num_detections):
    """
    Normalise raw boxes into an N x 4 float array.

    Args:
        boxes:
            Raw SAM3 boxes output.

        num_detections:
            Expected number of detections.

    Returns:
        np.ndarray:
            Float32 array with shape (num_detections, 4).
    """
    if num_detections <= 0:
        return np.zeros((0, 4), dtype=np.float32)

    if boxes is None:
        return np.zeros((num_detections, 4), dtype=np.float32)

    arr = to_numpy(boxes).astype(np.float32)

    if arr.ndim == 1 and arr.shape[0] == 4:
        arr = arr.reshape(1, 4)

    if arr.ndim != 2 or arr.shape[1] != 4:
        return np.zeros((num_detections, 4), dtype=np.float32)

    if arr.shape[0] < num_detections:
        pad = np.zeros(
            (num_detections - arr.shape[0], 4),
            dtype=np.float32,
        )

        arr = np.vstack([arr, pad])

    return arr[:num_detections]


def normalize_scores(scores, num_detections):
    """
    Normalise raw scores into a 1D float array.

    Args:
        scores:
            Raw SAM3 score output.

        num_detections:
            Expected number of detections.

    Returns:
        np.ndarray:
            Float32 array with shape (num_detections,).
    """
    if num_detections <= 0:
        return np.zeros((0,), dtype=np.float32)

    if scores is None:
        return np.zeros((num_detections,), dtype=np.float32)

    arr = to_numpy(scores).astype(np.float32).reshape(-1)

    if arr.size < num_detections:
        pad = np.zeros(
            (num_detections - arr.size,),
            dtype=np.float32,
        )

        arr = np.concatenate([arr, pad])

    return arr[:num_detections]


def normalize_masks(raw_masks, num_detections, image_h, image_w):
    """
    Normalise raw SAM3 masks into a list of 2D boolean masks.

    Args:
        raw_masks:
            Raw SAM3 masks output.

        num_detections:
            Expected number of detections.

        image_h:
            ROI image height. This is the first dimension of the expected mask
            shape.

        image_w:
            ROI image width. This is the second dimension of the expected mask
            shape.

    Returns:
        list:
            List of length num_detections. Each item is either a 2D boolean mask
            with shape (image_h, image_w), or None if no valid mask is available.
    """
    if num_detections <= 0:
        return []

    if raw_masks is None:
        return [None] * num_detections

    if isinstance(raw_masks, (list, tuple)):
        mask_items = list(raw_masks)

    else:
        arr = to_numpy(raw_masks)

        # ---------------------------------------------------------------------
        # Dev-fidelity guard:
        # Keep this explicit even though current to_numpy normally returns an
        # ndarray. It protects the batch if to_numpy is later changed or if SAM3
        # returns an unexpected nullable object.
        # ---------------------------------------------------------------------
        if arr is None:
            return [None] * num_detections

        if arr.ndim == 2:
            mask_items = [arr]

        elif arr.ndim == 3:
            mask_items = [
                arr[i]
                for i in range(min(arr.shape[0], num_detections))
            ]

        elif arr.ndim == 4:
            mask_items = [
                arr[i]
                for i in range(min(arr.shape[0], num_detections))
            ]

        else:
            return [None] * num_detections

    norm_masks = []

    for det_idx in range(num_detections):
        if det_idx >= len(mask_items):
            norm_masks.append(None)
            continue

        m = to_numpy(mask_items[det_idx])

        # ---------------------------------------------------------------------
        # Dev-fidelity guard:
        # Explicitly handle None elements inside raw mask lists.
        # ---------------------------------------------------------------------
        if m is None:
            norm_masks.append(None)
            continue

        m = np.squeeze(m)

        if m.ndim != 2:
            norm_masks.append(None)
            continue

        if m.shape != (image_h, image_w):
            norm_masks.append(None)
            continue

        # IMPORTANT:
        # Preserve tested dev behaviour until raw mask dtype/range is verified.
        mask_bool = m.copy() if m.dtype == bool else (m > 0)

        if mask_bool.sum() == 0:
            norm_masks.append(None)
        else:
            norm_masks.append(mask_bool.astype(bool))

    if len(norm_masks) < num_detections:
        norm_masks.extend([None] * (num_detections - len(norm_masks)))

    return norm_masks[:num_detections]



# =============================================================================
# 16.15 BOX / MASK GEOMETRY HELPERS
# =============================================================================
# EXPLANATION:
# These helpers provide reusable geometry operations for bounding boxes and
# binary masks.
#
# They are used later by:
#   - containment suppression
#   - pole-overlap filtering
#   - same-crossarm merging
#   - PCA cleanup
#   - X-split logic
#   - final dedupe
#
# Coordinate convention:
#   Boxes are expected in XYXY format:
#       [x1, y1, x2, y2]
#
# IMPORTANT:
#   These helpers intentionally avoid resizing masks. They either compute using
#   the mask geometry as-is or return a safe fallback value.
#
#   compute_box_overlap_with_mask keeps the tested development denominator:
#       overlap_pixels / original_float_box_area
#
#   Do not change this to clipped box area, because pole-overlap thresholds were
#   tuned against the original float box-area denominator.
# =============================================================================

def box_area_xyxy(box_xyxy):
    """
    Compute the area of an XYXY bounding box.

    Args:
        box_xyxy:
            Box coordinates in [x1, y1, x2, y2] format.

    Returns:
        float:
            Non-negative box area. Returns 0.0 for invalid input.
    """
    try:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    except Exception:
        return 0.0

    if not np.all(np.isfinite([x1, y1, x2, y2])):
        return 0.0

    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area_xyxy(box_a, box_b):
    """
    Compute the intersection area between two XYXY boxes.

    Args:
        box_a:
            First box in [x1, y1, x2, y2] format.

        box_b:
            Second box in [x1, y1, x2, y2] format.

    Returns:
        float:
            Non-negative intersection area. Returns 0.0 for invalid input.
    """
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
    except Exception:
        return 0.0

    if not np.all(np.isfinite([ax1, ay1, ax2, ay2, bx1, by1, bx2, by2])):
        return 0.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _safe_mask_containment(mask_j, mask_i):
    """
    Compute how much of mask_j is contained inside mask_i.

    Formula:
        (mask_j AND mask_i).sum() / mask_j.sum()

    Args:
        mask_j:
            Candidate mask being tested for containment.

        mask_i:
            Reference mask that may contain mask_j.

    Returns:
        tuple:
            (valid, fraction)

            valid:
                True only when both masks are 2D arrays with the same shape and
                mask_j has positive pixels.

            fraction:
                Containment fraction in [0, 1] when valid, otherwise np.nan.
    """
    if mask_j is None or mask_i is None:
        return False, np.nan

    if not isinstance(mask_j, np.ndarray) or not isinstance(mask_i, np.ndarray):
        return False, np.nan

    if mask_j.ndim != 2 or mask_i.ndim != 2:
        return False, np.nan

    if mask_j.shape != mask_i.shape:
        return False, np.nan

    mj = mask_j.astype(bool)
    mi = mask_i.astype(bool)

    mask_j_area = int(mj.sum())

    if mask_j_area <= 0:
        return False, np.nan

    inter = int((mj & mi).sum())

    return True, float(inter / mask_j_area)


def compute_box_overlap_with_mask(box_xyxy, binary_mask):
    """
    Compute how much of a box area overlaps a binary mask.

    Args:
        box_xyxy:
            Box coordinates in [x1, y1, x2, y2] format.

        binary_mask:
            2D binary/boolean mask.

    Returns:
        float:
            Mask pixels inside the clipped box slice divided by the original
            float box area. Returns 0.0 for invalid boxes or invalid masks.
    """
    if binary_mask is None:
        return 0.0

    mask = to_numpy(binary_mask)

    if mask.ndim != 2:
        return 0.0

    mask = mask.astype(bool)
    h, w = mask.shape

    try:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    except Exception:
        return 0.0

    if not np.all(np.isfinite([x1, y1, x2, y2])):
        return 0.0

    box_area = box_area_xyxy([x1, y1, x2, y2])

    if box_area <= 0:
        return 0.0

    ix1 = int(np.floor(max(0.0, min(float(w), x1))))
    iy1 = int(np.floor(max(0.0, min(float(h), y1))))
    ix2 = int(np.ceil(max(0.0, min(float(w), x2))))
    iy2 = int(np.ceil(max(0.0, min(float(h), y2))))

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    overlap_pixels = int(mask[iy1:iy2, ix1:ix2].sum())

    return float(overlap_pixels / box_area)


def compute_box_pole_mask_containment_fraction(
    box_xyxy,
    projected_pole_mask,
):
    """
    Compute how much of the selected-pole mask is contained inside a box.

    Formula:
        pole pixels inside detection box / total pole pixels

    This is different from compute_box_overlap_with_mask(...), which divides by
    box area. Here the denominator is the full selected-pole mask area.

    Args:
        box_xyxy:
            Detection box in [x1, y1, x2, y2] format.

        projected_pole_mask:
            Selected-pole mask projected into the fixed ROI canvas.

    Returns:
        float:
            Fraction of selected-pole pixels inside the box.
            Returns np.nan if the pole mask is unavailable.
    """
    if not (
        isinstance(projected_pole_mask, np.ndarray)
        and projected_pole_mask.ndim == 2
        and projected_pole_mask.any()
    ):
        return np.nan

    pole_mask_bool = projected_pole_mask.astype(bool)
    total_pole_px = int(pole_mask_bool.sum())

    if total_pole_px <= 0:
        return np.nan

    h, w = pole_mask_bool.shape

    try:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    except Exception:
        return np.nan

    if not np.all(np.isfinite([x1, y1, x2, y2])):
        return np.nan

    ix1 = int(np.floor(max(0.0, min(float(w), x1))))
    iy1 = int(np.floor(max(0.0, min(float(h), y1))))
    ix2 = int(np.ceil(max(0.0, min(float(w), x2))))
    iy2 = int(np.ceil(max(0.0, min(float(h), y2))))

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    pole_px_inside_box = int(
        pole_mask_bool[iy1:iy2, ix1:ix2].sum()
    )

    return float(pole_px_inside_box / max(total_pole_px, 1))


def crop_mask_to_box(mask_bool, box_xyxy):
    """
    Crop a binary mask to the region covered by an XYXY box.

    Args:
        mask_bool:
            2D binary/boolean mask.

        box_xyxy:
            Box coordinates in [x1, y1, x2, y2] format.

    Returns:
        np.ndarray:
            Cropped boolean mask. Returns an empty boolean array if invalid.
    """
    if mask_bool is None:
        return np.zeros((0, 0), dtype=bool)

    mask = to_numpy(mask_bool)

    if mask.ndim != 2:
        return np.zeros((0, 0), dtype=bool)

    mask = mask.astype(bool)
    h, w = mask.shape

    try:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    except Exception:
        return np.zeros((0, 0), dtype=bool)

    if not np.all(np.isfinite([x1, y1, x2, y2])):
        return np.zeros((0, 0), dtype=bool)

    ix1 = int(np.floor(max(0.0, min(float(w), x1))))
    iy1 = int(np.floor(max(0.0, min(float(h), y1))))
    ix2 = int(np.ceil(max(0.0, min(float(w), x2))))
    iy2 = int(np.ceil(max(0.0, min(float(h), y2))))

    if ix2 <= ix1 or iy2 <= iy1:
        return np.zeros((0, 0), dtype=bool)

    return mask[iy1:iy2, ix1:ix2]


def _bbox_from_mask(mask_bool, pad_px=0, image_w=None, image_h=None):
    """
    Build a tight XYXY bounding box from a binary mask.

    Args:
        mask_bool:
            2D binary/boolean mask.

        pad_px:
            Optional padding around the tight box.

        image_w:
            Optional image width for clipping.

        image_h:
            Optional image height for clipping.

    Returns:
        tuple:
            (valid, x1, y1, x2, y2)

            valid:
                True if a non-empty valid box could be created.
    """
    if mask_bool is None:
        return False, np.nan, np.nan, np.nan, np.nan

    mask = to_numpy(mask_bool)

    if mask.ndim != 2:
        return False, np.nan, np.nan, np.nan, np.nan

    mask = mask.astype(bool)
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return False, np.nan, np.nan, np.nan, np.nan

    x1 = int(xs.min()) - int(pad_px)
    y1 = int(ys.min()) - int(pad_px)
    x2 = int(xs.max()) + 1 + int(pad_px)
    y2 = int(ys.max()) + 1 + int(pad_px)

    if image_w is not None:
        x1 = max(0, min(int(image_w), x1))
        x2 = max(0, min(int(image_w), x2))

    if image_h is not None:
        y1 = max(0, min(int(image_h), y1))
        y2 = max(0, min(int(image_h), y2))

    if x2 <= x1 or y2 <= y1:
        return False, np.nan, np.nan, np.nan, np.nan

    return True, float(x1), float(y1), float(x2), float(y2)


def _box_overlap_fraction_of_smaller(row_a, row_b):
    """
    Compute box intersection divided by the smaller box area.

    Args:
        row_a:
            Row-like object containing x1, y1, x2, y2.

        row_b:
            Row-like object containing x1, y1, x2, y2.

    Returns:
        float:
            Intersection area divided by the smaller box area.
            Returns 0.0 when either box is invalid.
    """
    try:
        box_a = [row_a["x1"], row_a["y1"], row_a["x2"], row_a["y2"]]
        box_b = [row_b["x1"], row_b["y1"], row_b["x2"], row_b["y2"]]
    except Exception:
        return 0.0

    area_a = box_area_xyxy(box_a)
    area_b = box_area_xyxy(box_b)

    if area_a <= 0 or area_b <= 0:
        return 0.0

    denom = max(1e-9, min(area_a, area_b))
    inter = intersection_area_xyxy(box_a, box_b)

    return float(np.clip(inter / denom, 0.0, 1.0))


# =============================================================================
# 16.16 CONTAINMENT SUPPRESSION HELPERS
# =============================================================================
# EXPLANATION:
# This helper removes duplicate or fragment crossarm detections.
#
# It uses two rules:
#   1) Normal duplicate rule:
#        Remove a smaller detection if it is mostly contained inside a larger
#        detection, unless mask evidence says they are probably separate objects.
#
#   2) Near-total fragment rule:
#        Remove a smaller detection when both box containment and mask
#        containment are very high, even if the smaller detection has a strong
#        score.
#
# INPUT CONTRACT:
#   detections_df must contain:
#       x1, y1, x2, y2, score, orig_det_idx
#
# IMPORTANT:
#   If masks are not available, this safely falls back to box-based logic.
# =============================================================================

def suppress_contained_shorter_detections(
    detections_df,
    containment_threshold=0.80,
    min_area_ratio=1.20,
    min_score_advantage=0.0,
    crossarm_mask_lookup=None,
    mask_containment_filter_enabled=True,
    mask_containment_veto_threshold=0.30,
    near_total_box_containment_threshold=0.95,
    mask_containment_high=0.80,
    pair_debug_min_box_containment=0.50,
):
    """
    Remove duplicate or fragment detections using box containment and optional
    mask containment evidence.

    Args:
        detections_df:
            DataFrame of candidate detections. Must contain x1, y1, x2, y2,
            score, and orig_det_idx.

        containment_threshold:
            Minimum fraction of the smaller box that must be inside the larger
            box for the normal duplicate rule.

        min_area_ratio:
            Minimum area ratio required for the larger box to be considered a
            valid container.

        min_score_advantage:
            Minimum score difference required for the container to suppress the
            candidate in the normal duplicate rule. With the default 0.0, the
            container score must be greater than or equal to the candidate score.

        crossarm_mask_lookup:
            Optional dictionary mapping orig_det_idx to a 2D boolean mask.

        mask_containment_filter_enabled:
            If True, use mask containment to veto false duplicate removals and
            confirm near-total fragments.

        mask_containment_veto_threshold:
            If mask containment is valid but below this value, the normal
            duplicate removal is vetoed.

        near_total_box_containment_threshold:
            Box-containment threshold for the near-total fragment rule.

        mask_containment_high:
            Mask-containment threshold for confirming a near-total fragment.

        pair_debug_min_box_containment:
            Minimum box containment required before recording a pair-level
            diagnostic row.

    Returns:
        tuple:
            kept_df:
                Detections that survive containment suppression.

            removed_df:
                Detections removed by containment suppression, with diagnostic
                columns added.

            pair_debug_df:
                Pair-level diagnostic table showing containment-rule decisions.
    """
    base_cols = list(detections_df.columns)

    if detections_df.empty:
        empty_pair = pd.DataFrame(
            columns=[
                "candidate_orig_det_idx_j",
                "container_orig_det_idx_i",
                "candidate_score_j",
                "container_score_i",
                "candidate_area_j",
                "container_area_i",
                "box_containment_of_j_inside_i",
                "mask_containment_of_j_inside_i",
                "area_ratio_i_over_j",
                "score_advantage_i_minus_j",
                "has_valid_mask_containment",
                "base_box_rule",
                "score_guard_passed",
                "mask_veto_active",
                "near_total_box_rule",
                "normal_duplicate_rule",
                "near_total_fragment_rule",
                "should_remove_candidate",
            ]
        )

        return detections_df.copy(), detections_df.iloc[0:0].copy(), empty_pair

    if crossarm_mask_lookup is None:
        crossarm_mask_lookup = {}

    df = detections_df.copy().reset_index(drop=True)

    df["box_area"] = df.apply(
        lambda r: box_area_xyxy(
            [
                r["x1"],
                r["y1"],
                r["x2"],
                r["y2"],
            ]
        ),
        axis=1,
    )

    n = len(df)

    keep_mask = np.ones(n, dtype=bool)
    removal_reason = [None] * n
    removed_by_idx = [np.nan] * n

    per_det_box_containment = [np.nan] * n
    per_det_mask_containment = [np.nan] * n
    per_det_area_ratio = [np.nan] * n
    per_det_score_advantage = [np.nan] * n
    per_det_normal_rule = [False] * n
    per_det_near_total_rule = [False] * n
    per_det_mask_veto_active = [False] * n

    pair_rows = []

    for j in range(n):
        if not keep_mask[j]:
            continue

        area_j = float(df.loc[j, "box_area"])
        score_j = float(df.loc[j, "score"])
        orig_j = int(df.loc[j, "orig_det_idx"])

        box_j = [
            float(df.loc[j, "x1"]),
            float(df.loc[j, "y1"]),
            float(df.loc[j, "x2"]),
            float(df.loc[j, "y2"]),
        ]

        mask_j = crossarm_mask_lookup.get(orig_j, None)

        if area_j <= 0:
            keep_mask[j] = False
            removal_reason[j] = "invalid_box_area"
            continue

        for i in range(n):
            if i == j:
                continue

            area_i = float(df.loc[i, "box_area"])
            score_i = float(df.loc[i, "score"])
            orig_i = int(df.loc[i, "orig_det_idx"])

            box_i = [
                float(df.loc[i, "x1"]),
                float(df.loc[i, "y1"]),
                float(df.loc[i, "x2"]),
                float(df.loc[i, "y2"]),
            ]

            mask_i = crossarm_mask_lookup.get(orig_i, None)

            if area_i <= 0:
                continue

            # i must be at least as big as j to be a plausible container.
            if area_i < area_j:
                continue

            inter = intersection_area_xyxy(
                box_i,
                box_j,
            )

            box_containment_of_j_inside_i = (
                inter / area_j
                if area_j > 0
                else 0.0
            )

            area_ratio = (
                area_i / area_j
                if area_j > 0
                else 0.0
            )

            score_advantage = score_i - score_j

            # Skip uninformative pairs in the diagnostic log and decision.
            if box_containment_of_j_inside_i < pair_debug_min_box_containment:
                continue

            has_valid_mask_containment, mask_containment_of_j_inside_i = (
                _safe_mask_containment(
                    mask_j,
                    mask_i,
                )
            )

            # -----------------------------------------------------------------
            # Branch A: box duplicate rule, with optional mask veto
            # -----------------------------------------------------------------
            base_box_rule = bool(
                box_containment_of_j_inside_i >= containment_threshold
                and area_ratio >= min_area_ratio
            )

            score_guard_passed = bool(
                score_advantage >= min_score_advantage
            )

            # Veto only fires when:
            #   - mask filtering is enabled
            #   - valid mask containment exists for this pair
            #   - mask containment is clearly low
            #
            # If any condition is False, Branch A behaves like the box rule.
            mask_veto_active = bool(
                mask_containment_filter_enabled
                and has_valid_mask_containment
                and mask_containment_of_j_inside_i < mask_containment_veto_threshold
            )

            normal_duplicate_rule = bool(
                base_box_rule
                and score_guard_passed
                and not mask_veto_active
            )

            # -----------------------------------------------------------------
            # Branch B: near-total fragment rule, confirmed by mask evidence
            # -----------------------------------------------------------------
            near_total_box_rule = bool(
                box_containment_of_j_inside_i >= near_total_box_containment_threshold
                and area_ratio >= min_area_ratio
            )

            if mask_containment_filter_enabled and has_valid_mask_containment:
                mask_high_passed = bool(
                    mask_containment_of_j_inside_i >= mask_containment_high
                )

                near_total_fragment_rule = bool(
                    near_total_box_rule
                    and mask_high_passed
                )
            else:
                near_total_fragment_rule = False

            should_remove_candidate = bool(
                normal_duplicate_rule
                or near_total_fragment_rule
            )

            pair_rows.append(
                {
                    "candidate_orig_det_idx_j": orig_j,
                    "container_orig_det_idx_i": orig_i,
                    "candidate_score_j": float(score_j),
                    "container_score_i": float(score_i),
                    "candidate_area_j": float(area_j),
                    "container_area_i": float(area_i),
                    "box_containment_of_j_inside_i": float(
                        box_containment_of_j_inside_i
                    ),
                    "mask_containment_of_j_inside_i": (
                        float(mask_containment_of_j_inside_i)
                        if has_valid_mask_containment
                        else np.nan
                    ),
                    "area_ratio_i_over_j": float(area_ratio),
                    "score_advantage_i_minus_j": float(score_advantage),
                    "has_valid_mask_containment": bool(has_valid_mask_containment),
                    "base_box_rule": bool(base_box_rule),
                    "score_guard_passed": bool(score_guard_passed),
                    "mask_veto_active": bool(mask_veto_active),
                    "near_total_box_rule": bool(near_total_box_rule),
                    "normal_duplicate_rule": bool(normal_duplicate_rule),
                    "near_total_fragment_rule": bool(near_total_fragment_rule),
                    "should_remove_candidate": bool(should_remove_candidate),
                }
            )

            if should_remove_candidate and keep_mask[j]:
                keep_mask[j] = False

                if near_total_fragment_rule and not normal_duplicate_rule:
                    removal_reason[j] = f"fragment_of_orig_{orig_i}"
                else:
                    removal_reason[j] = f"contained_in_orig_{orig_i}"

                removed_by_idx[j] = orig_i
                per_det_box_containment[j] = float(
                    box_containment_of_j_inside_i
                )

                per_det_mask_containment[j] = (
                    float(mask_containment_of_j_inside_i)
                    if has_valid_mask_containment
                    else np.nan
                )

                per_det_area_ratio[j] = float(area_ratio)
                per_det_score_advantage[j] = float(score_advantage)
                per_det_normal_rule[j] = bool(normal_duplicate_rule)
                per_det_near_total_rule[j] = bool(near_total_fragment_rule)
                per_det_mask_veto_active[j] = bool(mask_veto_active)

                break

    kept_cols = list(base_cols)

    if "box_area" not in kept_cols:
        kept_cols.append("box_area")

    kept_df = (
        df.loc[keep_mask, kept_cols]
        .copy()
        .reset_index(drop=True)
        )

    df_diag = df.copy()

    df_diag["removal_reason"] = removal_reason
    df_diag["removed_by_orig_det_idx"] = removed_by_idx
    df_diag["box_containment_of_j_inside_i"] = per_det_box_containment
    df_diag["mask_containment_of_j_inside_i"] = per_det_mask_containment
    df_diag["area_ratio_i_over_j"] = per_det_area_ratio
    df_diag["score_advantage_i_minus_j"] = per_det_score_advantage
    df_diag["normal_duplicate_rule"] = per_det_normal_rule
    df_diag["near_total_fragment_rule"] = per_det_near_total_rule
    df_diag["mask_veto_active"] = per_det_mask_veto_active

    removed_df = (
        df_diag.loc[~keep_mask]
        .copy()
        .reset_index(drop=True)
    )

    pair_debug_df = pd.DataFrame(pair_rows)

    return kept_df, removed_df, pair_debug_df


# =============================================================================
# 16.17 MAIN-CLUSTER HELPERS
# =============================================================================
# EXPLANATION:
# These helpers remove isolated detections that are far away from the main group
# of crossarm candidates.
#
# The logic is:
#   1) Compute each detection centre point and box diagonal.
#   2) Build connected components using centre-to-centre distance.
#   3) Keep the main component.
#
# Main component selection:
#   - largest component size wins first
#   - if tied, highest total score wins
#
# INPUT CONTRACT:
#   detections_df must contain:
#       x1, y1, x2, y2, score
# =============================================================================

def compute_centers_and_scale(detections_df):
    """
    Add centre-point and scale columns to a detection DataFrame.

    Args:
        detections_df:
            DataFrame containing x1, y1, x2, y2 columns.

    Returns:
        tuple:
            df:
                Copy of detections_df with cx, cy, w, h, and diag columns added.

            median_diag:
                Median positive box diagonal. Returns 0.0 if no valid diagonals
                exist.
    """
    if detections_df.empty:
        df = detections_df.copy()
        df["cx"] = []
        df["cy"] = []
        df["w"] = []
        df["h"] = []
        df["diag"] = []

        return df, 0.0

    df = detections_df.copy().reset_index(drop=True)

    df["cx"] = (df["x1"] + df["x2"]) / 2.0
    df["cy"] = (df["y1"] + df["y2"]) / 2.0
    df["w"] = (df["x2"] - df["x1"]).clip(lower=0.0)
    df["h"] = (df["y2"] - df["y1"]).clip(lower=0.0)
    df["diag"] = np.sqrt(df["w"] ** 2 + df["h"] ** 2)

    positive_diags = df.loc[df["diag"] > 0, "diag"]

    median_diag = (
        float(positive_diags.median())
        if len(positive_diags) > 0
        else 0.0
    )

    return df, median_diag


def connected_components_from_center_distance(df, center_dist_factor=2.75):
    """
    Build connected components from detection centre-point distances.

    Args:
        df:
            DataFrame containing cx, cy, and diag columns.

        center_dist_factor:
            Multiplier applied to the median box diagonal to create the distance
            threshold.

    Returns:
        tuple:
            components:
                List of connected components. Each component is a list of row
                indices.

            center_dist_threshold:
                Distance threshold used to connect detections.
    """
    n = len(df)

    if n == 0:
        return [], 0.0

    if n == 1:
        return [
            [0]
        ], float(center_dist_factor * float(df["diag"].iloc[0]))

    positive_diags = df.loc[df["diag"] > 0, "diag"]

    median_diag = (
        float(positive_diags.median())
        if len(positive_diags) > 0
        else 0.0
    )

    center_dist_threshold = float(center_dist_factor * median_diag)

    adjacency = {
        i: []
        for i in range(n)
    }

    visited = [False] * n
    components = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = math.hypot(
                float(df.loc[i, "cx"] - df.loc[j, "cx"]),
                float(df.loc[i, "cy"] - df.loc[j, "cy"]),
            )

            if dist <= center_dist_threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)

    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        comp = []
        visited[start] = True

        while stack:
            node = stack.pop()
            comp.append(node)

            for nbr in adjacency[node]:
                if not visited[nbr]:
                    visited[nbr] = True
                    stack.append(nbr)

        components.append(sorted(comp))

    return components, center_dist_threshold


def keep_main_detection_cluster(detections_df, center_dist_factor=2.75):
    """
    Keep the main spatial cluster of detections and remove isolated detections.

    Args:
        detections_df:
            DataFrame containing x1, y1, x2, y2, and score columns.

        center_dist_factor:
            Multiplier applied to the median box diagonal to create the cluster
            distance threshold.

    Returns:
        tuple:
            kept_df:
                Detections inside the selected main component.

            removed_df:
                Detections outside the selected main component. If non-empty,
                removal_reason is set to "outside_main_cluster".

            cluster_threshold:
                Centre-distance threshold used for clustering.
    """
    if detections_df.empty:
        return (
            detections_df.copy(),
            detections_df.iloc[0:0].copy(),
            0.0,
        )

    if len(detections_df) == 1:
        df1, median_diag = compute_centers_and_scale(
            detections_df
        )

        return (
            df1.reset_index(drop=True),
            df1.iloc[0:0].copy(),
            center_dist_factor * median_diag,
        )

    df, _ = compute_centers_and_scale(
        detections_df
    )

    components, cluster_threshold = connected_components_from_center_distance(
        df,
        center_dist_factor=center_dist_factor,
    )

    best_component = None
    best_key = None

    for comp in components:
        comp_df = df.iloc[comp]

        comp_size = len(comp)
        comp_score_sum = float(comp_df["score"].sum())

        key = (
            comp_size,
            comp_score_sum,
        )

        if best_key is None or key > best_key:
            best_key = key
            best_component = comp

    keep_idx = set(best_component)

    kept_df = (
        df.iloc[sorted(keep_idx)]
        .copy()
        .reset_index(drop=True)
    )

    removed_df = (
        df.drop(index=sorted(keep_idx))
        .copy()
        .reset_index(drop=True)
    )

    if len(removed_df) > 0:
        removed_df["removal_reason"] = "outside_main_cluster"

    return kept_df, removed_df, cluster_threshold


# =============================================================================
# 16.18 POLE-MASK PROJECTION HELPERS
# =============================================================================
# EXPLANATION:
# These helpers support the pole-overlap / pole-corridor filtering stage.
#
# They do two related jobs:
#   1) Project the selected full-source-image pole mask into the ROI crop space.
#   2) Measure how much each crossarm detection overlaps the projected pole mask.
#
# INPUT CONTRACT:
#   project_pole_mask_to_roi needs the source-to-ROI geometry from CELL 14:
#       src_x1, src_y1, src_x2, src_y2, dst_x1, dst_y1
#
#   compute_detection_overlap_with_pole_mask expects each detection row to have:
#       orig_det_idx, x1, y1, x2, y2
#
# IMPORTANT:
#   The projection arithmetic is deliberately explicit. It clips both the source
#   pole mask crop and the destination ROI paste region so slice shapes remain
#   aligned even near image boundaries.
#
#   The returned projected pole mask is always ROI-space with shape:
#       (roi_h, roi_w)
#
#   In 16C, call this helper with keyword arguments for roi_w and roi_h because
#   this helper's signature takes roi_w before roi_h.
# =============================================================================

def project_pole_mask_to_roi(
    pole_mask,
    src_x1,
    src_y1,
    src_x2,
    src_y2,
    dst_x1,
    dst_y1,
    roi_w,
    roi_h,
):
    """
    Project a full-source-image pole mask into the selected ROI crop space.

    Args:
        pole_mask:
            Full-source-image pole mask. Can be a NumPy array, torch tensor, or
            list/tuple containing one mask.

        src_x1:
            Source crop left coordinate in full-image space.

        src_y1:
            Source crop top coordinate in full-image space.

        src_x2:
            Source crop right coordinate in full-image space.

        src_y2:
            Source crop bottom coordinate in full-image space.

        dst_x1:
            Destination paste left coordinate in ROI space.

        dst_y1:
            Destination paste top coordinate in ROI space.

        roi_w:
            ROI crop width.

        roi_h:
            ROI crop height.

    Returns:
        np.ndarray:
            Boolean pole mask with shape (roi_h, roi_w). If projection is not
            possible, returns an all-False mask.
    """
    roi_w = int(roi_w)
    roi_h = int(roi_h)

    roi_mask = np.zeros(
        (
            roi_h,
            roi_w,
        ),
        dtype=bool,
    )

    if pole_mask is None:
        return roi_mask

    arr = pole_mask

    if isinstance(arr, (list, tuple)):
        if len(arr) == 0:
            return roi_mask

        arr = arr[0]

    arr = to_numpy(arr)

    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 3:
        arr = arr[0]

    if arr.ndim != 2:
        return roi_mask

    arr = arr.astype(bool)

    src_h, src_w = arr.shape

    src_x1 = int(src_x1)
    src_y1 = int(src_y1)
    src_x2 = int(src_x2)
    src_y2 = int(src_y2)
    dst_x1 = int(dst_x1)
    dst_y1 = int(dst_y1)

    clip_src_x1 = max(0, min(src_w, src_x1))
    clip_src_y1 = max(0, min(src_h, src_y1))
    clip_src_x2 = max(0, min(src_w, src_x2))
    clip_src_y2 = max(0, min(src_h, src_y2))

    if clip_src_x2 <= clip_src_x1 or clip_src_y2 <= clip_src_y1:
        return roi_mask

    src_crop = arr[
        clip_src_y1:clip_src_y2,
        clip_src_x1:clip_src_x2,
    ]

    paste_x1 = dst_x1 + (clip_src_x1 - src_x1)
    paste_y1 = dst_y1 + (clip_src_y1 - src_y1)
    paste_x2 = paste_x1 + (clip_src_x2 - clip_src_x1)
    paste_y2 = paste_y1 + (clip_src_y2 - clip_src_y1)

    dst_clip_x1 = max(0, min(roi_w, paste_x1))
    dst_clip_y1 = max(0, min(roi_h, paste_y1))
    dst_clip_x2 = max(0, min(roi_w, paste_x2))
    dst_clip_y2 = max(0, min(roi_h, paste_y2))

    if dst_clip_x2 <= dst_clip_x1 or dst_clip_y2 <= dst_clip_y1:
        return roi_mask

    src_off_x1 = dst_clip_x1 - paste_x1
    src_off_y1 = dst_clip_y1 - paste_y1
    src_off_x2 = src_off_x1 + (dst_clip_x2 - dst_clip_x1)
    src_off_y2 = src_off_y1 + (dst_clip_y2 - dst_clip_y1)

    roi_mask[
        dst_clip_y1:dst_clip_y2,
        dst_clip_x1:dst_clip_x2,
    ] = src_crop[
        src_off_y1:src_off_y2,
        src_off_x1:src_off_x2,
    ]

    return roi_mask


def compute_detection_overlap_with_pole_mask(
    det_row,
    projected_pole_mask,
    crossarm_mask_lookup=None,
):
    """
    Compute how much of one crossarm detection overlaps the projected pole mask.

    Preferred behaviour:
        If a SAM3 crossarm mask exists for this detection, compute:

            overlap = (crossarm_mask AND pole_mask).sum() / crossarm_mask.sum()

        This directly answers:
            "How much of this detected crossarm is actually pole?"

    Fallback behaviour:
        If the crossarm mask is unavailable, fall back to the box-based helper:

            overlap = pole pixels inside detection box / detection box area

    Args:
        det_row:
            One row from the crossarm detections DataFrame. Must contain
            orig_det_idx, x1, y1, x2, and y2.

        projected_pole_mask:
            2D boolean pole mask projected into ROI coordinates.

        crossarm_mask_lookup:
            Optional dictionary keyed by orig_det_idx containing 2D crossarm
            masks.

    Returns:
        tuple:
            overlap_fraction:
                Fraction of the detection overlapping the projected pole mask.

            overlap_source:
                One of:
                    "mask"         -> used crossarm mask overlap
                    "box_fallback" -> used box overlap fallback
                    "none"         -> no usable pole mask
    """
    if projected_pole_mask is None:
        return 0.0, "none"

    pole_mask = to_numpy(projected_pole_mask)

    if pole_mask.ndim != 2 or not np.any(pole_mask):
        return 0.0, "none"

    pole_mask = pole_mask.astype(bool)

    orig_idx = int(det_row["orig_det_idx"])
    crossarm_mask = None

    if crossarm_mask_lookup is not None:
        crossarm_mask = crossarm_mask_lookup.get(
            orig_idx,
            None,
        )

    if (
        isinstance(crossarm_mask, np.ndarray)
        and crossarm_mask.ndim == 2
        and crossarm_mask.shape == pole_mask.shape
        and crossarm_mask.sum() > 0
    ):
        det_mask = crossarm_mask.astype(bool)

        overlap_pixels = int(
            (
                det_mask
                & pole_mask
            ).sum()
        )

        det_pixels = int(det_mask.sum())

        if det_pixels > 0:
            return float(overlap_pixels / det_pixels), "mask"

    # Fallback only when no usable crossarm mask exists.
    box_i = [
        det_row["x1"],
        det_row["y1"],
        det_row["x2"],
        det_row["y2"],
    ]

    return (
        float(
            compute_box_overlap_with_mask(
                box_i,
                pole_mask,
            )
        ),
        "box_fallback",
    )
    
    
# =============================================================================
# 16.19 PCA / MASK-SHAPE HELPERS
# =============================================================================
# EXPLANATION:
# These helpers compute PCA-based shape information for binary crossarm masks.
#
# They are used later by:
#   - single-box X-shaped crossarm split
#   - same-crossarm continuity merge
#   - PCA / axis cleanup for broad non-X masks
#   - targeted two-box X ownership
#
# IMPORTANT:
#   These helpers do not display anything and do not mutate run-level state.
#
#   _canonical_angle_deg is defined later in the X-split helper section. That is
#   safe because Python resolves the function name when these helpers are called,
#   not when they are defined.
# =============================================================================

def compute_binary_mask_pca_stats(mask_bool):
    """
    Compute PCA summary statistics for a binary mask.

    Args:
        mask_bool:
            2D binary/boolean mask.

    Returns:
        dict:
            PCA summary fields:
              - valid
              - num_pixels
              - pc1_ratio
              - pc2_ratio
              - anisotropy
              - perp_std
    """
    out = {
        "valid": False,
        "num_pixels": 0,
        "pc1_ratio": np.nan,
        "pc2_ratio": np.nan,
        "anisotropy": np.nan,
        "perp_std": np.nan,
    }

    if mask_bool is None:
        return out

    mask = to_numpy(mask_bool)

    if mask.ndim != 2:
        return out

    mask = mask.astype(bool)

    ys, xs = np.where(mask)

    num_pixels = int(len(xs))
    out["num_pixels"] = num_pixels

    if num_pixels < 3:
        return out

    coords = np.column_stack(
        [
            xs.astype(np.float32),
            ys.astype(np.float32),
        ]
    )

    if coords.shape[0] < 3:
        return out

    coords = coords - coords.mean(
        axis=0,
        keepdims=True,
    )

    try:
        cov = np.cov(
            coords,
            rowvar=False,
        )

        eigvals, _ = np.linalg.eigh(cov)

    except Exception:
        return out

    eigvals = np.sort(
        np.maximum(
            eigvals,
            0.0,
        )
    )[::-1]

    if len(eigvals) < 2:
        return out

    total_var = float(eigvals.sum())

    if total_var <= 0:
        return out

    pc1_ratio = float(eigvals[0] / total_var)
    pc2_ratio = float(eigvals[1] / total_var)
    anisotropy = float(eigvals[0] / max(eigvals[1], 1e-8))
    perp_std = float(np.sqrt(max(eigvals[1], 0.0)))

    out.update(
        {
            "valid": True,
            "pc1_ratio": pc1_ratio,
            "pc2_ratio": pc2_ratio,
            "anisotropy": anisotropy,
            "perp_std": perp_std,
        }
    )

    return out


def _mask_pca_signature(mask_bool):
    """
    Compute quick PCA shape signature for a single detection mask.

    Args:
        mask_bool:
            2D binary/boolean mask.

    Returns:
        dict:
            PCA signature fields:
              - valid
              - num_pixels
              - pc1_ratio
              - anisotropy
              - angle_deg
    """
    out = {
        "valid": False,
        "num_pixels": 0,
        "pc1_ratio": np.nan,
        "anisotropy": np.nan,
        "angle_deg": np.nan,
    }

    if mask_bool is None:
        return out

    mask = to_numpy(mask_bool)

    if mask.ndim != 2:
        return out

    mask = mask.astype(bool)

    ys, xs = np.where(mask)

    if len(xs) < XSPLIT_MIN_PARENT_MASK_PIXELS:
        out["num_pixels"] = int(len(xs))
        return out

    xs_f = xs.astype(np.float64)
    ys_f = ys.astype(np.float64)

    pts = np.column_stack(
        [
            xs_f,
            ys_f,
        ]
    )

    pts_centered = pts - pts.mean(
        axis=0,
        keepdims=True,
    )

    try:
        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

    except Exception:
        return out

    eigvals = np.maximum(
        eigvals,
        0.0,
    )

    if eigvals.size < 2 or float(eigvals.sum()) <= 0:
        return out

    order = np.argsort(eigvals)[::-1]

    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    pc1_ratio = float(
        eigvals[0] / max(float(eigvals.sum()), 1e-9)
    )

    anisotropy = float(
        eigvals[0] / max(float(eigvals[1]), 1e-9)
    )

    angle_deg = _canonical_angle_deg(
        math.degrees(
            math.atan2(
                float(eigvecs[1, 0]),
                float(eigvecs[0, 0]),
            )
        )
    )

    out.update(
        {
            "valid": True,
            "num_pixels": int(len(xs)),
            "pc1_ratio": pc1_ratio,
            "anisotropy": anisotropy,
            "angle_deg": angle_deg,
        }
    )

    return out


def _fit_line_model_from_points(xs, ys):
    """
    Fit one undirected PCA line model through a set of points.

    Args:
        xs:
            X coordinates.

        ys:
            Y coordinates.

    Returns:
        dict:
            Line model with centre, unit direction, angle, and validity flag.
    """
    out = {
        "valid": False,
        "cx": np.nan,
        "cy": np.nan,
        "ux": np.nan,
        "uy": np.nan,
        "angle_deg": np.nan,
        "num_pixels": 0,
    }

    xs = np.asarray(
        xs,
        dtype=np.float64,
    ).reshape(-1)

    ys = np.asarray(
        ys,
        dtype=np.float64,
    ).reshape(-1)

    if xs.size < 2 or ys.size < 2 or xs.size != ys.size:
        return out

    cx = float(xs.mean())
    cy = float(ys.mean())

    pts = np.column_stack(
        [
            xs,
            ys,
        ]
    )

    pts_centered = pts - pts.mean(
        axis=0,
        keepdims=True,
    )

    try:
        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

    except Exception:
        return out

    if eigvals.size < 2:
        return out

    order = np.argsort(eigvals)[::-1]
    eigvec = eigvecs[:, order[0]]

    ux = float(eigvec[0])
    uy = float(eigvec[1])

    norm = math.hypot(
        ux,
        uy,
    )

    if norm <= 0:
        return out

    ux = ux / norm
    uy = uy / norm

    if (ux < 0) or (abs(ux) < 1e-6 and uy < 0):
        ux = -ux
        uy = -uy

    angle_deg = _canonical_angle_deg(
        math.degrees(
            math.atan2(
                uy,
                ux,
            )
        )
    )

    out.update(
        {
            "valid": True,
            "cx": cx,
            "cy": cy,
            "ux": ux,
            "uy": uy,
            "angle_deg": float(angle_deg),
            "num_pixels": int(xs.size),
        }
    )

    return out


def _line_distance_for_points(xs, ys, model):
    """
    Compute perpendicular distance from points to a PCA line model.

    Args:
        xs:
            X coordinates.

        ys:
            Y coordinates.

        model:
            PCA line model containing cx, cy, ux, and uy.

    Returns:
        np.ndarray:
            Absolute perpendicular distances from points to the line model.
    """
    xs = np.asarray(
        xs,
        dtype=np.float64,
    )

    ys = np.asarray(
        ys,
        dtype=np.float64,
    )

    cx = float(model["cx"])
    cy = float(model["cy"])
    ux = float(model["ux"])
    uy = float(model["uy"])

    # Perpendicular unit vector.
    vx = -uy
    vy = ux

    return np.abs(
        (xs - cx) * vx
        + (ys - cy) * vy
    )


def _fit_axis_cleanup_model(mask_bool):
    """
    Fit a dominant PCA axis to one crossarm mask.

    Args:
        mask_bool:
            2D boolean SAM3 mask.

    Returns:
        dict:
            PCA axis model and quality metrics.
    """
    out = {
        "valid": False,
        "num_pixels": 0,
        "cx": np.nan,
        "cy": np.nan,
        "ux": np.nan,
        "uy": np.nan,
        "angle_deg": np.nan,
        "pc1_ratio": np.nan,
        "anisotropy": np.nan,
        "perp_std": np.nan,
        "perp_median_abs": np.nan,
        "perp_mad_abs": np.nan,
    }

    if mask_bool is None:
        return out

    mask = to_numpy(mask_bool)

    if mask.ndim != 2:
        return out

    mask = mask.astype(bool)

    ys, xs = np.where(mask)

    num_pixels = int(len(xs))
    out["num_pixels"] = num_pixels

    if num_pixels < int(AXIS_CLEANUP_MIN_MASK_PIXELS):
        return out

    xs_f = xs.astype(np.float64)
    ys_f = ys.astype(np.float64)

    cx = float(xs_f.mean())
    cy = float(ys_f.mean())

    pts = np.column_stack(
        [
            xs_f,
            ys_f,
        ]
    )

    pts_centered = pts - pts.mean(
        axis=0,
        keepdims=True,
    )

    try:
        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

    except Exception:
        return out

    eigvals = np.maximum(
        eigvals,
        0.0,
    )

    if eigvals.size < 2 or float(eigvals.sum()) <= 0:
        return out

    order = np.argsort(eigvals)[::-1]

    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    ux = float(eigvecs[0, 0])
    uy = float(eigvecs[1, 0])

    norm = math.hypot(
        ux,
        uy,
    )

    if norm <= 0:
        return out

    ux = ux / norm
    uy = uy / norm

    if (ux < 0) or (abs(ux) < 1e-6 and uy < 0):
        ux = -ux
        uy = -uy

    # Perpendicular vector.
    vx = -uy
    vy = ux

    perp = (
        (xs_f - cx) * vx
        + (ys_f - cy) * vy
    )

    abs_perp = np.abs(perp)

    pc1_ratio = float(
        eigvals[0] / max(float(eigvals.sum()), 1e-9)
    )

    anisotropy = float(
        eigvals[0] / max(float(eigvals[1]), 1e-9)
    )

    perp_std = float(
        np.sqrt(
            max(
                float(eigvals[1]),
                0.0,
            )
        )
    )

    perp_median_abs = float(np.median(abs_perp))

    perp_mad_abs = float(
        np.median(
            np.abs(abs_perp - perp_median_abs)
        )
    )

    angle_deg = _canonical_angle_deg(
        math.degrees(
            math.atan2(
                uy,
                ux,
            )
        )
    )

    out.update(
        {
            "valid": True,
            "cx": cx,
            "cy": cy,
            "ux": ux,
            "uy": uy,
            "angle_deg": angle_deg,
            "pc1_ratio": pc1_ratio,
            "anisotropy": anisotropy,
            "perp_std": perp_std,
            "perp_median_abs": perp_median_abs,
            "perp_mad_abs": perp_mad_abs,
        }
    )

    return out


def _axis_cleanup_mask(mask_bool, model, half_width_px):
    """
    Keep only mask pixels close to the dominant PCA axis.

    Args:
        mask_bool:
            2D boolean mask.

        model:
            PCA axis model from _fit_axis_cleanup_model.

        half_width_px:
            Half-width around the PCA axis to retain.

    Returns:
        np.ndarray:
            Cleaned boolean mask.
    """
    if mask_bool is None:
        return np.zeros((0, 0), dtype=bool)

    mask = to_numpy(mask_bool)

    if mask.ndim != 2:
        return np.zeros((0, 0), dtype=bool)

    mask = mask.astype(bool)

    if not bool(model.get("valid", False)):
        return mask.copy()

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return mask.copy()

    xs_f = xs.astype(np.float64)
    ys_f = ys.astype(np.float64)

    cx = float(model["cx"])
    cy = float(model["cy"])
    ux = float(model["ux"])
    uy = float(model["uy"])

    vx = -uy
    vy = ux

    perp = (
        (xs_f - cx) * vx
        + (ys_f - cy) * vy
    )

    keep_pix = np.abs(perp) <= float(half_width_px)

    cleaned = np.zeros_like(
        mask,
        dtype=bool,
    )

    cleaned[
        ys[keep_pix],
        xs[keep_pix],
    ] = True

    return cleaned


# =============================================================================
# 16.20 SAME-CROSSARM CONTINUITY MERGE HELPERS
# =============================================================================
# EXPLANATION:
# These helpers identify when two separate SAM3 crossarm detections are actually
# two fragments of the same physical crossarm.
#
# This is different from:
#   - Single-box X split:
#       one detection contains two crossed crossarms.
#
#   - Two-box X ownership:
#       two detections overlap at an X crossing.
#
# Here, the issue is:
#   - one physical crossarm is split into two boxes/masks, often because the pole
#     or another object interrupts the detection.
#
# IMPORTANT:
#   merge_same_crossarm_fragments mutates the per-ROI crossarm_mask_lookup only
#   when it creates a merged detection. The new merged mask is stored under a
#   new integer orig_det_idx.
#
#   This is safe because crossarm_mask_lookup is reset inside each ROI iteration
#   in 16C.
#
# FORWARD REFERENCES:
#   This section uses _append_reason and _angle_diff_undirected_deg, which are
#   defined later in the X-split helper section. This is safe as long as all 16B
#   helper sections are executed before the 16C batch loop starts.
# =============================================================================

def _same_xarm_get_mask(row, crossarm_mask_lookup):
    """
    Get a crossarm mask for one detection row.

    Args:
        row:
            Detection row containing orig_det_idx.

        crossarm_mask_lookup:
            Per-ROI dictionary keyed by integer orig_det_idx.

    Returns:
        np.ndarray or None:
            Matching crossarm mask when available, otherwise None.
    """
    if crossarm_mask_lookup is None:
        return None

    if "orig_det_idx" not in row:
        return None

    try:
        return crossarm_mask_lookup.get(
            int(row["orig_det_idx"]),
            None,
        )

    except Exception:
        return None


def _same_xarm_fit_line_model(row, crossarm_mask_lookup):
    """
    Fit a PCA line model for a possible crossarm fragment.

    Args:
        row:
            Detection row.

        crossarm_mask_lookup:
            Per-ROI dictionary keyed by integer orig_det_idx.

    Returns:
        dict:
            Line model with validity flag, reason, angle, centre, unit direction,
            and mask pixel count.
    """
    out = {
        "valid": False,
        "reason": "not_run",
        "cx": np.nan,
        "cy": np.nan,
        "ux": np.nan,
        "uy": np.nan,
        "angle_deg": np.nan,
        "num_pixels": 0,
    }

    mask_i = _same_xarm_get_mask(
        row=row,
        crossarm_mask_lookup=crossarm_mask_lookup,
    )

    if not isinstance(mask_i, np.ndarray) or mask_i.ndim != 2:
        out["reason"] = "missing_or_invalid_mask"
        return out

    mask_i = mask_i.astype(bool)

    ys, xs = np.where(mask_i)

    num_pixels = int(len(xs))
    out["num_pixels"] = num_pixels

    if num_pixels < int(SAME_XARM_MERGE_MIN_MASK_PIXELS):
        out["reason"] = "mask_too_small"
        return out

    model = _fit_line_model_from_points(
        xs,
        ys,
    )

    if not bool(model.get("valid", False)):
        out["reason"] = "line_fit_failed"
        return out

    model["reason"] = "ok"
    model["num_pixels"] = num_pixels

    return model


def _same_xarm_span_on_model(mask_bool, model):
    """
    Project mask pixels onto a fitted line model and return the along-axis span.

    Args:
        mask_bool:
            2D boolean mask.

        model:
            PCA line model with cx, cy, ux, and uy.

    Returns:
        dict:
            Span fields:
              - valid
              - t_min
              - t_max
              - length
    """
    out = {
        "valid": False,
        "t_min": np.nan,
        "t_max": np.nan,
        "length": np.nan,
    }

    if not isinstance(mask_bool, np.ndarray) or mask_bool.ndim != 2:
        return out

    if not bool(model.get("valid", False)):
        return out

    ys, xs = np.where(mask_bool.astype(bool))

    if len(xs) == 0:
        return out

    xs_f = xs.astype(np.float64)
    ys_f = ys.astype(np.float64)

    cx = float(model["cx"])
    cy = float(model["cy"])
    ux = float(model["ux"])
    uy = float(model["uy"])

    t = (
        (xs_f - cx) * ux
        + (ys_f - cy) * uy
    )

    t_min = float(np.min(t))
    t_max = float(np.max(t))

    out.update(
        {
            "valid": True,
            "t_min": t_min,
            "t_max": t_max,
            "length": float(t_max - t_min),
        }
    )

    return out


def _same_xarm_gap_between_spans(span_a, span_b):
    """
    Compute along-axis gap between two projected spans.

    Args:
        span_a:
            First span dictionary.

        span_b:
            Second span dictionary.

    Returns:
        float:
            Gap in pixels. Returns 0.0 when spans overlap.
            Returns np.nan when either span is invalid.
    """
    if not bool(span_a.get("valid", False)):
        return np.nan

    if not bool(span_b.get("valid", False)):
        return np.nan

    a_min = float(span_a["t_min"])
    a_max = float(span_a["t_max"])
    b_min = float(span_b["t_min"])
    b_max = float(span_b["t_max"])

    # If spans overlap, gap is zero.
    return float(
        max(
            0.0,
            max(a_min, b_min) - min(a_max, b_max),
        )
    )


def _same_xarm_attach_corridor(projected_pole_mask, roi_w):
    """
    Return the expanded pole attachment corridor x-range.

    Args:
        projected_pole_mask:
            Selected pole mask projected into ROI space.

        roi_w:
            ROI image width.

    Returns:
        dict:
            Corridor fields:
              - available
              - attach_x1
              - attach_x2
              - pole_x1
              - pole_x2
    """
    out = {
        "available": False,
        "attach_x1": np.nan,
        "attach_x2": np.nan,
        "pole_x1": np.nan,
        "pole_x2": np.nan,
    }

    if (
        not isinstance(projected_pole_mask, np.ndarray)
        or projected_pole_mask.ndim != 2
    ):
        return out

    pole_cols = np.where(
        projected_pole_mask.astype(bool).any(axis=0)
    )[0]

    if len(pole_cols) == 0:
        return out

    pole_x1 = int(pole_cols.min())
    pole_x2 = int(pole_cols.max())

    attach_margin_px = int(POLE_ATTACH_MARGIN_PX)

    attach_x1 = max(
        0,
        pole_x1 - attach_margin_px,
    )

    attach_x2 = min(
        int(roi_w) - 1,
        pole_x2 + attach_margin_px,
    )

    out.update(
        {
            "available": True,
            "attach_x1": int(attach_x1),
            "attach_x2": int(attach_x2),
            "pole_x1": int(pole_x1),
            "pole_x2": int(pole_x2),
        }
    )

    return out


def _same_xarm_pair_bridges_corridor(row_a, row_b, corridor_info):
    """
    Check whether the combined pair spans the expanded pole corridor.

    Args:
        row_a:
            First detection row.

        row_b:
            Second detection row.

        corridor_info:
            Corridor dictionary from _same_xarm_attach_corridor.

    Returns:
        bool:
            True when the pair spans the selected-pole attachment corridor.
    """
    if not bool(corridor_info.get("available", False)):
        return not bool(SAME_XARM_MERGE_REQUIRE_POLE_BRIDGE)

    attach_x1 = float(corridor_info["attach_x1"])
    attach_x2 = float(corridor_info["attach_x2"])

    pair_x1 = min(
        float(row_a["x1"]),
        float(row_b["x1"]),
    )

    pair_x2 = max(
        float(row_a["x2"]),
        float(row_b["x2"]),
    )

    # For continuity merging, the important question is whether the line/box
    # formed by the two fragments bridges the selected-pole corridor. The two
    # individual fragment boxes may each have tiny direct pole contact, but the
    # combined pair should span the pole region if they are one physical crossarm.
    pair_spans_corridor = bool(
        pair_x1 <= attach_x2
        and pair_x2 >= attach_x1
    )

    return bool(pair_spans_corridor)


def _same_xarm_pair_candidate(
    row_a,
    row_b,
    model_a,
    model_b,
    mask_a,
    mask_b,
    corridor_info,
):
    """
    Decide whether two detections should be merged as fragments of one crossarm.

    Args:
        row_a:
            First detection row.

        row_b:
            Second detection row.

        model_a:
            PCA line model for first detection.

        model_b:
            PCA line model for second detection.

        mask_a:
            First detection mask.

        mask_b:
            Second detection mask.

        corridor_info:
            Expanded pole-corridor information.

    Returns:
        dict:
            Pair candidate decision and diagnostic fields.
    """
    out = {
        "pair_candidate": False,
        "reason": "not_run",
        "angle_diff": np.nan,
        "perp_dist_ab": np.nan,
        "perp_dist_ba": np.nan,
        "perp_dist_max": np.nan,
        "gap_px": np.nan,
        "bridges_attach_corridor": False,
    }

    if not bool(model_a.get("valid", False)) or not bool(model_b.get("valid", False)):
        out["reason"] = "invalid_line_model"
        return out

    angle_diff = _angle_diff_undirected_deg(
        model_a.get("angle_deg", np.nan),
        model_b.get("angle_deg", np.nan),
    )

    out["angle_diff"] = float(angle_diff)

    if angle_diff > float(SAME_XARM_MERGE_MAX_ANGLE_DIFF_DEG):
        out["reason"] = "angle_diff_too_large"
        return out

    # Measure whether each fragment centre lies near the other fragment's axis.
    perp_ab = float(
        _line_distance_for_points(
            np.array([float(model_b["cx"])]),
            np.array([float(model_b["cy"])]),
            model_a,
        )[0]
    )

    perp_ba = float(
        _line_distance_for_points(
            np.array([float(model_a["cx"])]),
            np.array([float(model_a["cy"])]),
            model_b,
        )[0]
    )

    perp_max = max(
        perp_ab,
        perp_ba,
    )

    out["perp_dist_ab"] = perp_ab
    out["perp_dist_ba"] = perp_ba
    out["perp_dist_max"] = perp_max

    if perp_max > float(SAME_XARM_MERGE_MAX_PERP_DIST_PX):
        out["reason"] = "perpendicular_distance_too_large"
        return out

    # Compute the gap using model A as the common line direction.
    span_a = _same_xarm_span_on_model(
        mask_a,
        model_a,
    )

    span_b = _same_xarm_span_on_model(
        mask_b,
        model_a,
    )

    gap_px = _same_xarm_gap_between_spans(
        span_a,
        span_b,
    )

    out["gap_px"] = (
        float(gap_px)
        if np.isfinite(gap_px)
        else np.nan
    )

    if not np.isfinite(gap_px):
        out["reason"] = "invalid_axis_gap"
        return out

    if gap_px > float(SAME_XARM_MERGE_MAX_GAP_PX):
        out["reason"] = "gap_too_large"
        return out

    bridges_corridor = _same_xarm_pair_bridges_corridor(
        row_a,
        row_b,
        corridor_info,
    )

    out["bridges_attach_corridor"] = bool(bridges_corridor)

    if bool(SAME_XARM_MERGE_REQUIRE_POLE_BRIDGE) and not bridges_corridor:
        out["reason"] = "does_not_bridge_attach_corridor"
        return out

    out["pair_candidate"] = True
    out["reason"] = "same_crossarm_fragment_pair"

    return out


def merge_same_crossarm_fragments(
    detections_df,
    crossarm_mask_lookup,
    projected_pole_mask,
    roi_w,
    roi_h,
):
    """
    Merge separate same-line crossarm fragments into one detection.

    Args:
        detections_df:
            Detections after pole-overlap / pole-corridor filtering.

        crossarm_mask_lookup:
            Per-ROI dictionary of crossarm masks keyed by integer orig_det_idx.

        projected_pole_mask:
            Selected pole mask projected into ROI space.

        roi_w:
            ROI width.

        roi_h:
            ROI height.

    Returns:
        tuple:
            merged_df:
                Output detections after same-crossarm merge.

            removed_df:
                Original detections replaced by merged detections.

            pair_debug_df:
                Pair-level merge diagnostics.

            component_debug_df:
                Component-level merge diagnostics.
    """
    if detections_df is None or len(detections_df) == 0:
        empty_df = (
            detections_df.copy()
            if isinstance(detections_df, pd.DataFrame)
            else pd.DataFrame()
        )

        return (
            empty_df,
            empty_df.copy(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    if not bool(SAME_XARM_MERGE_ENABLED) or len(detections_df) < 2:
        df = detections_df.copy().reset_index(drop=True)

        df["same_xarm_merge_applied"] = False
        df["same_xarm_merge_group_id"] = np.nan
        df["same_xarm_merge_count"] = 1
        df["merged_from_orig_det_idxs"] = df["orig_det_idx"].astype(str)

        return (
            df,
            df.iloc[0:0].copy(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    df = detections_df.copy().reset_index(drop=True)

    corridor_info = _same_xarm_attach_corridor(
        projected_pole_mask,
        roi_w,
    )

    models = {}
    masks = {}

    for _, row_i in df.iterrows():
        orig_i = int(row_i["orig_det_idx"])

        masks[orig_i] = _same_xarm_get_mask(
            row=row_i,
            crossarm_mask_lookup=crossarm_mask_lookup,
        )

        models[orig_i] = _same_xarm_fit_line_model(
            row=row_i,
            crossarm_mask_lookup=crossarm_mask_lookup,
        )

    pair_debug_rows = []
    edges = []

    for i in range(len(df)):
        row_i = df.iloc[i]
        orig_i = int(row_i["orig_det_idx"])

        for j in range(i + 1, len(df)):
            row_j = df.iloc[j]
            orig_j = int(row_j["orig_det_idx"])

            result = _same_xarm_pair_candidate(
                row_a=row_i,
                row_b=row_j,
                model_a=models[orig_i],
                model_b=models[orig_j],
                mask_a=masks[orig_i],
                mask_b=masks[orig_j],
                corridor_info=corridor_info,
            )

            pair_debug_rows.append(
                {
                    "orig_i": orig_i,
                    "orig_j": orig_j,
                    "pair_candidate": bool(result["pair_candidate"]),
                    "reason": str(result["reason"]),
                    "angle_diff": (
                        float(result["angle_diff"])
                        if np.isfinite(result["angle_diff"])
                        else np.nan
                    ),
                    "perp_dist_ab": (
                        float(result["perp_dist_ab"])
                        if np.isfinite(result["perp_dist_ab"])
                        else np.nan
                    ),
                    "perp_dist_ba": (
                        float(result["perp_dist_ba"])
                        if np.isfinite(result["perp_dist_ba"])
                        else np.nan
                    ),
                    "perp_dist_max": (
                        float(result["perp_dist_max"])
                        if np.isfinite(result["perp_dist_max"])
                        else np.nan
                    ),
                    "gap_px": (
                        float(result["gap_px"])
                        if np.isfinite(result["gap_px"])
                        else np.nan
                    ),
                    "bridges_attach_corridor": bool(
                        result["bridges_attach_corridor"]
                    ),
                    "corridor_available": bool(
                        corridor_info.get("available", False)
                    ),
                    "attach_x1": corridor_info.get("attach_x1", np.nan),
                    "attach_x2": corridor_info.get("attach_x2", np.nan),
                }
            )

            if bool(result["pair_candidate"]):
                edges.append(
                    (
                        i,
                        j,
                    )
                )

    # Connected components from candidate merge edges.
    adjacency = {
        i: []
        for i in range(len(df))
    }

    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    visited = [False] * len(df)
    components = []

    for start in range(len(df)):
        if visited[start]:
            continue

        stack = [start]
        comp = []
        visited[start] = True

        while stack:
            node = stack.pop()
            comp.append(node)

            for nbr in adjacency[node]:
                if not visited[nbr]:
                    visited[nbr] = True
                    stack.append(nbr)

        components.append(sorted(comp))

    existing_orig_idxs = [
        int(v)
        for v in df["orig_det_idx"].dropna().tolist()
    ]

    existing_orig_idxs.extend(
        [
            int(k)
            for k in crossarm_mask_lookup.keys()
        ]
    )

    next_orig_idx = max(existing_orig_idxs + [-1]) + 1

    output_rows = []
    removed_rows = []
    component_debug_rows = []
    merge_group_id = 1

    for comp in components:
        comp_df = df.iloc[comp].copy().reset_index(drop=True)

        if len(comp) == 1:
            row_out = comp_df.iloc[0].copy()

            row_out["same_xarm_merge_applied"] = False
            row_out["same_xarm_merge_group_id"] = np.nan
            row_out["same_xarm_merge_count"] = 1
            row_out["merged_from_orig_det_idxs"] = str(
                int(row_out["orig_det_idx"])
            )

            output_rows.append(row_out)
            continue

        parent_orig_idxs = [
            int(v)
            for v in comp_df["orig_det_idx"].tolist()
        ]

        parent_masks = [
            crossarm_mask_lookup.get(
                int(orig_idx),
                None,
            )
            for orig_idx in parent_orig_idxs
        ]

        valid_masks = [
            mask_i.astype(bool)
            for mask_i in parent_masks
            if isinstance(mask_i, np.ndarray)
            and mask_i.ndim == 2
            and mask_i.shape == (int(roi_h), int(roi_w))
        ]

        union_mask = None

        if len(valid_masks) > 0:
            union_mask = np.zeros(
                (
                    int(roi_h),
                    int(roi_w),
                ),
                dtype=bool,
            )

            for mask_i in valid_masks:
                union_mask |= mask_i.astype(bool)

        best_row_idx = int(
            comp_df["score"].astype(float).idxmax()
        )

        merged_row = comp_df.loc[best_row_idx].copy()

        new_orig_idx = int(next_orig_idx)
        next_orig_idx += 1

        if union_mask is not None and union_mask.sum() > 0:
            valid_box, mx1, my1, mx2, my2 = _bbox_from_mask(
                union_mask,
                pad_px=SAME_XARM_MERGE_BOX_PAD_PX,
                image_w=roi_w,
                image_h=roi_h,
            )

        else:
            valid_box = True

            mx1 = float(comp_df["x1"].min()) - SAME_XARM_MERGE_BOX_PAD_PX
            my1 = float(comp_df["y1"].min()) - SAME_XARM_MERGE_BOX_PAD_PX
            mx2 = float(comp_df["x2"].max()) + SAME_XARM_MERGE_BOX_PAD_PX
            my2 = float(comp_df["y2"].max()) + SAME_XARM_MERGE_BOX_PAD_PX

            mx1 = max(
                0.0,
                min(float(roi_w), mx1),
            )

            mx2 = max(
                0.0,
                min(float(roi_w), mx2),
            )

            my1 = max(
                0.0,
                min(float(roi_h), my1),
            )

            my2 = max(
                0.0,
                min(float(roi_h), my2),
            )

        if not valid_box:
            # Safety fallback: keep originals unchanged.
            for _, row_keep in comp_df.iterrows():
                row_keep = row_keep.copy()

                row_keep["same_xarm_merge_applied"] = False
                row_keep["same_xarm_merge_group_id"] = np.nan
                row_keep["same_xarm_merge_count"] = 1
                row_keep["merged_from_orig_det_idxs"] = str(
                    int(row_keep["orig_det_idx"])
                )

                output_rows.append(row_keep)

            continue

        if str(SAME_XARM_MERGE_SCORE_MODE).lower() == "mean":
            merged_score = float(
                comp_df["score"].astype(float).mean()
            )

        else:
            merged_score = float(
                comp_df["score"].astype(float).max()
            )

        merged_row["orig_det_idx"] = new_orig_idx
        merged_row["score"] = merged_score
        merged_row["x1"] = float(mx1)
        merged_row["y1"] = float(my1)
        merged_row["x2"] = float(mx2)
        merged_row["y2"] = float(my2)
        merged_row["box_w"] = float(mx2 - mx1)
        merged_row["box_h"] = float(my2 - my1)
        merged_row["has_mask"] = bool(
            union_mask is not None
            and union_mask.sum() > 0
        )

        merged_row["same_xarm_merge_applied"] = True
        merged_row["same_xarm_merge_group_id"] = int(merge_group_id)
        merged_row["same_xarm_merge_count"] = int(len(comp))
        merged_row["merged_from_orig_det_idxs"] = ",".join(
            [
                str(v)
                for v in parent_orig_idxs
            ]
        )

        merged_row["review_reason"] = _append_reason(
            merged_row.get("review_reason", ""),
            "same_crossarm_continuity_merge",
        )

        # Recompute pole diagnostic fields for the merged detection where possible.
        if union_mask is not None and union_mask.sum() > 0:
            crossarm_mask_lookup[new_orig_idx] = union_mask.astype(bool)

            if (
                isinstance(projected_pole_mask, np.ndarray)
                and projected_pole_mask.ndim == 2
            ):
                overlap_frac, overlap_source = compute_detection_overlap_with_pole_mask(
                    det_row=merged_row,
                    projected_pole_mask=projected_pole_mask,
                    crossarm_mask_lookup=crossarm_mask_lookup,
                )

                merged_row["pole_overlap_fraction"] = float(overlap_frac)
                merged_row["pole_overlap_source"] = f"merged_{overlap_source}"

                merged_row["pole_touch_fraction"] = float(
                    compute_box_overlap_with_mask(
                        box_xyxy=[
                            merged_row["x1"],
                            merged_row["y1"],
                            merged_row["x2"],
                            merged_row["y2"],
                        ],
                        binary_mask=projected_pole_mask,
                    )
                )

                merged_row["pole_overlap_touches_min"] = bool(
                    merged_row["pole_touch_fraction"]
                    >= POLE_OVERLAP_MIN_FRACTION
                )

                merged_row["pole_overlap_under_max"] = bool(
                    merged_row["pole_overlap_fraction"]
                    <= POLE_OVERLAP_MAX_FRACTION
                )

                merged_row["pole_dominated_reject"] = bool(
                    merged_row["pole_overlap_fraction"]
                    > POLE_OVERLAP_MAX_FRACTION
                )

        output_rows.append(merged_row)

        for _, removed_row in comp_df.iterrows():
            removed_row = removed_row.copy()

            removed_row["removal_reason"] = (
                f"merged_into_same_crossarm_orig_{new_orig_idx}"
            )

            removed_row["same_xarm_merge_group_id"] = int(merge_group_id)

            removed_rows.append(removed_row)

        component_debug_rows.append(
            {
                "same_xarm_merge_group_id": int(merge_group_id),
                "new_orig_det_idx": int(new_orig_idx),
                "merged_from_orig_det_idxs": ",".join(
                    [
                        str(v)
                        for v in parent_orig_idxs
                    ]
                ),
                "merge_count": int(len(comp)),
                "merged_score": float(merged_score),
                "merged_box_x1": float(mx1),
                "merged_box_y1": float(my1),
                "merged_box_x2": float(mx2),
                "merged_box_y2": float(my2),
                "has_union_mask": bool(
                    union_mask is not None
                    and union_mask.sum() > 0
                ),
            }
        )

        merge_group_id += 1

    merged_df = pd.DataFrame(output_rows).reset_index(drop=True)

    removed_df = (
        pd.DataFrame(removed_rows).reset_index(drop=True)
        if len(removed_rows) > 0
        else df.iloc[0:0].copy()
    )

    pair_debug_df = pd.DataFrame(pair_debug_rows)
    component_debug_df = pd.DataFrame(component_debug_rows)

    return (
        merged_df,
        removed_df,
        pair_debug_df,
        component_debug_df,
    )
    
    
# =============================================================================
# 16.21 X-SPLIT HELPER UTILITIES
# =============================================================================
# EXPLANATION:
# These helpers support the single-box X-split stage.
#
# They detect X-shaped masks by:
#   1) skeletonising the parent SAM3 mask
#   2) running Hough line detection
#   3) clustering line segments into two angle groups
#   4) splitting parent mask pixels into two child masks using fitted line models
#
# IMPORTANT:
#   This section also defines shared helper utilities used earlier by 16.19 and
#   16.20:
#       _append_reason
#       _canonical_angle_deg
#       _angle_diff_undirected_deg
#
#   Those earlier helpers reference these names, but that is safe because Python
#   resolves function names at call time. The 16C batch loop runs only after all
#   16B helper sections have been executed.
#
# PRODUCTION NOTE:
#   _morphological_skeleton uses the CELL 16A production skimage variables:
#       CELL16_SKIMAGE_SKELETONIZE_AVAILABLE
#       _cell16_skimage_skeletonize
#
#   It does not use the old debug names:
#       SKIMAGE_SKELETONIZE_AVAILABLE
#       _skimage_skeletonize
# =============================================================================

def _append_reason(existing_reason, new_reason):
    """
    Append a semi-colon-separated review / diagnostic reason.

    Args:
        existing_reason:
            Existing reason value, possibly NaN, None, or empty.

        new_reason:
            New reason string to append.

    Returns:
        str:
            Combined reason string.
    """
    if new_reason is None or str(new_reason).strip() == "":
        return "" if pd.isna(existing_reason) else str(existing_reason)

    if (
        existing_reason is None
        or pd.isna(existing_reason)
        or str(existing_reason).strip() == ""
    ):
        return str(new_reason)

    existing_reason = str(existing_reason)
    new_reason = str(new_reason)

    if new_reason in existing_reason.split(";"):
        return existing_reason

    return existing_reason + ";" + new_reason


def _canonical_angle_deg(angle_deg):
    """
    Convert any angle to an undirected [0, 180) degree angle.

    Args:
        angle_deg:
            Input angle in degrees.

    Returns:
        float:
            Canonical undirected angle in [0, 180).
    """
    return float(angle_deg) % 180.0


def _angle_diff_undirected_deg(a, b):
    """
    Compute the smallest difference between two undirected line angles.

    Args:
        a:
            First angle in degrees.

        b:
            Second angle in degrees.

    Returns:
        float:
            Smallest angular difference in degrees.
    """
    a = _canonical_angle_deg(a)
    b = _canonical_angle_deg(b)

    d = abs(a - b) % 180.0

    return float(
        min(
            d,
            180.0 - d,
        )
    )


def _morphological_skeleton(mask_bool):
    """
    Skeletonise a binary mask without requiring opencv-contrib.

    This tries skimage first if available. If skimage is unavailable, it falls
    back to a pure OpenCV morphological skeleton loop.

    Args:
        mask_bool:
            2D boolean mask.

    Returns:
        np.ndarray:
            2D boolean skeleton mask.
    """
    mask = to_numpy(mask_bool)

    if mask.ndim != 2:
        return np.zeros_like(
            mask,
            dtype=bool,
        )

    mask = mask.astype(bool)

    if mask.sum() == 0:
        return np.zeros_like(
            mask,
            dtype=bool,
        )

    # Preferred option if scikit-image exists in the production environment.
    if (
        bool(CELL16_SKIMAGE_SKELETONIZE_AVAILABLE)
        and _cell16_skimage_skeletonize is not None
    ):
        try:
            return _cell16_skimage_skeletonize(mask).astype(bool)
        except Exception:
            pass

    # Fallback: OpenCV morphological skeleton. This avoids cv2.ximgproc.thinning,
    # which is not available in plain opencv-python.
    img = mask.astype(np.uint8) * 255
    skel = np.zeros_like(
        img,
        dtype=np.uint8,
    )

    element = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (
            3,
            3,
        ),
    )

    # Safety cap prevents infinite loops on unusual masks.
    max_iters = 512
    iter_count = 0

    while True:
        opened = cv2.morphologyEx(
            img,
            cv2.MORPH_OPEN,
            element,
        )

        temp = cv2.subtract(
            img,
            opened,
        )

        eroded = cv2.erode(
            img,
            element,
        )

        skel = cv2.bitwise_or(
            skel,
            temp,
        )

        img = eroded.copy()
        iter_count += 1

        if cv2.countNonZero(img) == 0 or iter_count >= max_iters:
            break

    return skel > 0


def _line_segment_angle_and_length(line):
    """
    Extract angle and length from one HoughLinesP segment.

    Args:
        line:
            Line segment in [x1, y1, x2, y2] format.

    Returns:
        tuple:
            angle:
                Canonical undirected angle in degrees.

            length:
                Segment length in pixels.
    """
    x1, y1, x2, y2 = [
        float(v)
        for v in line
    ]

    dx = x2 - x1
    dy = y2 - y1

    length = float(
        math.hypot(
            dx,
            dy,
        )
    )

    if length <= 0:
        return np.nan, 0.0

    angle = _canonical_angle_deg(
        math.degrees(
            math.atan2(
                dy,
                dx,
            )
        )
    )

    return angle, length


def _run_probabilistic_hough(binary_img, min_line_length_px):
    """
    Run cv2.HoughLinesP on a binary image and return line segment rows.

    Args:
        binary_img:
            2D boolean or uint8 image.

        min_line_length_px:
            Minimum segment length.

    Returns:
        list:
            Each item is [x1, y1, x2, y2].
    """
    if binary_img is None:
        return []

    arr = to_numpy(binary_img)

    if arr.ndim != 2:
        return []

    hough_input = arr.astype(bool).astype(np.uint8) * 255

    if int(hough_input.sum()) == 0:
        return []

    lines = cv2.HoughLinesP(
        hough_input,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(XSPLIT_HOUGH_THRESHOLD),
        minLineLength=max(
            5,
            int(min_line_length_px),
        ),
        maxLineGap=int(XSPLIT_HOUGH_MAX_LINE_GAP),
    )

    if lines is None:
        return []

    lines = np.asarray(lines).reshape(-1, 4)

    return [
        [
            int(v)
            for v in row
        ]
        for row in lines
    ]


def _cluster_hough_lines_into_two_angle_groups(lines):
    """
    Cluster Hough line segments into two undirected angle groups.

    The clustering is done in doubled-angle space, so angles near 0 and 180 are
    treated as the same orientation.

    Args:
        lines:
            List of [x1, y1, x2, y2] segments.

    Returns:
        dict:
            Validity, cluster angles, angle difference, and diagnostic stats.
    """
    out = {
        "valid": False,
        "reason": "not_enough_lines",
        "angle_a": np.nan,
        "angle_b": np.nan,
        "angle_diff": np.nan,
        "group_a_total_length": 0.0,
        "group_b_total_length": 0.0,
        "num_lines": int(len(lines)) if lines is not None else 0,
        "labels": [],
        "line_angles": [],
        "line_lengths": [],
    }

    if lines is None or len(lines) < 2:
        return out

    angles = []
    lengths = []

    for line in lines:
        angle, length = _line_segment_angle_and_length(line)

        if np.isfinite(angle) and length > 0:
            angles.append(float(angle))
            lengths.append(float(length))

    if len(angles) < 2:
        out["reason"] = "not_enough_valid_line_angles"
        return out

    angles_arr = np.asarray(
        angles,
        dtype=np.float64,
    )

    lengths_arr = np.asarray(
        lengths,
        dtype=np.float64,
    )

    # Doubled-angle embedding for undirected orientations.
    theta = np.deg2rad(2.0 * angles_arr)

    emb = np.column_stack(
        [
            np.cos(theta),
            np.sin(theta),
        ]
    )

    # Deterministic initialisation:
    # first centre is longest line, second is farthest orientation from it.
    first_idx = int(np.argmax(lengths_arr))

    dists = np.linalg.norm(
        emb - emb[first_idx],
        axis=1,
    )

    second_idx = int(np.argmax(dists))

    if first_idx == second_idx:
        out["reason"] = "unable_to_seed_two_angle_groups"
        return out

    centers = np.vstack(
        [
            emb[first_idx],
            emb[second_idx],
        ]
    ).astype(np.float64)

    labels = np.zeros(
        len(angles_arr),
        dtype=int,
    )

    for _ in range(12):
        d0 = np.linalg.norm(
            emb - centers[0],
            axis=1,
        )

        d1 = np.linalg.norm(
            emb - centers[1],
            axis=1,
        )

        new_labels = (d1 < d0).astype(int)

        if np.array_equal(
            new_labels,
            labels,
        ):
            labels = new_labels
            break

        labels = new_labels

        for k in [
            0,
            1,
        ]:
            idxs = np.where(labels == k)[0]

            if len(idxs) == 0:
                continue

            weighted = emb[idxs] * lengths_arr[idxs, None]
            centre = weighted.sum(axis=0)
            norm = float(np.linalg.norm(centre))

            if norm > 0:
                centers[k] = centre / norm

    group_lengths = []
    group_angles = []

    for k in [
        0,
        1,
    ]:
        idxs = np.where(labels == k)[0]

        if len(idxs) == 0:
            out["reason"] = "empty_angle_group"
            return out

        total_len = float(lengths_arr[idxs].sum())
        group_lengths.append(total_len)

        weighted = emb[idxs] * lengths_arr[idxs, None]
        centre = weighted.sum(axis=0)

        angle = 0.5 * math.degrees(
            math.atan2(
                float(centre[1]),
                float(centre[0]),
            )
        )

        group_angles.append(
            _canonical_angle_deg(angle)
        )

    angle_diff = _angle_diff_undirected_deg(
        group_angles[0],
        group_angles[1],
    )

    total_length = max(
        1e-9,
        float(lengths_arr.sum()),
    )

    out.update(
        {
            "angle_a": float(group_angles[0]),
            "angle_b": float(group_angles[1]),
            "angle_diff": float(angle_diff),
            "group_a_total_length": float(group_lengths[0]),
            "group_b_total_length": float(group_lengths[1]),
            "labels": labels.tolist(),
            "line_angles": angles_arr.tolist(),
            "line_lengths": lengths_arr.tolist(),
        }
    )

    min_group_frac = min(group_lengths) / total_length

    if angle_diff < XSPLIT_MIN_ANGLE_DIFF_DEG:
        out["reason"] = "angle_groups_too_similar"
        return out

    if angle_diff > XSPLIT_MAX_ANGLE_DIFF_DEG:
        out["reason"] = "angle_groups_too_opposite_or_unstable"
        return out

    if min_group_frac < XSPLIT_MIN_GROUP_LENGTH_FRAC:
        out["reason"] = "one_angle_group_too_weak"
        return out

    out["valid"] = True
    out["reason"] = "two_angle_groups_found"

    return out


def _detect_two_angle_groups_for_mask(mask_bool, image_np_rgb=None):
    """
    Detect two strong line directions inside one SAM3 mask.

    First method:
        Skeletonize the mask and run Hough on the skeleton.

    Fallback method:
        Run Canny on the actual image, restricted to the SAM3 mask region.

    Args:
        mask_bool:
            2D boolean SAM3 mask.

        image_np_rgb:
            RGB image array for fallback edge detection.

    Returns:
        dict:
            Hough / angle-group result.
    """
    out = {
        "valid": False,
        "source": "none",
        "reason": "not_run",
        "angle_a": np.nan,
        "angle_b": np.nan,
        "angle_diff": np.nan,
        "num_lines": 0,
        "lines": [],
    }

    mask = to_numpy(mask_bool)

    if mask.ndim != 2:
        out["reason"] = "invalid_mask_shape"
        return out

    mask = mask.astype(bool)

    if mask.sum() < XSPLIT_MIN_PARENT_MASK_PIXELS:
        out["reason"] = "mask_too_small"
        return out

    valid_box, x1, y1, x2, y2 = _bbox_from_mask(
        mask,
        pad_px=0,
        image_w=mask.shape[1],
        image_h=mask.shape[0],
    )

    if not valid_box:
        out["reason"] = "invalid_mask_box"
        return out

    box_w = max(
        1.0,
        float(x2 - x1),
    )

    box_h = max(
        1.0,
        float(y2 - y1),
    )

    min_line_len = max(
        8.0,
        XSPLIT_HOUGH_MIN_LINE_LENGTH_FRAC * max(box_w, box_h),
    )

    # -------------------------------------------------------------------------
    # Method 1: skeleton Hough. This is preferred because it finds centre-lines,
    # not outer mask boundaries.
    # -------------------------------------------------------------------------
    skeleton = _morphological_skeleton(mask)

    skeleton_lines = _run_probabilistic_hough(
        binary_img=skeleton,
        min_line_length_px=min_line_len,
    )

    skeleton_clusters = _cluster_hough_lines_into_two_angle_groups(
        skeleton_lines
    )

    if bool(skeleton_clusters.get("valid", False)):
        out.update(skeleton_clusters)
        out["valid"] = True
        out["source"] = "mask_skeleton_hough"
        out["lines"] = skeleton_lines

        return out

    # -------------------------------------------------------------------------
    # Method 2: image-edge Hough inside the SAM3 mask. This is only a fallback.
    # -------------------------------------------------------------------------
    if image_np_rgb is not None:
        img_arr = to_numpy(image_np_rgb)

        if img_arr.ndim == 3 and img_arr.shape[:2] == mask.shape:
            gray = cv2.cvtColor(
                img_arr.astype(np.uint8),
                cv2.COLOR_RGB2GRAY,
            )

            gray = cv2.GaussianBlur(
                gray,
                (
                    3,
                    3,
                ),
                0,
            )

            edges = cv2.Canny(
                gray,
                XSPLIT_EDGE_CANNY_LOW,
                XSPLIT_EDGE_CANNY_HIGH,
            )

            kernel = np.ones(
                (
                    3,
                    3,
                ),
                dtype=np.uint8,
            )

            mask_u8 = mask.astype(np.uint8) * 255

            mask_dilated = cv2.dilate(
                mask_u8,
                kernel,
                iterations=int(XSPLIT_EDGE_MASK_DILATE_ITER),
            )

            masked_edges = (
                (edges > 0)
                & (mask_dilated > 0)
            )

            edge_lines = _run_probabilistic_hough(
                binary_img=masked_edges,
                min_line_length_px=min_line_len,
            )

            edge_clusters = _cluster_hough_lines_into_two_angle_groups(
                edge_lines
            )

            if bool(edge_clusters.get("valid", False)):
                out.update(edge_clusters)
                out["valid"] = True
                out["source"] = "image_edges_inside_mask_hough"
                out["lines"] = edge_lines

                return out

            out.update(
                {
                    "reason": "skeleton_and_edge_hough_failed",
                    "skeleton_reason": skeleton_clusters.get("reason", "unknown"),
                    "edge_reason": edge_clusters.get("reason", "unknown"),
                    "num_lines": max(
                        int(skeleton_clusters.get("num_lines", 0)),
                        int(edge_clusters.get("num_lines", 0)),
                    ),
                }
            )

            return out

    out.update(
        {
            "reason": "skeleton_hough_failed_no_image_edge_fallback",
            "skeleton_reason": skeleton_clusters.get("reason", "unknown"),
            "num_lines": int(skeleton_clusters.get("num_lines", 0)),
        }
    )

    return out


def _split_mask_using_two_angles(mask_bool, angle_a, angle_b):
    """
    Split one mask into two child masks using two line directions.

    The two initial lines pass through the parent mask centroid. Then we run a
    few assign/refit iterations:
      - assign each parent mask pixel to the closer line
      - refit each line to its assigned pixels

    Args:
        mask_bool:
            2D boolean parent mask.

        angle_a:
            First line angle in degrees.

        angle_b:
            Second line angle in degrees.

    Returns:
        dict:
            Split result with child masks, line models, and success reason.
    """
    out = {
        "valid": False,
        "reason": "not_run",
        "child_mask_a": None,
        "child_mask_b": None,
        "model_a": None,
        "model_b": None,
        "child_a_pixels": 0,
        "child_b_pixels": 0,
        "parent_pixels": 0,
        "child_balance_ratio": np.nan,
    }

    mask = to_numpy(mask_bool)

    if mask.ndim != 2:
        out["reason"] = "invalid_mask_shape"
        return out

    mask = mask.astype(bool)

    ys, xs = np.where(mask)

    parent_pixels = int(len(xs))
    out["parent_pixels"] = parent_pixels

    if parent_pixels < XSPLIT_MIN_PARENT_MASK_PIXELS:
        out["reason"] = "parent_mask_too_small"
        return out

    xs_f = xs.astype(np.float64)
    ys_f = ys.astype(np.float64)

    cx = float(xs_f.mean())
    cy = float(ys_f.mean())

    def _model_from_angle(angle_deg):
        theta = math.radians(float(angle_deg))

        ux = math.cos(theta)
        uy = math.sin(theta)

        return {
            "valid": True,
            "cx": cx,
            "cy": cy,
            "ux": ux,
            "uy": uy,
            "angle_deg": _canonical_angle_deg(angle_deg),
            "num_pixels": parent_pixels,
        }

    model_a = _model_from_angle(angle_a)
    model_b = _model_from_angle(angle_b)

    labels = np.zeros(
        parent_pixels,
        dtype=np.int32,
    )

    for _ in range(4):
        dist_a = _line_distance_for_points(
            xs_f,
            ys_f,
            model_a,
        )

        dist_b = _line_distance_for_points(
            xs_f,
            ys_f,
            model_b,
        )

        labels = (dist_b < dist_a).astype(np.int32)

        idx_a = np.where(labels == 0)[0]
        idx_b = np.where(labels == 1)[0]

        if (
            len(idx_a) < XSPLIT_MIN_CHILD_MASK_PIXELS
            or len(idx_b) < XSPLIT_MIN_CHILD_MASK_PIXELS
        ):
            out["reason"] = "one_child_too_small_during_refit"
            return out

        model_a_new = _fit_line_model_from_points(
            xs_f[idx_a],
            ys_f[idx_a],
        )

        model_b_new = _fit_line_model_from_points(
            xs_f[idx_b],
            ys_f[idx_b],
        )

        if not model_a_new["valid"] or not model_b_new["valid"]:
            out["reason"] = "line_refit_failed"
            return out

        model_a = model_a_new
        model_b = model_b_new

    idx_a = np.where(labels == 0)[0]
    idx_b = np.where(labels == 1)[0]

    child_a_pixels = int(len(idx_a))
    child_b_pixels = int(len(idx_b))

    min_child_pixels = min(
        child_a_pixels,
        child_b_pixels,
    )

    max_child_pixels = max(
        child_a_pixels,
        child_b_pixels,
    )

    balance_ratio = float(
        min_child_pixels / max(max_child_pixels, 1)
    )

    if (
        child_a_pixels < XSPLIT_MIN_CHILD_MASK_PIXELS
        or child_b_pixels < XSPLIT_MIN_CHILD_MASK_PIXELS
    ):
        out["reason"] = "child_mask_too_small"
        return out

    if (
        child_a_pixels / max(parent_pixels, 1)
    ) < XSPLIT_MIN_CHILD_FRAC_OF_PARENT:
        out["reason"] = "child_a_too_small_fraction_of_parent"
        return out

    if (
        child_b_pixels / max(parent_pixels, 1)
    ) < XSPLIT_MIN_CHILD_FRAC_OF_PARENT:
        out["reason"] = "child_b_too_small_fraction_of_parent"
        return out

    if balance_ratio < XSPLIT_MIN_CHILD_BALANCE_RATIO:
        out["reason"] = "children_too_unbalanced"
        return out

    angle_diff = _angle_diff_undirected_deg(
        model_a["angle_deg"],
        model_b["angle_deg"],
    )

    if angle_diff < XSPLIT_MIN_ANGLE_DIFF_DEG:
        out["reason"] = "refit_child_angles_too_similar"
        return out

    child_mask_a = np.zeros_like(
        mask,
        dtype=bool,
    )

    child_mask_b = np.zeros_like(
        mask,
        dtype=bool,
    )

    child_mask_a[
        ys[idx_a],
        xs[idx_a],
    ] = True

    child_mask_b[
        ys[idx_b],
        xs[idx_b],
    ] = True

    out.update(
        {
            "valid": True,
            "reason": "split_success",
            "child_mask_a": child_mask_a,
            "child_mask_b": child_mask_b,
            "model_a": model_a,
            "model_b": model_b,
            "child_a_pixels": child_a_pixels,
            "child_b_pixels": child_b_pixels,
            "child_balance_ratio": balance_ratio,
        }
    )

    return out


# =============================================================================
# FINAL OUTPUT LINEAGE HELPERS
# =============================================================================
# EXPLANATION:
# These helpers support final Xarm labelling and lineage trace creation.
#
# They are defined once in 16B so the per-ROI batch loop does not redefine them
# for every ROI.
#
# IMPORTANT:
#   These helpers do not modify detection rows or masks.
#   They only normalise lineage fields such as:
#       orig_det_idx
#       parent_orig_det_idx
#       merged_from_orig_det_idxs
#       source_orig_det_idxs
#
# PLACE THIS BLOCK:
#   - after 16.21 X-split helper utilities
#   - before 16C. PER-ROI BATCH LOOP
# =============================================================================

def normalise_orig_idx_list(value):
    """
    Convert different stored lineage formats into a clean list of integers.

    Handles examples:
        3
        3.0
        "3"
        "3,6"
        "[3, 6]"
        [3, 6]
        np.nan
        None

    Args:
        value:
            Any lineage value from a row.

    Returns:
        list[int]:
            Clean ordered list of unique integer idx values.
    """
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        raw_items = list(value)

    else:
        text = str(value).strip()

        if text == "" or text.lower() in ["nan", "none", "null"]:
            return []

        text = (
            text.replace("[", " ")
            .replace("]", " ")
            .replace("(", " ")
            .replace(")", " ")
            .replace(",", " ")
        )

        raw_items = text.split()

    clean_items = []

    for item in raw_items:
        try:
            if item is None:
                continue

            try:
                if pd.isna(item):
                    continue
            except Exception:
                pass

            item_text = str(item).strip()

            if item_text == "" or item_text.lower() in [
                "nan",
                "none",
                "null",
            ]:
                continue

            clean_items.append(int(float(item_text)))

        except Exception:
            continue

    seen = set()
    unique_items = []

    for item in clean_items:
        item = int(item)

        if item not in seen:
            unique_items.append(item)
            seen.add(item)

    return unique_items


def collect_source_orig_det_idxs(row):
    """
    Pick the best available lineage source for a final detection.

    Priority:
        1) merged_from_orig_det_idxs
           Best for same-crossarm continuity merge cases.

        2) source_orig_det_idxs
           Used if a later stage has already created a clean lineage field.

        3) parent_orig_det_idx
           Useful for X-split children.

        4) orig_det_idx
           Fallback for normal detections that were never merged/split.

    Args:
        row:
            One final detection row.

    Returns:
        list[int]:
            Source idx values that contributed to the final detection.
    """
    if "merged_from_orig_det_idxs" in row:
        merged_idxs = normalise_orig_idx_list(
            row.get("merged_from_orig_det_idxs")
        )

        if len(merged_idxs) > 0:
            return merged_idxs

    if "source_orig_det_idxs" in row:
        source_idxs = normalise_orig_idx_list(
            row.get("source_orig_det_idxs")
        )

        if len(source_idxs) > 0:
            return source_idxs

    if "parent_orig_det_idx" in row:
        parent_idxs = normalise_orig_idx_list(
            row.get("parent_orig_det_idx")
        )

        if len(parent_idxs) > 0:
            return parent_idxs

    if "orig_det_idx" in row:
        return normalise_orig_idx_list(
            row.get("orig_det_idx")
        )

    return []


# =============================================================================
# FINAL REVIEW IMAGE DRAWING HELPERS
# =============================================================================
# EXPLANATION:
# These helpers draw final production review overlays onto Matplotlib axes.
#
# They are used later by 16.34 when saving final Gold review images:
#   - Final_Image_Real_Mask
#   - Final_Image_Display_Mask
#
# PRODUCTION RULE:
#   These helpers only draw onto an axis.
#   They do not call display().
#   They do not call plt.show().
#   They do not save files directly.
#
# IMPORTANT:
#   - crossarm_mask_lookup is passed in by the caller.
#   - final_display_mask_lookup may also be passed in by the caller.
#   - These helpers must not mutate masks, boxes, or DataFrames.
# =============================================================================

def draw_pole_label(ax, pole_mask):
    """
    Draw a small POLE label near the projected pole mask.

    Args:
        ax:
            Matplotlib axis to draw on.

        pole_mask:
            2D boolean projected pole mask.

    Returns:
        None.
    """
    if pole_mask is None:
        return

    if not isinstance(pole_mask, np.ndarray):
        return

    if pole_mask.ndim != 2:
        return

    if not np.any(pole_mask):
        return

    pole_ys, pole_xs = np.where(pole_mask.astype(bool))

    if len(pole_xs) == 0 or len(pole_ys) == 0:
        return

    pole_label_x = float(pole_xs.min())
    pole_label_y = float(max(8, pole_ys.min() - 6))

    ax.text(
        pole_label_x,
        pole_label_y,
        "POLE",
        color="white",
        fontsize=8,
        bbox=dict(
            facecolor="red",
            alpha=0.90,
            pad=0.3,
            edgecolor="none",
        ),
    )


def plot_stage_on_ax(
    ax,
    image,
    detections_df,
    title,
    projected_pole_mask=None,
    crossarm_mask_lookup=None,
    crossarm_mask_alpha=0.40,
    pole_mask_alpha=0.30,
    label_bg="#1E90FF",
    final_style=False,
):
    """
    Plot one crossarm / xarm pipeline stage on a Matplotlib axis.

    Args:
        ax:
            Matplotlib axis to draw on.

        image:
            RGB PIL image or image array for the ROI crop.

        detections_df:
            DataFrame of detections to draw. Expected columns are:
            orig_det_idx, x1, y1, x2, y2, and score.

        title:
            Title to display above the plot.

        projected_pole_mask:
            Optional 2D boolean selected-pole mask projected into ROI space.

        crossarm_mask_lookup:
            Optional dictionary keyed by orig_det_idx containing 2D crossarm
            masks. This can be the real canonical mask lookup or a display-only
            mask lookup.

        crossarm_mask_alpha:
            Alpha value for crossarm mask overlays.

        pole_mask_alpha:
            Alpha value for projected pole mask overlay.

        label_bg:
            Label background colour.

        final_style:
            If True and xarm_label/final_xarm_label exists, use the final
            display label instead of the technical orig_det_idx label.

    Returns:
        None.
    """
    ax.imshow(image)

    # -------------------------------------------------------------------------
    # Draw crossarm / xarm masks.
    # -------------------------------------------------------------------------
    if (
        detections_df is not None
        and isinstance(detections_df, pd.DataFrame)
        and len(detections_df) > 0
    ):
        try:
            cmap = plt.colormaps.get_cmap("tab10").resampled(
                max(len(detections_df), 1)
            )
        except Exception:
            cmap = plt.cm.get_cmap(
                "tab10",
                max(len(detections_df), 1),
            )

        for plot_idx, (_, det_row) in enumerate(detections_df.iterrows()):
            if "orig_det_idx" not in det_row:
                continue

            try:
                orig_idx = int(det_row["orig_det_idx"])
            except Exception:
                continue

            mask_i = (
                crossarm_mask_lookup.get(orig_idx, None)
                if isinstance(crossarm_mask_lookup, dict)
                else None
            )

            if (
                isinstance(mask_i, np.ndarray)
                and mask_i.ndim == 2
                and int(mask_i.sum()) > 0
            ):
                color_rgba = cmap(plot_idx)

                overlay = np.zeros(
                    (
                        mask_i.shape[0],
                        mask_i.shape[1],
                        4,
                    ),
                    dtype=np.float32,
                )

                overlay[..., 0] = float(color_rgba[0])
                overlay[..., 1] = float(color_rgba[1])
                overlay[..., 2] = float(color_rgba[2])
                overlay[..., 3] = (
                    mask_i.astype(np.float32)
                    * float(crossarm_mask_alpha)
                )

                ax.imshow(overlay)

    # -------------------------------------------------------------------------
    # Draw selected-pole mask context.
    # -------------------------------------------------------------------------
    if (
        projected_pole_mask is not None
        and isinstance(projected_pole_mask, np.ndarray)
        and projected_pole_mask.ndim == 2
        and np.any(projected_pole_mask)
    ):
        pole_mask = projected_pole_mask.astype(bool)

        pole_overlay = np.zeros(
            (
                pole_mask.shape[0],
                pole_mask.shape[1],
                4,
            ),
            dtype=np.float32,
        )

        pole_overlay[..., 0] = 1.0
        pole_overlay[..., 1] = 0.0
        pole_overlay[..., 2] = 0.0
        pole_overlay[..., 3] = (
            pole_mask.astype(np.float32)
            * float(pole_mask_alpha)
        )

        ax.imshow(pole_overlay)

        draw_pole_label(
            ax=ax,
            pole_mask=pole_mask,
        )

    # -------------------------------------------------------------------------
    # Draw boxes and labels.
    # -------------------------------------------------------------------------
    if (
        detections_df is not None
        and isinstance(detections_df, pd.DataFrame)
        and len(detections_df) > 0
    ):
        for _, det_row in detections_df.iterrows():
            required_box_cols = ["x1", "y1", "x2", "y2"]

            if not all(col_name in det_row for col_name in required_box_cols):
                continue

            try:
                x1 = float(det_row["x1"])
                y1 = float(det_row["y1"])
                x2 = float(det_row["x2"])
                y2 = float(det_row["y2"])
            except Exception:
                continue

            if not np.all(np.isfinite([x1, y1, x2, y2])):
                continue

            rect = patches.Rectangle(
                (x1, y1),
                max(0.0, x2 - x1),
                max(0.0, y2 - y1),
                linewidth=2.0,
                edgecolor="yellow",
                facecolor="none",
            )

            ax.add_patch(rect)

            if (
                bool(final_style)
                and "xarm_label" in det_row
                and pd.notna(det_row["xarm_label"])
            ):
                label_text = str(det_row["xarm_label"])

            elif (
                bool(final_style)
                and "final_xarm_label" in det_row
                and pd.notna(det_row["final_xarm_label"])
            ):
                label_text = str(det_row["final_xarm_label"])

            elif "orig_det_idx" in det_row and pd.notna(det_row["orig_det_idx"]):
                label_text = f"idx={int(det_row['orig_det_idx'])}"

            else:
                label_text = "idx=?"

            if "score" in det_row and pd.notna(det_row["score"]):
                label_text = f"{label_text} | {float(det_row['score']):.3f}"

            ax.text(
                x1,
                max(0.0, y1 - 5.0),
                label_text,
                color="white",
                fontsize=8,
                bbox=dict(
                    facecolor=label_bg,
                    alpha=0.85,
                    pad=0.35,
                    edgecolor="none",
                ),
            )

    ax.set_title(str(title))
    ax.axis("off")
    
 
    
# =============================================================================
# 16C. PER-ROI BATCH LOOP
# =============================================================================
# EXPLANATION:
# This section runs the locked crossarm / xarm detection pipeline for each
# selected pole-top ROI in cell16_roi_input_df.
#
# IMPORTANT:
#   Use cell16_roi_input_df from 16A.
#   Do not use roi_input_df from this point onward.
#
#   crossarm_mask_lookup is reset inside each ROI iteration.
#   It must not be shared across ROIs.
# =============================================================================


# =============================================================================
# 16.22 PER-ROI PROCESSING LOOP — START
# =============================================================================
# EXPLANATION:
# Each iteration processes one pole-top ROI crop.
#
# The first step inside the loop is to project the selected full-resolution pole
# mask from CELL 13 into the current ROI coordinate space.
#
# This creates:
#   projected_pole_mask
#
# with shape:
#   (roi_h, roi_w)
#
# If the live selected-pole mask is unavailable for a selected ROI row, the
# pipeline continues with an all-False projected pole mask for that ROI. The
# pole-overlap / pole-corridor stage will later skip pole-mask filtering and add
# the appropriate review reason.
#
# IMPORTANT:
#   This loop uses one try/except per ROI. Each ROI becomes either:
#       - one success row
#       - or one failure row
# =============================================================================

for roi_position, roi_row in cell16_roi_input_df.iterrows():
    roi_start_time = time.time()

    # -------------------------------------------------------------------------
    # 16.22.1 Initialise per-ROI working state
    # -------------------------------------------------------------------------
    # IMPORTANT:
    # crossarm_mask_lookup is deliberately reset per ROI.
    # Do not move this outside the loop.
    # -------------------------------------------------------------------------
    crossarm_mask_lookup = {}

    row_failure_reason = ""
    row_failure_detail = ""

    projected_pole_mask_available = False
    pole_mask_filter_applied = False
    original_pole_mask = None

    current_stage = "16.22_project_pole_mask"

    try:
        # ---------------------------------------------------------------------
        # 16.22.2 Extract stable ROI identity fields
        # ---------------------------------------------------------------------
        image_id = str(roi_row["image_id"])
        file_name = str(roi_row["file_name"])
        roi_file_name = str(roi_row["roi_file_name"])
        roi_image_path = str(roi_row["roi_image_path"])

        pole_prompt = str(roi_row["prompt"])
        pole_det_idx = int(roi_row["det_idx"])

        roi_w = int(roi_row["roi_w"])
        roi_h = int(roi_row["roi_h"])

        row_has_live_pole_mask = parse_bool(
            roi_row.get(
                "cell16_has_live_pole_mask",
                False,
            )
        )

        # ---------------------------------------------------------------------
        # 16.22.3 Initialise projected pole mask in ROI space
        # ---------------------------------------------------------------------
        # Shape must be:
        #   (roi_h, roi_w)
        #
        # This matches the crossarm masks returned later by:
        #   normalize_masks(raw_masks, num_detections, image_h=roi_h, image_w=roi_w)
        # ---------------------------------------------------------------------
        projected_pole_mask = np.zeros(
            (
                roi_h,
                roi_w,
            ),
            dtype=bool,
        )

        # ---------------------------------------------------------------------
        # 16.22.4 Fetch selected full-resolution pole mask from CELL 13 lookup
        # ---------------------------------------------------------------------
        # CELL 13 / CELL 14 key contract:
        #   (str(image_id), str(prompt), int(det_idx))
        #
        # Do not use the crossarm int orig_det_idx key here.
        #
        # IMPORTANT:
        #   cell16_has_live_pole_mask is the authoritative per-row check created
        #   in 16.6C from the current live pole_mask_lookup.
        # ---------------------------------------------------------------------
        pole_key = (
            str(image_id),
            str(pole_prompt),
            int(pole_det_idx),
        )

        if (
            CELL16_POLE_MASK_LOOKUP_AVAILABLE
            and row_has_live_pole_mask
            and pole_key in pole_mask_lookup
        ):
            original_pole_mask = pole_mask_lookup.get(
                pole_key,
                None,
            )

        # ---------------------------------------------------------------------
        # 16.22.5 Project selected pole mask into ROI coordinates
        # ---------------------------------------------------------------------
        # IMPORTANT:
        # Use keyword arguments for roi_w and roi_h.
        #
        # project_pole_mask_to_roi signature uses:
        #   roi_w, roi_h
        #
        # normalize_masks later uses:
        #   image_h, image_w
        #
        # Avoid positional argument mistakes.
        # ---------------------------------------------------------------------
        if original_pole_mask is not None:
            projected_pole_mask = project_pole_mask_to_roi(
                pole_mask=original_pole_mask,
                src_x1=int(roi_row["src_x1"]),
                src_y1=int(roi_row["src_y1"]),
                src_x2=int(roi_row["src_x2"]),
                src_y2=int(roi_row["src_y2"]),
                dst_x1=int(roi_row["dst_x1"]),
                dst_y1=int(roi_row["dst_y1"]),
                roi_w=roi_w,
                roi_h=roi_h,
            )

            if (
                isinstance(projected_pole_mask, np.ndarray)
                and projected_pole_mask.shape == (roi_h, roi_w)
                and projected_pole_mask.any()
            ):
                projected_pole_mask_available = True

            else:
                # Defensive fallback:
                # keep shape stable even if projection returns an invalid mask.
                projected_pole_mask = np.zeros(
                    (
                        roi_h,
                        roi_w,
                    ),
                    dtype=bool,
                )

        # ---------------------------------------------------------------------
        # 16.22.6 Decide whether pole-mask filtering can run for this ROI
        # ---------------------------------------------------------------------
        # The actual pole-overlap / pole-corridor filter runs later.
        # This flag only records whether a projected pole mask is available.
        # ---------------------------------------------------------------------
        pole_mask_filter_applied = bool(
            POLE_MASK_FILTER_ENABLED
            and projected_pole_mask_available
        )


        # =============================================================================
        # 16.23 LOAD ROI CROP
        # =============================================================================
        # EXPLANATION:
        # Load the current fixed-canvas pole-top ROI crop from disk.
        #
        # IMPORTANT:
        #   The loaded image size must match the roi_w / roi_h values from
        #   cell16_roi_input_df. Crossarm masks later must have shape:
        #
        #       (roi_h, roi_w)
        #
        #   If the saved image dimensions do not match the metadata, fail this
        #   ROI before SAM3 inference.
        # =============================================================================

        current_stage = "16.23_load_roi_crop"

        with Image.open(roi_image_path) as img:
            roi_original_mode = img.mode

            if roi_original_mode != "RGB":
                image = img.convert("RGB")
            else:
                image = img.copy()

            image.load()

        loaded_roi_w, loaded_roi_h = image.size

        if int(loaded_roi_w) != int(roi_w) or int(loaded_roi_h) != int(roi_h):
            raise ValueError(
                "Loaded ROI image size does not match cell16_roi_input_df metadata.\n"
                f"roi_image_path : {roi_image_path}\n"
                f"metadata size  : {roi_w} x {roi_h}\n"
                f"loaded size    : {loaded_roi_w} x {loaded_roi_h}"
            )


        # =============================================================================
        # 16.24 RUN SAM3 INFERENCE — STATEFUL PROCESSOR PATH
        # =============================================================================
        # EXPLANATION:
        # Run SAM3 on the current ROI crop using the production crossarm prompt.
        #
        # The stateful processor path matches the working notebook setup:
        #   set_image
        #   reset_all_prompts
        #   set_text_prompt
        #
        # PRODUCTION RULE:
        #   No plot_results diagnostic.
        #   No plt.show().
        #   No per-ROI prints.
        # =============================================================================

        current_stage = "16.24_run_sam3_inference"

        if hasattr(processor, "device"):
            processor.device = CELL16_RUN_DEVICE

        if hasattr(processor, "set_confidence_threshold"):
            processor.set_confidence_threshold(CELL16_TEXT_THRESHOLD)

        state = {}

        state = processor.set_image(
            image,
            state=state,
        )

        reset_result = processor.reset_all_prompts(state)

        if reset_result is not None:
            state = reset_result

        state = processor.set_text_prompt(
            CELL16_PROMPT_TEXT,
            state,
        )

        raw_boxes = state.get(
            "boxes",
            None,
        )

        raw_scores = state.get(
            "scores",
            None,
        )

        raw_masks = state.get(
            "masks",
            None,
        )
        
        
        # =============================================================================
        # 16.25 NORMALISE SAM3 OUTPUTS + BUILD PER-ROI CROSSARM MASK LOOKUP
        # =============================================================================
        # EXPLANATION:
        # Convert raw SAM3 outputs into standard arrays/lists:
        #   - boxes    -> shape (N, 4)
        #   - scores   -> shape (N,)
        #   - masks_2d -> list of 2D masks or None
        #
        # This section also rebuilds:
        #   crossarm_mask_lookup
        #
        # IMPORTANT CONTRACT FOR 16.26:
        #   crossarm_mask_lookup is keyed by int(det_idx).
        #
        #   The next section must create raw detections with:
        #       orig_det_idx = det_idx
        #
        #   Do not use DataFrame row index after sorting as orig_det_idx.
        #   That would break downstream mask lookup logic.
        #
        # PRODUCTION RULE:
        #   No per-ROI prints.
        #   Counts are stored in variables and optional stage-summary rows.
        # =============================================================================

        current_stage = "16.25_normalise_outputs"

        num_detections = infer_num_detections(
            raw_boxes=raw_boxes,
            raw_scores=raw_scores,
            raw_masks=raw_masks,
        )

        boxes = normalize_boxes(
            boxes=raw_boxes,
            num_detections=num_detections,
        )

        scores = normalize_scores(
            scores=raw_scores,
            num_detections=num_detections,
        )

        masks_2d = normalize_masks(
            raw_masks=raw_masks,
            num_detections=num_detections,
            image_h=roi_h,
            image_w=roi_w,
        )

        # ---------------------------------------------------------------------
        # 16.25.1 Rebuild per-ROI crossarm mask lookup
        # ---------------------------------------------------------------------
        # IMPORTANT:
        # This lookup is local to the current ROI iteration.
        # It must remain keyed by int(det_idx), matching raw SAM3 detection
        # order before any sorting or filtering.
        # ---------------------------------------------------------------------
        crossarm_mask_lookup = {}

        for det_idx in range(num_detections):
            mask_i = (
                masks_2d[det_idx]
                if det_idx < len(masks_2d)
                else None
            )

            has_mask = bool(
                isinstance(mask_i, np.ndarray)
                and mask_i.ndim == 2
                and mask_i.shape == (roi_h, roi_w)
                and int(mask_i.sum()) > 0
            )

            if has_mask:
                crossarm_mask_lookup[int(det_idx)] = mask_i.astype(bool)

        has_any_valid_crossarm_masks = bool(
            len(crossarm_mask_lookup) > 0
        )

        # ---------------------------------------------------------------------
        # 16.25.2 Optional lightweight stage summary row
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # This records the raw SAM3 output count for production auditability.
        # It does not save anything yet; 16F will save accumulated tables.
        # ---------------------------------------------------------------------
        crossarm_stage_summary_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "image_id": image_id,
                "file_name": file_name,
                "roi_file_name": roi_file_name,
                "roi_image_path": roi_image_path,
                "processing_order": roi_row.get("processing_order", np.nan),

                "stage_order": 1,
                "stage_name": "raw_sam3_outputs",
                "input_count": np.nan,
                "output_count": int(num_detections),
                "removed_count": 0,

                "num_detections": int(num_detections),
                "num_valid_crossarm_masks": int(len(crossarm_mask_lookup)),
                "has_any_valid_crossarm_masks": bool(has_any_valid_crossarm_masks),

                "projected_pole_mask_available": bool(projected_pole_mask_available),
                "pole_mask_filter_applied": bool(pole_mask_filter_applied),
            }
        )
        
        
        # =============================================================================
        # 16.26 BUILD RAW DETECTIONS + RAW SCORE PREFILTER + CONTAINMENT SUPPRESSION
        # =============================================================================
        # EXPLANATION:
        # This section starts the post-processing pipeline for the current ROI.
        #
        # It does three things:
        #   1) Builds the raw detection table from normalised SAM3 boxes/scores.
        #   2) Removes weak detections using the raw score prefilter.
        #   3) Runs mask-veto containment suppression on the remaining detections.
        #
        # IMPORTANT CONTRACT:
        #   16.25 stores masks using:
        #       crossarm_mask_lookup[int(det_idx)]
        #
        #   Therefore this section must create:
        #       orig_det_idx = det_idx
        #
        #   Do not replace orig_det_idx with the DataFrame row index after
        #   sorting. Downstream mask lookups depend on the raw SAM3 index.
        #
        # PRODUCTION RULE:
        #   No display().
        #   No _safe_display().
        #   No per-ROI print().
        #   Save lightweight audit counts into crossarm_stage_summary_rows.
        #   Save removed-detection audit rows into crossarm_trace_rows.
        # =============================================================================

        current_stage = "16.26_raw_score_and_containment"

        raw_detection_columns = [
            "orig_det_idx",
            "score",
            "x1",
            "y1",
            "x2",
            "y2",
            "has_mask",
        ]

        # ---------------------------------------------------------------------
        # 16.26.1 Build raw detection table
        # ---------------------------------------------------------------------
        # IMPORTANT:
        # orig_det_idx deliberately preserves the original SAM3 detection index.
        # This is the key used to retrieve the matching mask from
        # crossarm_mask_lookup.
        # ---------------------------------------------------------------------
        if num_detections == 0:
            raw_detections_df = pd.DataFrame(
                columns=raw_detection_columns
            )

        else:
            raw_detections_df = pd.DataFrame(
                {
                    "orig_det_idx": np.arange(
                        num_detections,
                        dtype=int,
                    ),
                    "score": scores.astype(float),
                    "x1": boxes[:, 0].astype(float),
                    "y1": boxes[:, 1].astype(float),
                    "x2": boxes[:, 2].astype(float),
                    "y2": boxes[:, 3].astype(float),
                    "has_mask": [
                        int(det_idx) in crossarm_mask_lookup
                        for det_idx in range(num_detections)
                    ],
                }
            )

            raw_detections_df = raw_detections_df.sort_values(
                by="score",
                ascending=False,
                kind="mergesort",
            ).reset_index(drop=True)

        # Keep a stage snapshot for the next production stage.
        stage_raw_df = raw_detections_df.copy()

        # ---------------------------------------------------------------------
        # 16.26.2A Raw score prefilter
        # ---------------------------------------------------------------------
        # Rule:
        #   score >  CROSSARM_RAW_SCORE_REMOVE_MAX  -> keep
        #   score <= CROSSARM_RAW_SCORE_REMOVE_MAX  -> remove
        # ---------------------------------------------------------------------
        if raw_detections_df.empty:
            kept_after_raw_score_df = raw_detections_df.copy()
            removed_by_raw_score_df = raw_detections_df.copy()

        else:
            raw_score_keep_mask = (
                raw_detections_df["score"].astype(float)
                > float(CROSSARM_RAW_SCORE_REMOVE_MAX)
            )

            kept_after_raw_score_df = raw_detections_df.loc[
                raw_score_keep_mask
            ].copy().reset_index(drop=True)

            removed_by_raw_score_df = raw_detections_df.loc[
                ~raw_score_keep_mask
            ].copy().reset_index(drop=True)

            if len(removed_by_raw_score_df) > 0:
                removed_by_raw_score_df["removal_reason"] = (
                    f"raw_score_lte_{float(CROSSARM_RAW_SCORE_REMOVE_MAX):.2f}"
                )

                for _, removed_row in removed_by_raw_score_df.iterrows():
                    crossarm_trace_rows.append(
                        {
                            "run_id": RUN_ID,
                            "run_timestamp": RUN_TIMESTAMP,
                            "image_id": image_id,
                            "file_name": file_name,
                            "roi_file_name": roi_file_name,
                            "roi_image_path": roi_image_path,
                            "processing_order": roi_row.get(
                                "processing_order",
                                np.nan,
                            ),

                            "stage_order": 2,
                            "stage_name": "raw_score_prefilter",
                            "orig_det_idx": int(removed_row["orig_det_idx"]),
                            "score": float(removed_row["score"]),
                            "x1": float(removed_row["x1"]),
                            "y1": float(removed_row["y1"]),
                            "x2": float(removed_row["x2"]),
                            "y2": float(removed_row["y2"]),
                            "has_mask": bool(removed_row.get("has_mask", False)),

                            "action": "removed",
                            "removal_reason": str(
                                removed_row.get(
                                    "removal_reason",
                                    "",
                                )
                            ),
                            "removed_by_orig_det_idx": np.nan,
                            "review_reason": "",
                        }
                    )

        stage_raw_score_prefilter_df = kept_after_raw_score_df.copy()

        crossarm_stage_summary_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "image_id": image_id,
                "file_name": file_name,
                "roi_file_name": roi_file_name,
                "roi_image_path": roi_image_path,
                "processing_order": roi_row.get("processing_order", np.nan),

                "stage_order": 2,
                "stage_name": "raw_score_prefilter",
                "input_count": int(len(raw_detections_df)),
                "output_count": int(len(kept_after_raw_score_df)),
                "removed_count": int(len(removed_by_raw_score_df)),

                "threshold_name": "CROSSARM_RAW_SCORE_REMOVE_MAX",
                "threshold_value": float(CROSSARM_RAW_SCORE_REMOVE_MAX),
                "keep_rule": (
                    f"score > {float(CROSSARM_RAW_SCORE_REMOVE_MAX):.2f}"
                ),
                "has_any_valid_crossarm_masks": bool(
                    has_any_valid_crossarm_masks
                ),
                "projected_pole_mask_available": bool(
                    projected_pole_mask_available
                ),
                "pole_mask_filter_applied": bool(
                    pole_mask_filter_applied
                ),
            }
        )
        
        
        # ---------------------------------------------------------------------
        # 16.26.2B Full-pole container veto before containment
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # Some SAM3 crossarm detections create a large false box that contains:
        #   - the real crossarm
        #   - almost the full selected pole mask
        #
        # If that box reaches containment suppression, it can incorrectly remove
        # the smaller true crossarm box because the true box sits inside it.
        #
        # This stage removes those full-pole container boxes before containment.
        #
        # Rule:
        #   Remove a candidate if:
        #       1) its box contains almost all selected-pole mask pixels
        #       2) its vertical span is a large fraction of the visible pole span
        #
        # IMPORTANT:
        #   This stage only runs when projected_pole_mask_available=True.
        #   If no selected-pole mask is available, the input passes through
        #   unchanged.
        # ---------------------------------------------------------------------

        current_stage = "16.26_full_pole_container_veto"

        removed_by_pole_trunk_container_veto_df = (
            kept_after_raw_score_df
            .iloc[0:0]
            .copy()
        )

        if (
            bool(POLE_TRUNK_CONTAINER_VETO_ENABLED)
            and bool(projected_pole_mask_available)
            and isinstance(projected_pole_mask, np.ndarray)
            and projected_pole_mask.ndim == 2
            and projected_pole_mask.any()
            and len(kept_after_raw_score_df) > 0
        ):
            pole_rows = np.where(projected_pole_mask.astype(bool).any(axis=1))[0]

            if len(pole_rows) > 0:
                pole_vertical_span_px = float(
                    int(pole_rows.max()) - int(pole_rows.min()) + 1
                )
            else:
                pole_vertical_span_px = 0.0

            tmp_pole_trunk_df = (
                kept_after_raw_score_df
                .copy()
                .reset_index(drop=True)
            )

            tmp_pole_trunk_df["box_h"] = (
                tmp_pole_trunk_df["y2"].astype(float)
                - tmp_pole_trunk_df["y1"].astype(float)
            ).clip(lower=0.0)

            tmp_pole_trunk_df["pole_mask_containment_fraction"] = (
                tmp_pole_trunk_df.apply(
                    lambda r: compute_box_pole_mask_containment_fraction(
                        box_xyxy=[
                            r["x1"],
                            r["y1"],
                            r["x2"],
                            r["y2"],
                        ],
                        projected_pole_mask=projected_pole_mask,
                    ),
                    axis=1,
                )
            )

            tmp_pole_trunk_df["pole_vertical_span_px"] = float(
                pole_vertical_span_px
            )

            tmp_pole_trunk_df["vertical_span_to_pole_span"] = (
                tmp_pole_trunk_df["box_h"].astype(float)
                / max(float(pole_vertical_span_px), 1.0)
            )

            tmp_pole_trunk_df["pole_trunk_container_veto_candidate"] = (
                (
                    tmp_pole_trunk_df[
                        "pole_mask_containment_fraction"
                    ].astype(float)
                    >= float(POLE_TRUNK_CONTAINER_VETO_MIN_POLE_CONTAINMENT)
                )
                & (
                    tmp_pole_trunk_df[
                        "vertical_span_to_pole_span"
                    ].astype(float)
                    >= float(POLE_TRUNK_CONTAINER_VETO_MIN_VERTICAL_SPAN_RATIO)
                )
            )

            removed_by_pole_trunk_container_veto_df = (
                tmp_pole_trunk_df[
                    tmp_pole_trunk_df["pole_trunk_container_veto_candidate"]
                ]
                .copy()
                .reset_index(drop=True)
            )

            kept_after_pole_trunk_container_veto_df = (
                tmp_pole_trunk_df[
                    ~tmp_pole_trunk_df["pole_trunk_container_veto_candidate"]
                ]
                .copy()
                .reset_index(drop=True)
            )

            if len(removed_by_pole_trunk_container_veto_df) > 0:
                removed_by_pole_trunk_container_veto_df["removal_reason"] = (
                    "pole_trunk_full_pole_container_veto"
                )

        else:
            kept_after_pole_trunk_container_veto_df = (
                kept_after_raw_score_df
                .copy()
                .reset_index(drop=True)
            )

            kept_after_pole_trunk_container_veto_df[
                "pole_mask_containment_fraction"
            ] = np.nan

            kept_after_pole_trunk_container_veto_df[
                "pole_vertical_span_px"
            ] = np.nan

            kept_after_pole_trunk_container_veto_df[
                "vertical_span_to_pole_span"
            ] = np.nan

            kept_after_pole_trunk_container_veto_df[
                "pole_trunk_container_veto_candidate"
            ] = False

        if len(removed_by_pole_trunk_container_veto_df) > 0:
            for _, removed_row in removed_by_pole_trunk_container_veto_df.iterrows():
                crossarm_trace_rows.append(
                    {
                        "run_id": RUN_ID,
                        "run_timestamp": RUN_TIMESTAMP,
                        "image_id": image_id,
                        "file_name": file_name,
                        "roi_file_name": roi_file_name,
                        "roi_image_path": roi_image_path,
                        "processing_order": roi_row.get(
                            "processing_order",
                            np.nan,
                        ),

                        "stage_order": 3,
                        "stage_name": "full_pole_container_veto",

                        "orig_det_idx": int(removed_row["orig_det_idx"]),
                        "score": float(removed_row["score"]),
                        "x1": float(removed_row["x1"]),
                        "y1": float(removed_row["y1"]),
                        "x2": float(removed_row["x2"]),
                        "y2": float(removed_row["y2"]),
                        "has_mask": bool(removed_row.get("has_mask", False)),

                        "action": "removed",
                        "removal_reason": str(
                            removed_row.get(
                                "removal_reason",
                                "pole_trunk_full_pole_container_veto",
                            )
                        ),
                        "removed_by_orig_det_idx": np.nan,

                        "pole_mask_containment_fraction": removed_row.get(
                            "pole_mask_containment_fraction",
                            np.nan,
                        ),
                        "pole_vertical_span_px": removed_row.get(
                            "pole_vertical_span_px",
                            np.nan,
                        ),
                        "vertical_span_to_pole_span": removed_row.get(
                            "vertical_span_to_pole_span",
                            np.nan,
                        ),

                        "review_reason": "",
                    }
                )

        stage_pole_trunk_container_veto_df = (
            kept_after_pole_trunk_container_veto_df
            .copy()
            .reset_index(drop=True)
        )

        crossarm_stage_summary_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "image_id": image_id,
                "file_name": file_name,
                "roi_file_name": roi_file_name,
                "roi_image_path": roi_image_path,
                "processing_order": roi_row.get("processing_order", np.nan),

                "stage_order": 3,
                "stage_name": "full_pole_container_veto",
                "input_count": int(len(kept_after_raw_score_df)),
                "output_count": int(len(kept_after_pole_trunk_container_veto_df)),
                "removed_count": int(
                    len(removed_by_pole_trunk_container_veto_df)
                ),

                "veto_enabled": bool(POLE_TRUNK_CONTAINER_VETO_ENABLED),
                "projected_pole_mask_available": bool(
                    projected_pole_mask_available
                ),
                "min_pole_containment": float(
                    POLE_TRUNK_CONTAINER_VETO_MIN_POLE_CONTAINMENT
                ),
                "min_vertical_span_ratio": float(
                    POLE_TRUNK_CONTAINER_VETO_MIN_VERTICAL_SPAN_RATIO
                ),
            }
        )
        
        
        
        # ---------------------------------------------------------------------
        # 16.26.3 Containment suppression
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # Containment receives only detections that survived:
        #   1) raw-score prefilter
        #   2) full-pole-container veto
        #
        # This call depends on:
        #   - orig_det_idx matching crossarm_mask_lookup keys
        #   - x1, y1, x2, y2, score being present
        #
        # Both conditions are satisfied by kept_after_pole_trunk_container_veto_df above.
        #
        # PRODUCTION RULE:
        #   No display().
        #   No _safe_display().
        #   No per-ROI print().
        # ---------------------------------------------------------------------

        current_stage = "16.26_containment_suppression"

        (
            kept_after_containment_df,
            removed_by_containment_df,
            containment_pair_debug_df,
        ) = suppress_contained_shorter_detections(
            detections_df=kept_after_pole_trunk_container_veto_df,
            containment_threshold=float(CONTAINMENT_THRESHOLD),
            min_area_ratio=float(MIN_AREA_RATIO),
            min_score_advantage=float(MIN_SCORE_ADVANTAGE),
            crossarm_mask_lookup=crossarm_mask_lookup,
            mask_containment_filter_enabled=bool(MASK_CONTAINMENT_FILTER_ENABLED),
            mask_containment_veto_threshold=float(MASK_CONTAINMENT_VETO_THRESHOLD),
            near_total_box_containment_threshold=float(
                NEAR_TOTAL_BOX_CONTAINMENT_THRESHOLD
            ),
            mask_containment_high=float(MASK_CONTAINMENT_HIGH),
            pair_debug_min_box_containment=float(PAIR_DEBUG_MIN_BOX_CONTAINMENT),
        )

        kept_after_containment_df = (
            kept_after_containment_df
            .copy()
            .reset_index(drop=True)
        )

        removed_by_containment_df = (
            removed_by_containment_df
            .copy()
            .reset_index(drop=True)
        )

        containment_pair_debug_df = (
            containment_pair_debug_df
            .copy()
            .reset_index(drop=True)
        )

        # ---------------------------------------------------------------------
        # 16.26.4 Add removed containment rows to trace output
        # ---------------------------------------------------------------------
        if len(removed_by_containment_df) > 0:
            for _, removed_row in removed_by_containment_df.iterrows():
                crossarm_trace_rows.append(
                    {
                        "run_id": RUN_ID,
                        "run_timestamp": RUN_TIMESTAMP,
                        "image_id": image_id,
                        "file_name": file_name,
                        "roi_file_name": roi_file_name,
                        "roi_image_path": roi_image_path,
                        "processing_order": roi_row.get(
                            "processing_order",
                            np.nan,
                        ),

                        "stage_order": 3,
                        "stage_name": "containment_suppression",

                        "orig_det_idx": int(removed_row["orig_det_idx"]),
                        "score": float(removed_row["score"]),
                        "x1": float(removed_row["x1"]),
                        "y1": float(removed_row["y1"]),
                        "x2": float(removed_row["x2"]),
                        "y2": float(removed_row["y2"]),
                        "has_mask": bool(removed_row.get("has_mask", False)),

                        "action": "removed",
                        "removal_reason": str(
                            removed_row.get(
                                "removal_reason",
                                "",
                            )
                        ),
                        "removed_by_orig_det_idx": removed_row.get(
                            "removed_by_orig_det_idx",
                            np.nan,
                        ),

                        "box_area": removed_row.get("box_area", np.nan),
                        "box_containment_of_j_inside_i": removed_row.get(
                            "box_containment_of_j_inside_i",
                            np.nan,
                        ),
                        "mask_containment_of_j_inside_i": removed_row.get(
                            "mask_containment_of_j_inside_i",
                            np.nan,
                        ),
                        "area_ratio_i_over_j": removed_row.get(
                            "area_ratio_i_over_j",
                            np.nan,
                        ),
                        "score_advantage_i_minus_j": removed_row.get(
                            "score_advantage_i_minus_j",
                            np.nan,
                        ),
                        "mask_veto_active": parse_bool(
                            removed_row.get("mask_veto_active", False)
                        ),
                        "normal_duplicate_rule": parse_bool(
                            removed_row.get("normal_duplicate_rule", False)
                        ),
                        "near_total_fragment_rule": parse_bool(
                            removed_row.get("near_total_fragment_rule", False)
                        ),

                        "review_reason": "",
                    }
                )

        # ---------------------------------------------------------------------
        # 16.26.5 Save containment stage snapshot for downstream sections
        # ---------------------------------------------------------------------
        stage_containment_df = kept_after_containment_df.copy()

        crossarm_stage_summary_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "image_id": image_id,
                "file_name": file_name,
                "roi_file_name": roi_file_name,
                "roi_image_path": roi_image_path,
                "processing_order": roi_row.get("processing_order", np.nan),

                "stage_order": 3,
                "stage_name": "containment_suppression",
                "input_count": int(len(kept_after_pole_trunk_container_veto_df)),
                "output_count": int(len(kept_after_containment_df)),
                "removed_count": int(len(removed_by_containment_df)),

                "pair_debug_count": int(len(containment_pair_debug_df)),
                "has_any_valid_crossarm_masks": bool(has_any_valid_crossarm_masks),

                "containment_threshold": float(CONTAINMENT_THRESHOLD),
                "min_area_ratio": float(MIN_AREA_RATIO),
                "min_score_advantage": float(MIN_SCORE_ADVANTAGE),
                "mask_containment_filter_enabled": bool(
                    MASK_CONTAINMENT_FILTER_ENABLED
                ),
                "mask_containment_veto_threshold": float(
                    MASK_CONTAINMENT_VETO_THRESHOLD
                ),
                "near_total_box_containment_threshold": float(
                    NEAR_TOTAL_BOX_CONTAINMENT_THRESHOLD
                ),
                "mask_containment_high": float(MASK_CONTAINMENT_HIGH),
                "pair_debug_min_box_containment": float(
                    PAIR_DEBUG_MIN_BOX_CONTAINMENT
                ),

                "projected_pole_mask_available": bool(
                    projected_pole_mask_available
                ),
                "pole_mask_filter_applied": bool(pole_mask_filter_applied),
            }
        )
        
        
        # =============================================================================
        # 16.27 MAIN-CLUSTER FILTERING
        # =============================================================================
        # EXPLANATION:
        # Remove isolated detections that are far away from the main crossarm
        # candidate group.
        #
        # Input:
        #   kept_after_containment_df
        #
        # Output:
        #   kept_after_cluster_df
        #   removed_by_cluster_df
        #   stage_cluster_df
        #
        # The helper appends centre/scale columns such as:
        #   cx, cy, w, h, diag
        #
        # These extra columns are harmless and may be useful for audit/debug
        # tables later.
        #
        # PRODUCTION RULE:
        #   No display().
        #   No _safe_display().
        #   No per-ROI print().
        # =============================================================================

        current_stage = "16.27_main_cluster_filtering"

        if len(kept_after_containment_df) == 0:
            kept_after_cluster_df = kept_after_containment_df.copy()
            removed_by_cluster_df = kept_after_containment_df.copy()
            cluster_threshold_used = 0.0

        else:
            (
                kept_after_cluster_df,
                removed_by_cluster_df,
                cluster_threshold_used,
            ) = keep_main_detection_cluster(
                detections_df=kept_after_containment_df,
                center_dist_factor=float(CENTER_DIST_FACTOR),
            )

        kept_after_cluster_df = (
            kept_after_cluster_df
            .copy()
            .reset_index(drop=True)
        )

        removed_by_cluster_df = (
            removed_by_cluster_df
            .copy()
            .reset_index(drop=True)
        )

        # ---------------------------------------------------------------------
        # 16.27.1 Add removed cluster rows to trace output
        # ---------------------------------------------------------------------
        if len(removed_by_cluster_df) > 0:
            for _, removed_row in removed_by_cluster_df.iterrows():
                crossarm_trace_rows.append(
                    {
                        "run_id": RUN_ID,
                        "run_timestamp": RUN_TIMESTAMP,
                        "image_id": image_id,
                        "file_name": file_name,
                        "roi_file_name": roi_file_name,
                        "roi_image_path": roi_image_path,
                        "processing_order": roi_row.get(
                            "processing_order",
                            np.nan,
                        ),

                        "stage_order": 4,
                        "stage_name": "main_cluster_filtering",

                        "orig_det_idx": int(removed_row["orig_det_idx"]),
                        "score": float(removed_row["score"]),
                        "x1": float(removed_row["x1"]),
                        "y1": float(removed_row["y1"]),
                        "x2": float(removed_row["x2"]),
                        "y2": float(removed_row["y2"]),
                        "has_mask": bool(removed_row.get("has_mask", False)),

                        "cx": removed_row.get("cx", np.nan),
                        "cy": removed_row.get("cy", np.nan),
                        "w": removed_row.get("w", np.nan),
                        "h": removed_row.get("h", np.nan),
                        "diag": removed_row.get("diag", np.nan),

                        "action": "removed",
                        "removal_reason": str(
                            removed_row.get(
                                "removal_reason",
                                "isolated_from_main_cluster",
                            )
                        ),
                        "removed_by_orig_det_idx": np.nan,
                        "review_reason": "",
                    }
                )

        # ---------------------------------------------------------------------
        # 16.27.2 Save stage snapshot for the next section
        # ---------------------------------------------------------------------
        stage_cluster_df = kept_after_cluster_df.copy()

        crossarm_stage_summary_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "image_id": image_id,
                "file_name": file_name,
                "roi_file_name": roi_file_name,
                "roi_image_path": roi_image_path,
                "processing_order": roi_row.get("processing_order", np.nan),

                "stage_order": 4,
                "stage_name": "main_cluster_filtering",
                "input_count": int(len(kept_after_containment_df)),
                "output_count": int(len(kept_after_cluster_df)),
                "removed_count": int(len(removed_by_cluster_df)),

                "center_dist_factor": float(CENTER_DIST_FACTOR),
                "cluster_threshold_used": float(cluster_threshold_used),

                "has_any_valid_crossarm_masks": bool(
                    has_any_valid_crossarm_masks
                ),
                "projected_pole_mask_available": bool(
                    projected_pole_mask_available
                ),
                "pole_mask_filter_applied": bool(pole_mask_filter_applied),
            }
        )
        
        
        # =============================================================================
        # 16.28 POLE-OVERLAP / POLE-CORRIDOR FILTER
        # =============================================================================
        # EXPLANATION:
        # This step keeps real crossarms that are attached to the selected pole,
        # while removing:
        # 1) crossarm-like detections far away from the selected pole
        # 2) detections that are mostly the pole itself
        # 3) one-sided lower thin-arm detections, such as streetlight arms
                    #
        # FINAL COMBINED RULE:
        #   Keep a detection if:
        #
        #       A) it is attached to the pole region:
        #             - direct pole touch / overlap is enough
        #               OR
        #             - it touches the expanded pole corridor
        #
        #          OR
        #
        #       B) it passes top-of-pole rescue:
        #             - touches the pole attachment corridor
        #             - touches the top-of-pole band
        #             - is wide enough relative to other candidates
        #
        #   AND:
        #
        #       C) it is NOT pole-dominated.
        #
        # WHY:
        #   Direct selected-pole-mask overlap can be tiny for valid crossarms, so
        #   the expanded pole corridor protects real crossarms whose boxes/masks
        #   barely intersect the selected pole mask.
        #
        # PRODUCTION RULE:
        #   No display().
        #   No _safe_display().
        #   No per-ROI print().
        # =============================================================================

        current_stage = "16.28_pole_overlap_corridor_filter"

        removed_by_pole_mask_df = kept_after_cluster_df.iloc[0:0].copy()

        # ---------------------------------------------------------------------
        # 16.28.1 Empty input path
        # ---------------------------------------------------------------------
        if len(kept_after_cluster_df) == 0:
            final_kept_detections_df = kept_after_cluster_df.copy()
            removed_by_pole_mask_df = kept_after_cluster_df.copy()

        # ---------------------------------------------------------------------
        # 16.28.2 Normal pole-mask / pole-corridor path
        # ---------------------------------------------------------------------
        elif bool(POLE_MASK_FILTER_ENABLED) and bool(projected_pole_mask_available):
            pole_mask_filter_applied = True

            tmp_df = kept_after_cluster_df.copy().reset_index(drop=True)

            # -----------------------------------------------------------------
            # 16.28.2A Compute pole-overlap / pole-touch evidence
            # -----------------------------------------------------------------
            overlap_fracs = []
            overlap_sources = []
            touch_fracs = []

            for _, det_row in tmp_df.iterrows():
                overlap_frac, overlap_source = compute_detection_overlap_with_pole_mask(
                    det_row=det_row,
                    projected_pole_mask=projected_pole_mask,
                    crossarm_mask_lookup=crossarm_mask_lookup,
                )

                box_i = [
                    det_row["x1"],
                    det_row["y1"],
                    det_row["x2"],
                    det_row["y2"],
                ]

                touch_frac = compute_box_overlap_with_mask(
                    box_xyxy=box_i,
                    binary_mask=projected_pole_mask,
                )

                overlap_fracs.append(float(overlap_frac))
                overlap_sources.append(str(overlap_source))
                touch_fracs.append(float(touch_frac))

            tmp_df["pole_overlap_fraction"] = overlap_fracs
            tmp_df["pole_overlap_source"] = overlap_sources
            tmp_df["pole_touch_fraction"] = touch_fracs

            # -----------------------------------------------------------------
            # 16.28.2B Normal pole-touch / pole-overlap band
            # -----------------------------------------------------------------
            tmp_df["pole_overlap_touches_min"] = (
                tmp_df["pole_touch_fraction"].astype(float)
                >= float(POLE_OVERLAP_MIN_FRACTION)
            )

            tmp_df["pole_overlap_under_max"] = (
                tmp_df["pole_overlap_fraction"].astype(float)
                <= float(POLE_OVERLAP_MAX_FRACTION)
            )

            tmp_df["pole_dominated_reject"] = (
                tmp_df["pole_overlap_fraction"].astype(float)
                > float(POLE_OVERLAP_MAX_FRACTION)
            )

            tmp_df["attached_to_pole_region"] = tmp_df["pole_overlap_touches_min"]

            tmp_df["kept_by_pole_overlap"] = (
                tmp_df["attached_to_pole_region"]
                & tmp_df["pole_overlap_under_max"]
            )

            # -----------------------------------------------------------------
            # 16.28.2C Default top-rescue columns
            # -----------------------------------------------------------------
            tmp_df["touches_attach_corridor"] = False
            tmp_df["touches_top_band"] = False
            tmp_df["top_attach_rescue_candidate"] = False
            tmp_df["rescued_by_top_attach"] = False

            tmp_df["box_w"] = (
                tmp_df["x2"].astype(float) - tmp_df["x1"].astype(float)
            ).clip(lower=0.0)

            max_box_w = (
                float(tmp_df["box_w"].max())
                if len(tmp_df) > 0
                else 0.0
            )

            if max_box_w > 0:
                tmp_df["relative_width_to_max"] = (
                    tmp_df["box_w"] / max_box_w
                )
            else:
                tmp_df["relative_width_to_max"] = 0.0
                
                
            # -----------------------------------------------------------------
            # 16.28.2D Pole corridor + top-of-pole rescue
            # -----------------------------------------------------------------
            pole_cols = np.where(projected_pole_mask.any(axis=0))[0]
            pole_rows = np.where(projected_pole_mask.any(axis=1))[0]

            pole_corridor_available = bool(
                len(pole_cols) > 0
                and len(pole_rows) > 0
            )

            if pole_corridor_available:
                pole_x1 = int(pole_cols.min())
                pole_x2 = int(pole_cols.max())
                pole_top_y = int(pole_rows.min())

                attach_x1 = max(
                    0,
                    pole_x1 - int(POLE_ATTACH_MARGIN_PX),
                )

                attach_x2 = min(
                    int(roi_w) - 1,
                    pole_x2 + int(POLE_ATTACH_MARGIN_PX),
                )

                top_band_y1 = max(
                    0,
                    pole_top_y - int(TOP_BAND_ABOVE),
                )

                top_band_y2 = min(
                    int(roi_h) - 1,
                    pole_top_y + int(TOP_BAND_BELOW),
                )

                tmp_df["touches_attach_corridor"] = (
                    (tmp_df["x2"].astype(float) >= float(attach_x1))
                    & (tmp_df["x1"].astype(float) <= float(attach_x2))
                )

                # Treat expanded pole-corridor contact as pole attachment.
                tmp_df["attached_to_pole_region"] = (
                    tmp_df["pole_overlap_touches_min"]
                    | tmp_df["touches_attach_corridor"]
                )

                tmp_df["kept_by_pole_overlap"] = (
                    tmp_df["attached_to_pole_region"]
                    & tmp_df["pole_overlap_under_max"]
                )

                tmp_df["touches_top_band"] = (
                    (tmp_df["y2"].astype(float) >= float(top_band_y1))
                    & (tmp_df["y1"].astype(float) <= float(top_band_y2))
                )

                tmp_df["top_attach_rescue_candidate"] = (
                    tmp_df["touches_attach_corridor"]
                    & tmp_df["touches_top_band"]
                    & (
                        tmp_df["relative_width_to_max"].astype(float)
                        >= float(MIN_RELATIVE_WIDTH_TO_MAX)
                    )
                )

                tmp_df["rescued_by_top_attach"] = (
                    (~tmp_df["kept_by_pole_overlap"])
                    & tmp_df["top_attach_rescue_candidate"]
                    & (~tmp_df["pole_dominated_reject"])
                )

                # -------------------------------------------------------------
                # 16.28.2D.1 One-sided lower thin-arm veto
                # -------------------------------------------------------------
                # EXPLANATION:
                # Removes streetlight-arm / cantilever-arm false positives that
                # look like crossarms but:
                #   - sit lower than the main/top crossarm candidate
                #   - are thin / rod-like
                #   - extend mostly to one side of the selected pole
                #   - touch the pole attachment corridor
                #
                # IMPORTANT:
                # This is intentionally part of the pole-overlap / corridor
                # filter so no downstream labels, stage names, or merge inputs
                # need to change.
                # -------------------------------------------------------------
                pole_cx = 0.5 * (float(pole_x1) + float(pole_x2))

                tmp_df["box_w"] = (
                    tmp_df["x2"].astype(float)
                    - tmp_df["x1"].astype(float)
                ).clip(lower=0.0)

                tmp_df["box_h"] = (
                    tmp_df["y2"].astype(float)
                    - tmp_df["y1"].astype(float)
                ).clip(lower=0.0)

                tmp_df["box_cx"] = (
                    0.5
                    * (
                        tmp_df["x1"].astype(float)
                        + tmp_df["x2"].astype(float)
                    )
                )

                tmp_df["box_cy"] = (
                    0.5
                    * (
                        tmp_df["y1"].astype(float)
                        + tmp_df["y2"].astype(float)
                    )
                )

                tmp_df["box_aspect_w_over_h"] = (
                    tmp_df["box_w"].astype(float)
                    / tmp_df["box_h"].astype(float).clip(lower=1.0)
                )

                median_box_h = (
                    float(tmp_df["box_h"].median())
                    if len(tmp_df) > 0
                    else 0.0
                )

                # Use the topmost candidate as the primary/top crossarm context.
                # This is only used as a relative height reference.
                primary_crossarm_cy = (
                    float(tmp_df["box_cy"].min())
                    if len(tmp_df) > 0
                    else np.nan
                )

                y_gap_threshold = max(
                    float(ONE_SIDED_LOWER_ARM_MIN_Y_GAP_PX),
                    float(ONE_SIDED_LOWER_ARM_MIN_Y_GAP_FACTOR)
                    * max(float(median_box_h), 1.0),
                )

                tmp_df["lower_than_primary_crossarm"] = (
                    tmp_df["box_cy"].astype(float)
                    > float(primary_crossarm_cy) + float(y_gap_threshold)
                )

                tmp_df["left_extent_from_pole_px"] = (
                    float(pole_cx) - tmp_df["x1"].astype(float)
                ).clip(lower=0.0)

                tmp_df["right_extent_from_pole_px"] = (
                    tmp_df["x2"].astype(float) - float(pole_cx)
                ).clip(lower=0.0)

                tmp_df["larger_side_from_pole_px"] = (
                    tmp_df[
                        [
                            "left_extent_from_pole_px",
                            "right_extent_from_pole_px",
                        ]
                    ]
                    .astype(float)
                    .max(axis=1)
                    .clip(lower=1.0)
                )

                tmp_df["smaller_side_from_pole_px"] = (
                    tmp_df[
                        [
                            "left_extent_from_pole_px",
                            "right_extent_from_pole_px",
                        ]
                    ]
                    .astype(float)
                    .min(axis=1)
                )

                tmp_df["pole_side_balance"] = (
                    tmp_df["smaller_side_from_pole_px"].astype(float)
                    / tmp_df["larger_side_from_pole_px"].astype(float)
                )

                tmp_df["pole_short_side_fraction"] = (
                    tmp_df["smaller_side_from_pole_px"].astype(float)
                    / tmp_df["box_w"].astype(float).clip(lower=1.0)
                )

                tmp_df["one_sided_from_pole"] = (
                    (
                        tmp_df["pole_side_balance"].astype(float)
                        <= float(ONE_SIDED_LOWER_ARM_SIDE_BALANCE_MAX)
                    )
                    & (
                        tmp_df["pole_short_side_fraction"].astype(float)
                        <= float(ONE_SIDED_LOWER_ARM_SHORT_SIDE_FRAC_MAX)
                    )
                )

                tmp_df["thin_arm_candidate"] = (
                    (
                        tmp_df["box_aspect_w_over_h"].astype(float)
                        >= float(ONE_SIDED_LOWER_ARM_ASPECT_MIN)
                    )
                    & (
                        tmp_df["box_h"].astype(float)
                        <= (
                            float(ONE_SIDED_LOWER_ARM_HEIGHT_TO_MEDIAN_MAX)
                            * max(float(median_box_h), 1.0)
                        )
                    )
                )

                if (
                    bool(ONE_SIDED_LOWER_ARM_VETO_ENABLED)
                    and len(tmp_df) >= 2
                ):
                    tmp_df["one_sided_lower_thin_arm_veto"] = (
                        tmp_df["lower_than_primary_crossarm"]
                        & tmp_df["one_sided_from_pole"]
                        & tmp_df["thin_arm_candidate"]
                        & tmp_df["touches_attach_corridor"]
                    )

                else:
                    tmp_df["one_sided_lower_thin_arm_veto"] = False

            else:
                tmp_df["touches_attach_corridor"] = False

                tmp_df["attached_to_pole_region"] = (
                    tmp_df["pole_overlap_touches_min"]
                )

                tmp_df["kept_by_pole_overlap"] = (
                    tmp_df["attached_to_pole_region"]
                    & tmp_df["pole_overlap_under_max"]
                )

                tmp_df["touches_top_band"] = False
                tmp_df["top_attach_rescue_candidate"] = False
                tmp_df["rescued_by_top_attach"] = False

                tmp_df["box_w"] = (
                    tmp_df["x2"].astype(float)
                    - tmp_df["x1"].astype(float)
                ).clip(lower=0.0)

                tmp_df["box_h"] = (
                    tmp_df["y2"].astype(float)
                    - tmp_df["y1"].astype(float)
                ).clip(lower=0.0)

                tmp_df["box_cx"] = np.nan
                tmp_df["box_cy"] = np.nan
                tmp_df["box_aspect_w_over_h"] = np.nan

                tmp_df["lower_than_primary_crossarm"] = False
                tmp_df["left_extent_from_pole_px"] = np.nan
                tmp_df["right_extent_from_pole_px"] = np.nan
                tmp_df["larger_side_from_pole_px"] = np.nan
                tmp_df["smaller_side_from_pole_px"] = np.nan
                tmp_df["pole_side_balance"] = np.nan
                tmp_df["pole_short_side_fraction"] = np.nan

                tmp_df["one_sided_from_pole"] = False
                tmp_df["thin_arm_candidate"] = False
                tmp_df["one_sided_lower_thin_arm_veto"] = False


            # -----------------------------------------------------------------
            # 16.28.2E Final combined keep rule
            # -----------------------------------------------------------------
            # EXPLANATION:
            # Keep detections that are attached to / rescued by the selected
            # pole region, while rejecting:
            #   1) pole-dominated detections
            #   2) one-sided lower thin-arm false positives such as streetlight
            #      arms
            # -----------------------------------------------------------------
            keep_flags_arr = (
                (
                    tmp_df["kept_by_pole_overlap"]
                    | tmp_df["rescued_by_top_attach"]
                )
                & (~tmp_df["pole_dominated_reject"])
                & (~tmp_df["one_sided_lower_thin_arm_veto"])
            ).to_numpy(dtype=bool)

            final_kept_detections_df = (
                tmp_df[keep_flags_arr]
                .copy()
                .reset_index(drop=True)
            )

            removed_by_pole_mask_df = (
                tmp_df[~keep_flags_arr]
                .copy()
                .reset_index(drop=True)
            )

            # -----------------------------------------------------------------
            # 16.28.2F Add removal reason
            # -----------------------------------------------------------------
            if len(removed_by_pole_mask_df) > 0:
                removal_reasons = []

                for _, det_row in removed_by_pole_mask_df.iterrows():
                    if parse_bool(
                        det_row.get("one_sided_lower_thin_arm_veto", False)
                    ):
                        removal_reasons.append(
                            "one_sided_lower_thin_arm_veto"
                        )

                    elif parse_bool(
                        det_row.get("pole_dominated_reject", False)
                    ):
                        removal_reasons.append(
                            f"pole_overlap_gt_max_{float(POLE_OVERLAP_MAX_FRACTION):.3f}"
                        )

                    elif not parse_bool(
                        det_row.get("attached_to_pole_region", False)
                    ):
                        removal_reasons.append(
                            "failed_low_overlap_no_corridor_and_not_top_rescued"
                        )

                    else:
                        removal_reasons.append(
                            "failed_pole_overlap_band_or_top_attach_rescue"
                        )

                removed_by_pole_mask_df["removal_reason"] = removal_reasons

        # ---------------------------------------------------------------------
        # 16.28.3 No projected pole mask available / pole filter disabled path
        # ---------------------------------------------------------------------
        else:
            final_kept_detections_df = kept_after_cluster_df.copy().reset_index(
                drop=True
            )

            removed_by_pole_mask_df = (
                kept_after_cluster_df
                .iloc[0:0]
                .copy()
                .reset_index(drop=True)
            )

            if bool(POLE_MASK_FILTER_ENABLED):
                no_pole_source = "no_projected_pole_mask"
                no_pole_review_reason = CELL16_REVIEW_POLE_MASK_UNAVAILABLE
            else:
                no_pole_source = "pole_mask_filter_disabled"
                no_pole_review_reason = ""

            final_kept_detections_df["pole_overlap_fraction"] = np.nan
            final_kept_detections_df["pole_overlap_source"] = no_pole_source
            final_kept_detections_df["pole_touch_fraction"] = np.nan

            final_kept_detections_df["pole_overlap_touches_min"] = False
            final_kept_detections_df["pole_overlap_under_max"] = False
            final_kept_detections_df["pole_dominated_reject"] = False
            final_kept_detections_df["attached_to_pole_region"] = False
            final_kept_detections_df["kept_by_pole_overlap"] = False

            final_kept_detections_df["touches_attach_corridor"] = False
            final_kept_detections_df["touches_top_band"] = False
            final_kept_detections_df["top_attach_rescue_candidate"] = False
            final_kept_detections_df["rescued_by_top_attach"] = False
            final_kept_detections_df["box_h"] = np.nan
            final_kept_detections_df["box_cx"] = np.nan
            final_kept_detections_df["box_cy"] = np.nan
            final_kept_detections_df["box_aspect_w_over_h"] = np.nan

            final_kept_detections_df["lower_than_primary_crossarm"] = False
            final_kept_detections_df["left_extent_from_pole_px"] = np.nan
            final_kept_detections_df["right_extent_from_pole_px"] = np.nan
            final_kept_detections_df["larger_side_from_pole_px"] = np.nan
            final_kept_detections_df["smaller_side_from_pole_px"] = np.nan
            final_kept_detections_df["pole_side_balance"] = np.nan
            final_kept_detections_df["pole_short_side_fraction"] = np.nan

            final_kept_detections_df["one_sided_from_pole"] = False
            final_kept_detections_df["thin_arm_candidate"] = False
            final_kept_detections_df["one_sided_lower_thin_arm_veto"] = False

            if len(final_kept_detections_df) > 0:
                final_kept_detections_df["box_w"] = (
                    final_kept_detections_df["x2"].astype(float)
                    - final_kept_detections_df["x1"].astype(float)
                ).clip(lower=0.0)

                max_box_w = float(final_kept_detections_df["box_w"].max())

                if max_box_w > 0:
                    final_kept_detections_df["relative_width_to_max"] = (
                        final_kept_detections_df["box_w"] / max_box_w
                    )
                else:
                    final_kept_detections_df["relative_width_to_max"] = 0.0

                if no_pole_review_reason:
                    if "review_reason" not in final_kept_detections_df.columns:
                        final_kept_detections_df["review_reason"] = ""

                    final_kept_detections_df["review_reason"] = (
                        final_kept_detections_df["review_reason"].apply(
                            lambda old_reason: _append_reason(
                                old_reason,
                                no_pole_review_reason,
                            )
                        )
                    )

            else:
                final_kept_detections_df["box_w"] = pd.Series(dtype=float)
                final_kept_detections_df["relative_width_to_max"] = pd.Series(
                    dtype=float
                )

        final_kept_detections_df = (
            final_kept_detections_df
            .copy()
            .reset_index(drop=True)
        )

        removed_by_pole_mask_df = (
            removed_by_pole_mask_df
            .copy()
            .reset_index(drop=True)
        )

        # ---------------------------------------------------------------------
        # 16.28.4 Add removed pole-overlap rows to trace output
        # ---------------------------------------------------------------------
        if len(removed_by_pole_mask_df) > 0:
            for _, removed_row in removed_by_pole_mask_df.iterrows():
                crossarm_trace_rows.append(
                    {
                        "run_id": RUN_ID,
                        "run_timestamp": RUN_TIMESTAMP,
                        "image_id": image_id,
                        "file_name": file_name,
                        "roi_file_name": roi_file_name,
                        "roi_image_path": roi_image_path,
                        "processing_order": roi_row.get(
                            "processing_order",
                            np.nan,
                        ),

                        "stage_order": 5,
                        "stage_name": "pole_overlap_corridor_filter",

                        "orig_det_idx": int(removed_row["orig_det_idx"]),
                        "score": float(removed_row["score"]),
                        "x1": float(removed_row["x1"]),
                        "y1": float(removed_row["y1"]),
                        "x2": float(removed_row["x2"]),
                        "y2": float(removed_row["y2"]),
                        "has_mask": bool(removed_row.get("has_mask", False)),

                        "pole_overlap_fraction": removed_row.get(
                            "pole_overlap_fraction",
                            np.nan,
                        ),
                        "pole_overlap_source": str(
                            removed_row.get(
                                "pole_overlap_source",
                                "",
                            )
                        ),
                        "pole_touch_fraction": removed_row.get(
                            "pole_touch_fraction",
                            np.nan,
                        ),
                        "pole_overlap_touches_min": parse_bool(
                            removed_row.get(
                                "pole_overlap_touches_min",
                                False,
                            )
                        ),
                        "pole_overlap_under_max": parse_bool(
                            removed_row.get(
                                "pole_overlap_under_max",
                                False,
                            )
                        ),
                        "pole_dominated_reject": parse_bool(
                            removed_row.get(
                                "pole_dominated_reject",
                                False,
                            )
                        ),
                        "attached_to_pole_region": parse_bool(
                            removed_row.get(
                                "attached_to_pole_region",
                                False,
                            )
                        ),
                        "kept_by_pole_overlap": parse_bool(
                            removed_row.get(
                                "kept_by_pole_overlap",
                                False,
                            )
                        ),
                        "touches_attach_corridor": parse_bool(
                            removed_row.get(
                                "touches_attach_corridor",
                                False,
                            )
                        ),
                        "touches_top_band": parse_bool(
                            removed_row.get(
                                "touches_top_band",
                                False,
                            )
                        ),
                        "top_attach_rescue_candidate": parse_bool(
                            removed_row.get(
                                "top_attach_rescue_candidate",
                                False,
                            )
                        ),
                        "rescued_by_top_attach": parse_bool(
                            removed_row.get(
                                "rescued_by_top_attach",
                                False,
                            )
                        ),

                        "box_w": removed_row.get("box_w", np.nan),
                        "relative_width_to_max": removed_row.get(
                            "relative_width_to_max",
                            np.nan,
                        ),

                        # -----------------------------------------------------
                        # One-sided lower thin-arm veto audit fields
                        # -----------------------------------------------------
                        "box_h": removed_row.get("box_h", np.nan),
                        "box_cx": removed_row.get("box_cx", np.nan),
                        "box_cy": removed_row.get("box_cy", np.nan),
                        "box_aspect_w_over_h": removed_row.get(
                            "box_aspect_w_over_h",
                            np.nan,
                        ),

                        "lower_than_primary_crossarm": parse_bool(
                            removed_row.get(
                                "lower_than_primary_crossarm",
                                False,
                            )
                        ),
                        "one_sided_from_pole": parse_bool(
                            removed_row.get(
                                "one_sided_from_pole",
                                False,
                            )
                        ),
                        "thin_arm_candidate": parse_bool(
                            removed_row.get(
                                "thin_arm_candidate",
                                False,
                            )
                        ),
                        "one_sided_lower_thin_arm_veto": parse_bool(
                            removed_row.get(
                                "one_sided_lower_thin_arm_veto",
                                False,
                            )
                        ),

                        "left_extent_from_pole_px": removed_row.get(
                            "left_extent_from_pole_px",
                            np.nan,
                        ),
                        "right_extent_from_pole_px": removed_row.get(
                            "right_extent_from_pole_px",
                            np.nan,
                        ),
                        "pole_side_balance": removed_row.get(
                            "pole_side_balance",
                            np.nan,
                        ),
                        "pole_short_side_fraction": removed_row.get(
                            "pole_short_side_fraction",
                            np.nan,
                        ),

                        "action": "removed",
                        "removal_reason": str(
                            removed_row.get(
                                "removal_reason",
                                "",
                            )
                        ),
                        "removed_by_orig_det_idx": np.nan,
                        "review_reason": "",
                    }
                )
        # ---------------------------------------------------------------------
        # 16.28.5 Save stage snapshot for same-crossarm continuity merge
        # ---------------------------------------------------------------------
        stage_pole_overlap_df = final_kept_detections_df.copy()

        crossarm_stage_summary_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "image_id": image_id,
                "file_name": file_name,
                "roi_file_name": roi_file_name,
                "roi_image_path": roi_image_path,
                "processing_order": roi_row.get("processing_order", np.nan),

                "stage_order": 5,
                "stage_name": "pole_overlap_corridor_filter",
                "input_count": int(len(kept_after_cluster_df)),
                "output_count": int(len(final_kept_detections_df)),
                "removed_count": int(len(removed_by_pole_mask_df)),

                "one_sided_lower_thin_arm_removed_count": int(
                    (
                        removed_by_pole_mask_df["removal_reason"].astype(str)
                        == "one_sided_lower_thin_arm_veto"
                    ).sum()
                )
                if (
                    isinstance(removed_by_pole_mask_df, pd.DataFrame)
                    and "removal_reason" in removed_by_pole_mask_df.columns
                )
                else 0,

                "pole_mask_filter_enabled": bool(POLE_MASK_FILTER_ENABLED),
                "pole_mask_filter_applied": bool(pole_mask_filter_applied),
                "projected_pole_mask_available": bool(
                    projected_pole_mask_available
                ),

                "pole_overlap_min_fraction": float(
                    POLE_OVERLAP_MIN_FRACTION
                ),
                "pole_overlap_max_fraction": float(
                    POLE_OVERLAP_MAX_FRACTION
                ),
                "min_relative_width_to_max": float(
                    MIN_RELATIVE_WIDTH_TO_MAX
                ),
                "pole_attach_margin_px": float(POLE_ATTACH_MARGIN_PX),
                "top_band_above": float(TOP_BAND_ABOVE),
                "top_band_below": float(TOP_BAND_BELOW),

                "one_sided_lower_arm_veto_enabled": bool(
                    ONE_SIDED_LOWER_ARM_VETO_ENABLED
                ),
                "one_sided_lower_arm_side_balance_max": float(
                    ONE_SIDED_LOWER_ARM_SIDE_BALANCE_MAX
                ),
                "one_sided_lower_arm_short_side_frac_max": float(
                    ONE_SIDED_LOWER_ARM_SHORT_SIDE_FRAC_MAX
                ),
                "one_sided_lower_arm_aspect_min": float(
                    ONE_SIDED_LOWER_ARM_ASPECT_MIN
                ),
                "one_sided_lower_arm_height_to_median_max": float(
                    ONE_SIDED_LOWER_ARM_HEIGHT_TO_MEDIAN_MAX
                ),
                "one_sided_lower_arm_min_y_gap_px": float(
                    ONE_SIDED_LOWER_ARM_MIN_Y_GAP_PX
                ),
                "one_sided_lower_arm_min_y_gap_factor": float(
                    ONE_SIDED_LOWER_ARM_MIN_Y_GAP_FACTOR
                ),
            }
        )
    
        
        # =============================================================================
        # 16.29 SAME-CROSSARM CONTINUITY MERGE
        # =============================================================================
        # EXPLANATION:
        # This stage merges separate SAM3 detections when they look like two
        # fragments of the same physical crossarm.
        #
        # Example:
        #   idx 3 = left side of one crossarm
        #   idx 6 = right side of the same crossarm
        #
        # The stage checks:
        #   1) same / very similar angle
        #   2) low perpendicular distance between fitted axes
        #   3) reasonable gap between fragments
        #   4) pair bridges the expanded pole corridor
        #
        # It runs after pole-overlap / pole-corridor filtering and before
        # X-split / PCA.
        #
        # IMPORTANT:
        #   merge_same_crossarm_fragments updates crossarm_mask_lookup in place
        #   when it creates a merged detection.
        #
        #   The merged row receives a new orig_det_idx, and:
        #       crossarm_mask_lookup[new_orig_idx]
        #   stores the merged union mask.
        #
        # PRODUCTION RULE:
        #   No display().
        #   No _safe_display().
        #   No per-ROI print().
        # =============================================================================

        current_stage = "16.29_same_crossarm_continuity_merge"

        stage_same_xarm_merge_input_df = (
            final_kept_detections_df
            .copy()
            .reset_index(drop=True)
        )

        (
            final_kept_detections_df,
            removed_by_same_xarm_merge_df,
            same_xarm_pair_debug_df,
            same_xarm_component_debug_df,
        ) = merge_same_crossarm_fragments(
            detections_df=stage_same_xarm_merge_input_df,
            crossarm_mask_lookup=crossarm_mask_lookup,
            projected_pole_mask=(
                projected_pole_mask
                if projected_pole_mask_available
                else None
            ),
            roi_w=int(roi_w),
            roi_h=int(roi_h),
        )

        final_kept_detections_df = (
            final_kept_detections_df
            .copy()
            .reset_index(drop=True)
        )

        removed_by_same_xarm_merge_df = (
            removed_by_same_xarm_merge_df
            .copy()
            .reset_index(drop=True)
        )

        same_xarm_pair_debug_df = (
            same_xarm_pair_debug_df
            .copy()
            .reset_index(drop=True)
            if isinstance(same_xarm_pair_debug_df, pd.DataFrame)
            else pd.DataFrame()
        )

        same_xarm_component_debug_df = (
            same_xarm_component_debug_df
            .copy()
            .reset_index(drop=True)
            if isinstance(same_xarm_component_debug_df, pd.DataFrame)
            else pd.DataFrame()
        )

        # ---------------------------------------------------------------------
        # 16.29.1 Add replaced original detections to trace output
        # ---------------------------------------------------------------------
        if len(removed_by_same_xarm_merge_df) > 0:
            for _, removed_row in removed_by_same_xarm_merge_df.iterrows():
                crossarm_trace_rows.append(
                    {
                        "run_id": RUN_ID,
                        "run_timestamp": RUN_TIMESTAMP,
                        "image_id": image_id,
                        "file_name": file_name,
                        "roi_file_name": roi_file_name,
                        "roi_image_path": roi_image_path,
                        "processing_order": roi_row.get(
                            "processing_order",
                            np.nan,
                        ),

                        "stage_order": 6,
                        "stage_name": "same_crossarm_continuity_merge",

                        "orig_det_idx": int(removed_row["orig_det_idx"]),
                        "score": float(removed_row["score"]),
                        "x1": float(removed_row["x1"]),
                        "y1": float(removed_row["y1"]),
                        "x2": float(removed_row["x2"]),
                        "y2": float(removed_row["y2"]),
                        "has_mask": bool(removed_row.get("has_mask", False)),

                        "same_xarm_merge_applied": parse_bool(
                            removed_row.get(
                                "same_xarm_merge_applied",
                                True,
                            )
                        ),
                        "same_xarm_merge_group_id": removed_row.get(
                            "same_xarm_merge_group_id",
                            np.nan,
                        ),
                        "same_xarm_merge_count": removed_row.get(
                            "same_xarm_merge_count",
                            np.nan,
                        ),
                        "merged_from_orig_det_idxs": str(
                            removed_row.get(
                                "merged_from_orig_det_idxs",
                                "",
                            )
                        ),

                        "action": "removed",
                        "removal_reason": str(
                            removed_row.get(
                                "removal_reason",
                                "replaced_by_same_crossarm_merge",
                            )
                        ),
                        "removed_by_orig_det_idx": np.nan,
                        "review_reason": str(
                            removed_row.get(
                                "review_reason",
                                "",
                            )
                        ),
                    }
                )

        # ---------------------------------------------------------------------
        # 16.29.2 Save stage snapshot for the next section
        # ---------------------------------------------------------------------
        stage_same_xarm_merge_df = (
            final_kept_detections_df
            .copy()
            .reset_index(drop=True)
        )

        crossarm_stage_summary_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "image_id": image_id,
                "file_name": file_name,
                "roi_file_name": roi_file_name,
                "roi_image_path": roi_image_path,
                "processing_order": roi_row.get("processing_order", np.nan),

                "stage_order": 6,
                "stage_name": "same_crossarm_continuity_merge",
                "input_count": int(len(stage_same_xarm_merge_input_df)),
                "output_count": int(len(stage_same_xarm_merge_df)),
                "removed_count": int(len(removed_by_same_xarm_merge_df)),

                "same_xarm_merge_enabled": bool(SAME_XARM_MERGE_ENABLED),
                "merged_detections_created": int(
                    len(same_xarm_component_debug_df)
                ),
                "pair_debug_count": int(len(same_xarm_pair_debug_df)),

                "same_xarm_merge_min_mask_pixels": int(
                    SAME_XARM_MERGE_MIN_MASK_PIXELS
                ),
                "same_xarm_merge_max_angle_diff_deg": float(
                    SAME_XARM_MERGE_MAX_ANGLE_DIFF_DEG
                ),
                "same_xarm_merge_max_perp_dist_px": float(
                    SAME_XARM_MERGE_MAX_PERP_DIST_PX
                ),
                "same_xarm_merge_max_gap_px": float(
                    SAME_XARM_MERGE_MAX_GAP_PX
                ),
                "same_xarm_merge_require_pole_bridge": bool(
                    SAME_XARM_MERGE_REQUIRE_POLE_BRIDGE
                ),
                "same_xarm_merge_box_pad_px": int(
                    SAME_XARM_MERGE_BOX_PAD_PX
                ),
                "same_xarm_merge_score_mode": str(
                    SAME_XARM_MERGE_SCORE_MODE
                ),

                "projected_pole_mask_available": bool(
                    projected_pole_mask_available
                ),
                "pole_mask_filter_applied": bool(pole_mask_filter_applied),
            }
        )
        
        
        # =============================================================================
        # 16.30 SINGLE-BOX X-SHAPED CROSSARM SPLIT
        # =============================================================================
        # EXPLANATION:
        # This stage handles the case where SAM3 returns ONE large crossarm
        # detection, but that one mask/box visually contains TWO crossed
        # crossarms.
        #
        # This is different from the targeted X-ownership stage later:
        #   - single-box X split:   1 SAM3 detection contains 2 diagonals
        #   - two-box X ownership:  2 SAM3 detections overlap at the crossing
        #
        # STRATEGY:
        #   1) Only inspect detections that look suspiciously large / tall /
        #      square-ish.
        #   2) Skeletonize the SAM3 mask so the X becomes thin centre-lines.
        #   3) Run HoughLinesP on the skeleton to find line directions.
        #   4) If skeleton Hough fails, fall back to image edges inside the
        #      SAM3 mask.
        #   5) If two strong angle groups exist, split mask pixels by nearest
        #      line.
        #   6) Replace the parent detection with two child detections only if
        #      the split passes conservative safety checks.
        #   7) If X-like but split fails, keep the parent, flag it for review,
        #      and skip PCA axis cleanup so the X is not mangled.
        #
        # IMPORTANT:
        #   New child detections receive new orig_det_idx values greater than
        #   any existing DataFrame orig_det_idx or crossarm_mask_lookup key.
        #
        #   Each child mask is registered back into crossarm_mask_lookup using
        #   that new orig_det_idx, so downstream stages can continue using the
        #   same lookup pattern.
        #
        # PRODUCTION RULE:
        #   No display().
        #   No _safe_display().
        #   No per-ROI print().
        # =============================================================================

        current_stage = "16.30_single_box_xsplit"

        # ---------------------------------------------------------------------
        # 16.30.1 Prepare input snapshot and preserve pre-split masks
        # ---------------------------------------------------------------------
        stage_single_xsplit_input_df = (
            final_kept_detections_df
            .copy()
            .reset_index(drop=True)
        )

        single_xsplit_input_mask_lookup = {
            int(k): (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in crossarm_mask_lookup.items()
        }

        single_xsplit_debug_rows = []
        single_xsplit_child_rows = []
        single_xsplit_parent_replaced_rows = []

        single_xsplit_applied_count = 0
        single_xsplit_failed_count = 0
        single_xsplit_xlike_count = 0

        # ---------------------------------------------------------------------
        # 16.30.2 Convert current ROI image to RGB array for fallback edge Hough
        # ---------------------------------------------------------------------
        try:
            if isinstance(image, Image.Image):
                image_np_for_xsplit = np.array(image.convert("RGB"))

            else:
                image_np_for_xsplit = np.asarray(image)

                if (
                    image_np_for_xsplit.ndim == 3
                    and image_np_for_xsplit.shape[2] >= 3
                ):
                    image_np_for_xsplit = (
                        image_np_for_xsplit[:, :, :3]
                        .astype(np.uint8)
                    )

        except Exception:
            image_np_for_xsplit = None

        # ---------------------------------------------------------------------
        # 16.30.3 Run single-box X split
        # ---------------------------------------------------------------------
        if (
            bool(SINGLE_XSPLIT_ENABLED)
            and len(stage_single_xsplit_input_df) > 0
        ):
            tmp_df = (
                stage_single_xsplit_input_df
                .copy()
                .reset_index(drop=True)
            )

            # Initialise new review/debug columns without clobbering existing
            # values.
            for col, default_value in [
                ("is_x_like", False),
                ("xsplit_attempted", False),
                ("xsplit_applied", False),
                ("xsplit_reason", ""),
                ("xsplit_source", ""),
                ("skip_axis_cleanup", False),
                ("review_reason", ""),
                ("parent_orig_det_idx", np.nan),
                ("xsplit_child_rank", np.nan),
            ]:
                if col not in tmp_df.columns:
                    tmp_df[col] = default_value

            tmp_df["box_w"] = (
                tmp_df["x2"].astype(float)
                - tmp_df["x1"].astype(float)
            ).clip(lower=0.0)

            tmp_df["box_h"] = (
                tmp_df["y2"].astype(float)
                - tmp_df["y1"].astype(float)
            ).clip(lower=0.0)

            tmp_df["box_aspect_w_over_h"] = (
                tmp_df["box_w"] / tmp_df["box_h"].clip(lower=1.0)
            )

            tmp_df["box_area_xsplit"] = tmp_df.apply(
                lambda r: box_area_xyxy(
                    [
                        r["x1"],
                        r["y1"],
                        r["x2"],
                        r["y2"],
                    ]
                ),
                axis=1,
            )

            positive_heights = tmp_df.loc[
                tmp_df["box_h"] > 0,
                "box_h",
            ]

            positive_areas = tmp_df.loc[
                tmp_df["box_area_xsplit"] > 0,
                "box_area_xsplit",
            ]

            median_box_h = (
                float(positive_heights.median())
                if len(positive_heights) > 0
                else 0.0
            )

            median_box_area = (
                float(positive_areas.median())
                if len(positive_areas) > 0
                else 0.0
            )

            tmp_df["xsplit_height_to_median"] = (
                tmp_df["box_h"] / max(median_box_h, 1.0)
                if median_box_h > 0
                else 0.0
            )

            tmp_df["xsplit_area_to_median"] = (
                tmp_df["box_area_xsplit"] / max(median_box_area, 1.0)
                if median_box_area > 0
                else 0.0
            )

            existing_orig_idxs = []

            if "orig_det_idx" in tmp_df.columns:
                existing_orig_idxs.extend(
                    [
                        int(v)
                        for v in tmp_df["orig_det_idx"].dropna().tolist()
                    ]
                )

            existing_orig_idxs.extend(
                [
                    int(k)
                    for k in crossarm_mask_lookup.keys()
                ]
            )

            next_child_orig_idx = max(existing_orig_idxs + [-1]) + 1

            output_rows = []

            for _, det_row in tmp_df.iterrows():
                row_out = det_row.copy()

                orig_idx = int(det_row["orig_det_idx"])
                mask_i = crossarm_mask_lookup.get(orig_idx, None)

                pca_sig = _mask_pca_signature(mask_i)

                shape_suspicious = bool(
                    (
                        float(det_row.get("xsplit_height_to_median", 0.0))
                        >= float(XSPLIT_HEIGHT_TO_MEDIAN_TRIGGER)
                    )
                    or (
                        float(det_row.get("xsplit_area_to_median", 0.0))
                        >= float(XSPLIT_AREA_TO_MEDIAN_TRIGGER)
                    )
                    or (
                        float(det_row.get("box_aspect_w_over_h", 999.0))
                        <= float(XSPLIT_MAX_ASPECT_FOR_SUSPICIOUS)
                    )
                )

                pca_not_single_clean_axis = bool(
                    pca_sig.get("valid", False)
                    and float(pca_sig.get("pc1_ratio", 1.0))
                    <= float(XSPLIT_PC1_RATIO_MAX_FOR_XLIKE)
                )

                angle_result = {
                    "valid": False,
                    "reason": "not_run",
                    "source": "none",
                    "angle_a": np.nan,
                    "angle_b": np.nan,
                    "angle_diff": np.nan,
                    "num_lines": 0,
                }

                split_result = {
                    "valid": False,
                    "reason": "not_run",
                }

                x_like = False
                split_applied = False

                has_usable_mask = bool(
                    isinstance(mask_i, np.ndarray)
                    and mask_i.ndim == 2
                    and int(mask_i.sum()) >= int(XSPLIT_MIN_PARENT_MASK_PIXELS)
                )

                should_test_for_x = bool(
                    has_usable_mask
                    and (shape_suspicious or pca_not_single_clean_axis)
                )

                if should_test_for_x:
                    angle_result = _detect_two_angle_groups_for_mask(
                        mask_bool=mask_i,
                        image_np_rgb=image_np_for_xsplit,
                    )

                    x_like = bool(angle_result.get("valid", False))

                if x_like:
                    single_xsplit_xlike_count += 1

                    row_out["is_x_like"] = True
                    row_out["xsplit_attempted"] = True
                    row_out["xsplit_source"] = str(
                        angle_result.get("source", "")
                    )

                    split_result = _split_mask_using_two_angles(
                        mask_bool=mask_i,
                        angle_a=float(angle_result["angle_a"]),
                        angle_b=float(angle_result["angle_b"]),
                    )

                    if bool(split_result.get("valid", False)):
                        child_masks = [
                            split_result["child_mask_a"],
                            split_result["child_mask_b"],
                        ]

                        child_models = [
                            split_result["model_a"],
                            split_result["model_b"],
                        ]

                        child_rows_for_parent = []
                        child_valid = True
                        child_invalid_reason = ""

                        for child_rank, child_mask in enumerate(
                            child_masks,
                            start=1,
                        ):
                            valid_box, cx1, cy1, cx2, cy2 = _bbox_from_mask(
                                child_mask,
                                pad_px=int(XSPLIT_CHILD_BOX_PAD_PX),
                                image_w=int(roi_w),
                                image_h=int(roi_h),
                            )

                            if not valid_box:
                                child_valid = False
                                child_invalid_reason = (
                                    f"child_{child_rank}_invalid_box"
                                )
                                break

                            child_orig_idx = int(next_child_orig_idx)
                            next_child_orig_idx += 1

                            child_row = det_row.copy()
                            child_row["orig_det_idx"] = child_orig_idx
                            child_row["x1"] = float(cx1)
                            child_row["y1"] = float(cy1)
                            child_row["x2"] = float(cx2)
                            child_row["y2"] = float(cy2)
                            child_row["box_w"] = float(cx2 - cx1)
                            child_row["box_h"] = float(cy2 - cy1)
                            child_row["has_mask"] = True

                            child_row["is_x_like"] = True
                            child_row["xsplit_attempted"] = True
                            child_row["xsplit_applied"] = True
                            child_row["xsplit_reason"] = (
                                "single_box_xsplit_success"
                            )
                            child_row["xsplit_source"] = str(
                                angle_result.get("source", "")
                            )

                            child_row["skip_axis_cleanup"] = True
                            child_row["parent_orig_det_idx"] = orig_idx
                            child_row["xsplit_child_rank"] = int(child_rank)

                            child_row["review_reason"] = _append_reason(
                                child_row.get("review_reason", ""),
                                "single_box_xsplit_child",
                            )

                            child_row["xsplit_child_angle_deg"] = float(
                                child_models[child_rank - 1].get(
                                    "angle_deg",
                                    np.nan,
                                )
                            )

                            child_row["xsplit_parent_pixels"] = int(
                                split_result.get("parent_pixels", 0)
                            )

                            child_row["xsplit_child_pixels"] = int(
                                child_mask.sum()
                            )

                            child_row["xsplit_child_balance_ratio"] = float(
                                split_result.get(
                                    "child_balance_ratio",
                                    np.nan,
                                )
                            )

                            crossarm_mask_lookup[child_orig_idx] = (
                                child_mask.astype(bool)
                            )

                            child_rows_for_parent.append(child_row)

                            single_xsplit_child_rows.append(
                                {
                                    "parent_orig_det_idx": int(orig_idx),
                                    "child_orig_det_idx": int(child_orig_idx),
                                    "xsplit_child_rank": int(child_rank),
                                    "xsplit_child_pixels": int(child_mask.sum()),
                                    "xsplit_child_angle_deg": float(
                                        child_models[child_rank - 1].get(
                                            "angle_deg",
                                            np.nan,
                                        )
                                    ),
                                    "xsplit_source": str(
                                        angle_result.get("source", "")
                                    ),
                                }
                            )

                        if child_valid and len(child_rows_for_parent) == 2:
                            output_rows.extend(child_rows_for_parent)
                            single_xsplit_applied_count += 1
                            split_applied = True

                            parent_trace_row = row_out.copy()
                            parent_trace_row["xsplit_applied"] = True
                            parent_trace_row["xsplit_reason"] = (
                                "single_box_xsplit_parent_replaced"
                            )
                            parent_trace_row["removal_reason"] = (
                                "replaced_by_single_box_xsplit_children"
                            )
                            single_xsplit_parent_replaced_rows.append(
                                parent_trace_row
                            )

                        else:
                            row_out["xsplit_applied"] = False
                            row_out["xsplit_reason"] = (
                                child_invalid_reason
                                or "split_child_validation_failed"
                            )
                            row_out["skip_axis_cleanup"] = True
                            row_out["review_reason"] = _append_reason(
                                row_out.get("review_reason", ""),
                                "x_like_split_failed",
                            )

                            output_rows.append(row_out)
                            single_xsplit_failed_count += 1

                    else:
                        row_out["xsplit_applied"] = False
                        row_out["xsplit_reason"] = str(
                            split_result.get("reason", "split_failed")
                        )
                        row_out["skip_axis_cleanup"] = True
                        row_out["review_reason"] = _append_reason(
                            row_out.get("review_reason", ""),
                            "x_like_split_failed",
                        )

                        output_rows.append(row_out)
                        single_xsplit_failed_count += 1

                else:
                    row_out["is_x_like"] = False
                    row_out["xsplit_attempted"] = False
                    row_out["xsplit_applied"] = False
                    row_out["xsplit_reason"] = str(
                        angle_result.get("reason", "not_x_like")
                    )
                    row_out["skip_axis_cleanup"] = bool(
                        row_out.get("skip_axis_cleanup", False)
                    )

                    output_rows.append(row_out)

                single_xsplit_debug_rows.append(
                    {
                        "orig_det_idx": int(orig_idx),
                        "has_usable_mask": bool(has_usable_mask),
                        "shape_suspicious": bool(shape_suspicious),
                        "pca_not_single_clean_axis": bool(
                            pca_not_single_clean_axis
                        ),
                        "should_test_for_x": bool(should_test_for_x),
                        "is_x_like": bool(x_like),
                        "split_applied": bool(split_applied),
                        "split_reason": str(
                            split_result.get("reason", "not_run")
                        ),
                        "hough_source": str(
                            angle_result.get("source", "none")
                        ),
                        "hough_reason": str(
                            angle_result.get("reason", "not_run")
                        ),
                        "hough_num_lines": int(
                            angle_result.get("num_lines", 0)
                        ),
                        "angle_a": float(
                            angle_result.get("angle_a", np.nan)
                        ),
                        "angle_b": float(
                            angle_result.get("angle_b", np.nan)
                        ),
                        "angle_diff": float(
                            angle_result.get("angle_diff", np.nan)
                        ),
                        "mask_pixels": int(pca_sig.get("num_pixels", 0)),
                        "pc1_ratio": float(
                            pca_sig.get("pc1_ratio", np.nan)
                        ),
                        "box_w": float(det_row.get("box_w", np.nan)),
                        "box_h": float(det_row.get("box_h", np.nan)),
                        "box_aspect_w_over_h": float(
                            det_row.get(
                                "box_aspect_w_over_h",
                                np.nan,
                            )
                        ),
                        "height_to_median": float(
                            det_row.get(
                                "xsplit_height_to_median",
                                np.nan,
                            )
                        ),
                        "area_to_median": float(
                            det_row.get(
                                "xsplit_area_to_median",
                                np.nan,
                            )
                        ),
                    }
                )

            if len(output_rows) > 0:
                final_kept_detections_df = (
                    pd.DataFrame(output_rows)
                    .reset_index(drop=True)
                )
            else:
                final_kept_detections_df = tmp_df.iloc[0:0].copy()

        else:
            final_kept_detections_df = (
                stage_single_xsplit_input_df
                .copy()
                .reset_index(drop=True)
            )

            for col, default_value in [
                ("is_x_like", False),
                ("xsplit_attempted", False),
                ("xsplit_applied", False),
                ("xsplit_reason", "single_xsplit_disabled_or_no_detections"),
                ("xsplit_source", ""),
                ("skip_axis_cleanup", False),
                ("review_reason", ""),
                ("parent_orig_det_idx", np.nan),
                ("xsplit_child_rank", np.nan),
            ]:
                if col not in final_kept_detections_df.columns:
                    final_kept_detections_df[col] = default_value

        # ---------------------------------------------------------------------
        # 16.30.4 Save X-split diagnostics and recompute width fields
        # ---------------------------------------------------------------------
        single_xsplit_debug_df = pd.DataFrame(single_xsplit_debug_rows)
        single_xsplit_child_df = pd.DataFrame(single_xsplit_child_rows)

        final_kept_detections_df = (
            final_kept_detections_df
            .copy()
            .reset_index(drop=True)
        )

        stage_single_xsplit_df = (
            final_kept_detections_df
            .copy()
            .reset_index(drop=True)
        )

        if len(final_kept_detections_df) > 0:
            final_kept_detections_df["box_w"] = (
                final_kept_detections_df["x2"].astype(float)
                - final_kept_detections_df["x1"].astype(float)
            ).clip(lower=0.0)

            final_kept_detections_df["box_h"] = (
                final_kept_detections_df["y2"].astype(float)
                - final_kept_detections_df["y1"].astype(float)
            ).clip(lower=0.0)

            max_box_w_after_xsplit = float(
                final_kept_detections_df["box_w"].max()
            )

            final_kept_detections_df["relative_width_to_max"] = (
                final_kept_detections_df["box_w"]
                / max(max_box_w_after_xsplit, 1.0)
            )

            stage_single_xsplit_df = (
                final_kept_detections_df
                .copy()
                .reset_index(drop=True)
            )

        # ---------------------------------------------------------------------
        # 16.30.5 Add replaced parent detections to trace output
        # ---------------------------------------------------------------------
        if len(single_xsplit_parent_replaced_rows) > 0:
            for parent_row in single_xsplit_parent_replaced_rows:
                crossarm_trace_rows.append(
                    {
                        "run_id": RUN_ID,
                        "run_timestamp": RUN_TIMESTAMP,
                        "image_id": image_id,
                        "file_name": file_name,
                        "roi_file_name": roi_file_name,
                        "roi_image_path": roi_image_path,
                        "processing_order": roi_row.get(
                            "processing_order",
                            np.nan,
                        ),

                        "stage_order": 7,
                        "stage_name": "single_box_xsplit",

                        "orig_det_idx": int(parent_row["orig_det_idx"]),
                        "score": float(parent_row["score"]),
                        "x1": float(parent_row["x1"]),
                        "y1": float(parent_row["y1"]),
                        "x2": float(parent_row["x2"]),
                        "y2": float(parent_row["y2"]),
                        "has_mask": bool(parent_row.get("has_mask", False)),

                        "is_x_like": parse_bool(
                            parent_row.get("is_x_like", True)
                        ),
                        "xsplit_attempted": parse_bool(
                            parent_row.get("xsplit_attempted", True)
                        ),
                        "xsplit_applied": parse_bool(
                            parent_row.get("xsplit_applied", True)
                        ),
                        "xsplit_reason": str(
                            parent_row.get(
                                "xsplit_reason",
                                "single_box_xsplit_parent_replaced",
                            )
                        ),
                        "xsplit_source": str(
                            parent_row.get("xsplit_source", "")
                        ),

                        "action": "removed",
                        "removal_reason": str(
                            parent_row.get(
                                "removal_reason",
                                "replaced_by_single_box_xsplit_children",
                            )
                        ),
                        "removed_by_orig_det_idx": np.nan,
                        "review_reason": str(
                            parent_row.get("review_reason", "")
                        ),
                    }
                )

        # ---------------------------------------------------------------------
        # 16.30.6 Save stage summary
        # ---------------------------------------------------------------------
        crossarm_stage_summary_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "image_id": image_id,
                "file_name": file_name,
                "roi_file_name": roi_file_name,
                "roi_image_path": roi_image_path,
                "processing_order": roi_row.get("processing_order", np.nan),

                "stage_order": 7,
                "stage_name": "single_box_xsplit",
                "input_count": int(len(stage_single_xsplit_input_df)),
                "output_count": int(len(stage_single_xsplit_df)),
                "removed_count": int(len(single_xsplit_parent_replaced_rows)),

                "single_xsplit_enabled": bool(SINGLE_XSPLIT_ENABLED),
                "single_xsplit_xlike_count": int(single_xsplit_xlike_count),
                "single_xsplit_applied_count": int(single_xsplit_applied_count),
                "single_xsplit_failed_count": int(single_xsplit_failed_count),
                "single_xsplit_child_count": int(len(single_xsplit_child_df)),
                "single_xsplit_debug_count": int(len(single_xsplit_debug_df)),

                "xsplit_height_to_median_trigger": float(
                    XSPLIT_HEIGHT_TO_MEDIAN_TRIGGER
                ),
                "xsplit_area_to_median_trigger": float(
                    XSPLIT_AREA_TO_MEDIAN_TRIGGER
                ),
                "xsplit_max_aspect_for_suspicious": float(
                    XSPLIT_MAX_ASPECT_FOR_SUSPICIOUS
                ),
                "xsplit_pc1_ratio_max_for_xlike": float(
                    XSPLIT_PC1_RATIO_MAX_FOR_XLIKE
                ),
                "xsplit_min_parent_mask_pixels": int(
                    XSPLIT_MIN_PARENT_MASK_PIXELS
                ),
                "xsplit_child_box_pad_px": int(XSPLIT_CHILD_BOX_PAD_PX),

                "projected_pole_mask_available": bool(
                    projected_pole_mask_available
                ),
                "pole_mask_filter_applied": bool(pole_mask_filter_applied),
            }
        )
        
        # =============================================================================
        # 16.31 PCA AXIS CLEANUP + TIGHT BOX REBUILD, NON-X ONLY
        # =============================================================================
        # EXPLANATION:
        # This stage cleans a single broad / messy crossarm mask by fitting ONE
        # dominant PCA axis, trimming pixels far away from that axis, and
        # rebuilding a tighter bounding box.
        #
        # IMPORTANT SAFETY GUARD:
        #   If a detection was tagged as is_x_like=True or
        #   skip_axis_cleanup=True, this section leaves it unchanged.
        #
        #   This prevents PCA cleanup from damaging an X-shaped crossarm mask
        #   that has two valid directions.
        #
        # IMPORTANT:
        #   This section does not create new detections.
        #
        #   If cleanup succeeds, it updates crossarm_mask_lookup in place using
        #   the same orig_det_idx.
        #
        # PRODUCTION RULE:
        #   No display().
        #   No _safe_display().
        #   No per-ROI print().
        # =============================================================================

        current_stage = "16.31_axis_cleanup_non_x_only"

        # ---------------------------------------------------------------------
        # 16.31.1 Prepare input snapshot and preserve pre-cleanup masks
        # ---------------------------------------------------------------------
        stage_axis_cleanup_input_df = (
            final_kept_detections_df
            .copy()
            .reset_index(drop=True)
        )

        axis_cleanup_input_mask_lookup = {
            int(k): (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in crossarm_mask_lookup.items()
        }

        axis_cleanup_debug_rows = []
        axis_cleanup_applied_count = 0
        axis_cleanup_skipped_x_count = 0
        stage_axis_cleanup_candidate_rows = []

        # ---------------------------------------------------------------------
        # 16.31.2 Run PCA cleanup only for non-X detections
        # ---------------------------------------------------------------------
        if (
            bool(AXIS_CLEANUP_ENABLED)
            and len(stage_axis_cleanup_input_df) > 0
        ):
            tmp_df = (
                stage_axis_cleanup_input_df
                .copy()
                .reset_index(drop=True)
            )

            tmp_df["box_w"] = (
                tmp_df["x2"].astype(float)
                - tmp_df["x1"].astype(float)
            ).clip(lower=0.0)

            tmp_df["box_h"] = (
                tmp_df["y2"].astype(float)
                - tmp_df["y1"].astype(float)
            ).clip(lower=0.0)

            # Median height calculation excludes X-like detections so a
            # successful split does not inflate the median and cause real long
            # crossarms to be incorrectly flagged as suspicious.
            non_x_flags = (
                tmp_df.get(
                    "is_x_like",
                    pd.Series(
                        [False] * len(tmp_df),
                        index=tmp_df.index,
                    ),
                )
                .fillna(False)
                .astype(bool)
                .to_numpy()
            )

            non_x_heights = tmp_df.loc[
                (tmp_df["box_h"] > 0) & (~non_x_flags),
                "box_h",
            ]

            if len(non_x_heights) > 0:
                median_box_h = float(non_x_heights.median())

            else:
                # Safety fallback:
                # If every surviving detection is X-like, fall back to all
                # positive heights so this stage still has a non-zero median
                # reference.
                positive_heights = tmp_df.loc[
                    tmp_df["box_h"] > 0,
                    "box_h",
                ]

                median_box_h = (
                    float(positive_heights.median())
                    if len(positive_heights) > 0
                    else 0.0
                )

            tmp_df["axis_height_to_median"] = (
                tmp_df["box_h"] / max(median_box_h, 1.0)
                if median_box_h > 0
                else 0.0
            )

            cleaned_rows = []

            for row_idx, det_row in tmp_df.iterrows():
                row_out = det_row.copy()

                orig_idx = int(det_row["orig_det_idx"])
                mask_i = crossarm_mask_lookup.get(orig_idx, None)

                row_is_x_like = parse_bool(
                    det_row.get("is_x_like", False)
                )

                row_skip_axis_cleanup = parse_bool(
                    det_row.get("skip_axis_cleanup", False)
                )

                # -------------------------------------------------------------
                # 16.31.2A Skip X-like / guarded detections
                # -------------------------------------------------------------
                if row_is_x_like or row_skip_axis_cleanup:
                    row_out["axis_cleanup_candidate"] = False
                    row_out["axis_cleanup_applied"] = False
                    row_out["axis_cleanup_reason"] = (
                        "skipped_x_like_or_guarded"
                    )
                    row_out["skip_axis_cleanup"] = True

                    row_out["review_reason"] = _append_reason(
                        row_out.get("review_reason", ""),
                        "axis_cleanup_skipped_x_like",
                    )

                    cleaned_rows.append(row_out)
                    axis_cleanup_skipped_x_count += 1

                    axis_cleanup_debug_rows.append(
                        {
                            "orig_det_idx": int(orig_idx),
                            "axis_cleanup_candidate": False,
                            "axis_cleanup_applied": False,
                            "axis_cleanup_reason": (
                                "skipped_x_like_or_guarded"
                            ),
                            "is_x_like": bool(row_is_x_like),
                            "skip_axis_cleanup": bool(row_skip_axis_cleanup),
                        }
                    )

                    continue

                # -------------------------------------------------------------
                # 16.31.2B Fit PCA axis and decide whether cleanup is needed
                # -------------------------------------------------------------
                model_axis = _fit_axis_cleanup_model(mask_i)

                trigger_height = bool(
                    float(det_row.get("axis_height_to_median", 0.0))
                    >= float(AXIS_CLEANUP_HEIGHT_TO_MEDIAN_TRIGGER)
                )

                trigger_perp = bool(
                    model_axis.get("valid", False)
                    and float(model_axis.get("perp_std", 0.0))
                    >= float(AXIS_CLEANUP_PERP_STD_TRIGGER_PX)
                )

                max_overlap_smaller = 0.0

                for other_idx, other_row in tmp_df.iterrows():
                    if int(other_idx) == int(row_idx):
                        continue

                    try:
                        max_overlap_smaller = max(
                            max_overlap_smaller,
                            _box_overlap_fraction_of_smaller(
                                det_row,
                                other_row,
                            ),
                        )
                    except Exception:
                        pass

                trigger_overlap = bool(
                    max_overlap_smaller
                    >= float(AXIS_CLEANUP_BOX_OVERLAP_FRAC_TRIGGER)
                )

                axis_candidate = bool(
                    model_axis.get("valid", False)
                    and (trigger_height or trigger_perp or trigger_overlap)
                )

                row_out["axis_cleanup_candidate"] = axis_candidate
                row_out["axis_cleanup_applied"] = False
                row_out["axis_cleanup_reason"] = "not_candidate"

                row_out["axis_angle_deg"] = float(
                    model_axis.get("angle_deg", np.nan)
                )
                row_out["axis_pc1_ratio"] = float(
                    model_axis.get("pc1_ratio", np.nan)
                )
                row_out["axis_perp_std"] = float(
                    model_axis.get("perp_std", np.nan)
                )
                row_out["axis_max_overlap_smaller"] = float(
                    max_overlap_smaller
                )

                # -------------------------------------------------------------
                # 16.31.2C Apply cleanup only if candidate passes safety checks
                # -------------------------------------------------------------
                if axis_candidate:
                    stage_axis_cleanup_candidate_rows.append(row_out.copy())

                    half_width_px = float(
                        np.clip(
                            float(model_axis.get("perp_median_abs", 0.0))
                            + 2.0 * float(
                                model_axis.get("perp_mad_abs", 0.0)
                            )
                            + float(AXIS_CLEANUP_HALF_WIDTH_EXTRA_PX),
                            float(AXIS_CLEANUP_MIN_HALF_WIDTH_PX),
                            float(AXIS_CLEANUP_MAX_HALF_WIDTH_PX),
                        )
                    )

                    original_pixels = (
                        int(mask_i.sum())
                        if isinstance(mask_i, np.ndarray)
                        else 0
                    )

                    cleaned_mask = _axis_cleanup_mask(
                        mask_i,
                        model_axis,
                        half_width_px,
                    )

                    cleaned_pixels = int(cleaned_mask.sum())
                    retained_frac = float(
                        cleaned_pixels / max(original_pixels, 1)
                    )

                    valid_box, nx1, ny1, nx2, ny2 = _bbox_from_mask(
                        cleaned_mask,
                        pad_px=int(AXIS_CLEANUP_BOX_PAD_PX),
                        image_w=int(roi_w),
                        image_h=int(roi_h),
                    )

                    cleanup_passed_safety = bool(
                        valid_box
                        and cleaned_pixels >= int(AXIS_CLEANUP_MIN_MASK_PIXELS)
                        and retained_frac
                        >= float(AXIS_CLEANUP_MIN_RETAINED_FRAC)
                    )

                    if cleanup_passed_safety:
                        # Update the existing mask in place under the same
                        # orig_det_idx.
                        crossarm_mask_lookup[orig_idx] = (
                            cleaned_mask.astype(bool)
                        )

                        row_out["x1"] = float(nx1)
                        row_out["y1"] = float(ny1)
                        row_out["x2"] = float(nx2)
                        row_out["y2"] = float(ny2)

                        row_out["box_w"] = float(nx2 - nx1)
                        row_out["box_h"] = float(ny2 - ny1)

                        row_out["axis_cleanup_applied"] = True
                        row_out["axis_cleanup_reason"] = (
                            "axis_cleanup_success"
                        )
                        row_out["axis_cleanup_half_width_px"] = half_width_px
                        row_out["axis_cleanup_original_pixels"] = int(
                            original_pixels
                        )
                        row_out["axis_cleanup_cleaned_pixels"] = int(
                            cleaned_pixels
                        )
                        row_out["axis_cleanup_retained_frac"] = float(
                            retained_frac
                        )

                        axis_cleanup_applied_count += 1

                    else:
                        # Safety failed:
                        # Keep the original mask and box unchanged.
                        row_out["axis_cleanup_applied"] = False
                        row_out["axis_cleanup_reason"] = (
                            "cleanup_failed_safety_check"
                        )
                        row_out["axis_cleanup_half_width_px"] = half_width_px
                        row_out["axis_cleanup_original_pixels"] = int(
                            original_pixels
                        )
                        row_out["axis_cleanup_cleaned_pixels"] = int(
                            cleaned_pixels
                        )
                        row_out["axis_cleanup_retained_frac"] = float(
                            retained_frac
                        )

                        row_out["review_reason"] = _append_reason(
                            row_out.get("review_reason", ""),
                            "axis_cleanup_failed",
                        )

                cleaned_rows.append(row_out)

                # -------------------------------------------------------------
                # 16.31.2D Save per-row cleanup diagnostics
                # -------------------------------------------------------------
                axis_cleanup_debug_rows.append(
                    {
                        "orig_det_idx": int(orig_idx),
                        "axis_cleanup_candidate": bool(axis_candidate),
                        "axis_cleanup_applied": bool(
                            row_out.get("axis_cleanup_applied", False)
                        ),
                        "axis_cleanup_reason": str(
                            row_out.get("axis_cleanup_reason", "")
                        ),
                        "trigger_height": bool(trigger_height),
                        "trigger_perp": bool(trigger_perp),
                        "trigger_overlap": bool(trigger_overlap),
                        "height_to_median": float(
                            det_row.get("axis_height_to_median", np.nan)
                        ),
                        "perp_std": float(
                            model_axis.get("perp_std", np.nan)
                        ),
                        "pc1_ratio": float(
                            model_axis.get("pc1_ratio", np.nan)
                        ),
                        "max_overlap_smaller": float(max_overlap_smaller),
                    }
                )

            final_kept_detections_df = (
                pd.DataFrame(cleaned_rows)
                .reset_index(drop=True)
            )

        else:
            final_kept_detections_df = (
                stage_axis_cleanup_input_df
                .copy()
                .reset_index(drop=True)
            )

            if "axis_cleanup_candidate" not in final_kept_detections_df.columns:
                final_kept_detections_df["axis_cleanup_candidate"] = False

            if "axis_cleanup_applied" not in final_kept_detections_df.columns:
                final_kept_detections_df["axis_cleanup_applied"] = False

            if "axis_cleanup_reason" not in final_kept_detections_df.columns:
                final_kept_detections_df["axis_cleanup_reason"] = (
                    "axis_cleanup_disabled_or_no_detections"
                )

        # ---------------------------------------------------------------------
        # 16.31.3 Save cleanup diagnostics and stage snapshots
        # ---------------------------------------------------------------------
        axis_cleanup_debug_df = pd.DataFrame(axis_cleanup_debug_rows)

        stage_axis_cleanup_candidate_df = (
            pd.DataFrame(stage_axis_cleanup_candidate_rows)
            .reset_index(drop=True)
            if len(stage_axis_cleanup_candidate_rows) > 0
            else final_kept_detections_df.iloc[0:0].copy()
        )

        final_kept_detections_df = (
            final_kept_detections_df
            .copy()
            .reset_index(drop=True)
        )

        stage_axis_cleanup_df = (
            final_kept_detections_df
            .copy()
            .reset_index(drop=True)
        )

        # ---------------------------------------------------------------------
        # 16.31.4 Recompute relative width after cleanup
        # ---------------------------------------------------------------------
        if len(final_kept_detections_df) > 0:
            final_kept_detections_df["box_w"] = (
                final_kept_detections_df["x2"].astype(float)
                - final_kept_detections_df["x1"].astype(float)
            ).clip(lower=0.0)

            final_kept_detections_df["box_h"] = (
                final_kept_detections_df["y2"].astype(float)
                - final_kept_detections_df["y1"].astype(float)
            ).clip(lower=0.0)

            max_box_w_after_axis = float(
                final_kept_detections_df["box_w"].max()
            )

            final_kept_detections_df["relative_width_to_max"] = (
                final_kept_detections_df["box_w"]
                / max(max_box_w_after_axis, 1.0)
            )

            stage_axis_cleanup_df = (
                final_kept_detections_df
                .copy()
                .reset_index(drop=True)
            )

        # ---------------------------------------------------------------------
        # 16.31.5 Save stage summary
        # ---------------------------------------------------------------------
        crossarm_stage_summary_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "image_id": image_id,
                "file_name": file_name,
                "roi_file_name": roi_file_name,
                "roi_image_path": roi_image_path,
                "processing_order": roi_row.get("processing_order", np.nan),

                "stage_order": 8,
                "stage_name": "axis_cleanup_non_x_only",
                "input_count": int(len(stage_axis_cleanup_input_df)),
                "output_count": int(len(stage_axis_cleanup_df)),
                "removed_count": 0,

                "axis_cleanup_enabled": bool(AXIS_CLEANUP_ENABLED),
                "axis_cleanup_skipped_x_count": int(
                    axis_cleanup_skipped_x_count
                ),
                "axis_cleanup_candidate_count": int(
                    len(stage_axis_cleanup_candidate_df)
                ),
                "axis_cleanup_applied_count": int(
                    axis_cleanup_applied_count
                ),
                "axis_cleanup_debug_count": int(len(axis_cleanup_debug_df)),

                "axis_cleanup_height_to_median_trigger": float(
                    AXIS_CLEANUP_HEIGHT_TO_MEDIAN_TRIGGER
                ),
                "axis_cleanup_box_overlap_frac_trigger": float(
                    AXIS_CLEANUP_BOX_OVERLAP_FRAC_TRIGGER
                ),
                "axis_cleanup_perp_std_trigger_px": float(
                    AXIS_CLEANUP_PERP_STD_TRIGGER_PX
                ),
                "axis_cleanup_min_mask_pixels": int(
                    AXIS_CLEANUP_MIN_MASK_PIXELS
                ),
                "axis_cleanup_min_retained_frac": float(
                    AXIS_CLEANUP_MIN_RETAINED_FRAC
                ),
                "axis_cleanup_box_pad_px": int(AXIS_CLEANUP_BOX_PAD_PX),

                "projected_pole_mask_available": bool(
                    projected_pole_mask_available
                ),
                "pole_mask_filter_applied": bool(pole_mask_filter_applied),
            }
        )
        
        
        # =============================================================================
        # 16.32 TARGETED TWO-BOX X OWNERSHIP, ONLY IF NEEDED
        # =============================================================================
        # EXPLANATION:
        # This stage handles the older X-crossarm failure mode where SAM3 already
        # produced TWO detections, but their masks overlap at the X intersection.
        #
        # It is deliberately conservative:
        #   1) requires two usable masks
        #   2) requires shared mask pixels
        #   3) requires different PCA directions
        #   4) only reassigns shared pixels
        #   5) keeps both detections only if both masks pass safety checks
        #
        # IMPORTANT:
        #   This section does not create new detections.
        #
        #   It modifies affected masks in place under the same orig_det_idx keys.
        #
        # PRODUCTION RULE:
        #   No display().
        #   No _safe_display().
        #   No per-ROI print().
        # =============================================================================

        current_stage = "16.32_targeted_two_box_xownership"

        # ---------------------------------------------------------------------
        # 16.32.1 Prepare input snapshot and preserve pre-ownership masks
        # ---------------------------------------------------------------------
        stage_xownership_input_df = (
            final_kept_detections_df
            .copy()
            .reset_index(drop=True)
        )

        xownership_input_mask_lookup = {
            int(k): (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in crossarm_mask_lookup.items()
        }

        xownership_pair_debug_rows = []
        xownership_detection_debug_rows = []
        xownership_candidate_orig_idxs = set()
        xownership_applied_pairs = 0
        affected_orig_idxs = set()

        # ---------------------------------------------------------------------
        # 16.32.2 Run targeted two-box X ownership
        # ---------------------------------------------------------------------
        if (
            bool(TWO_BOX_XOWNERSHIP_ENABLED)
            and len(stage_xownership_input_df) >= 2
        ):
            tmp_df = (
                stage_xownership_input_df
                .copy()
                .reset_index(drop=True)
            )

            # Keep this boolean column clean for all rows, including unaffected
            # rows.
            if "xownership_applied" not in tmp_df.columns:
                tmp_df["xownership_applied"] = False
            else:
                tmp_df["xownership_applied"] = (
                    tmp_df["xownership_applied"]
                    .fillna(False)
                    .apply(parse_bool)
                )

            # Ensure review_reason exists before appending ownership notes.
            if "review_reason" not in tmp_df.columns:
                tmp_df["review_reason"] = ""

            # -----------------------------------------------------------------
            # 16.32.2A Fit one PCA line model per detection
            # -----------------------------------------------------------------
            line_models = {}
            mask_sizes = {}

            for _, det_row in tmp_df.iterrows():
                orig_idx = int(det_row["orig_det_idx"])
                mask_i = crossarm_mask_lookup.get(orig_idx, None)
                model_i = _fit_axis_cleanup_model(mask_i)

                line_models[orig_idx] = model_i
                mask_sizes[orig_idx] = (
                    int(mask_i.sum())
                    if isinstance(mask_i, np.ndarray)
                    else 0
                )

            # -----------------------------------------------------------------
            # 16.32.2B Build candidate pairs sorted by shared-pixel strength
            # -----------------------------------------------------------------
            candidate_pairs = []

            for i in range(len(tmp_df)):
                row_i = tmp_df.iloc[i]
                orig_i = int(row_i["orig_det_idx"])
                mask_i = crossarm_mask_lookup.get(orig_i, None)
                model_i = line_models.get(orig_i, {})

                if (
                    not isinstance(mask_i, np.ndarray)
                    or mask_i.ndim != 2
                    or int(mask_i.sum()) == 0
                ):
                    continue

                if not bool(model_i.get("valid", False)):
                    continue

                for j in range(i + 1, len(tmp_df)):
                    row_j = tmp_df.iloc[j]
                    orig_j = int(row_j["orig_det_idx"])
                    mask_j = crossarm_mask_lookup.get(orig_j, None)
                    model_j = line_models.get(orig_j, {})

                    # Do not run two-box ownership on children created by the
                    # single-box X-split stage. They have already been
                    # geometrically separated.
                    if (
                        parse_bool(row_i.get("xsplit_applied", False))
                        or parse_bool(row_j.get("xsplit_applied", False))
                    ):
                        xownership_pair_debug_rows.append(
                            {
                                "orig_i": int(orig_i),
                                "orig_j": int(orig_j),
                                "shared_pixels": np.nan,
                                "shared_frac_smaller": np.nan,
                                "angle_i": float(
                                    model_i.get("angle_deg", np.nan)
                                ),
                                "angle_j": float(
                                    model_j.get("angle_deg", np.nan)
                                ),
                                "angle_diff": np.nan,
                                "pair_candidate": False,
                                "skip_reason": "single_box_xsplit_child",
                            }
                        )
                        continue

                    if (
                        not isinstance(mask_j, np.ndarray)
                        or mask_j.ndim != 2
                        or int(mask_j.sum()) == 0
                    ):
                        continue

                    if mask_i.shape != mask_j.shape:
                        continue

                    if not bool(model_j.get("valid", False)):
                        continue

                    shared = mask_i.astype(bool) & mask_j.astype(bool)

                    shared_pixels = int(shared.sum())
                    smaller_pixels = max(
                        1,
                        min(
                            int(mask_i.sum()),
                            int(mask_j.sum()),
                        ),
                    )
                    shared_frac_smaller = float(
                        shared_pixels / smaller_pixels
                    )

                    angle_diff = _angle_diff_undirected_deg(
                        model_i.get("angle_deg", np.nan),
                        model_j.get("angle_deg", np.nan),
                    )

                    pair_candidate = bool(
                        shared_pixels >= int(XOWN_MIN_SHARED_PIXELS)
                        and shared_frac_smaller
                        >= float(XOWN_MIN_SHARED_FRAC_OF_SMALLER)
                        and angle_diff >= float(XOWN_MIN_ANGLE_DIFF_DEG)
                    )

                    xownership_pair_debug_rows.append(
                        {
                            "orig_i": int(orig_i),
                            "orig_j": int(orig_j),
                            "shared_pixels": int(shared_pixels),
                            "shared_frac_smaller": float(
                                shared_frac_smaller
                            ),
                            "angle_i": float(
                                model_i.get("angle_deg", np.nan)
                            ),
                            "angle_j": float(
                                model_j.get("angle_deg", np.nan)
                            ),
                            "angle_diff": float(angle_diff),
                            "pair_candidate": bool(pair_candidate),
                        }
                    )

                    if pair_candidate:
                        candidate_pairs.append(
                            {
                                "orig_i": int(orig_i),
                                "orig_j": int(orig_j),
                                "shared_pixels": int(shared_pixels),
                                "shared_frac_smaller": float(
                                    shared_frac_smaller
                                ),
                                "angle_diff": float(angle_diff),
                            }
                        )

            candidate_pairs = sorted(
                candidate_pairs,
                key=lambda x: (
                    x["shared_pixels"],
                    x["shared_frac_smaller"],
                ),
                reverse=True,
            )

            # -----------------------------------------------------------------
            # 16.32.2C Apply conservative one-pair-per-detection ownership
            # reassignment
            # -----------------------------------------------------------------
            already_used = set()

            for pair in candidate_pairs:
                orig_i = int(pair["orig_i"])
                orig_j = int(pair["orig_j"])

                if orig_i in already_used or orig_j in already_used:
                    continue

                mask_i = crossarm_mask_lookup.get(orig_i, None)
                mask_j = crossarm_mask_lookup.get(orig_j, None)

                model_i = line_models.get(orig_i, {})
                model_j = line_models.get(orig_j, {})

                if (
                    not isinstance(mask_i, np.ndarray)
                    or not isinstance(mask_j, np.ndarray)
                ):
                    continue

                shared = mask_i.astype(bool) & mask_j.astype(bool)
                ys, xs = np.where(shared)

                if len(xs) < int(XOWN_MIN_SHARED_PIXELS):
                    continue

                dist_i = _line_distance_for_points(
                    xs,
                    ys,
                    model_i,
                )

                dist_j = _line_distance_for_points(
                    xs,
                    ys,
                    model_j,
                )

                assign_i = dist_i <= dist_j
                assign_j = ~assign_i

                new_mask_i = mask_i.astype(bool).copy()
                new_mask_j = mask_j.astype(bool).copy()

                # Remove all shared pixels from both masks.
                new_mask_i[ys, xs] = False
                new_mask_j[ys, xs] = False

                # Add shared pixels back to the closest owner.
                new_mask_i[ys[assign_i], xs[assign_i]] = True
                new_mask_j[ys[assign_j], xs[assign_j]] = True

                old_i_pixels = int(mask_i.sum())
                old_j_pixels = int(mask_j.sum())

                new_i_pixels = int(new_mask_i.sum())
                new_j_pixels = int(new_mask_j.sum())

                valid_i = bool(
                    new_i_pixels >= int(XOWN_MIN_CHILD_PIXELS_AFTER)
                    and (
                        new_i_pixels / max(old_i_pixels, 1)
                    ) >= float(XOWN_MIN_RETAINED_FRAC_AFTER)
                )

                valid_j = bool(
                    new_j_pixels >= int(XOWN_MIN_CHILD_PIXELS_AFTER)
                    and (
                        new_j_pixels / max(old_j_pixels, 1)
                    ) >= float(XOWN_MIN_RETAINED_FRAC_AFTER)
                )

                if not (valid_i and valid_j):
                    xownership_detection_debug_rows.append(
                        {
                            "orig_i": int(orig_i),
                            "orig_j": int(orig_j),
                            "xownership_applied": False,
                            "reason": "safety_check_failed",
                            "old_i_pixels": int(old_i_pixels),
                            "old_j_pixels": int(old_j_pixels),
                            "new_i_pixels": int(new_i_pixels),
                            "new_j_pixels": int(new_j_pixels),
                        }
                    )
                    continue

                # Safety passed:
                # Update affected masks in place under the same orig_det_idx
                # keys.
                crossarm_mask_lookup[orig_i] = new_mask_i.astype(bool)
                crossarm_mask_lookup[orig_j] = new_mask_j.astype(bool)

                # -------------------------------------------------------------
                # 16.32.2D Rebuild boxes for affected detections
                # -------------------------------------------------------------
                for affected_orig, affected_mask in [
                    (orig_i, new_mask_i),
                    (orig_j, new_mask_j),
                ]:
                    valid_box, bx1, by1, bx2, by2 = _bbox_from_mask(
                        affected_mask,
                        pad_px=int(XOWN_BOX_PAD_PX),
                        image_w=int(roi_w),
                        image_h=int(roi_h),
                    )

                    if valid_box:
                        row_mask = (
                            tmp_df["orig_det_idx"].astype(int)
                            == int(affected_orig)
                        )

                        tmp_df.loc[row_mask, "x1"] = float(bx1)
                        tmp_df.loc[row_mask, "y1"] = float(by1)
                        tmp_df.loc[row_mask, "x2"] = float(bx2)
                        tmp_df.loc[row_mask, "y2"] = float(by2)

                        tmp_df.loc[row_mask, "box_w"] = float(bx2 - bx1)
                        tmp_df.loc[row_mask, "box_h"] = float(by2 - by1)

                        tmp_df.loc[row_mask, "xownership_applied"] = True

                        tmp_df.loc[row_mask, "review_reason"] = (
                            tmp_df.loc[row_mask, "review_reason"].apply(
                                lambda old_reason: _append_reason(
                                    old_reason,
                                    "two_box_xownership_applied",
                                )
                            )
                        )

                already_used.add(orig_i)
                already_used.add(orig_j)

                affected_orig_idxs.add(orig_i)
                affected_orig_idxs.add(orig_j)

                xownership_candidate_orig_idxs.add(orig_i)
                xownership_candidate_orig_idxs.add(orig_j)

                xownership_applied_pairs += 1

                xownership_detection_debug_rows.append(
                    {
                        "orig_i": int(orig_i),
                        "orig_j": int(orig_j),
                        "xownership_applied": True,
                        "reason": "ownership_success",
                        "old_i_pixels": int(old_i_pixels),
                        "old_j_pixels": int(old_j_pixels),
                        "new_i_pixels": int(new_i_pixels),
                        "new_j_pixels": int(new_j_pixels),
                    }
                )

            final_kept_detections_df = (
                tmp_df
                .copy()
                .reset_index(drop=True)
            )

        else:
            final_kept_detections_df = (
                stage_xownership_input_df
                .copy()
                .reset_index(drop=True)
            )

        # ---------------------------------------------------------------------
        # 16.32.3 Ensure output/debug columns exist
        # ---------------------------------------------------------------------
        if "xownership_applied" not in final_kept_detections_df.columns:
            final_kept_detections_df["xownership_applied"] = False
        else:
            final_kept_detections_df["xownership_applied"] = (
                final_kept_detections_df["xownership_applied"]
                .fillna(False)
                .apply(parse_bool)
            )

        # ---------------------------------------------------------------------
        # 16.32.4 Save ownership diagnostics and stage snapshots
        # ---------------------------------------------------------------------
        xownership_pair_debug_df = pd.DataFrame(xownership_pair_debug_rows)
        xownership_detection_debug_df = pd.DataFrame(
            xownership_detection_debug_rows
        )

        if (
            len(final_kept_detections_df) > 0
            and len(xownership_candidate_orig_idxs) > 0
            and "orig_det_idx" in final_kept_detections_df.columns
        ):
            xownership_candidate_df = (
                final_kept_detections_df[
                    final_kept_detections_df["orig_det_idx"]
                    .astype(int)
                    .isin(xownership_candidate_orig_idxs)
                ]
                .copy()
                .reset_index(drop=True)
            )

        else:
            xownership_candidate_df = (
                final_kept_detections_df
                .iloc[0:0]
                .copy()
                .reset_index(drop=True)
            )

        stage_xownership_df = (
            final_kept_detections_df
            .copy()
            .reset_index(drop=True)
        )

        candidate_pair_count = (
            int(
                len(
                    xownership_pair_debug_df[
                        xownership_pair_debug_df["pair_candidate"] == True
                    ]
                )
            )
            if (
                len(xownership_pair_debug_df) > 0
                and "pair_candidate" in xownership_pair_debug_df.columns
            )
            else 0
        )

        # ---------------------------------------------------------------------
        # 16.32.5 Save stage summary
        # ---------------------------------------------------------------------
        crossarm_stage_summary_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "image_id": image_id,
                "file_name": file_name,
                "roi_file_name": roi_file_name,
                "roi_image_path": roi_image_path,
                "processing_order": roi_row.get("processing_order", np.nan),

                "stage_order": 9,
                "stage_name": "targeted_two_box_xownership",
                "input_count": int(len(stage_xownership_input_df)),
                "output_count": int(len(stage_xownership_df)),
                "removed_count": 0,

                "two_box_xownership_enabled": bool(
                    TWO_BOX_XOWNERSHIP_ENABLED
                ),
                "candidate_pair_count": int(candidate_pair_count),
                "xownership_applied_pairs": int(xownership_applied_pairs),
                "xownership_candidate_detection_count": int(
                    len(xownership_candidate_df)
                ),
                "xownership_pair_debug_count": int(
                    len(xownership_pair_debug_df)
                ),
                "xownership_detection_debug_count": int(
                    len(xownership_detection_debug_df)
                ),

                "xown_min_shared_pixels": int(XOWN_MIN_SHARED_PIXELS),
                "xown_min_shared_frac_of_smaller": float(
                    XOWN_MIN_SHARED_FRAC_OF_SMALLER
                ),
                "xown_min_angle_diff_deg": float(XOWN_MIN_ANGLE_DIFF_DEG),
                "xown_min_child_pixels_after": int(
                    XOWN_MIN_CHILD_PIXELS_AFTER
                ),
                "xown_min_retained_frac_after": float(
                    XOWN_MIN_RETAINED_FRAC_AFTER
                ),
                "xown_box_pad_px": int(XOWN_BOX_PAD_PX),

                "projected_pole_mask_available": bool(
                    projected_pole_mask_available
                ),
                "pole_mask_filter_applied": bool(pole_mask_filter_applied),
            }
        )
        
        
        # =============================================================================
        # 16.33 FINAL DEDUPE + REVIEW FLAGS
        # =============================================================================
        # EXPLANATION:
        # This final stage starts turning the post-processing output for the
        # current ROI into clean final results.
        #
        # This section does three things:
        #   1) Runs one final lightweight dedupe pass.
        #   2) Adds stable review/debug schema columns.
        #   3) Adds an ROI-level review reason.
        #
        # IMPORTANT:
        #   - Do NOT overwrite orig_det_idx.
        #   - orig_det_idx remains the technical tracking ID used across the
        #     pipeline.
        #   - Final xarm labels are added in the next section, not here.
        #
        # PRODUCTION RULE:
        #   No display().
        #   No _safe_display().
        #   No per-ROI print().
        # =============================================================================

        current_stage = "16.33_final_dedupe_review_flags"

        # ---------------------------------------------------------------------
        # 16.33.1 Final dedupe
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # Run the containment suppressor one final time to catch duplicate /
        # contained boxes that may have survived previous geometry stages.
        #
        # This is wrapped in try/except because final output should remain
        # available even if the final dedupe pass fails.
        # ---------------------------------------------------------------------
        stage_final_dedupe_input_df = (
            final_kept_detections_df
            .copy()
            .reset_index(drop=True)
        )

        final_dedupe_failed = False
        final_dedupe_error = ""

        if (
            bool(FINAL_DEDUPE_ENABLED)
            and len(stage_final_dedupe_input_df) > 1
        ):
            try:
                (
                    final_detections_df,
                    removed_by_final_dedupe_df,
                    final_dedupe_pair_debug_df,
                ) = suppress_contained_shorter_detections(
                    detections_df=stage_final_dedupe_input_df,
                    containment_threshold=float(CONTAINMENT_THRESHOLD),
                    min_area_ratio=float(MIN_AREA_RATIO),
                    min_score_advantage=float(MIN_SCORE_ADVANTAGE),
                    crossarm_mask_lookup=crossarm_mask_lookup,
                    mask_containment_filter_enabled=bool(
                        MASK_CONTAINMENT_FILTER_ENABLED
                    ),
                    mask_containment_veto_threshold=float(
                        MASK_CONTAINMENT_VETO_THRESHOLD
                    ),
                    near_total_box_containment_threshold=float(
                        NEAR_TOTAL_BOX_CONTAINMENT_THRESHOLD
                    ),
                    mask_containment_high=float(MASK_CONTAINMENT_HIGH),
                    pair_debug_min_box_containment=float(
                        PAIR_DEBUG_MIN_BOX_CONTAINMENT
                    ),
                )

            except Exception as exc:
                final_dedupe_failed = True
                final_dedupe_error = str(exc)

                final_detections_df = stage_final_dedupe_input_df.copy()
                removed_by_final_dedupe_df = (
                    stage_final_dedupe_input_df
                    .iloc[0:0]
                    .copy()
                )
                final_dedupe_pair_debug_df = pd.DataFrame()

        else:
            final_detections_df = stage_final_dedupe_input_df.copy()
            removed_by_final_dedupe_df = (
                stage_final_dedupe_input_df
                .iloc[0:0]
                .copy()
            )
            final_dedupe_pair_debug_df = pd.DataFrame()

        final_detections_df = (
            final_detections_df
            .copy()
            .reset_index(drop=True)
        )

        removed_by_final_dedupe_df = (
            removed_by_final_dedupe_df
            .copy()
            .reset_index(drop=True)
        )

        final_dedupe_pair_debug_df = (
            final_dedupe_pair_debug_df
            .copy()
            .reset_index(drop=True)
            if isinstance(final_dedupe_pair_debug_df, pd.DataFrame)
            else pd.DataFrame()
        )

        # ---------------------------------------------------------------------
        # 16.33.2 Add removed final-dedupe rows to trace output
        # ---------------------------------------------------------------------
        if len(removed_by_final_dedupe_df) > 0:
            for _, removed_row in removed_by_final_dedupe_df.iterrows():
                crossarm_trace_rows.append(
                    {
                        "run_id": RUN_ID,
                        "run_timestamp": RUN_TIMESTAMP,
                        "image_id": image_id,
                        "file_name": file_name,
                        "roi_file_name": roi_file_name,
                        "roi_image_path": roi_image_path,
                        "processing_order": roi_row.get(
                            "processing_order",
                            np.nan,
                        ),

                        "stage_order": 10,
                        "stage_name": "final_dedupe",

                        "orig_det_idx": int(removed_row["orig_det_idx"]),
                        "score": float(removed_row["score"]),
                        "x1": float(removed_row["x1"]),
                        "y1": float(removed_row["y1"]),
                        "x2": float(removed_row["x2"]),
                        "y2": float(removed_row["y2"]),
                        "has_mask": bool(removed_row.get("has_mask", False)),

                        "action": "removed",
                        "removal_reason": str(
                            removed_row.get(
                                "removal_reason",
                                "removed_by_final_dedupe",
                            )
                        ),
                        "removed_by_orig_det_idx": removed_row.get(
                            "removed_by_orig_det_idx",
                            np.nan,
                        ),

                        "box_containment_of_j_inside_i": removed_row.get(
                            "box_containment_of_j_inside_i",
                            np.nan,
                        ),
                        "mask_containment_of_j_inside_i": removed_row.get(
                            "mask_containment_of_j_inside_i",
                            np.nan,
                        ),
                        "area_ratio_i_over_j": removed_row.get(
                            "area_ratio_i_over_j",
                            np.nan,
                        ),
                        "score_advantage_i_minus_j": removed_row.get(
                            "score_advantage_i_minus_j",
                            np.nan,
                        ),

                        "review_reason": str(
                            removed_row.get(
                                "review_reason",
                                "",
                            )
                        ),
                    }
                )

        # ---------------------------------------------------------------------
        # 16.33.3 Add final schema columns consistently
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # These columns are expected by final trace/debug outputs. If an upstream
        # stage did not create one of them, create it with a safe default.
        # ---------------------------------------------------------------------
        for col, default_value in [
            ("is_x_like", False),
            ("xsplit_attempted", False),
            ("xsplit_applied", False),
            ("skip_axis_cleanup", False),
            ("axis_cleanup_applied", False),
            ("xownership_applied", False),
            ("review_reason", ""),
        ]:
            if col not in final_detections_df.columns:
                final_detections_df[col] = default_value
            else:
                final_detections_df[col] = (
                    final_detections_df[col]
                    .fillna(default_value)
                )

        for bool_col in [
            "is_x_like",
            "xsplit_attempted",
            "xsplit_applied",
            "skip_axis_cleanup",
            "axis_cleanup_applied",
            "xownership_applied",
        ]:
            if bool_col in final_detections_df.columns:
                final_detections_df[bool_col] = (
                    final_detections_df[bool_col]
                    .apply(parse_bool)
                )

        # ---------------------------------------------------------------------
        # 16.33.4 ROI-level review flag
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # One simple status string is stored on every final row so a later batch
        # run can quickly filter ROIs that need human review.
        # ---------------------------------------------------------------------
        final_crossarm_count = int(len(final_detections_df))

        if final_crossarm_count == 0:
            final_roi_review_reason = "no_final_crossarms_found"

        elif final_crossarm_count > int(EXPECTED_MAX_CROSSARMS_FOR_DEBUG):
            final_roi_review_reason = "too_many_final_crossarms_review"

        elif int(single_xsplit_failed_count) > 0:
            final_roi_review_reason = "x_like_split_failed_review"

        elif bool(final_dedupe_failed):
            final_roi_review_reason = "final_dedupe_failed_review"

        else:
            final_roi_review_reason = "ok"

        final_detections_df["final_crossarm_count_for_roi"] = (
            final_crossarm_count
        )
        final_detections_df["final_roi_review_reason"] = (
            final_roi_review_reason
        )

        if bool(final_dedupe_failed):
            if "review_reason" not in final_detections_df.columns:
                final_detections_df["review_reason"] = ""

            final_detections_df["review_reason"] = (
                final_detections_df["review_reason"].apply(
                    lambda old_reason: _append_reason(
                        old_reason,
                        "final_dedupe_failed",
                    )
                )
            )

        # ---------------------------------------------------------------------
        # 16.33.5 Save stage summary
        # ---------------------------------------------------------------------
        crossarm_stage_summary_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "image_id": image_id,
                "file_name": file_name,
                "roi_file_name": roi_file_name,
                "roi_image_path": roi_image_path,
                "processing_order": roi_row.get("processing_order", np.nan),

                "stage_order": 10,
                "stage_name": "final_dedupe_review_flags",
                "input_count": int(len(stage_final_dedupe_input_df)),
                "output_count": int(len(final_detections_df)),
                "removed_count": int(len(removed_by_final_dedupe_df)),

                "final_dedupe_enabled": bool(FINAL_DEDUPE_ENABLED),
                "final_dedupe_failed": bool(final_dedupe_failed),
                "final_dedupe_error": str(final_dedupe_error),
                "final_dedupe_pair_debug_count": int(
                    len(final_dedupe_pair_debug_df)
                ),

                "final_crossarm_count": int(final_crossarm_count),
                "final_roi_review_reason": str(final_roi_review_reason),
                "expected_max_crossarms_for_debug": int(
                    EXPECTED_MAX_CROSSARMS_FOR_DEBUG
                ),

                "single_xsplit_failed_count": int(
                    single_xsplit_failed_count
                ),
                "projected_pole_mask_available": bool(
                    projected_pole_mask_available
                ),
                "pole_mask_filter_applied": bool(pole_mask_filter_applied),
            }
        )
        
        
        # =============================================================================
        # 16.34 FINAL XARM LABELLING + SUCCESS OUTPUTS + FINAL IMAGE SAVES
        # =============================================================================
        # EXPLANATION:
        # This section converts the final post-processed detections for the
        # current ROI into production outputs.
        #
        # This section does six things:
        #   1) Assigns stable final labels:
        #          Xarm_1, Xarm_2, Xarm_3, ...
        #
        #   2) Preserves technical tracking IDs:
        #          orig_det_idx is NOT overwritten.
        #
        #   3) Builds final per-ROI output / trace tables.
        #
        #   4) Builds display-only final masks for human-friendly images.
        #
        #   5) Saves the three final production images:
        #          Final_Image_Real_Mask
        #          Final_Image_Display_Mask
        #          Display_Image_3
        #
        #   6) Appends success rows to run-level accumulators.
        #
        # IMPORTANT:
        #   - Do NOT overwrite orig_det_idx.
        #   - crossarm_mask_lookup remains the canonical real mask lookup.
        #   - final_display_mask_lookup is display-only.
        #   - No display().
        #   - No _safe_display().
        #   - No plt.show().
        #   - No per-ROI print().
        # =============================================================================

        current_stage = "16.34_final_xarm_labelling_outputs"

        # ---------------------------------------------------------------------
        # 16.34.1 Assign stable final Xarm labels
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # Final labels are assigned in visual order:
        #   1) top-to-bottom
        #   2) left-to-right
        #   3) orig_det_idx as deterministic tie-breaker
        #
        # The technical orig_det_idx remains unchanged.
        # ---------------------------------------------------------------------
        if len(final_detections_df) > 0:
            final_detections_df = (
                final_detections_df
                .copy()
                .reset_index(drop=True)
            )

            final_detections_df["box_w"] = (
                final_detections_df["x2"].astype(float)
                - final_detections_df["x1"].astype(float)
            ).clip(lower=0.0)

            final_detections_df["box_h"] = (
                final_detections_df["y2"].astype(float)
                - final_detections_df["y1"].astype(float)
            ).clip(lower=0.0)

            final_detections_df["final_cx"] = (
                final_detections_df["x1"].astype(float)
                + final_detections_df["x2"].astype(float)
            ) / 2.0

            final_detections_df["final_cy"] = (
                final_detections_df["y1"].astype(float)
                + final_detections_df["y2"].astype(float)
            ) / 2.0

            final_detections_df = (
                final_detections_df
                .sort_values(
                    by=[
                        "final_cy",
                        "final_cx",
                        "orig_det_idx",
                    ],
                    ascending=[
                        True,
                        True,
                        True,
                    ],
                    kind="mergesort",
                )
                .reset_index(drop=True)
            )

            final_detections_df["final_xarm_number"] = np.arange(
                1,
                len(final_detections_df) + 1,
                dtype=int,
            )

            final_detections_df["final_xarm_label"] = (
                final_detections_df["final_xarm_number"]
                .apply(lambda n: f"Xarm_{int(n)}")
            )

            # Alias used by plot_stage_on_ax when final_style=True.
            final_detections_df["xarm_label"] = (
                final_detections_df["final_xarm_label"]
            )

            final_detections_df["source_orig_det_idxs"] = (
                final_detections_df.apply(
                    collect_source_orig_det_idxs,
                    axis=1,
                )
            )

            final_detections_df["source_orig_det_idxs_text"] = (
                final_detections_df["source_orig_det_idxs"]
                .apply(lambda xs: ",".join([str(int(v)) for v in xs]))
            )

        else:
            final_detections_df = final_detections_df.copy()

            for col_name, dtype_name in [
                ("box_w", "float64"),
                ("box_h", "float64"),
                ("final_cx", "float64"),
                ("final_cy", "float64"),
                ("final_xarm_number", "int64"),
                ("final_xarm_label", "object"),
                ("xarm_label", "object"),
                ("source_orig_det_idxs", "object"),
                ("source_orig_det_idxs_text", "object"),
            ]:
                if col_name not in final_detections_df.columns:
                    final_detections_df[col_name] = pd.Series(
                        dtype=dtype_name,
                    )

        # ---------------------------------------------------------------------
        # 16.34.2 Create final per-ROI output aliases
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # stage_final_df is the canonical final per-ROI detection table after
        # final labelling.
        # ---------------------------------------------------------------------
        final_kept_detections_df = (
            final_detections_df
            .copy()
            .reset_index(drop=True)
        )

        stage_final_df = (
            final_kept_detections_df
            .copy()
            .reset_index(drop=True)
        )

        final_xarm_output_cols = [
            "final_xarm_label",
            "final_xarm_number",
            "score",
            "x1",
            "y1",
            "x2",
            "y2",
            "box_w",
            "box_h",
            "final_cx",
            "final_cy",
            "final_crossarm_count_for_roi",
            "final_roi_review_reason",
        ]

        final_xarm_output_df = stage_final_df[
            [
                col_name
                for col_name in final_xarm_output_cols
                if col_name in stage_final_df.columns
            ]
        ].copy()

        final_xarm_trace_cols = [
            "final_xarm_label",
            "final_xarm_number",
            "orig_det_idx",
            "source_orig_det_idxs",
            "source_orig_det_idxs_text",
            "merged_from_orig_det_idxs",
            "parent_orig_det_idx",
            "score",
            "x1",
            "y1",
            "x2",
            "y2",
            "box_w",
            "box_h",
            "final_cx",
            "final_cy",
            "same_xarm_merge_applied",
            "same_xarm_merge_group_id",
            "same_xarm_merge_count",
            "xsplit_attempted",
            "xsplit_applied",
            "xsplit_reason",
            "axis_cleanup_applied",
            "xownership_applied",
            "is_x_like",
            "skip_axis_cleanup",
            "review_reason",
            "final_roi_review_reason",
        ]

        final_xarm_trace_df = stage_final_df[
            [
                col_name
                for col_name in final_xarm_trace_cols
                if col_name in stage_final_df.columns
            ]
        ].copy()

        # ---------------------------------------------------------------------
        # 16.34.3 Build display-only final mask lookup
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # final_display_mask_lookup is used only for saved human-friendly
        # final images.
        #
        # IMPORTANT:
        #   Do NOT overwrite crossarm_mask_lookup.
        # ---------------------------------------------------------------------
        final_display_mask_lookup = {}

        if len(stage_final_df) > 0:
            for _, final_row in stage_final_df.iterrows():
                if "orig_det_idx" not in final_row:
                    continue

                final_orig_det_idx = int(final_row["orig_det_idx"])

                candidate_orig_idxs = []

                for lineage_col in [
                    "source_orig_det_idxs",
                    "merged_from_orig_det_idxs",
                    "parent_orig_det_idx",
                    "orig_det_idx",
                ]:
                    if lineage_col in final_row:
                        candidate_orig_idxs.extend(
                            normalise_orig_idx_list(
                                final_row.get(lineage_col)
                            )
                        )

                seen_candidate_idxs = set()
                ordered_candidate_orig_idxs = []

                for candidate_idx in candidate_orig_idxs:
                    candidate_idx = int(candidate_idx)

                    if candidate_idx not in seen_candidate_idxs:
                        ordered_candidate_orig_idxs.append(candidate_idx)
                        seen_candidate_idxs.add(candidate_idx)

                union_mask = np.zeros(
                    (int(roi_h), int(roi_w)),
                    dtype=bool,
                )

                for candidate_idx in ordered_candidate_orig_idxs:
                    source_mask = crossarm_mask_lookup.get(
                        int(candidate_idx),
                        None,
                    )

                    if (
                        isinstance(source_mask, np.ndarray)
                        and source_mask.ndim == 2
                        and source_mask.shape == (int(roi_h), int(roi_w))
                    ):
                        union_mask |= source_mask.astype(bool)

                box_x1 = int(np.floor(float(final_row["x1"]))) - 6
                box_y1 = int(np.floor(float(final_row["y1"]))) - 6
                box_x2 = int(np.ceil(float(final_row["x2"]))) + 6
                box_y2 = int(np.ceil(float(final_row["y2"]))) + 6

                box_x1 = max(0, min(int(roi_w), box_x1))
                box_y1 = max(0, min(int(roi_h), box_y1))
                box_x2 = max(0, min(int(roi_w), box_x2))
                box_y2 = max(0, min(int(roi_h), box_y2))

                bbox_mask = np.zeros(
                    (int(roi_h), int(roi_w)),
                    dtype=bool,
                )

                if box_x2 > box_x1 and box_y2 > box_y1:
                    bbox_mask[box_y1:box_y2, box_x1:box_x2] = True

                if int(union_mask.sum()) == 0:
                    display_mask = bbox_mask

                else:
                    display_mask = union_mask & bbox_mask

                    if int(display_mask.sum()) == 0:
                        display_mask = union_mask.copy()

                    if int(display_mask.sum()) == 0:
                        display_mask = bbox_mask

                if int(display_mask.sum()) > 0:
                    multi_source = len(ordered_candidate_orig_idxs) > 1

                    box_w_display = max(1, box_x2 - box_x1)
                    box_h_display = max(1, box_y2 - box_y1)
                    box_min_dim = max(
                        1,
                        min(box_w_display, box_h_display),
                    )

                    if multi_source:
                        close_k = max(5, int(round(box_min_dim * 0.30)))
                        dilate_k = max(3, int(round(box_min_dim * 0.01)))
                    else:
                        close_k = max(3, int(round(box_min_dim * 0.015)))
                        dilate_k = max(3, int(round(box_min_dim * 0.005)))

                    if close_k % 2 == 0:
                        close_k += 1

                    if dilate_k % 2 == 0:
                        dilate_k += 1

                    try:
                        display_u8 = display_mask.astype(np.uint8) * 255

                        close_kernel = cv2.getStructuringElement(
                            cv2.MORPH_RECT,
                            (close_k, close_k),
                        )

                        dilate_kernel = cv2.getStructuringElement(
                            cv2.MORPH_RECT,
                            (dilate_k, dilate_k),
                        )

                        closed_u8 = cv2.morphologyEx(
                            display_u8,
                            cv2.MORPH_CLOSE,
                            close_kernel,
                        )

                        dilated_u8 = cv2.dilate(
                            closed_u8,
                            dilate_kernel,
                            iterations=1,
                        )

                        display_mask = (dilated_u8 > 0) & bbox_mask

                        if int(display_mask.sum()) == 0:
                            display_mask = bbox_mask

                    except Exception:
                        display_mask = display_mask & bbox_mask

                        if int(display_mask.sum()) == 0:
                            display_mask = bbox_mask

                final_display_mask_lookup[final_orig_det_idx] = (
                    display_mask.astype(bool)
                )

        # ---------------------------------------------------------------------
        # 16.34.4 Save final three images
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # Save only final production images. Do not display them.
        #
        # Images:
        #   1) Final_Image_Real_Mask
        #   2) Final_Image_Display_Mask
        #   3) Display_Image_3
        #
        # IMPORTANT:
        #   Image-save failures are non-fatal.
        #
        #   If a final PNG fails to save, the ROI still commits its successful
        #   detection rows. The image-save failure is recorded in:
        #       final_image_save_failed
        #       final_image_save_error
        #       final_image_save_failed_images
        #
        #   Do not raise from these image-save except blocks.
        # ---------------------------------------------------------------------
        saved_final_image_paths = {
            "Final_Image_Real_Mask": None,
            "Final_Image_Display_Mask": None,
            "Display_Image_3": None,
        }

        expected_final_image_count = int(len(saved_final_image_paths))

        final_image_save_failed = False
        final_image_save_error = ""
        final_image_save_failed_images = []

        current_saved_image_rows = []

        final_projected_pole_mask = (
            projected_pole_mask
            if (
                bool(projected_pole_mask_available)
                and isinstance(projected_pole_mask, np.ndarray)
                and projected_pole_mask.ndim == 2
                and projected_pole_mask.any()
            )
            else None
        )

        cell16_crossarm_mask_alpha = float(
            globals().get(
                "CROSSARM_MASK_ALPHA",
                0.40,
            )
        )

        cell16_pole_mask_alpha = float(
            globals().get(
                "POLE_MASK_ALPHA",
                0.30,
            )
        )

        cell16_label_bg = str(
            globals().get(
                "LABEL_BG",
                "#1E90FF",
            )
        )

        save_final_images_for_roi = bool(
            CELL16_SAVE_GOLD_FINAL_IMAGES
            and should_save_gold_final_images(
                final_roi_review_reason=final_roi_review_reason
            )
        )

        if save_final_images_for_roi:
            # -----------------------------------------------------------------
            # 16.34.4A Save Final_Image_Real_Mask
            # -----------------------------------------------------------------
            fig_real = None

            try:
                fig_real, ax_real = plt.subplots(
                    1,
                    1,
                    figsize=(10, 10),
                )

                plot_stage_on_ax(
                    ax=ax_real,
                    image=image,
                    detections_df=stage_final_df,
                    title="Final_Image_Real_Mask",
                    projected_pole_mask=final_projected_pole_mask,
                    crossarm_mask_lookup=crossarm_mask_lookup,
                    crossarm_mask_alpha=cell16_crossarm_mask_alpha,
                    pole_mask_alpha=cell16_pole_mask_alpha,
                    label_bg=cell16_label_bg,
                    final_style=False,
                )

                fig_real.tight_layout()

                saved_path = save_final_image(
                    fig=fig_real,
                    image_id=image_id,
                    roi_file_name=roi_file_name,
                    image_name="Final_Image_Real_Mask",
                )

                saved_final_image_paths["Final_Image_Real_Mask"] = saved_path

                current_saved_image_rows.append(
                    {
                        "run_id": RUN_ID,
                        "run_timestamp": RUN_TIMESTAMP,
                        "image_id": image_id,
                        "file_name": file_name,
                        "roi_file_name": roi_file_name,
                        "roi_image_path": roi_image_path,
                        "processing_order": roi_row.get(
                            "processing_order",
                            np.nan,
                        ),
                        "image_name": "Final_Image_Real_Mask",
                        "image_path": saved_path,
                        "final_crossarm_count": int(final_crossarm_count),
                        "final_roi_review_reason": str(final_roi_review_reason),
                    }
                )

            except Exception as image_save_exc:
                final_image_save_failed = True
                final_image_save_failed_images.append("Final_Image_Real_Mask")

                error_text = (
                    f"Final_Image_Real_Mask: "
                    f"{type(image_save_exc).__name__}: {image_save_exc}"
                )

                final_image_save_error = (
                    error_text
                    if len(final_image_save_error) == 0
                    else f"{final_image_save_error} | {error_text}"
                )

                saved_final_image_paths["Final_Image_Real_Mask"] = None

                if fig_real is not None:
                    try:
                        plt.close(fig_real)
                    except Exception:
                        pass

            # -----------------------------------------------------------------
            # 16.34.4B Save Final_Image_Display_Mask
            # -----------------------------------------------------------------
            fig_display = None

            try:
                fig_display, ax_display = plt.subplots(
                    1,
                    1,
                    figsize=(10, 10),
                )

                plot_stage_on_ax(
                    ax=ax_display,
                    image=image,
                    detections_df=stage_final_df,
                    title="Final_Image_Display_Mask",
                    projected_pole_mask=final_projected_pole_mask,
                    crossarm_mask_lookup=final_display_mask_lookup,
                    crossarm_mask_alpha=cell16_crossarm_mask_alpha,
                    pole_mask_alpha=cell16_pole_mask_alpha,
                    label_bg=cell16_label_bg,
                    final_style=True,
                )

                fig_display.tight_layout()

                saved_path = save_final_image(
                    fig=fig_display,
                    image_id=image_id,
                    roi_file_name=roi_file_name,
                    image_name="Final_Image_Display_Mask",
                )

                saved_final_image_paths["Final_Image_Display_Mask"] = saved_path

                current_saved_image_rows.append(
                    {
                        "run_id": RUN_ID,
                        "run_timestamp": RUN_TIMESTAMP,
                        "image_id": image_id,
                        "file_name": file_name,
                        "roi_file_name": roi_file_name,
                        "roi_image_path": roi_image_path,
                        "processing_order": roi_row.get(
                            "processing_order",
                            np.nan,
                        ),
                        "image_name": "Final_Image_Display_Mask",
                        "image_path": saved_path,
                        "final_crossarm_count": int(final_crossarm_count),
                        "final_roi_review_reason": str(final_roi_review_reason),
                    }
                )

            except Exception as image_save_exc:
                final_image_save_failed = True
                final_image_save_failed_images.append(
                    "Final_Image_Display_Mask"
                )

                error_text = (
                    f"Final_Image_Display_Mask: "
                    f"{type(image_save_exc).__name__}: {image_save_exc}"
                )

                final_image_save_error = (
                    error_text
                    if len(final_image_save_error) == 0
                    else f"{final_image_save_error} | {error_text}"
                )

                saved_final_image_paths["Final_Image_Display_Mask"] = None

                if fig_display is not None:
                    try:
                        plt.close(fig_display)
                    except Exception:
                        pass

            # -----------------------------------------------------------------
            # 16.34.4C Save Display_Image_3
            # -----------------------------------------------------------------
            fig_display_3 = None

            try:
                fig_display_3, ax_display_3 = plt.subplots(
                    1,
                    1,
                    figsize=(10, 10),
                )

                ax_display_3.imshow(image)

                # Draw pole context lightly in red.
                if final_projected_pole_mask is not None:
                    pole_overlay = np.zeros(
                        (
                            final_projected_pole_mask.shape[0],
                            final_projected_pole_mask.shape[1],
                            4,
                        ),
                        dtype=np.float32,
                    )

                    pole_overlay[..., 0] = 1.0
                    pole_overlay[..., 1] = 0.0
                    pole_overlay[..., 2] = 0.0
                    pole_overlay[..., 3] = (
                        final_projected_pole_mask.astype(np.float32)
                        * 0.20
                    )

                    ax_display_3.imshow(pole_overlay)

                display_image_3_mask_lookup = (
                    final_display_mask_lookup
                    if len(final_display_mask_lookup) > 0
                    else crossarm_mask_lookup
                )

                if len(stage_final_df) > 0:
                    for _, xarm_row in stage_final_df.iterrows():
                        if "orig_det_idx" not in xarm_row:
                            continue

                        xarm_idx = int(xarm_row["orig_det_idx"])

                        mask_i = display_image_3_mask_lookup.get(
                            xarm_idx,
                            None,
                        )

                        if (
                            isinstance(mask_i, np.ndarray)
                            and mask_i.ndim == 2
                            and int(mask_i.sum()) > 0
                        ):
                            xarm_overlay = np.zeros(
                                (
                                    mask_i.shape[0],
                                    mask_i.shape[1],
                                    4,
                                ),
                                dtype=np.float32,
                            )

                            # Purple mask overlay.
                            xarm_overlay[..., 0] = 0.55
                            xarm_overlay[..., 1] = 0.00
                            xarm_overlay[..., 2] = 1.00
                            xarm_overlay[..., 3] = (
                                mask_i.astype(np.float32)
                                * 0.35
                            )

                            ax_display_3.imshow(xarm_overlay)

                    for _, xarm_row in stage_final_df.iterrows():
                        if not all(
                            col_name in xarm_row
                            for col_name in ["x1", "y1", "x2", "y2"]
                        ):
                            continue

                        x1 = float(xarm_row["x1"])
                        y1 = float(xarm_row["y1"])
                        x2 = float(xarm_row["x2"])
                        y2 = float(xarm_row["y2"])

                        rect = patches.Rectangle(
                            (x1, y1),
                            max(0.0, x2 - x1),
                            max(0.0, y2 - y1),
                            linewidth=1.8,
                            edgecolor="purple",
                            facecolor="none",
                            linestyle="--",
                        )

                        ax_display_3.add_patch(rect)

                        if (
                            "final_xarm_label" in xarm_row
                            and pd.notna(xarm_row["final_xarm_label"])
                        ):
                            xarm_label = str(xarm_row["final_xarm_label"])

                        elif (
                            "xarm_label" in xarm_row
                            and pd.notna(xarm_row["xarm_label"])
                        ):
                            xarm_label = str(xarm_row["xarm_label"])

                        elif (
                            "orig_det_idx" in xarm_row
                            and pd.notna(xarm_row["orig_det_idx"])
                        ):
                            xarm_label = (
                                f"Xarm_idx_{int(xarm_row['orig_det_idx'])}"
                            )

                        else:
                            xarm_label = "Xarm"

                        ax_display_3.text(
                            x1,
                            max(0.0, y1 - 5.0),
                            xarm_label,
                            color="white",
                            fontsize=9,
                            bbox=dict(
                                facecolor="purple",
                                alpha=0.80,
                                pad=0.35,
                                edgecolor="none",
                            ),
                        )

                ax_display_3.set_title("Display_Image_3")
                ax_display_3.axis("off")

                fig_display_3.tight_layout()

                saved_path = save_final_image(
                    fig=fig_display_3,
                    image_id=image_id,
                    roi_file_name=roi_file_name,
                    image_name="Display_Image_3",
                )

                saved_final_image_paths["Display_Image_3"] = saved_path

                current_saved_image_rows.append(
                    {
                        "run_id": RUN_ID,
                        "run_timestamp": RUN_TIMESTAMP,
                        "image_id": image_id,
                        "file_name": file_name,
                        "roi_file_name": roi_file_name,
                        "roi_image_path": roi_image_path,
                        "processing_order": roi_row.get(
                            "processing_order",
                            np.nan,
                        ),
                        "image_name": "Display_Image_3",
                        "image_path": saved_path,
                        "final_crossarm_count": int(final_crossarm_count),
                        "final_roi_review_reason": str(final_roi_review_reason),
                    }
                )

            except Exception as image_save_exc:
                final_image_save_failed = True
                final_image_save_failed_images.append("Display_Image_3")

                error_text = (
                    f"Display_Image_3: "
                    f"{type(image_save_exc).__name__}: {image_save_exc}"
                )

                final_image_save_error = (
                    error_text
                    if len(final_image_save_error) == 0
                    else f"{final_image_save_error} | {error_text}"
                )

                saved_final_image_paths["Display_Image_3"] = None

                if fig_display_3 is not None:
                    try:
                        plt.close(fig_display_3)
                    except Exception:
                        pass

        # ---------------------------------------------------------------------
        # 16.34.5 Build per-ROI success row buffers
        # ---------------------------------------------------------------------
        # EXPLANATION:
        # Build local row buffers first. Commit them to the run-level
        # accumulators after final image-save attempts complete.
        #
        # IMPORTANT:
        #   Image-save failures are non-fatal and are recorded as metadata.
        #   Successful detections still commit as success rows.
        # ---------------------------------------------------------------------
        roi_processing_seconds = float(time.time() - roi_start_time)

        current_final_detection_rows = []
        current_final_trace_rows = []

        base_success_fields = {
            "run_id": RUN_ID,
            "run_timestamp": RUN_TIMESTAMP,
            "image_id": image_id,
            "file_name": file_name,
            "roi_file_name": roi_file_name,
            "roi_image_path": roi_image_path,
            "processing_order": roi_row.get("processing_order", np.nan),
            "roi_w": int(roi_w),
            "roi_h": int(roi_h),
            "final_crossarm_count": int(final_crossarm_count),
            "final_roi_review_reason": str(final_roi_review_reason),
            "projected_pole_mask_available": bool(projected_pole_mask_available),
            "pole_mask_filter_applied": bool(pole_mask_filter_applied),
        }

        final_detection_export_cols = [
            "final_xarm_label",
            "final_xarm_number",
            "orig_det_idx",
            "source_orig_det_idxs",
            "source_orig_det_idxs_text",
            "merged_from_orig_det_idxs",
            "parent_orig_det_idx",
            "score",
            "x1",
            "y1",
            "x2",
            "y2",
            "box_w",
            "box_h",
            "final_cx",
            "final_cy",
            "same_xarm_merge_applied",
            "same_xarm_merge_group_id",
            "same_xarm_merge_count",
            "is_x_like",
            "xsplit_attempted",
            "xsplit_applied",
            "xsplit_reason",
            "axis_cleanup_applied",
            "xownership_applied",
            "skip_axis_cleanup",
            "review_reason",
            "final_crossarm_count_for_roi",
            "final_roi_review_reason",
        ]

        if len(stage_final_df) > 0:
            for _, final_row in stage_final_df.iterrows():
                final_detection_record = dict(base_success_fields)

                for col_name in final_detection_export_cols:
                    if col_name in stage_final_df.columns:
                        final_detection_record[col_name] = final_row.get(
                            col_name,
                            np.nan,
                        )

                current_final_detection_rows.append(final_detection_record)

                final_trace_record = dict(base_success_fields)
                final_trace_record.update(
                    {
                        "stage_order": 11,
                        "stage_name": "final_xarm_labelling",
                        "action": "kept_final",
                    }
                )

                for col_name in final_xarm_trace_cols:
                    if col_name in stage_final_df.columns:
                        final_trace_record[col_name] = final_row.get(
                            col_name,
                            np.nan,
                        )

                current_final_trace_rows.append(final_trace_record)

        current_image_row = {
            "run_id": RUN_ID,
            "run_timestamp": RUN_TIMESTAMP,
            "image_id": image_id,
            "file_name": file_name,
            "roi_file_name": roi_file_name,
            "roi_image_path": roi_image_path,
            "processing_order": roi_row.get("processing_order", np.nan),

            "processing_status": "success",
            "failure_reason": "",
            "failure_stage": "",

            "roi_w": int(roi_w),
            "roi_h": int(roi_h),
            "processing_seconds": roi_processing_seconds,

            "raw_detection_count": int(len(stage_raw_df)),
            "raw_score_prefilter_count": int(len(stage_raw_score_prefilter_df)),
            "containment_count": int(len(stage_containment_df)),
            "main_cluster_count": int(len(stage_cluster_df)),
            "pole_overlap_count": int(len(stage_pole_overlap_df)),
            "same_xarm_merge_count": int(len(stage_same_xarm_merge_df)),
            "single_xsplit_count": int(len(stage_single_xsplit_df)),
            "axis_cleanup_count": int(len(stage_axis_cleanup_df)),
            "xownership_count": int(len(stage_xownership_df)),
            "final_crossarm_count": int(final_crossarm_count),

            "final_roi_review_reason": str(final_roi_review_reason),
            "single_xsplit_failed_count": int(single_xsplit_failed_count),
            "final_dedupe_failed": bool(final_dedupe_failed),
            "final_dedupe_error": str(final_dedupe_error),

            "projected_pole_mask_available": bool(projected_pole_mask_available),
            "pole_mask_filter_applied": bool(pole_mask_filter_applied),

            "final_images_requested": bool(save_final_images_for_roi),
            "final_images_saved": bool(
                save_final_images_for_roi
                and not final_image_save_failed
                and len(current_saved_image_rows) == expected_final_image_count
            ),
            "final_images_expected_count": int(expected_final_image_count),
            "final_images_saved_count": int(len(current_saved_image_rows)),
            "final_image_save_failed": bool(final_image_save_failed),
            "final_image_save_error": str(final_image_save_error),
            "final_image_save_failed_images": ",".join(
                [str(v) for v in final_image_save_failed_images]
            ),
            "final_image_real_mask_path": saved_final_image_paths[
                "Final_Image_Real_Mask"
            ],
            "final_image_display_mask_path": saved_final_image_paths[
                "Final_Image_Display_Mask"
            ],
            "display_image_3_path": saved_final_image_paths[
                "Display_Image_3"
            ],
        }

        # ---------------------------------------------------------------------
        # 16.34.6 Commit per-ROI success rows
        # ---------------------------------------------------------------------
        crossarm_final_detection_rows.extend(current_final_detection_rows)
        crossarm_trace_rows.extend(current_final_trace_rows)
        crossarm_saved_image_rows.extend(current_saved_image_rows)
        crossarm_image_rows.append(current_image_row)

        # ---------------------------------------------------------------------
        # 16.34.7 Save final labelling / output stage summary
        # ---------------------------------------------------------------------
        crossarm_stage_summary_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "image_id": image_id,
                "file_name": file_name,
                "roi_file_name": roi_file_name,
                "roi_image_path": roi_image_path,
                "processing_order": roi_row.get("processing_order", np.nan),

                "stage_order": 11,
                "stage_name": "final_xarm_labelling_outputs",
                "input_count": int(len(final_detections_df)),
                "output_count": int(len(stage_final_df)),
                "removed_count": 0,

                "final_crossarm_count": int(final_crossarm_count),
                "final_roi_review_reason": str(final_roi_review_reason),

                "final_detection_rows_added": int(
                    len(current_final_detection_rows)
                ),
                "final_trace_rows_added": int(len(current_final_trace_rows)),
                "final_images_requested": bool(save_final_images_for_roi),
                "final_images_saved": bool(
                    save_final_images_for_roi
                    and not final_image_save_failed
                    and len(current_saved_image_rows) == expected_final_image_count
                ),
                "final_images_expected_count": int(expected_final_image_count),
                "final_images_saved_count": int(len(current_saved_image_rows)),
                "final_image_save_failed": bool(final_image_save_failed),
                "final_image_save_error": str(final_image_save_error),
                "final_image_save_failed_images": ",".join(
                    [str(v) for v in final_image_save_failed_images]
                ),
                "final_image_real_mask_path": saved_final_image_paths[
                    "Final_Image_Real_Mask"
                ],
                "final_image_display_mask_path": saved_final_image_paths[
                    "Final_Image_Display_Mask"
                ],
                "display_image_3_path": saved_final_image_paths[
                    "Display_Image_3"
                ],

                "projected_pole_mask_available": bool(
                    projected_pole_mask_available
                ),
                "pole_mask_filter_applied": bool(pole_mask_filter_applied),
                "processing_seconds": roi_processing_seconds,
            }
        )
        

    except Exception as exc:
        row_failure_reason = "cell16_roi_processing_failed"

        crossarm_failure_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "image_id": str(roi_row.get("image_id", "")),
                "file_name": str(roi_row.get("file_name", "")),
                "roi_file_name": str(roi_row.get("roi_file_name", "")),
                "roi_image_path": str(roi_row.get("roi_image_path", "")),
                "processing_order": roi_row.get("processing_order", np.nan),

                # Stage where this ROI failed.
                "failure_stage": current_stage,

                # CELL 16-specific high-level failure label.
                "failure_reason": row_failure_reason,

                # Match CELL 13 / CELL 14 failure schema semantics.
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_traceback": traceback.format_exc(),
            }
        )

        if CELL16_STOP_ON_ROW_FAILURE:
            raise

        continue
    
    
# =============================================================================
# 16D. ASSEMBLE RUN-LEVEL OUTPUT TABLES
# =============================================================================
# EXPLANATION:
# Convert the CELL 16 run-level row accumulators into stable pandas DataFrames.
#
# IMPORTANT:
#   This section runs once after the 16C per-ROI loop has completed.
#
#   Do NOT place this section inside the ROI loop.
#
#   16C appends lightweight dictionaries to these run-level accumulators:
#       crossarm_image_rows
#       crossarm_final_detection_rows
#       crossarm_trace_rows
#       crossarm_failure_rows
#       crossarm_stage_summary_rows
#       crossarm_saved_image_rows
#
#   16D only assembles those lists into DataFrames.
#
#   16E will perform reconciliation / consistency checks.
#   16F will save Gold/Silver output tables.
# =============================================================================

# =============================================================================
# 16.35 RUN-LEVEL TABLE ASSEMBLY HELPERS
# =============================================================================
# EXPLANATION:
# These helpers keep DataFrame assembly consistent and make empty output tables
# predictable.
# =============================================================================

def records_to_dataframe(records, expected_columns=None):
    """
    Convert a list of row dictionaries into a stable DataFrame.

    Args:
        records:
            List of dictionaries accumulated during CELL 16.

        expected_columns:
            Optional ordered list of columns expected in the output table.

    Returns:
        pd.DataFrame:
            DataFrame with expected columns present and ordered first.
    """
    if expected_columns is None:
        expected_columns = []

    if records is None:
        records = []

    if len(records) == 0:
        output_df = pd.DataFrame(columns=list(expected_columns))

    else:
        output_df = pd.DataFrame(records)

        for col_name in expected_columns:
            if col_name not in output_df.columns:
                output_df[col_name] = np.nan

        ordered_cols = (
            list(expected_columns)
            + [
                col_name
                for col_name in output_df.columns
                if col_name not in expected_columns
            ]
        )

        output_df = output_df[ordered_cols].copy()

    return output_df.reset_index(drop=True)


def sort_if_columns_exist(df, sort_columns):
    """
    Sort a DataFrame only by columns that exist.

    Args:
        df:
            DataFrame to sort.

        sort_columns:
            Preferred sort columns.

    Returns:
        pd.DataFrame:
            Sorted DataFrame when possible, otherwise a reset copy.
    """
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    if df.empty:
        return df.copy().reset_index(drop=True)

    available_sort_columns = [
        col_name
        for col_name in sort_columns
        if col_name in df.columns
    ]

    if len(available_sort_columns) == 0:
        return df.copy().reset_index(drop=True)

    return (
        df
        .sort_values(
            by=available_sort_columns,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


# =============================================================================
# 16.36 DEFINE RUN-LEVEL OUTPUT SCHEMAS
# =============================================================================
# EXPLANATION:
# These are the main expected output columns for each table.
#
# Extra columns are preserved automatically by records_to_dataframe().
# =============================================================================

crossarm_image_result_cols = [
    "run_id",
    "run_timestamp",
    "image_id",
    "file_name",
    "roi_file_name",
    "roi_image_path",
    "processing_order",

    "processing_status",
    "failure_reason",
    "failure_stage",

    "roi_w",
    "roi_h",
    "processing_seconds",

    "raw_detection_count",
    "raw_score_prefilter_count",
    "containment_count",
    "main_cluster_count",
    "pole_overlap_count",
    "same_xarm_merge_count",
    "single_xsplit_count",
    "axis_cleanup_count",
    "xownership_count",
    "final_crossarm_count",

    "final_roi_review_reason",
    "single_xsplit_failed_count",
    "final_dedupe_failed",
    "final_dedupe_error",

    "projected_pole_mask_available",
    "pole_mask_filter_applied",

    "final_images_requested",
    "final_images_saved",
    "final_images_expected_count",
    "final_images_saved_count",
    "final_image_save_failed",
    "final_image_save_error",
    "final_image_save_failed_images",

    "final_image_real_mask_path",
    "final_image_display_mask_path",
    "display_image_3_path",
]

crossarm_final_detection_cols = [
    "run_id",
    "run_timestamp",
    "image_id",
    "file_name",
    "roi_file_name",
    "roi_image_path",
    "processing_order",

    "roi_w",
    "roi_h",

    "final_crossarm_count",
    "final_crossarm_count_for_roi",
    "final_roi_review_reason",

    "projected_pole_mask_available",
    "pole_mask_filter_applied",

    "final_xarm_label",
    "final_xarm_number",

    "orig_det_idx",
    "source_orig_det_idxs",
    "source_orig_det_idxs_text",
    "merged_from_orig_det_idxs",
    "parent_orig_det_idx",

    "score",
    "x1",
    "y1",
    "x2",
    "y2",
    "box_w",
    "box_h",
    "final_cx",
    "final_cy",

    "same_xarm_merge_applied",
    "same_xarm_merge_group_id",
    "same_xarm_merge_count",

    "is_x_like",
    "xsplit_attempted",
    "xsplit_applied",
    "xsplit_reason",

    "axis_cleanup_applied",
    "xownership_applied",
    "skip_axis_cleanup",

    "review_reason",
]

crossarm_trace_cols = [
    "run_id",
    "run_timestamp",
    "image_id",
    "file_name",
    "roi_file_name",
    "roi_image_path",
    "processing_order",

    "stage_order",
    "stage_name",
    "action",
    "removal_reason",

    "orig_det_idx",
    "source_orig_det_idxs",
    "source_orig_det_idxs_text",
    "merged_from_orig_det_idxs",
    "parent_orig_det_idx",

    "score",
    "x1",
    "y1",
    "x2",
    "y2",
    "box_w",
    "box_h",
    "has_mask",

    "final_xarm_label",
    "final_xarm_number",
    "final_cx",
    "final_cy",

    "removed_by_orig_det_idx",
    "box_containment_of_j_inside_i",
    "mask_containment_of_j_inside_i",
    "area_ratio_i_over_j",
    "score_advantage_i_minus_j",

    "same_xarm_merge_applied",
    "same_xarm_merge_group_id",
    "same_xarm_merge_count",

    "xsplit_attempted",
    "xsplit_applied",
    "xsplit_reason",
    "axis_cleanup_applied",
    "xownership_applied",
    "is_x_like",
    "skip_axis_cleanup",

    "review_reason",
    "final_roi_review_reason",
]

crossarm_failure_cols = [
    "run_id",
    "run_timestamp",
    "image_id",
    "file_name",
    "roi_file_name",
    "roi_image_path",
    "processing_order",

    "failure_stage",
    "failure_reason",
    "error_type",
    "error_message",
    "error_traceback",
]

crossarm_stage_summary_cols = [
    "run_id",
    "run_timestamp",
    "image_id",
    "file_name",
    "roi_file_name",
    "roi_image_path",
    "processing_order",

    "stage_order",
    "stage_name",
    "input_count",
    "output_count",
    "removed_count",
]

crossarm_saved_image_cols = [
    "run_id",
    "run_timestamp",
    "image_id",
    "file_name",
    "roi_file_name",
    "roi_image_path",
    "processing_order",

    "image_name",
    "image_path",
    "final_crossarm_count",
    "final_roi_review_reason",
]


# =============================================================================
# 16.37 ASSEMBLE RUN-LEVEL DATAFRAMES
# =============================================================================
# EXPLANATION:
# Build the final in-memory output tables from row accumulators.
# =============================================================================

crossarm_image_results_df = records_to_dataframe(
    records=crossarm_image_rows,
    expected_columns=crossarm_image_result_cols,
)

crossarm_final_detections_df = records_to_dataframe(
    records=crossarm_final_detection_rows,
    expected_columns=crossarm_final_detection_cols,
)

crossarm_trace_df = records_to_dataframe(
    records=crossarm_trace_rows,
    expected_columns=crossarm_trace_cols,
)

crossarm_failures_df = records_to_dataframe(
    records=crossarm_failure_rows,
    expected_columns=crossarm_failure_cols,
)

crossarm_stage_summary_df = records_to_dataframe(
    records=crossarm_stage_summary_rows,
    expected_columns=crossarm_stage_summary_cols,
)

crossarm_saved_images_df = records_to_dataframe(
    records=crossarm_saved_image_rows,
    expected_columns=crossarm_saved_image_cols,
)


# =============================================================================
# 16.38 SORT RUN-LEVEL DATAFRAMES
# =============================================================================
# EXPLANATION:
# Sort tables deterministically for easier review, reproducibility, and saving.
# =============================================================================

crossarm_image_results_df = sort_if_columns_exist(
    df=crossarm_image_results_df,
    sort_columns=[
        "processing_order",
        "image_id",
    ],
)

crossarm_final_detections_df = sort_if_columns_exist(
    df=crossarm_final_detections_df,
    sort_columns=[
        "processing_order",
        "image_id",
        "final_xarm_number",
        "orig_det_idx",
    ],
)

crossarm_trace_df = sort_if_columns_exist(
    df=crossarm_trace_df,
    sort_columns=[
        "processing_order",
        "image_id",
        "stage_order",
        "orig_det_idx",
    ],
)

crossarm_failures_df = sort_if_columns_exist(
    df=crossarm_failures_df,
    sort_columns=[
        "processing_order",
        "image_id",
        "failure_stage",
    ],
)

crossarm_stage_summary_df = sort_if_columns_exist(
    df=crossarm_stage_summary_df,
    sort_columns=[
        "processing_order",
        "image_id",
        "stage_order",
        "stage_name",
    ],
)

crossarm_saved_images_df = sort_if_columns_exist(
    df=crossarm_saved_images_df,
    sort_columns=[
        "processing_order",
        "image_id",
        "image_name",
    ],
)


# =============================================================================
# 16.39 COMPACT OUTPUT COUNTS
# =============================================================================
# EXPLANATION:
# Store lightweight counts for 16E reconciliation and 16F summary.
# =============================================================================

cell16_output_counts = {
    "input_roi_count": int(len(cell16_roi_input_df)),
    "success_image_row_count": int(len(crossarm_image_results_df)),
    "failure_row_count": int(len(crossarm_failures_df)),
    "final_detection_row_count": int(len(crossarm_final_detections_df)),
    "trace_row_count": int(len(crossarm_trace_df)),
    "stage_summary_row_count": int(len(crossarm_stage_summary_df)),
    "saved_image_row_count": int(len(crossarm_saved_images_df)),
}

if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print("CELL 16D table assembly complete.")
    print(f"  input ROI rows              : {cell16_output_counts['input_roi_count']}")
    print(f"  success image result rows   : {cell16_output_counts['success_image_row_count']}")
    print(f"  failure rows                : {cell16_output_counts['failure_row_count']}")
    print(f"  final detection rows        : {cell16_output_counts['final_detection_row_count']}")
    print(f"  trace rows                  : {cell16_output_counts['trace_row_count']}")
    print(f"  stage summary rows          : {cell16_output_counts['stage_summary_row_count']}")
    print(f"  saved image manifest rows   : {cell16_output_counts['saved_image_row_count']}")
    



# =============================================================================
# 16E. RECONCILIATION + CONSISTENCY CHECKS
# =============================================================================
# EXPLANATION:
# Validate that CELL 16 produced complete, consistent run-level outputs.
#
# IMPORTANT:
#   This section runs once after:
#       16C. Per-ROI batch loop
#       16D. Assemble run-level output tables
#
#   Do NOT place this section inside the ROI loop.
#
# RECONCILIATION RULE:
#   A successful ROI contributes one row to:
#       crossarm_image_results_df
#
#   A failed ROI contributes one row to:
#       crossarm_failures_df
#
#   Therefore:
#       len(crossarm_image_results_df) + len(crossarm_failures_df)
#       ==
#       len(cell16_roi_input_df)
#
#   Do NOT reconcile using crossarm_final_detections_df because a successful ROI
#   can legitimately have zero final crossarms.
#
# IMAGE-SAVE RULE:
#   final_image_save_failed=True does NOT mean the ROI failed.
#   It remains a successful ROI and is recorded in crossarm_image_results_df.
# =============================================================================


# =============================================================================
# 16.40 RECONCILIATION HELPER FUNCTIONS
# =============================================================================
# EXPLANATION:
# Small helpers used only by 16E to keep checks readable.
# =============================================================================

def required_dataframe(df_name):
    """
    Fetch and validate a required run-level DataFrame by name.

    Args:
        df_name:
            Name of a DataFrame expected in globals().

    Returns:
        pd.DataFrame:
            The validated DataFrame.
    """
    if df_name not in globals():
        raise NameError(
            f"CELL 16E expected {df_name} to exist after 16D."
        )

    df = globals()[df_name]

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{df_name} exists but is not a pandas DataFrame."
        )

    return df


def safe_int(value, default_value=0):
    """
    Convert a value to int with a safe default.

    Args:
        value:
            Value to convert.

        default_value:
            Value returned when conversion fails.

    Returns:
        int:
            Converted integer or default_value.
    """
    try:
        value_num = pd.to_numeric(value, errors="coerce")

        if pd.isna(value_num):
            return int(default_value)

        return int(value_num)

    except Exception:
        return int(default_value)


def string_id_set(df, id_col="image_id"):
    """
    Build a set of string IDs from a DataFrame column.

    Args:
        df:
            Input DataFrame.

        id_col:
            Column name containing IDs.

    Returns:
        set:
            Set of non-null string IDs.
    """
    if not isinstance(df, pd.DataFrame):
        return set()

    if id_col not in df.columns:
        return set()

    return set(
        df[id_col]
        .dropna()
        .astype(str)
        .tolist()
    )


def append_check(check_rows, check_name, passed, details, observed_count=0):
    """
    Append one reconciliation check row.

    Args:
        check_rows:
            List to append to.

        check_name:
            Name of the check.

        passed:
            Boolean check result.

        details:
            Human-readable detail string.

        observed_count:
            Optional count related to the check.

    Returns:
        None.
    """
    check_rows.append(
        {
            "run_id": RUN_ID,
            "run_timestamp": RUN_TIMESTAMP,
            "check_name": str(check_name),
            "check_status": "pass" if bool(passed) else "fail",
            "passed": bool(passed),
            "observed_count": int(observed_count),
            "details": str(details),
        }
    )


# =============================================================================
# 16.41 VALIDATE REQUIRED RUN-LEVEL DATAFRAMES
# =============================================================================
# EXPLANATION:
# Confirm 16D created all expected tables before running deeper checks.
# =============================================================================

crossarm_image_results_df = required_dataframe(
    "crossarm_image_results_df"
)

crossarm_final_detections_df = required_dataframe(
    "crossarm_final_detections_df"
)

crossarm_trace_df = required_dataframe(
    "crossarm_trace_df"
)

crossarm_failures_df = required_dataframe(
    "crossarm_failures_df"
)

crossarm_stage_summary_df = required_dataframe(
    "crossarm_stage_summary_df"
)

crossarm_saved_images_df = required_dataframe(
    "crossarm_saved_images_df"
)

if "cell16_output_counts" not in globals():
    raise NameError(
        "CELL 16E expected cell16_output_counts to exist after 16D."
    )

if not isinstance(cell16_output_counts, dict):
    raise TypeError(
        "cell16_output_counts exists but is not a dictionary."
    )


# =============================================================================
# 16.42 BASIC COUNT RECONCILIATION
# =============================================================================
# EXPLANATION:
# Check the primary run-level invariant:
#
#   success ROI rows + failure ROI rows == input ROI rows
# =============================================================================

cell16_reconciliation_rows = []

input_roi_count = int(len(cell16_roi_input_df))
success_image_row_count = int(len(crossarm_image_results_df))
failure_row_count = int(len(crossarm_failures_df))
final_detection_row_count = int(len(crossarm_final_detections_df))
trace_row_count = int(len(crossarm_trace_df))
stage_summary_row_count = int(len(crossarm_stage_summary_df))
saved_image_row_count = int(len(crossarm_saved_images_df))

reconciled_roi_count = int(success_image_row_count + failure_row_count)

append_check(
    check_rows=cell16_reconciliation_rows,
    check_name="success_plus_failure_equals_input",
    passed=(reconciled_roi_count == input_roi_count),
    details=(
        f"success_image_row_count={success_image_row_count}, "
        f"failure_row_count={failure_row_count}, "
        f"input_roi_count={input_roi_count}"
    ),
    observed_count=reconciled_roi_count,
)


# =============================================================================
# 16.43 IMAGE_ID SET RECONCILIATION
# =============================================================================
# EXPLANATION:
# Check that every input ROI image_id appears exactly once as either:
#   - success in crossarm_image_results_df
#   - failure in crossarm_failures_df
#
# Also check that success and failure sets do not overlap.
# =============================================================================

input_image_ids = string_id_set(
    cell16_roi_input_df,
    id_col="image_id",
)

success_image_ids = string_id_set(
    crossarm_image_results_df,
    id_col="image_id",
)

failure_image_ids = string_id_set(
    crossarm_failures_df,
    id_col="image_id",
)

output_image_ids = success_image_ids | failure_image_ids

missing_output_image_ids = sorted(
    list(input_image_ids - output_image_ids)
)

extra_output_image_ids = sorted(
    list(output_image_ids - input_image_ids)
)

overlap_success_failure_image_ids = sorted(
    list(success_image_ids & failure_image_ids)
)

append_check(
    check_rows=cell16_reconciliation_rows,
    check_name="all_input_image_ids_have_output",
    passed=(len(missing_output_image_ids) == 0),
    details=(
        "Missing output image_ids: "
        f"{missing_output_image_ids[:20]}"
    ),
    observed_count=len(missing_output_image_ids),
)

append_check(
    check_rows=cell16_reconciliation_rows,
    check_name="no_extra_output_image_ids",
    passed=(len(extra_output_image_ids) == 0),
    details=(
        "Extra output image_ids not present in input: "
        f"{extra_output_image_ids[:20]}"
    ),
    observed_count=len(extra_output_image_ids),
)

append_check(
    check_rows=cell16_reconciliation_rows,
    check_name="success_and_failure_image_ids_disjoint",
    passed=(len(overlap_success_failure_image_ids) == 0),
    details=(
        "Image_ids present in both success and failure outputs: "
        f"{overlap_success_failure_image_ids[:20]}"
    ),
    observed_count=len(overlap_success_failure_image_ids),
)


# =============================================================================
# 16.44 DUPLICATE ROW CHECKS
# =============================================================================
# EXPLANATION:
# Success/failure tables should contain at most one row per image_id.
# Final detections may contain many rows per image_id.
# =============================================================================

duplicate_success_image_ids = []

if (
    "image_id" in crossarm_image_results_df.columns
    and len(crossarm_image_results_df) > 0
):
    duplicate_success_image_ids = (
        crossarm_image_results_df
        .loc[
            crossarm_image_results_df["image_id"].astype(str).duplicated(),
            "image_id",
        ]
        .astype(str)
        .tolist()
    )

duplicate_failure_image_ids = []

if (
    "image_id" in crossarm_failures_df.columns
    and len(crossarm_failures_df) > 0
):
    duplicate_failure_image_ids = (
        crossarm_failures_df
        .loc[
            crossarm_failures_df["image_id"].astype(str).duplicated(),
            "image_id",
        ]
        .astype(str)
        .tolist()
    )

append_check(
    check_rows=cell16_reconciliation_rows,
    check_name="no_duplicate_success_image_rows",
    passed=(len(duplicate_success_image_ids) == 0),
    details=(
        "Duplicate success image_ids: "
        f"{duplicate_success_image_ids[:20]}"
    ),
    observed_count=len(duplicate_success_image_ids),
)

append_check(
    check_rows=cell16_reconciliation_rows,
    check_name="no_duplicate_failure_image_rows",
    passed=(len(duplicate_failure_image_ids) == 0),
    details=(
        "Duplicate failure image_ids: "
        f"{duplicate_failure_image_ids[:20]}"
    ),
    observed_count=len(duplicate_failure_image_ids),
)


# =============================================================================
# 16.45 RUN_ID CONSISTENCY CHECKS
# =============================================================================
# EXPLANATION:
# Every non-empty run-level output table should belong to the active RUN_ID.
# =============================================================================

run_id_check_tables = {
    "crossarm_image_results_df": crossarm_image_results_df,
    "crossarm_final_detections_df": crossarm_final_detections_df,
    "crossarm_trace_df": crossarm_trace_df,
    "crossarm_failures_df": crossarm_failures_df,
    "crossarm_stage_summary_df": crossarm_stage_summary_df,
    "crossarm_saved_images_df": crossarm_saved_images_df,
}

for table_name, table_df in run_id_check_tables.items():
    if (
        isinstance(table_df, pd.DataFrame)
        and len(table_df) > 0
        and "run_id" in table_df.columns
    ):
        table_run_ids = (
            table_df["run_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        table_run_id_ok = (
            len(table_run_ids) == 1
            and table_run_ids[0] == str(RUN_ID)
        )

        append_check(
            check_rows=cell16_reconciliation_rows,
            check_name=f"{table_name}_run_id_matches_active_run",
            passed=table_run_id_ok,
            details=(
                f"{table_name} run_ids={table_run_ids}, "
                f"active RUN_ID={RUN_ID}"
            ),
            observed_count=len(table_run_ids),
        )


# =============================================================================
# 16.46 FINAL DETECTION CONSISTENCY CHECKS
# =============================================================================
# EXPLANATION:
# Final detections must belong only to successful ROI rows.
#
# For every successful ROI:
#   final_crossarm_count in crossarm_image_results_df
#   must match the number of final detection rows for that image_id.
# =============================================================================

final_detection_image_ids = string_id_set(
    crossarm_final_detections_df,
    id_col="image_id",
)

final_detection_without_success_ids = sorted(
    list(final_detection_image_ids - success_image_ids)
)

append_check(
    check_rows=cell16_reconciliation_rows,
    check_name="final_detections_only_for_successful_rois",
    passed=(len(final_detection_without_success_ids) == 0),
    details=(
        "Final detection image_ids without success image row: "
        f"{final_detection_without_success_ids[:20]}"
    ),
    observed_count=len(final_detection_without_success_ids),
)

final_detection_counts_by_image_id = {}

if (
    len(crossarm_final_detections_df) > 0
    and "image_id" in crossarm_final_detections_df.columns
):
    final_detection_counts_by_image_id = (
        crossarm_final_detections_df["image_id"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

final_count_mismatch_rows = []

if (
    len(crossarm_image_results_df) > 0
    and "image_id" in crossarm_image_results_df.columns
    and "final_crossarm_count" in crossarm_image_results_df.columns
):
    for _, image_row in crossarm_image_results_df.iterrows():
        image_id_text = str(image_row["image_id"])

        expected_final_count = safe_int(
            image_row.get("final_crossarm_count", 0),
            default_value=0,
        )

        actual_final_count = int(
            final_detection_counts_by_image_id.get(
                image_id_text,
                0,
            )
        )

        if expected_final_count != actual_final_count:
            final_count_mismatch_rows.append(
                {
                    "image_id": image_id_text,
                    "expected_final_crossarm_count": int(expected_final_count),
                    "actual_final_detection_rows": int(actual_final_count),
                }
            )

append_check(
    check_rows=cell16_reconciliation_rows,
    check_name="final_detection_count_matches_image_results",
    passed=(len(final_count_mismatch_rows) == 0),
    details=(
        "Final count mismatch examples: "
        f"{final_count_mismatch_rows[:20]}"
    ),
    observed_count=len(final_count_mismatch_rows),
)


# =============================================================================
# 16.47 FINAL XARM LABEL CONSISTENCY CHECKS
# =============================================================================
# EXPLANATION:
# Within each successful ROI, final_xarm_number should be sequential:
#   1, 2, 3, ...
#
# final_xarm_label should not duplicate within the same image_id.
# =============================================================================

xarm_number_sequence_errors = []
duplicate_final_label_rows = []

if len(crossarm_final_detections_df) > 0:
    if all(
        col_name in crossarm_final_detections_df.columns
        for col_name in [
            "image_id",
            "final_xarm_number",
        ]
    ):
        for image_id_text, group_df in (
            crossarm_final_detections_df
            .assign(
                image_id_text=(
                    crossarm_final_detections_df["image_id"].astype(str)
                )
            )
            .groupby("image_id_text", sort=False)
        ):
            xarm_numbers = (
                pd.to_numeric(
                    group_df["final_xarm_number"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .sort_values()
                .tolist()
            )

            expected_numbers = list(
                range(
                    1,
                    len(group_df) + 1,
                )
            )

            if xarm_numbers != expected_numbers:
                xarm_number_sequence_errors.append(
                    {
                        "image_id": str(image_id_text),
                        "observed_numbers": xarm_numbers,
                        "expected_numbers": expected_numbers,
                    }
                )

    if all(
        col_name in crossarm_final_detections_df.columns
        for col_name in [
            "image_id",
            "final_xarm_label",
        ]
    ):
        duplicated_label_mask = (
            crossarm_final_detections_df[
                [
                    "image_id",
                    "final_xarm_label",
                ]
            ]
            .astype(str)
            .duplicated()
        )

        if duplicated_label_mask.any():
            duplicate_final_label_rows = (
                crossarm_final_detections_df
                .loc[
                    duplicated_label_mask,
                    [
                        "image_id",
                        "final_xarm_label",
                    ],
                ]
                .astype(str)
                .head(20)
                .to_dict("records")
            )

append_check(
    check_rows=cell16_reconciliation_rows,
    check_name="final_xarm_numbers_are_sequential_per_image",
    passed=(len(xarm_number_sequence_errors) == 0),
    details=(
        "Xarm number sequence error examples: "
        f"{xarm_number_sequence_errors[:20]}"
    ),
    observed_count=len(xarm_number_sequence_errors),
)

append_check(
    check_rows=cell16_reconciliation_rows,
    check_name="no_duplicate_final_xarm_labels_per_image",
    passed=(len(duplicate_final_label_rows) == 0),
    details=(
        "Duplicate final_xarm_label examples: "
        f"{duplicate_final_label_rows[:20]}"
    ),
    observed_count=len(duplicate_final_label_rows),
)


# =============================================================================
# 16.48 SAVED IMAGE MANIFEST CONSISTENCY CHECKS
# =============================================================================
# EXPLANATION:
# Saved image rows must belong only to successful ROI rows.
#
# The number of saved image manifest rows for each successful ROI should match
# final_images_saved_count in crossarm_image_results_df.
# =============================================================================

saved_image_ids = string_id_set(
    crossarm_saved_images_df,
    id_col="image_id",
)

saved_images_without_success_ids = sorted(
    list(saved_image_ids - success_image_ids)
)

append_check(
    check_rows=cell16_reconciliation_rows,
    check_name="saved_images_only_for_successful_rois",
    passed=(len(saved_images_without_success_ids) == 0),
    details=(
        "Saved-image image_ids without success image row: "
        f"{saved_images_without_success_ids[:20]}"
    ),
    observed_count=len(saved_images_without_success_ids),
)

duplicate_saved_image_rows = []

if (
    len(crossarm_saved_images_df) > 0
    and all(
        col_name in crossarm_saved_images_df.columns
        for col_name in [
            "image_id",
            "image_name",
        ]
    )
):
    duplicated_saved_mask = (
        crossarm_saved_images_df[
            [
                "image_id",
                "image_name",
            ]
        ]
        .astype(str)
        .duplicated()
    )

    if duplicated_saved_mask.any():
        duplicate_saved_image_rows = (
            crossarm_saved_images_df
            .loc[
                duplicated_saved_mask,
                [
                    "image_id",
                    "image_name",
                ],
            ]
            .astype(str)
            .head(20)
            .to_dict("records")
        )

append_check(
    check_rows=cell16_reconciliation_rows,
    check_name="no_duplicate_saved_image_manifest_rows",
    passed=(len(duplicate_saved_image_rows) == 0),
    details=(
        "Duplicate saved image rows: "
        f"{duplicate_saved_image_rows[:20]}"
    ),
    observed_count=len(duplicate_saved_image_rows),
)

saved_image_counts_by_image_id = {}

if (
    len(crossarm_saved_images_df) > 0
    and "image_id" in crossarm_saved_images_df.columns
):
    saved_image_counts_by_image_id = (
        crossarm_saved_images_df["image_id"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

saved_image_count_mismatch_rows = []
final_image_saved_flag_errors = []

if (
    len(crossarm_image_results_df) > 0
    and "image_id" in crossarm_image_results_df.columns
):
    for _, image_row in crossarm_image_results_df.iterrows():
        image_id_text = str(image_row["image_id"])

        actual_saved_image_count = int(
            saved_image_counts_by_image_id.get(
                image_id_text,
                0,
            )
        )

        recorded_saved_image_count = safe_int(
            image_row.get("final_images_saved_count", 0),
            default_value=0,
        )

        expected_final_image_count = safe_int(
            image_row.get("final_images_expected_count", 0),
            default_value=0,
        )

        final_images_requested = parse_bool(
            image_row.get("final_images_requested", False)
        )

        final_images_saved = parse_bool(
            image_row.get("final_images_saved", False)
        )

        final_image_save_failed = parse_bool(
            image_row.get("final_image_save_failed", False)
        )

        if actual_saved_image_count != recorded_saved_image_count:
            saved_image_count_mismatch_rows.append(
                {
                    "image_id": image_id_text,
                    "actual_saved_image_rows": int(actual_saved_image_count),
                    "recorded_final_images_saved_count": int(
                        recorded_saved_image_count
                    ),
                }
            )

        if not final_images_requested and actual_saved_image_count != 0:
            final_image_saved_flag_errors.append(
                {
                    "image_id": image_id_text,
                    "reason": "images_not_requested_but_manifest_rows_exist",
                    "actual_saved_image_rows": int(actual_saved_image_count),
                }
            )

        if final_images_saved:
            if (
                not final_images_requested
                or final_image_save_failed
                or actual_saved_image_count != expected_final_image_count
            ):
                final_image_saved_flag_errors.append(
                    {
                        "image_id": image_id_text,
                        "reason": "final_images_saved_flag_inconsistent",
                        "final_images_requested": bool(final_images_requested),
                        "final_image_save_failed": bool(final_image_save_failed),
                        "actual_saved_image_rows": int(actual_saved_image_count),
                        "expected_final_image_count": int(
                            expected_final_image_count
                        ),
                    }
                )

        if final_image_save_failed and final_images_saved:
            final_image_saved_flag_errors.append(
                {
                    "image_id": image_id_text,
                    "reason": "save_failed_but_final_images_saved_true",
                }
            )

append_check(
    check_rows=cell16_reconciliation_rows,
    check_name="saved_image_manifest_count_matches_image_results",
    passed=(len(saved_image_count_mismatch_rows) == 0),
    details=(
        "Saved image count mismatch examples: "
        f"{saved_image_count_mismatch_rows[:20]}"
    ),
    observed_count=len(saved_image_count_mismatch_rows),
)

append_check(
    check_rows=cell16_reconciliation_rows,
    check_name="final_image_saved_flags_are_consistent",
    passed=(len(final_image_saved_flag_errors) == 0),
    details=(
        "Final image saved flag error examples: "
        f"{final_image_saved_flag_errors[:20]}"
    ),
    observed_count=len(final_image_saved_flag_errors),
)


# =============================================================================
# 16.49 SAVED IMAGE FILE EXISTENCE CHECK
# =============================================================================
# EXPLANATION:
# Every row in crossarm_saved_images_df should point to a file that exists.
# =============================================================================

missing_saved_image_files = []

if (
    len(crossarm_saved_images_df) > 0
    and "image_path" in crossarm_saved_images_df.columns
):
    for _, saved_image_row in crossarm_saved_images_df.iterrows():
        saved_image_path = str(
            saved_image_row.get(
                "image_path",
                "",
            )
        ).strip()

        if saved_image_path == "" or saved_image_path.lower() in [
            "nan",
            "none",
            "null",
        ]:
            missing_saved_image_files.append(
                {
                    "image_id": str(saved_image_row.get("image_id", "")),
                    "image_name": str(saved_image_row.get("image_name", "")),
                    "image_path": saved_image_path,
                    "reason": "blank_image_path",
                }
            )

        elif not os.path.exists(saved_image_path):
            missing_saved_image_files.append(
                {
                    "image_id": str(saved_image_row.get("image_id", "")),
                    "image_name": str(saved_image_row.get("image_name", "")),
                    "image_path": saved_image_path,
                    "reason": "path_does_not_exist",
                }
            )

append_check(
    check_rows=cell16_reconciliation_rows,
    check_name="saved_image_files_exist",
    passed=(len(missing_saved_image_files) == 0),
    details=(
        "Missing saved image file examples: "
        f"{missing_saved_image_files[:20]}"
    ),
    observed_count=len(missing_saved_image_files),
)


# =============================================================================
# 16.50 FAILURE TABLE CONSISTENCY CHECKS
# =============================================================================
# EXPLANATION:
# Failure rows should have enough information to debug the failed ROI.
# =============================================================================

failure_rows_missing_required_text = []

if len(crossarm_failures_df) > 0:
    for _, failure_row in crossarm_failures_df.iterrows():
        image_id_text = str(
            failure_row.get(
                "image_id",
                "",
            )
        )

        failure_stage_text = str(
            failure_row.get(
                "failure_stage",
                "",
            )
        ).strip()

        error_message_text = str(
            failure_row.get(
                "error_message",
                "",
            )
        ).strip()

        if (
            failure_stage_text == ""
            or failure_stage_text.lower() in ["nan", "none", "null"]
            or error_message_text == ""
            or error_message_text.lower() in ["nan", "none", "null"]
        ):
            failure_rows_missing_required_text.append(
                {
                    "image_id": image_id_text,
                    "failure_stage": failure_stage_text,
                    "error_message": error_message_text,
                }
            )

append_check(
    check_rows=cell16_reconciliation_rows,
    check_name="failure_rows_have_stage_and_error_message",
    passed=(len(failure_rows_missing_required_text) == 0),
    details=(
        "Failure rows missing required text examples: "
        f"{failure_rows_missing_required_text[:20]}"
    ),
    observed_count=len(failure_rows_missing_required_text),
)


# =============================================================================
# 16.51 BUILD RECONCILIATION CHECK TABLE
# =============================================================================
# EXPLANATION:
# Store all reconciliation checks in a run-level table for saving in 16F.
# =============================================================================

cell16_reconciliation_checks_df = pd.DataFrame(
    cell16_reconciliation_rows
).reset_index(drop=True)

failed_cell16_reconciliation_checks_df = (
    cell16_reconciliation_checks_df[
        cell16_reconciliation_checks_df["passed"] == False
    ]
    .copy()
    .reset_index(drop=True)
)


# =============================================================================
# 16.52 FAIL FAST IF RECONCILIATION FAILED
# =============================================================================
# EXPLANATION:
# Raise after all checks have run so the error message includes all failures.
# =============================================================================

cell16_reconciliation_passed = bool(
    len(failed_cell16_reconciliation_checks_df) == 0
)

cell16_output_counts.update(
    {
        "reconciled_roi_count": int(reconciled_roi_count),
        "reconciliation_check_count": int(
            len(cell16_reconciliation_checks_df)
        ),
        "failed_reconciliation_check_count": int(
            len(failed_cell16_reconciliation_checks_df)
        ),
        "cell16_reconciliation_passed": bool(cell16_reconciliation_passed),
    }
)

if not cell16_reconciliation_passed:
    raise RuntimeError(
        "CELL 16E reconciliation failed.\n"
        "Please inspect failed_cell16_reconciliation_checks_df for details.\n\n"
        f"{failed_cell16_reconciliation_checks_df.to_string(index=False)}"
    )


# =============================================================================
# 16.53 RECONCILIATION SUMMARY
# =============================================================================
# EXPLANATION:
# Print a compact reconciliation summary when PRINT_CONFIG_SUMMARY is enabled.
# =============================================================================

if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print("CELL 16E reconciliation checks complete.")
    print(f"  input ROI rows             : {input_roi_count}")
    print(f"  success image result rows  : {success_image_row_count}")
    print(f"  failure rows               : {failure_row_count}")
    print(f"  reconciled ROI rows        : {reconciled_roi_count}")
    print(f"  final detection rows       : {final_detection_row_count}")
    print(f"  trace rows                 : {trace_row_count}")
    print(f"  stage summary rows         : {stage_summary_row_count}")
    print(f"  saved image rows           : {saved_image_row_count}")
    print(
        f"  reconciliation checks      : "
        f"{len(cell16_reconciliation_checks_df)}"
    )
    print(
        f"  failed checks              : "
        f"{len(failed_cell16_reconciliation_checks_df)}"
    )
    
    
# =============================================================================
# 16F. SAVE GOLD/SILVER OUTPUTS + FINAL SUMMARY
# =============================================================================
# EXPLANATION:
# Save the final CELL 16 run-level outputs after reconciliation has passed.
#
# IMPORTANT:
#   This section runs once after:
#       16C. Per-ROI batch loop
#       16D. Assemble run-level output tables
#       16E. Reconciliation + consistency checks
#
#   Do NOT place this section inside the ROI loop.
#
#   Final review images are already saved inside 16C / 16.34 using:
#       save_final_image(...)
#
#   This section saves only run-level tables using:
#       save_run_table(...)
#
# OUTPUT POLICY:
#   Gold tables:
#       - crossarm_image_results_df
#       - crossarm_final_detections_df
#       - crossarm_trace_df
#       - crossarm_failures_df
#       - crossarm_saved_images_df
#
#   Silver stage/audit tables:
#       - crossarm_stage_summary_df
#       - cell16_reconciliation_checks_df
#
# NOTE:
#   save_run_table() safely skips empty DataFrames and returns None.
# =============================================================================


# =============================================================================
# 16.54 VALIDATE SAVE PREREQUISITES
# =============================================================================
# EXPLANATION:
# Confirm that 16D/16E completed before saving outputs.
# =============================================================================

required_cell16f_objects = [
    "crossarm_image_results_df",
    "crossarm_final_detections_df",
    "crossarm_trace_df",
    "crossarm_failures_df",
    "crossarm_stage_summary_df",
    "crossarm_saved_images_df",
    "cell16_reconciliation_checks_df",
    "failed_cell16_reconciliation_checks_df",
    "cell16_reconciliation_passed",
    "cell16_output_counts",
]

missing_cell16f_objects = [
    object_name
    for object_name in required_cell16f_objects
    if object_name not in globals()
]

if missing_cell16f_objects:
    raise NameError(
        "CELL 16F requires 16D and 16E to complete before saving outputs.\n"
        f"Missing objects: {missing_cell16f_objects}"
    )

if not bool(cell16_reconciliation_passed):
    raise RuntimeError(
        "CELL 16F will not save outputs because CELL 16E reconciliation did "
        "not pass.\n"
        "Please inspect failed_cell16_reconciliation_checks_df first."
    )

if not isinstance(cell16_output_counts, dict):
    raise TypeError(
        "cell16_output_counts exists but is not a dictionary."
    )


# =============================================================================
# 16.55 SAVE GOLD OUTPUT TABLES
# =============================================================================
# EXPLANATION:
# Save final production crossarm / xarm output tables to the run-scoped Gold
# table directory.
#
# IMPORTANT:
#   save_run_table() returns None when a table is empty.
# =============================================================================

cell16_saved_paths = {}

if bool(CELL16_SAVE_GOLD_TABLES):
    cell16_saved_paths["crossarm_image_results"] = save_run_table(
        df=crossarm_image_results_df,
        out_dir=RUN_GOLD_TABLES_DIR,
        table_name="crossarm_image_results",
    )

    cell16_saved_paths["crossarm_final_detections"] = save_run_table(
        df=crossarm_final_detections_df,
        out_dir=RUN_GOLD_TABLES_DIR,
        table_name="crossarm_final_detections",
    )

    cell16_saved_paths["crossarm_trace"] = save_run_table(
        df=crossarm_trace_df,
        out_dir=RUN_GOLD_TABLES_DIR,
        table_name="crossarm_trace",
    )

    cell16_saved_paths["crossarm_failures"] = save_run_table(
        df=crossarm_failures_df,
        out_dir=RUN_GOLD_TABLES_DIR,
        table_name="crossarm_failures",
    )

    cell16_saved_paths["crossarm_saved_images"] = save_run_table(
        df=crossarm_saved_images_df,
        out_dir=RUN_GOLD_TABLES_DIR,
        table_name="crossarm_saved_images",
    )

else:
    cell16_saved_paths["crossarm_image_results"] = None
    cell16_saved_paths["crossarm_final_detections"] = None
    cell16_saved_paths["crossarm_trace"] = None
    cell16_saved_paths["crossarm_failures"] = None
    cell16_saved_paths["crossarm_saved_images"] = None


# =============================================================================
# 16.56 SAVE SILVER STAGE / AUDIT TABLES
# =============================================================================
# EXPLANATION:
# Save lightweight stage-summary and reconciliation-check tables to the
# run-scoped Silver stage table directory when enabled.
# =============================================================================

if bool(CELL16_SAVE_SILVER_STAGE_TABLES):
    cell16_saved_paths["crossarm_stage_summary"] = save_run_table(
        df=crossarm_stage_summary_df,
        out_dir=RUN_SILVER_STAGE_TABLES_DIR,
        table_name="crossarm_stage_summary",
    )

    cell16_saved_paths["cell16_reconciliation_checks"] = save_run_table(
        df=cell16_reconciliation_checks_df,
        out_dir=RUN_SILVER_STAGE_TABLES_DIR,
        table_name="cell16_reconciliation_checks",
    )

else:
    cell16_saved_paths["crossarm_stage_summary"] = None
    cell16_saved_paths["cell16_reconciliation_checks"] = None


# =============================================================================
# 16.57 VALIDATE SAVED TABLE PATHS
# =============================================================================
# EXPLANATION:
# For every table path returned by save_run_table(), confirm the file exists.
#
# NOTE:
#   None is valid when:
#     - saving is disabled
#     - the DataFrame was empty and save_run_table() skipped it
# =============================================================================

missing_saved_table_paths = []

for table_name, saved_path in cell16_saved_paths.items():
    if saved_path is None:
        continue

    saved_path_text = str(saved_path).strip()

    if saved_path_text == "":
        missing_saved_table_paths.append(
            {
                "table_name": str(table_name),
                "saved_path": saved_path_text,
                "reason": "blank_saved_path",
            }
        )

    elif not os.path.exists(saved_path_text):
        missing_saved_table_paths.append(
            {
                "table_name": str(table_name),
                "saved_path": saved_path_text,
                "reason": "path_does_not_exist",
            }
        )

if len(missing_saved_table_paths) > 0:
    missing_saved_table_paths_df = pd.DataFrame(
        missing_saved_table_paths
    )

    raise FileNotFoundError(
        "Some CELL 16 saved table paths do not exist on disk.\n"
        f"{missing_saved_table_paths_df.to_string(index=False)}"
    )


# =============================================================================
# 16.58 UPDATE FINAL OUTPUT COUNTS
# =============================================================================
# EXPLANATION:
# Add save-path and final table-save status into the compact run summary.
# =============================================================================

cell16_output_counts.update(
    {
        "gold_tables_save_enabled": bool(CELL16_SAVE_GOLD_TABLES),
        "silver_stage_tables_save_enabled": bool(
            CELL16_SAVE_SILVER_STAGE_TABLES
        ),
        "saved_table_count": int(
            sum(
                1
                for saved_path in cell16_saved_paths.values()
                if saved_path is not None
            )
        ),
        "saved_table_missing_path_count": int(
            len(missing_saved_table_paths)
        ),
        "cell16_save_completed": True,
    }
)


# =============================================================================
# 16.59 FINAL CELL 16 SUMMARY
# =============================================================================
# EXPLANATION:
# Print a compact final run summary when PRINT_CONFIG_SUMMARY is enabled.
# =============================================================================

if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print("CELL 16 completed successfully.")

    print("\nCELL 16 OUTPUT COUNTS")
    print("-" * 100)
    print(f"  input ROI rows                 : {cell16_output_counts['input_roi_count']}")
    print(f"  success image result rows      : {cell16_output_counts['success_image_row_count']}")
    print(f"  failure rows                   : {cell16_output_counts['failure_row_count']}")
    print(f"  reconciled ROI rows            : {cell16_output_counts['reconciled_roi_count']}")
    print(f"  final detection rows           : {cell16_output_counts['final_detection_row_count']}")
    print(f"  trace rows                     : {cell16_output_counts['trace_row_count']}")
    print(f"  stage summary rows             : {cell16_output_counts['stage_summary_row_count']}")
    print(f"  saved image manifest rows      : {cell16_output_counts['saved_image_row_count']}")
    print(f"  reconciliation checks          : {cell16_output_counts['reconciliation_check_count']}")
    print(f"  failed reconciliation checks   : {cell16_output_counts['failed_reconciliation_check_count']}")
    print(f"  saved table count              : {cell16_output_counts['saved_table_count']}")

    print("\nCELL 16 SAVE PATHS")
    print("-" * 100)

    for table_name, saved_path in cell16_saved_paths.items():
        print(f"  {table_name:<32}: {saved_path}")

    print("\nCELL 16 FINAL STATUS")
    print("-" * 100)
    print(f"  reconciliation passed          : {cell16_reconciliation_passed}")
    print(f"  save completed                 : {cell16_output_counts['cell16_save_completed']}")
    print(f"  Gold table dir                 : {RUN_GOLD_TABLES_DIR}")
    print(f"  Gold image dir                 : {RUN_GOLD_IMAGES_DIR}")
    print(f"  Silver stage table dir         : {RUN_SILVER_STAGE_TABLES_DIR}")


# =============================================================================
# 16.60 FINAL HARD OUTPUT CHECK
# =============================================================================
# EXPLANATION:
# Fail loudly if CELL 16 produced no successful ROI rows at all.
#
# IMPORTANT:
#   crossarm_final_detections_df is allowed to be empty because successful ROIs
#   can legitimately have zero final crossarms.
#
#   crossarm_image_results_df should not be empty unless every ROI failed.
# =============================================================================

if len(crossarm_image_results_df) == 0:
    raise RuntimeError(
        "CELL 16 completed, but crossarm_image_results_df is empty.\n"
        "This means no ROI completed successfully.\n"
        "Please inspect crossarm_failures_df and earlier CELL 16 logs."
    )
    
    
    
    
# =============================================================================
# CELL 17 — GENERIC INSULATOR CANDIDATE DETECTION
# =============================================================================
# OVERVIEW:
# This cell runs one generic insulator prompt over every selected fixed-canvas
# pole-top ROI created by CELL 14.
#
# CURRENT SCOPE:
#   1) read selected saved ROI images from pole_rois_df
#   2) run the generic SAM3 prompt: "electrical insulator"
#   3) retain all raw SAM3 detections returned at the configured text threshold
#   4) build a per-ROI int-keyed mask lookup for QA rendering only
#   5) discard masks before moving to the next ROI
#   6) save lightweight raw-candidate, image-result, and failure tables
#
# INTENTIONALLY NOT INCLUDED YET:
#   - no pole-overlap filtering
#   - no crossarm-overlap filtering
#   - no transformer-overlap filtering
#   - no containment suppression
#   - no duplicate suppression
#   - no mask merging
#   - no insulator type classification
#   - no tiled inference
#   - no saved mask files
#
# IMPORTANT:
# An insulator can legitimately touch or overlap a pole, crossarm, transformer,
# conductor, or mounting hardware. Overlap is therefore not a rejection rule at
# this raw candidate-discovery stage.
#
# INPUT:
#   pole_rois_df
#
# OUTPUTS:
#   insulator_roi_input_df
#   insulator_raw_detections_df
#   insulator_image_results_df
#   insulator_failures_df
#   insulator_output_counts
#   insulator_saved_paths
#
# MASK CONTRACT:
#   insulator_mask_lookup is per-ROI working state only.
#
#   It is keyed by int(orig_det_idx), contains boolean ROI-space masks with
#   shape (roi_h, roi_w), and is discarded before processing the next ROI.
# =============================================================================


# =============================================================================
# 17A. SAFETY CHECKS + ROI INPUT PREPARATION
# =============================================================================


# =============================================================================
# 17.1 REQUIRED SETUP CHECKS
# =============================================================================
# EXPLANATION:
# CELL 17 reads its detection and QA configuration from CELL 3B, its save
# controls from CELL 10B, and reuses the tested SAM3 output-normalisation and
# DataFrame assembly helpers already defined in CELL 16.
# =============================================================================

required_cell17_globals = [
    # Core libraries.
    "os",
    "gc",
    "time",
    "traceback",
    "pd",
    "np",
    "torch",
    "Image",
    "ImageDraw",

    # SAM3 runtime.
    "model",
    "processor",
    "DEVICE",

    # CELL 14 ROI manifest.
    "pole_rois_df",

    # Active production run identity.
    "RUN_ID",
    "RUN_TIMESTAMP",

    # Generic asset output folders from CELL 10.
    "SILVER_ASSET_PROMPT_RUNS",
    "SILVER_ASSET_OVERLAYS",

    # Save helpers from CELL 10B.
    "save_run_table",
    "make_safe_path_part",

    # Reused SAM3 output-normalisation helpers from CELL 16.
    "infer_num_detections",
    "normalize_boxes",
    "normalize_scores",
    "normalize_masks",

    # Reused DataFrame helpers from CELL 16.
    "records_to_dataframe",
    "sort_if_columns_exist",

    # Fixed ROI contract from CELL 3B.
    "FIXED_ROI_WIDTH",
    "FIXED_ROI_HEIGHT",

    # Insulator prompt settings from CELL 3B.
    "INSULATOR_PROMPT_TEXT",
    "INSULATOR_TEXT_THRESHOLD",

    # Insulator QA styling from CELL 3B.
    "INSULATOR_MASK_ALPHA",
    "INSULATOR_BOX_RGB",
    "INSULATOR_BOX_LINEWIDTH",
    "INSULATOR_MASK_RGB",
    "INSULATOR_LABEL_RGB",
    "INSULATOR_LABEL_BACKGROUND_RGB",
    "INSULATOR_LABEL_PADDING_PX",

    # Insulator save controls from CELL 10B.
    "SAVE_INSULATOR_RAW_TABLES",
    "SAVE_INSULATOR_QA_OVERLAYS",
]

missing_cell17_globals = [
    name
    for name in required_cell17_globals
    if name not in globals()
]

if missing_cell17_globals:
    raise NameError(
        "CELL 17 requires the production setup, CELL 3B configuration, "
        "CELL 10B save controls, CELL 14 ROI output, and CELL 16 helper "
        "definitions to be available.\n"
        f"Missing globals: {missing_cell17_globals}"
    )

if not isinstance(pole_rois_df, pd.DataFrame):
    raise TypeError(
        "pole_rois_df exists but is not a pandas DataFrame."
    )

if pole_rois_df.empty:
    raise ValueError(
        "pole_rois_df is empty.\n"
        "Please check CELL 14 before running CELL 17."
    )


# =============================================================================
# 17.2 VALIDATE REQUIRED ROI COLUMNS
# =============================================================================
# EXPLANATION:
# Validate only the CELL 14 columns required by the insulator branch and its
# production lineage.
#
# CELL 17 does not depend on CELL 16 crossarm-specific columns or processing
# state.
# =============================================================================

required_roi_columns = [
    "run_id",
    "run_timestamp",
    "processing_order",
    "selection_status",
    "image_id",
    "file_name",
    "roi_file_name",
    "roi_image_path",
    "roi_w",
    "roi_h",
]

missing_roi_columns = [
    column_name
    for column_name in required_roi_columns
    if column_name not in pole_rois_df.columns
]

if missing_roi_columns:
    raise ValueError(
        "pole_rois_df is missing columns required by CELL 17.\n"
        "Please check the CELL 14 output schema.\n"
        f"Missing columns: {missing_roi_columns}"
    )


# =============================================================================
# 17.3 VALIDATE RUN IDENTITY
# =============================================================================
# EXPLANATION:
# Confirm that pole_rois_df belongs to the active production run before saving
# insulator outputs under RUN_ID.
#
# This does not prevent repeated CELL 17 tuning runs while RUN_ID remains
# unchanged. It prevents stale ROI rows from being saved under a newly created
# run identity if CELL 10B is rerun without rebuilding CELL 14.
# =============================================================================

if pole_rois_df["run_id"].isna().any():
    raise RuntimeError(
        "pole_rois_df contains missing run_id values.\n"
        "Please check CELL 14 output."
    )

if pole_rois_df["run_timestamp"].isna().any():
    raise RuntimeError(
        "pole_rois_df contains missing run_timestamp values.\n"
        "Please check CELL 14 output."
    )

roi_run_ids = (
    pole_rois_df["run_id"]
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

if len(roi_run_ids) != 1:
    raise RuntimeError(
        "pole_rois_df must contain exactly one run_id.\n"
        f"Found run_ids: {roi_run_ids}"
    )

if roi_run_ids[0] != str(RUN_ID):
    raise RuntimeError(
        "pole_rois_df run_id does not match the active RUN_ID.\n\n"
        f"pole_rois_df run_id : {roi_run_ids[0]}\n"
        f"active RUN_ID       : {RUN_ID}\n\n"
        "CELL 10B may have been rerun after CELL 14.\n"
        "Rerun CELL 14 for the active run, or restore the intended RUN_ID."
    )


# =============================================================================
# 17.4 PREPARE INSULATOR ROI INPUT TABLE
# =============================================================================
# EXPLANATION:
# Prepare the insulator-branch input directly from pole_rois_df.
#
# This step:
#   1) keeps rows with a saved ROI image path
#   2) keeps selected-pole ROI rows only
#   3) validates and normalises required text columns
#   4) sorts rows deterministically
#   5) requires one selected ROI per image
#   6) prevents duplicate saved ROI paths
#   7) validates CELL 14 ROI filename integrity
#
# IMPORTANT:
# Use insulator_roi_input_df throughout CELL 17. Do not reuse the generic
# roi_input_df name from CELL 16.
# =============================================================================

insulator_roi_input_df = pole_rois_df.copy()

insulator_roi_input_df = insulator_roi_input_df[
    insulator_roi_input_df["roi_image_path"].notna()
].copy()

insulator_roi_input_df = insulator_roi_input_df[
    insulator_roi_input_df["selection_status"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("selected")
].copy()

if insulator_roi_input_df.empty:
    raise ValueError(
        "No usable selected-pole ROI rows were found in pole_rois_df.\n"
        "Please check CELL 14 output."
    )


# -----------------------------------------------------------------------------
# 17.4.1 Validate and normalise required text columns
# -----------------------------------------------------------------------------

required_text_columns = [
    "image_id",
    "file_name",
    "roi_file_name",
    "roi_image_path",
]

for column_name in required_text_columns:
    if insulator_roi_input_df[column_name].isna().any():
        raise ValueError(
            f"insulator_roi_input_df contains missing {column_name} values."
        )

    insulator_roi_input_df[column_name] = (
        insulator_roi_input_df[column_name]
        .astype(str)
        .str.strip()
    )

    invalid_text_mask = (
        insulator_roi_input_df[column_name].eq("")
        | insulator_roi_input_df[column_name].str.lower().eq("nan")
    )

    if invalid_text_mask.any():
        raise ValueError(
            "insulator_roi_input_df contains blank or invalid "
            f"{column_name} values."
        )


# -----------------------------------------------------------------------------
# 17.4.2 Sort selected ROI rows deterministically
# -----------------------------------------------------------------------------

insulator_roi_input_df = (
    insulator_roi_input_df
    .sort_values(
        by=["processing_order", "image_id"],
        kind="mergesort",
    )
    .reset_index(drop=True)
)


# -----------------------------------------------------------------------------
# 17.4.3 Validate selected ROI uniqueness
# -----------------------------------------------------------------------------

if insulator_roi_input_df["image_id"].duplicated().any():
    duplicate_image_ids = (
        insulator_roi_input_df
        .loc[
            insulator_roi_input_df["image_id"].duplicated(keep=False),
            "image_id",
        ]
        .drop_duplicates()
        .tolist()
    )

    raise RuntimeError(
        "CELL 17 assumes one selected ROI per image_id.\n"
        "insulator_roi_input_df contains duplicate image_id values.\n"
        f"Duplicate examples: {duplicate_image_ids[:10]}"
    )

if insulator_roi_input_df["roi_image_path"].duplicated().any():
    duplicate_roi_paths = (
        insulator_roi_input_df
        .loc[
            insulator_roi_input_df["roi_image_path"].duplicated(keep=False),
            "roi_image_path",
        ]
        .drop_duplicates()
        .tolist()
    )

    raise RuntimeError(
        "insulator_roi_input_df contains duplicate roi_image_path values.\n"
        f"Duplicate examples: {duplicate_roi_paths[:10]}"
    )

# CELL 14 is expected to produce a unique ROI filename for every image.
# This validates CELL 14 manifest integrity. It is not required to prevent
# overlay overwrites because CELL 17 output names also contain image_id.
if insulator_roi_input_df["roi_file_name"].duplicated().any():
    duplicate_roi_file_names = (
        insulator_roi_input_df
        .loc[
            insulator_roi_input_df["roi_file_name"].duplicated(keep=False),
            "roi_file_name",
        ]
        .drop_duplicates()
        .tolist()
    )

    raise RuntimeError(
        "insulator_roi_input_df contains duplicate roi_file_name values.\n"
        "CELL 14 is expected to produce a unique ROI filename for every image.\n"
        f"Duplicate examples: {duplicate_roi_file_names[:10]}"
    )


# =============================================================================
# 17.5 VALIDATE ROI FILES + FIXED-CANVAS METADATA
# =============================================================================
# EXPLANATION:
# Validate the complete selected ROI input before starting SAM3 inference.
#
# This checks:
#   1) every saved ROI path exists
#   2) ROI width and height metadata are numeric integers
#   3) every ROI row matches the active fixed-canvas dimensions
#
# The loaded PIL image dimensions are checked again inside the processing loop
# because the stored file contents may differ from their metadata.
# =============================================================================


# -----------------------------------------------------------------------------
# 17.5.1 Validate saved ROI files
# -----------------------------------------------------------------------------

missing_roi_files = [
    roi_path
    for roi_path in insulator_roi_input_df["roi_image_path"].tolist()
    if not os.path.isfile(roi_path)
]

if missing_roi_files:
    raise FileNotFoundError(
        "Some ROI image files listed in pole_rois_df do not exist.\n"
        f"Missing file examples: {missing_roi_files[:10]}"
    )


# -----------------------------------------------------------------------------
# 17.5.2 Validate ROI dimension columns
# -----------------------------------------------------------------------------

for column_name in ["roi_w", "roi_h"]:
    numeric_values = pd.to_numeric(
        insulator_roi_input_df[column_name],
        errors="coerce",
    )

    if numeric_values.isna().any():
        raise ValueError(
            "insulator_roi_input_df contains missing or non-numeric values in "
            f"{column_name}."
        )

    numeric_array = numeric_values.to_numpy(dtype=float)

    if not np.isclose(numeric_array, np.round(numeric_array)).all():
        raise ValueError(
            "insulator_roi_input_df contains non-integer values in "
            f"{column_name}."
        )

    insulator_roi_input_df[column_name] = numeric_values.astype(int)


# -----------------------------------------------------------------------------
# 17.5.3 Validate fixed-canvas ROI dimensions
# -----------------------------------------------------------------------------

invalid_roi_size_df = insulator_roi_input_df[
    (insulator_roi_input_df["roi_w"] != int(FIXED_ROI_WIDTH))
    | (insulator_roi_input_df["roi_h"] != int(FIXED_ROI_HEIGHT))
].copy()

if not invalid_roi_size_df.empty:
    invalid_roi_size_examples = (
        invalid_roi_size_df[
            [
                "image_id",
                "roi_w",
                "roi_h",
                "roi_image_path",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    raise ValueError(
        "Some ROI rows do not match the expected fixed ROI size.\n"
        f"Expected ROI size: {FIXED_ROI_WIDTH} x {FIXED_ROI_HEIGHT}\n"
        "Please rerun CELL 14 so saved ROI crops and metadata match the active "
        "CELL 3B configuration.\n"
        f"Examples:\n{invalid_roi_size_examples}"
    )


# =============================================================================
# 17.6 RESOLVE PRODUCTION INSULATOR CONFIG
# =============================================================================
# EXPLANATION:
# CELL 17 reads its prompt, text threshold, QA styling, and save controls from
# the notebook configuration cells.
#
# IMPORTANT:
#   Do not locally retune the insulator prompt or text threshold in CELL 17.
#   Tune the named constants in CELL 3B so the active detection configuration
#   remains visible and auditable.
#
#   Output-save behaviour is controlled by CELL 10B.
# =============================================================================

CELL17_RUN_DEVICE = str(DEVICE)

CELL17_PROMPT_TEXT = str(INSULATOR_PROMPT_TEXT).strip()
CELL17_TEXT_THRESHOLD = float(INSULATOR_TEXT_THRESHOLD)

CELL17_SAVE_RAW_TABLES = bool(SAVE_INSULATOR_RAW_TABLES)
CELL17_SAVE_QA_OVERLAYS = bool(SAVE_INSULATOR_QA_OVERLAYS)

# Continue processing the remaining ROIs when one ROI fails.
CELL17_STOP_ON_ROW_FAILURE = False


# =============================================================================
# 17.6.1 VALIDATE INSULATOR PROMPT + TEXT THRESHOLD
# =============================================================================

if len(CELL17_PROMPT_TEXT) == 0:
    raise ValueError(
        "INSULATOR_PROMPT_TEXT is empty."
    )

if not np.isfinite(CELL17_TEXT_THRESHOLD):
    raise ValueError(
        "INSULATOR_TEXT_THRESHOLD must be finite. "
        f"Got: {CELL17_TEXT_THRESHOLD}"
    )

if CELL17_TEXT_THRESHOLD < 0.0 or CELL17_TEXT_THRESHOLD > 1.0:
    raise ValueError(
        "INSULATOR_TEXT_THRESHOLD should be in [0, 1]. "
        f"Got: {CELL17_TEXT_THRESHOLD}"
    )


# =============================================================================
# 17.7 PREPARE RUN-SCOPED OUTPUT DIRECTORIES
# =============================================================================
# EXPLANATION:
# Derive Cell 17's run-scoped Silver output folders from the stable asset
# candidate roots created in CELL 10.
# =============================================================================

RUN_INSULATOR_PROMPT_TABLES_DIR = os.path.join(
    SILVER_ASSET_PROMPT_RUNS,
    RUN_ID,
    "generic_insulator",
    "tables",
)

RUN_INSULATOR_OVERLAYS_DIR = os.path.join(
    SILVER_ASSET_OVERLAYS,
    RUN_ID,
    "generic_insulator",
)

insulator_output_dirs = [
    RUN_INSULATOR_PROMPT_TABLES_DIR,
    RUN_INSULATOR_OVERLAYS_DIR,
]

for output_dir in insulator_output_dirs:
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

missing_insulator_output_dirs = [
    output_dir
    for output_dir in insulator_output_dirs
    if not os.path.isdir(output_dir)
]

if missing_insulator_output_dirs:
    raise RuntimeError(
        "Some CELL 17 output directories were not created successfully.\n"
        f"Missing directories: {missing_insulator_output_dirs}"
    )


# =============================================================================
# 17.8 MODEL / DEVICE SANITY CHECK
# =============================================================================
# EXPLANATION:
# Make sure the SAM3 model is in eval mode and CUDA is available before running
# insulator inference.
# =============================================================================

if hasattr(model, "eval"):
    model.eval()

if CELL17_RUN_DEVICE != "cuda":
    raise RuntimeError(
        "CELL 17 expects DEVICE='cuda' for SAM3 inference.\n"
        f"Current DEVICE: {CELL17_RUN_DEVICE}"
    )

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available before CELL 17 inference."
    )


# =============================================================================
# 17B. QA HELPER
# =============================================================================


# =============================================================================
# 17.9 RENDER RAW INSULATOR OVERLAY
# =============================================================================
# EXPLANATION:
# Render one lightweight QA image for the current ROI.
#
# The helper receives the current ROI's row dictionaries directly, avoiding a
# redundant per-ROI DataFrame construction.
#
# The mask blend is applied only to mask pixels instead of converting two full
# 2600 x 3500 RGB images to float32.
# =============================================================================

def render_raw_insulator_overlay(
    image,
    detection_records,
    mask_lookup,
):
    """
    Render raw generic-insulator masks, boxes, and scores over one ROI.

    Args:
        image:
            PIL RGB image for the current ROI.

        detection_records:
            List of current-ROI raw detection dictionaries.

        mask_lookup:
            Per-ROI dict keyed by int(orig_det_idx) containing boolean
            ROI-space masks.

    Returns:
        PIL.Image.Image:
            RGB overlay image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(
            "image must be a PIL Image."
        )

    if detection_records is None:
        detection_records = []

    if mask_lookup is None:
        mask_lookup = {}

    output_array = np.asarray(
        image.convert("RGB"),
        dtype=np.uint8,
    ).copy()

    combined_mask = np.zeros(
        output_array.shape[:2],
        dtype=bool,
    )

    for mask in mask_lookup.values():
        if (
            isinstance(mask, np.ndarray)
            and mask.ndim == 2
            and mask.shape == combined_mask.shape
            and mask.any()
        ):
            combined_mask |= mask

    if combined_mask.any():
        source_pixels = output_array[combined_mask].astype(
            np.float32,
            copy=False,
        )

        mask_colour = np.asarray(
            INSULATOR_MASK_RGB,
            dtype=np.float32,
        )

        blended_pixels = (
            source_pixels * (1.0 - float(INSULATOR_MASK_ALPHA))
            + mask_colour * float(INSULATOR_MASK_ALPHA)
        )

        output_array[combined_mask] = np.clip(
            blended_pixels,
            0,
            255,
        ).astype(np.uint8)

    output_image = Image.fromarray(output_array)
    draw = ImageDraw.Draw(output_image)

    label_padding_px = int(INSULATOR_LABEL_PADDING_PX)

    for detection_row in detection_records:
        orig_det_idx = int(detection_row["orig_det_idx"])

        x1 = int(round(float(detection_row["x1"])))
        y1 = int(round(float(detection_row["y1"])))
        x2 = int(round(float(detection_row["x2"])))
        y2 = int(round(float(detection_row["y2"])))

        score = float(detection_row["score"])

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=INSULATOR_BOX_RGB,
            width=int(INSULATOR_BOX_LINEWIDTH),
        )

        label_text = f"insulator {orig_det_idx} {score:.3f}"

        text_bbox = draw.textbbox(
            (0, 0),
            label_text,
        )

        text_w = int(text_bbox[2] - text_bbox[0])
        text_h = int(text_bbox[3] - text_bbox[1])

        label_y1 = max(
            0,
            y1 - text_h - (2 * label_padding_px),
        )

        draw.rectangle(
            [
                x1,
                label_y1,
                x1 + text_w + (2 * label_padding_px),
                label_y1 + text_h + (2 * label_padding_px),
            ],
            fill=INSULATOR_LABEL_BACKGROUND_RGB,
        )

        draw.text(
            (
                x1 + label_padding_px,
                label_y1 + label_padding_px,
            ),
            label_text,
            fill=INSULATOR_LABEL_RGB,
        )

    return output_image


# =============================================================================
# 17C. PER-ROI BATCH LOOP
# =============================================================================


# =============================================================================
# 17.10 INITIALISE RUN-LEVEL ACCUMULATORS
# =============================================================================
# EXPLANATION:
# Only lightweight dictionaries are retained across ROIs.
#
# Full-resolution masks remain per-ROI working state and are never accumulated
# across the batch.
# =============================================================================

insulator_raw_detection_rows = []
insulator_image_result_rows = []
insulator_failure_rows = []

insulator_valid_mask_count = 0


# =============================================================================
# 17.11 PROCESS EVERY SELECTED SAVED ROI
# =============================================================================

for _, roi_row in insulator_roi_input_df.iterrows():
    roi_start_time = time.time()
    current_stage = "17.11_initialise_roi"

    # Per-ROI heavy working state.
    image = None
    state = None
    raw_boxes = None
    raw_scores = None
    raw_masks = None
    boxes = None
    scores = None
    masks_2d = None
    mask_i = None
    mask_bool = None
    overlay_image = None
    current_roi_detection_rows = None
    current_roi_mask_pixel_counts = None
    insulator_mask_lookup = {}

    try:
        # ---------------------------------------------------------------------
        # 17.11.1 Extract stable ROI identity fields
        # ---------------------------------------------------------------------
        image_id = str(roi_row["image_id"])
        file_name = str(roi_row["file_name"])
        roi_file_name = str(roi_row["roi_file_name"])
        roi_image_path = str(roi_row["roi_image_path"])
        processing_order = roi_row.get("processing_order", np.nan)

        roi_w = int(roi_row["roi_w"])
        roi_h = int(roi_row["roi_h"])


        # =====================================================================
        # 17.11.2 LOAD SAVED CLEAN ROI
        # =====================================================================

        current_stage = "17.11.2_load_saved_roi"

        with Image.open(roi_image_path) as loaded_image:
            if loaded_image.mode != "RGB":
                image = loaded_image.convert("RGB")
            else:
                image = loaded_image.copy()

            image.load()

        loaded_roi_w, loaded_roi_h = image.size

        if int(loaded_roi_w) != roi_w or int(loaded_roi_h) != roi_h:
            raise ValueError(
                "Loaded ROI dimensions do not match "
                "insulator_roi_input_df metadata.\n"
                f"roi_image_path : {roi_image_path}\n"
                f"metadata size  : {roi_w} x {roi_h}\n"
                f"loaded size    : {loaded_roi_w} x {loaded_roi_h}"
            )


        # =====================================================================
        # 17.11.3 RUN GENERIC SAM3 INSULATOR PROMPT
        # =====================================================================
        # EXPLANATION:
        # Preserve the tested stateful processor path used by CELL 16:
        #   set_image
        #   reset_all_prompts
        #   set_text_prompt
        #
        # PRODUCTION RULE:
        #   No plot_results diagnostic.
        #   No plt.show().
        #   No per-ROI prints.
        # =====================================================================

        current_stage = "17.11.3_run_generic_insulator_prompt"

        if hasattr(processor, "device"):
            processor.device = CELL17_RUN_DEVICE

        if hasattr(processor, "set_confidence_threshold"):
            processor.set_confidence_threshold(CELL17_TEXT_THRESHOLD)

        state = {}

        state = processor.set_image(
            image,
            state=state,
        )

        reset_result = processor.reset_all_prompts(state)

        if reset_result is not None:
            state = reset_result

        state = processor.set_text_prompt(
            CELL17_PROMPT_TEXT,
            state,
        )

        raw_boxes = state.get("boxes", None)
        raw_scores = state.get("scores", None)
        raw_masks = state.get("masks", None)


        # =====================================================================
        # 17.11.4 NORMALISE OUTPUTS + BUILD PER-ROI MASK LOOKUP
        # =====================================================================
        # EXPLANATION:
        # Convert raw SAM3 outputs into standard arrays/lists:
        #   - boxes    -> shape (N, 4)
        #   - scores   -> shape (N,)
        #   - masks_2d -> list of 2D boolean masks or None
        #
        # insulator_mask_lookup is keyed by int(det_idx), which is also preserved
        # as orig_det_idx in the raw candidate table.
        # =====================================================================

        current_stage = "17.11.4_normalise_outputs"

        num_detections = infer_num_detections(
            raw_boxes=raw_boxes,
            raw_scores=raw_scores,
            raw_masks=raw_masks,
        )

        boxes = normalize_boxes(
            boxes=raw_boxes,
            num_detections=num_detections,
        )

        scores = normalize_scores(
            scores=raw_scores,
            num_detections=num_detections,
        )

        masks_2d = normalize_masks(
            raw_masks=raw_masks,
            num_detections=num_detections,
            image_h=roi_h,
            image_w=roi_w,
        )

        current_roi_mask_pixel_counts = {}

        for det_idx in range(num_detections):
            mask_i = (
                masks_2d[det_idx]
                if det_idx < len(masks_2d)
                else None
            )

            mask_pixel_count = 0

            if (
                isinstance(mask_i, np.ndarray)
                and mask_i.ndim == 2
                and mask_i.shape == (roi_h, roi_w)
            ):
                mask_bool = mask_i.astype(
                    bool,
                    copy=False,
                )

                mask_pixel_count = int(
                    np.count_nonzero(mask_bool)
                )

                if mask_pixel_count > 0:
                    insulator_mask_lookup[int(det_idx)] = mask_bool

            current_roi_mask_pixel_counts[int(det_idx)] = int(
                mask_pixel_count
            )


        # =====================================================================
        # 17.11.5 BUILD RAW CANDIDATE ROWS
        # =====================================================================
        # EXPLANATION:
        # orig_det_idx preserves the raw SAM3 detection index and is the per-ROI
        # mask lookup key.
        #
        # No sorting, filtering, or reindexing changes this identity.
        # =====================================================================

        current_stage = "17.11.5_build_raw_candidate_rows"

        current_roi_detection_rows = []

        for det_idx in range(num_detections):
            raw_box = boxes[det_idx]

            x1 = float(np.clip(raw_box[0], 0, roi_w))
            y1 = float(np.clip(raw_box[1], 0, roi_h))
            x2 = float(np.clip(raw_box[2], 0, roi_w))
            y2 = float(np.clip(raw_box[3], 0, roi_h))

            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)

            current_roi_detection_rows.append(
                {
                    "run_id": RUN_ID,
                    "run_timestamp": RUN_TIMESTAMP,
                    "image_id": image_id,
                    "file_name": file_name,
                    "roi_file_name": roi_file_name,
                    "roi_image_path": roi_image_path,
                    "processing_order": processing_order,

                    "asset_class": "generic_insulator",
                    "prompt_text": CELL17_PROMPT_TEXT,

                    "orig_det_idx": int(det_idx),
                    "score": float(scores[det_idx]),

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,

                    "box_w": box_w,
                    "box_h": box_h,
                    "box_area": float(box_w * box_h),
                    "box_cx": float((x1 + x2) / 2.0),
                    "box_cy": float((y1 + y2) / 2.0),

                    "has_mask": bool(
                        int(det_idx) in insulator_mask_lookup
                    ),

                    "mask_pixel_count": int(
                        current_roi_mask_pixel_counts.get(
                            int(det_idx),
                            0,
                        )
                    ),
                }
            )


        # =====================================================================
        # 17.11.6 CREATE OPTIONAL RAW QA OVERLAY
        # =====================================================================
        # EXPLANATION:
        # Overlay failure is non-fatal. Detection rows still commit successfully.
        #
        # No overlay is saved when num_detections == 0 because that would only
        # duplicate the clean ROI already stored by CELL 14.
        # =====================================================================

        current_stage = "17.11.6_create_raw_qa_overlay"

        overlay_requested = bool(
            CELL17_SAVE_QA_OVERLAYS
            and num_detections > 0
        )

        overlay_saved = False
        overlay_path = ""
        overlay_save_failed = False
        overlay_save_error = ""

        if overlay_requested:
            try:
                overlay_image = render_raw_insulator_overlay(
                    image=image,
                    detection_records=current_roi_detection_rows,
                    mask_lookup=insulator_mask_lookup,
                )

                safe_image_id = make_safe_path_part(image_id)

                safe_roi_name = make_safe_path_part(
                    os.path.splitext(roi_file_name)[0]
                )

                overlay_file_name = (
                    f"{safe_image_id}"
                    f"__{safe_roi_name}"
                    "__generic_insulator_raw.png"
                )

                overlay_path = os.path.join(
                    RUN_INSULATOR_OVERLAYS_DIR,
                    overlay_file_name,
                )

                overlay_image.save(
                    overlay_path,
                    format="PNG",
                )

                overlay_saved = True

            except Exception as overlay_exc:
                overlay_saved = False
                overlay_path = ""
                overlay_save_failed = True
                overlay_save_error = (
                    f"{type(overlay_exc).__name__}: {overlay_exc}"
                )


        # =====================================================================
        # 17.11.7 COMMIT SUCCESSFUL ROI
        # =====================================================================
        # EXPLANATION:
        # Build all lightweight rows first, then commit them together at the end
        # of the successful ROI path.
        #
        # This prevents detection rows from a failed ROI leaking into the final
        # run-level candidate table.
        # =====================================================================

        current_stage = "17.11.7_commit_successful_roi"

        roi_processing_seconds = float(
            time.time() - roi_start_time
        )

        current_image_result_row = {
            "run_id": RUN_ID,
            "run_timestamp": RUN_TIMESTAMP,
            "image_id": image_id,
            "file_name": file_name,
            "roi_file_name": roi_file_name,
            "roi_image_path": roi_image_path,
            "processing_order": processing_order,

            "processing_status": (
                "raw_candidates_found"
                if num_detections > 0
                else "no_raw_candidates"
            ),

            "roi_w": roi_w,
            "roi_h": roi_h,
            "processing_seconds": roi_processing_seconds,

            "prompt_text": CELL17_PROMPT_TEXT,
            "text_threshold": float(CELL17_TEXT_THRESHOLD),

            "raw_detection_count": int(num_detections),

            "detections_with_mask_count": int(
                len(insulator_mask_lookup)
            ),

            "overlay_requested": bool(overlay_requested),
            "overlay_saved": bool(overlay_saved),
            "overlay_path": str(overlay_path),
            "overlay_save_failed": bool(overlay_save_failed),
            "overlay_save_error": str(overlay_save_error),
        }

        insulator_raw_detection_rows.extend(
            current_roi_detection_rows
        )

        insulator_image_result_rows.append(
            current_image_result_row
        )

        insulator_valid_mask_count += int(
            len(insulator_mask_lookup)
        )

    except Exception as exc:
        insulator_failure_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "image_id": str(roi_row.get("image_id", "")),
                "file_name": str(roi_row.get("file_name", "")),
                "roi_file_name": str(roi_row.get("roi_file_name", "")),
                "roi_image_path": str(roi_row.get("roi_image_path", "")),
                "processing_order": roi_row.get(
                    "processing_order",
                    np.nan,
                ),

                "failure_stage": current_stage,
                "failure_reason": "cell17_roi_processing_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_traceback": traceback.format_exc(),
            }
        )

        if CELL17_STOP_ON_ROW_FAILURE:
            raise

        continue

    finally:
        # Release all per-ROI heavy state. No full-resolution mask survives into
        # the next ROI iteration.
        image = None
        state = None
        raw_boxes = None
        raw_scores = None
        raw_masks = None
        boxes = None
        scores = None
        masks_2d = None
        mask_i = None
        mask_bool = None
        overlay_image = None
        current_roi_detection_rows = None
        current_roi_mask_pixel_counts = None
        insulator_mask_lookup = None


# Release any final Python / CUDA references once after the full batch.
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# =============================================================================
# 17D. ASSEMBLE RUN-LEVEL OUTPUT TABLES
# =============================================================================


# =============================================================================
# 17.12 DEFINE OUTPUT SCHEMAS
# =============================================================================

insulator_raw_detection_cols = [
    "run_id",
    "run_timestamp",
    "image_id",
    "file_name",
    "roi_file_name",
    "roi_image_path",
    "processing_order",

    "asset_class",
    "prompt_text",

    "orig_det_idx",
    "score",

    "x1",
    "y1",
    "x2",
    "y2",

    "box_w",
    "box_h",
    "box_area",
    "box_cx",
    "box_cy",

    "has_mask",
    "mask_pixel_count",
]

insulator_image_result_cols = [
    "run_id",
    "run_timestamp",
    "image_id",
    "file_name",
    "roi_file_name",
    "roi_image_path",
    "processing_order",

    "processing_status",

    "roi_w",
    "roi_h",
    "processing_seconds",

    "prompt_text",
    "text_threshold",

    "raw_detection_count",
    "detections_with_mask_count",

    "overlay_requested",
    "overlay_saved",
    "overlay_path",
    "overlay_save_failed",
    "overlay_save_error",
]

insulator_failure_cols = [
    "run_id",
    "run_timestamp",
    "image_id",
    "file_name",
    "roi_file_name",
    "roi_image_path",
    "processing_order",

    "failure_stage",
    "failure_reason",
    "error_type",
    "error_message",
    "error_traceback",
]


# =============================================================================
# 17.13 BUILD + SORT OUTPUT DATAFRAMES
# =============================================================================

insulator_raw_detections_df = records_to_dataframe(
    records=insulator_raw_detection_rows,
    expected_columns=insulator_raw_detection_cols,
)

insulator_image_results_df = records_to_dataframe(
    records=insulator_image_result_rows,
    expected_columns=insulator_image_result_cols,
)

insulator_failures_df = records_to_dataframe(
    records=insulator_failure_rows,
    expected_columns=insulator_failure_cols,
)

insulator_raw_detections_df = sort_if_columns_exist(
    df=insulator_raw_detections_df,
    sort_columns=[
        "processing_order",
        "image_id",
        "roi_file_name",
        "orig_det_idx",
    ],
)

insulator_image_results_df = sort_if_columns_exist(
    df=insulator_image_results_df,
    sort_columns=[
        "processing_order",
        "image_id",
        "roi_file_name",
    ],
)

insulator_failures_df = sort_if_columns_exist(
    df=insulator_failures_df,
    sort_columns=[
        "processing_order",
        "image_id",
        "roi_file_name",
    ],
)


# =============================================================================
# 17.14 BUILD OUTPUT COUNTS
# =============================================================================

insulator_output_counts = {
    "roi_input_count": int(len(insulator_roi_input_df)),

    "successful_roi_count": int(
        len(insulator_image_results_df)
    ),

    "failed_roi_count": int(
        len(insulator_failures_df)
    ),

    "raw_detection_count": int(
        len(insulator_raw_detections_df)
    ),

    "detections_with_mask_count": int(
        insulator_raw_detections_df["has_mask"]
        .fillna(False)
        .astype(bool)
        .sum()
        if not insulator_raw_detections_df.empty
        else 0
    ),

    "valid_mask_count_accumulated": int(
        insulator_valid_mask_count
    ),

    "overlay_requested_count": int(
        insulator_image_results_df["overlay_requested"]
        .fillna(False)
        .astype(bool)
        .sum()
        if not insulator_image_results_df.empty
        else 0
    ),

    "overlay_saved_count": int(
        insulator_image_results_df["overlay_saved"]
        .fillna(False)
        .astype(bool)
        .sum()
        if not insulator_image_results_df.empty
        else 0
    ),

    "overlay_save_failure_count": int(
        insulator_image_results_df["overlay_save_failed"]
        .fillna(False)
        .astype(bool)
        .sum()
        if not insulator_image_results_df.empty
        else 0
    ),
}


# =============================================================================
# 17E. RECONCILIATION + CONSISTENCY CHECKS
# =============================================================================


# =============================================================================
# 17.15 RECONCILE ROI OUTCOMES
# =============================================================================

expected_processed_roi_count = (
    insulator_output_counts["successful_roi_count"]
    + insulator_output_counts["failed_roi_count"]
)

if expected_processed_roi_count != insulator_output_counts["roi_input_count"]:
    raise RuntimeError(
        "CELL 17 ROI reconciliation failed.\n"
        f"Input ROIs : {insulator_output_counts['roi_input_count']}\n"
        f"Successful : {insulator_output_counts['successful_roi_count']}\n"
        f"Failed     : {insulator_output_counts['failed_roi_count']}"
    )

successful_image_ids = set(
    insulator_image_results_df["image_id"]
    .astype(str)
    .tolist()
)

failed_image_ids = set(
    insulator_failures_df["image_id"]
    .astype(str)
    .tolist()
)

overlapping_outcome_image_ids = sorted(
    successful_image_ids.intersection(
        failed_image_ids
    )
)

if overlapping_outcome_image_ids:
    raise RuntimeError(
        "CELL 17 outcome reconciliation failed.\n"
        "Some image_id values appear in both success and failure tables.\n"
        f"Examples: {overlapping_outcome_image_ids[:20]}"
    )


# =============================================================================
# 17.16 RECONCILE RAW DETECTION + MASK COUNTS
# =============================================================================

recorded_raw_detection_count = int(
    pd.to_numeric(
        insulator_image_results_df["raw_detection_count"],
        errors="coerce",
    )
    .fillna(0)
    .sum()
    if not insulator_image_results_df.empty
    else 0
)

if recorded_raw_detection_count != insulator_output_counts["raw_detection_count"]:
    raise RuntimeError(
        "CELL 17 raw-detection reconciliation failed.\n"
        f"Detection table rows       : "
        f"{insulator_output_counts['raw_detection_count']}\n"
        f"Image-result recorded total: {recorded_raw_detection_count}"
    )

recorded_mask_count = int(
    pd.to_numeric(
        insulator_image_results_df["detections_with_mask_count"],
        errors="coerce",
    )
    .fillna(0)
    .sum()
    if not insulator_image_results_df.empty
    else 0
)

if not (
    insulator_output_counts["detections_with_mask_count"]
    == recorded_mask_count
    == insulator_output_counts["valid_mask_count_accumulated"]
):
    raise RuntimeError(
        "CELL 17 mask reconciliation failed.\n"
        f"Detection rows with masks : "
        f"{insulator_output_counts['detections_with_mask_count']}\n"
        f"Image-result mask total   : {recorded_mask_count}\n"
        f"Accumulated mask count    : "
        f"{insulator_output_counts['valid_mask_count_accumulated']}"
    )

raw_detection_image_ids = set(
    insulator_raw_detections_df["image_id"]
    .astype(str)
    .tolist()
)

raw_rows_without_success = sorted(
    raw_detection_image_ids.difference(
        successful_image_ids
    )
)

if raw_rows_without_success:
    raise RuntimeError(
        "CELL 17 raw-detection lineage reconciliation failed.\n"
        "Raw detections exist for image_id values without a successful ROI row.\n"
        f"Examples: {raw_rows_without_success[:20]}"
    )


# =============================================================================
# 17.17 VALIDATE SAVED QA OVERLAYS
# =============================================================================

saved_overlay_paths = (
    insulator_image_results_df.loc[
        insulator_image_results_df["overlay_saved"]
        .fillna(False)
        .astype(bool),
        "overlay_path",
    ]
    .astype(str)
    .tolist()
    if not insulator_image_results_df.empty
    else []
)

missing_saved_overlay_paths = [
    overlay_path
    for overlay_path in saved_overlay_paths
    if overlay_path == ""
    or not os.path.isfile(overlay_path)
]

if missing_saved_overlay_paths:
    raise RuntimeError(
        "CELL 17 overlay reconciliation failed.\n"
        "Some rows marked overlay_saved=True do not point to an existing file.\n"
        f"Missing path examples: {missing_saved_overlay_paths[:10]}"
    )


# =============================================================================
# 17F. SAVE OUTPUTS + FINAL SUMMARY
# =============================================================================


# =============================================================================
# 17.18 SAVE CELL 17 TABLES
# =============================================================================
# EXPLANATION:
# Save only after all reconciliation checks have passed.
#
# save_run_table skips empty DataFrames and returns None.
# =============================================================================

insulator_saved_paths = {
    "insulator_raw_detections": None,
    "insulator_image_results": None,
    "insulator_failures": None,
}

if CELL17_SAVE_RAW_TABLES:
    insulator_saved_paths["insulator_raw_detections"] = save_run_table(
        df=insulator_raw_detections_df,
        out_dir=RUN_INSULATOR_PROMPT_TABLES_DIR,
        table_name="insulator_raw_detections",
    )

    insulator_saved_paths["insulator_image_results"] = save_run_table(
        df=insulator_image_results_df,
        out_dir=RUN_INSULATOR_PROMPT_TABLES_DIR,
        table_name="insulator_image_results",
    )

    insulator_saved_paths["insulator_failures"] = save_run_table(
        df=insulator_failures_df,
        out_dir=RUN_INSULATOR_PROMPT_TABLES_DIR,
        table_name="insulator_failures",
    )


# =============================================================================
# 17.19 VALIDATE SAVED TABLE PATHS
# =============================================================================

insulator_tables_by_name = {
    "insulator_raw_detections": insulator_raw_detections_df,
    "insulator_image_results": insulator_image_results_df,
    "insulator_failures": insulator_failures_df,
}

if CELL17_SAVE_RAW_TABLES:
    for table_name, table_df in insulator_tables_by_name.items():
        saved_path = insulator_saved_paths.get(
            table_name,
            None,
        )

        if len(table_df) > 0 and saved_path is None:
            raise RuntimeError(
                f"CELL 17 expected {table_name} to be saved, but "
                "save_run_table returned None."
            )

        if saved_path is not None and not os.path.isfile(saved_path):
            raise RuntimeError(
                f"CELL 17 saved path does not exist for {table_name}.\n"
                f"Path: {saved_path}"
            )

insulator_output_counts["saved_table_count"] = int(
    sum(
        saved_path is not None
        for saved_path in insulator_saved_paths.values()
    )
)


# =============================================================================
# 17.20 FINAL SUMMARY
# =============================================================================

if bool(globals().get("PRINT_CONFIG_SUMMARY", True)):
    print("CELL 17 — GENERIC INSULATOR CANDIDATE DETECTION COMPLETE\n")

    print("=" * 100)
    print("CELL 17 OUTPUT COUNTS")
    print("=" * 100)

    for count_name, count_value in insulator_output_counts.items():
        print(f"  {count_name:<34}: {count_value}")

    print("\nCELL 17 SAVE PATHS")
    print("-" * 100)

    for table_name, saved_path in insulator_saved_paths.items():
        print(f"  {table_name:<34}: {saved_path}")

    print("\nCELL 17 FINAL STATUS")
    print("-" * 100)
    print("  ROI reconciliation passed         : True")
    print("  Raw-detection reconciliation      : True")
    print("  Mask reconciliation               : True")
    print("  Overlay reconciliation            : True")
    print(f"  Prompt table dir                  : {RUN_INSULATOR_PROMPT_TABLES_DIR}")
    print(f"  QA overlay dir                    : {RUN_INSULATOR_OVERLAYS_DIR}")


# =============================================================================
# 17.21 FINAL HARD OUTPUT CHECK
# =============================================================================
# EXPLANATION:
# Zero raw insulator detections is valid. Zero successful ROI rows is not.
#
# This check runs after saving so the failure table remains available when table
# saving is enabled and every ROI fails.
# =============================================================================

if len(insulator_image_results_df) == 0:
    raise RuntimeError(
        "CELL 17 completed, but insulator_image_results_df is empty.\n"
        "This means no ROI completed successfully.\n"
        "Please inspect insulator_failures_df and its saved table."
    )
