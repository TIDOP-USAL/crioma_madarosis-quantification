# Objective Quantification of Chemotherapy-Induced Madarosis

**Implementation of the automated computer vision pipeline for eyebrow density assessment.**

This repository contains the source code and methodology validation described in the research paper:

> **"Objective Quantification of Chemotherapy-Induced Madarosis: Validation of an Automated Computer Vision Pipeline for Eyebrow Density Assessment"**
> *TIDOP Research Group, Escuela Politécnica Superior de Ávila.*

## Abstract

Chemotherapy-induced madarosis (loss of eyebrows and eyelashes) significantly impacts patient quality of life, yet current assessment methods rely heavily on subjective grading. This study presents a novel automated computer vision pipeline for objectively quantifying periocular hair density changes.

The methodology integrates facial landmark detection, multi-temporal registration, and deep learning-based hair segmentation with custom morphological filtering to isolate hair structures from noise. This framework provides a robust, operator-independent metric for monitoring madarosis and evaluating preventive strategies such as localized cryotherapy.

## Methodology

The pipeline implemented in this repository follows the workflow validated in the study:

1.  **Periocular Landmark Detection:** Automatic extraction of the Region of Interest (ROI) using MediaPipe Face Mesh to stabilize inference and ensure consistent framing.
2.  **ROI Normalization:** Photometric normalization in the HSV color space to reduce lighting variability across sessions.
3.  **Multi-temporal Registration:** A two-stage alignment process to register follow-up images (T2-T4) to the baseline (T1) coordinate system:
    * *Coarse Alignment:* Homography estimation based on facial landmarks.
    * *Fine Registration:* Refinement using R2D2 (Repeatable and Reliable Detector and Descriptor) keypoints to correct residual local misalignments.
4.  **Trimap-Guided Segmentation:** Deep learning segmentation using DAM-Net, guided by geometrically generated trimaps derived from the baseline image.
5.  **Morphological Filtering:** Custom automated filtering algorithms (opening, dilation, and connected component analysis) to remove false-positive artifacts and isolate the main hair component.
6.  **Quantification:** Calculation of hair density percentage relative to the baseline (T1) for longitudinal monitoring.

## Repository Structure

The project is structured as follows:

* `main_pipeline.py`: The main script that orchestrates the data loading, registration, segmentation, and quantification steps.
* `utils.py`: Contains helper functions for image processing, geometric transformations, and filtering logic.
* `data/`: Directory where patient images should be stored.
* `models/`: Directory where external pre-trained models must be placed.

## Installation and Dependencies

1.  Clone this repository:
    ```bash
    git clone [https://github.com/Tidop-Group/crioma_madarosis-quantification.git](https://github.com/Tidop-Group/crioma_madarosis-quantification.git)
    cd crioma_madarosis-quantification
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **External Models Setup:**
    This pipeline relies on external pre-trained models for segmentation and feature extraction. You must download and place them in the `models/` directory exactly as follows:

    * **DAM-Net:** Place the DAM-Net implementation and its `best_model.pth` checkpoint inside `models/DAM-Net/`.
    * **R2D2:** Place the R2D2 extraction script and the `r2d2_WASF_N16.pt` model inside `models/r2d2-master/`.

## Usage

### Data Preparation
Organize your dataset in the `data/` folder. The pipeline expects a folder per patient, containing subfolders for each time point or direct image files named according to the treatment phase (e.g., T1, T2, T3, T4).

### Running the Pipeline
To process the data and generate the quantification report, run:

```bash
python main_pipeline.py
