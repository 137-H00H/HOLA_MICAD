import os
import numpy as np
import SimpleITK as sitk
DATASET_ROOTS = {
    "base": "/gpfs/scratch/ec25141/medsam/data/data_processed/_preprocess_/DATA",
    "aortaseg60": "/gpfs/DERI-ecgai/001_CTA_Segmention/data/AortaSeg 60/_preprocess_/DATA",
    "sega": "/gpfs/DERI-ecgai/001_CTA_Segmention/data/SEGA/SEGA/_preprocess_/DATA",
    "cisunet": "/gpfs/DERI-ecgai/001_CTA_Segmention/data/CIS_UNet_Data/_preprocess_/DATA",
    "tbad": "/gpfs/scratch/ec25141/medsam/data/TBAD_extracted/imageTBAD/_preprocess_/DATA",
    "dissection": "/gpfs/DERI-ecgai/001_CTA_Segmention/data/Aortic Dissection Segmentations/Aortic Dissection Segmentations/_preprocess_/DATA",
}


def _patient_paths(dataset_name, patient_id):
    root = DATASET_ROOTS[dataset_name]
    patient_dir = os.path.join(root, patient_id)
    img_path = os.path.join(patient_dir, f"{patient_id}_image.nii.gz")
    seg_path = os.path.join(patient_dir, f"{patient_id}_label.nii.gz")
    return img_path, seg_path


def list_patients(dataset_name):
    root = DATASET_ROOTS[dataset_name]
    if not os.path.isdir(root):
        return []
    pts = []
    for d in sorted(os.listdir(root)):
        if d.startswith("_") or d.startswith("."):
            continue
        img_path, seg_path = _patient_paths(dataset_name, d)
        if os.path.exists(img_path) and os.path.exists(seg_path):
            pts.append(f"{dataset_name}::{d}")
    return pts


def load_patient(dataset_name, patient_id):
    img_path, seg_path = _patient_paths(dataset_name, patient_id)
    img = sitk.ReadImage(img_path)
    seg = sitk.ReadImage(seg_path)
    ia = sitk.GetArrayFromImage(img).astype(np.float32)
    ca = (sitk.GetArrayFromImage(seg) > 0).astype(np.float32)
    return ia, ca


def list_all_usable_patients():
    return list_patients("base")


def list_aortaseg60_patients():
    return list_patients("aortaseg60")


def list_sega_patients():
    return list_patients("sega")


def list_cisunet_patients():
    return list_patients("cisunet")


def list_tbad_patients():
    return list_patients("tbad")


def list_dissection_patients():
    return list_patients("dissection")


def load_patient_any(patient_id, dataset_name=None):
    if dataset_name is not None:
        return load_patient(dataset_name, patient_id)
    dataset_name, raw_id = patient_id.split("::", 1)
    return load_patient(dataset_name, raw_id)