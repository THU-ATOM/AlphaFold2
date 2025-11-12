#!/usr/bin/env python3
"""
Simplified AlphaFold2 Monomer Structure Prediction Script

This script performs AlphaFold2 structure prediction for protein monomers.
It supports optional template search and uses externally provided MSA files.

Usage:
    python predict.py --sequence SEQUENCE --a3m_path PATH --output_dir DIR [OPTIONS]
"""

import argparse
import os
import sys
import pickle as pkl
import json
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict
from absl import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import AlphaFold modules
from alphafold.data.from_msa_to_feature import (
    MonomerMSAFeatureProcessor,
    load_msa_from_path,
)
from alphafold.data import templates, parsers
from alphafold.data.tools import hhsearch
from alphafold.model import features, config, data, model
from alphafold.relax import relax
from alphafold.common import residue_constants, protein
import dataclasses
import subprocess
import shutil

logging.set_verbosity(logging.INFO)


def check_and_download_data(
    params_dir: str,
    use_templates: bool = False,
    pdb70_database_path: str = None,
    template_mmcif_dir: str = None,
) -> None:
    """
    Check if required data files exist and trigger download if missing.
    
    Args:
        params_dir: Directory containing AlphaFold model parameters
        use_templates: Whether template search is enabled
        pdb70_database_path: Path to PDB70 database
        template_mmcif_dir: Directory containing mmCIF files
    """
    script_dir = Path(__file__).parent / "scripts"
    
    # Check AlphaFold parameters
    params_path = Path(params_dir) / "params"
    required_params = [
        "params_model_1_ptm.npz",
        "params_model_2_ptm.npz", 
        "params_model_3_ptm.npz",
        "params_model_4_ptm.npz",
        "params_model_5_ptm.npz",
    ]
    
    missing_params = [p for p in required_params if not (params_path / p).exists()]
    
    if missing_params:
        logging.warning("=" * 60)
        logging.warning("AlphaFold parameters not found!")
        logging.warning(f"Missing files: {', '.join(missing_params)}")
        logging.warning("=" * 60)
        
        # Check if aria2c is available
        if not shutil.which("aria2c"):
            logging.error("aria2c is required for downloading. Please install it:")
            logging.error("  Ubuntu/Debian: sudo apt install aria2")
            logging.error("  macOS: brew install aria2")
            sys.exit(1)
        
        download_script = script_dir / "download_alphafold_params.sh"
        if not download_script.exists():
            logging.error(f"Download script not found: {download_script}")
            sys.exit(1)
        
        logging.info("Triggering automatic download of AlphaFold parameters...")
        logging.info(f"Download directory: {params_dir}")
        logging.info("This may take some time (approximately 5.3GB)...")
        
        try:
            subprocess.run(
                ["bash", str(download_script), params_dir],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            logging.info("✓ AlphaFold parameters downloaded successfully!")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to download AlphaFold parameters: {e}")
            logging.error(e.output)
            sys.exit(1)
    else:
        logging.info("✓ AlphaFold parameters found")
    
    # Check template databases if template search is enabled
    if use_templates:
        logging.info("Template search is enabled, checking template databases...")
        
        # Check PDB70
        if pdb70_database_path:
            pdb70_files = [
                f"{pdb70_database_path}_a3m.ffdata",
                f"{pdb70_database_path}_a3m.ffindex",
                f"{pdb70_database_path}_hhm.ffdata",
                f"{pdb70_database_path}_hhm.ffindex",
            ]
            
            missing_pdb70 = [f for f in pdb70_files if not Path(f).exists()]
            
            if missing_pdb70:
                logging.warning("=" * 60)
                logging.warning("PDB70 database not found!")
                logging.warning("=" * 60)
                
                download_script = script_dir / "download_pdb70.sh"
                if not download_script.exists():
                    logging.error(f"Download script not found: {download_script}")
                    sys.exit(1)
                
                # Extract base directory from pdb70_database_path
                pdb70_base_dir = str(Path(pdb70_database_path).parent.parent)
                
                logging.info("Triggering automatic download of PDB70 database...")
                logging.info(f"Download directory: {pdb70_base_dir}")
                logging.info("This may take some time (approximately 56GB)...")
                
                try:
                    subprocess.run(
                        ["bash", str(download_script), pdb70_base_dir],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                    logging.info("✓ PDB70 database downloaded successfully!")
                except subprocess.CalledProcessError as e:
                    logging.error(f"Failed to download PDB70: {e}")
                    logging.error(e.output)
                    sys.exit(1)
            else:
                logging.info("✓ PDB70 database found")
        
        # Check mmCIF files
        if template_mmcif_dir:
            mmcif_path = Path(template_mmcif_dir)
            if not mmcif_path.exists() or not any(mmcif_path.glob("*.cif")):
                logging.warning("=" * 60)
                logging.warning("PDB mmCIF database not found!")
                logging.warning("=" * 60)
                
                download_script = script_dir / "download_pdb_mmcif.sh"
                if not download_script.exists():
                    logging.error(f"Download script not found: {download_script}")
                    sys.exit(1)
                
                # Extract base directory
                mmcif_base_dir = str(mmcif_path.parent.parent)
                
                logging.info("Triggering automatic download of PDB mmCIF database...")
                logging.info(f"Download directory: {mmcif_base_dir}")
                logging.info("This will take a LONG time (approximately 200GB)...")
                logging.warning("Consider using --no_templates if you don't need template search!")
                
                user_input = input("Do you want to proceed with the download? (yes/no): ")
                if user_input.lower() not in ['yes', 'y']:
                    logging.info("Download cancelled. Exiting...")
                    sys.exit(0)
                
                try:
                    subprocess.run(
                        ["bash", str(download_script), mmcif_base_dir],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                    logging.info("✓ PDB mmCIF database downloaded successfully!")
                except subprocess.CalledProcessError as e:
                    logging.error(f"Failed to download PDB mmCIF: {e}")
                    logging.error(e.output)
                    sys.exit(1)
            else:
                logging.info("✓ PDB mmCIF database found")


def search_template(
    sequence: str,
    msa_path: str,
    pdb70_database_path: str,
    hhsearch_binary_path: str = "hhsearch",
):
    """Search for structural templates using HHsearch against PDB70."""
    logging.info("Searching for templates...")
    
    template_searcher = hhsearch.HHSearch(
        binary_path=hhsearch_binary_path,
        databases=[pdb70_database_path]
    )
    
    # Load MSA for template search
    msa_for_templates = load_msa_from_path(msa_path, "a3m", max_depth=None)
    
    # Run template search
    pdb_templates_result = template_searcher.query(a3m=msa_for_templates["a3m"])
    pdb_template_hits = template_searcher.get_template_hits(
        output_string=pdb_templates_result,
        input_sequence=sequence
    )
    
    logging.info(f"Found {len(pdb_template_hits)} template hits")
    return [dataclasses.asdict(h) for h in pdb_template_hits]


def make_template_features(
    sequence: str,
    template_hits,
    template_mmcif_dir: str,
    max_template_hits: int = 20,
    max_template_date: str = "2022-12-31",
    obsolete_pdbs_path: str = None,
    kalign_binary_path: str = "kalign",
):
    """Convert template hits to features."""
    logging.info("Creating template features...")
    
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=template_mmcif_dir,
        max_template_date=max_template_date,
        max_hits=max_template_hits,
        kalign_binary_path=kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=obsolete_pdbs_path,
    )
    
    # Convert dict back to TemplateHit objects
    template_hits_objects = [parsers.TemplateHit(**h) for h in template_hits]
    
    templates_result = template_featurizer.get_templates(
        query_sequence=sequence,
        hits=template_hits_objects
    )
    
    num_templates = templates_result.features["template_domain_names"].shape[0]
    logging.info(f"Generated features for {num_templates} templates")
    
    return {**templates_result.features}


def create_empty_template_features(sequence_length: int):
    """Create empty template features when templates are not used."""
    return {
        'template_aatype': np.zeros((1, sequence_length, 22), dtype=np.int32),
        'template_all_atom_masks': np.zeros((1, sequence_length, 37), dtype=np.float32),
        'template_all_atom_positions': np.zeros((1, sequence_length, 37, 3), dtype=np.float32),
        'template_domain_names': np.array(['none'], dtype=object),
        'template_sequence': np.array([b''], dtype=object),
        'template_sum_probs': np.zeros((1,), dtype=np.float32),
    }


def process_msa_to_features(
    sequence: str,
    target_name: str,
    msa_path: str,
    template_features: dict,
    model_name: str,
    random_seed: int,
    num_ensemble: int = 1,
    max_recycles: int = 3,
    max_msa_clusters: int = 512,
    max_extra_msa: int = 5120,
):
    """Process MSA and templates into model features."""
    logging.info("Processing MSA to features...")
    
    # Get model configuration
    model_config = config.model_config(model_name)
    model_config.data.eval.num_ensemble = num_ensemble
    model_config.model.num_recycle = max_recycles
    model_config.data.common.num_recycle = max_recycles
    model_config.data.eval.max_msa_clusters = max_msa_clusters
    model_config.data.common.max_extra_msa = max_extra_msa
    model_config.data.common.reduce_msa_clusters_by_max_templates = False
    
    # Process MSA
    data_pipe = MonomerMSAFeatureProcessor(msa_paths=[msa_path])
    raw_features = data_pipe.process(
        input_sequence=sequence,
        input_description=target_name
    )
    
    # Merge with template features
    raw_features_with_template = {**raw_features, **template_features}
    
    # Convert to model input features
    processed_features = features.np_example_to_features(
        np_example=raw_features_with_template,
        config=model_config,
        random_seed=random_seed,
    )
    
    return processed_features, model_config


def predict_structure(
    target_name: str,
    processed_features,
    model_config,
    model_name: str,
    params_dir: str,
    random_seed: int,
):
    """Run structure prediction using AlphaFold2."""
    logging.info(f"Running structure prediction with {model_name}...")
    
    # Load model parameters
    model_params = data.get_model_haiku_params(
        model_name=model_name,
        data_dir=params_dir
    )
    
    # Create model runner
    model_runner = model.RunModel(
        model_config,
        model_params,
        return_representations=True,
    )
    
    # Run prediction
    prediction_result = model_runner.predict(
        processed_features,
        random_seed=random_seed
    )
    
    # Extract all confidence metrics
    confidence_metrics = extract_confidence_metrics(
        prediction_result,
        sequence_length=len(processed_features['aatype'])
    )
    
    # Extract pLDDT scores
    plddt = prediction_result["plddt"]
    mean_plddt = np.mean(plddt)
    logging.info(f"Mean pLDDT: {mean_plddt:.2f}")
    
    # Log other metrics if available
    if "ptm" in confidence_metrics:
        logging.info(f"pTM score: {confidence_metrics['ptm']:.4f}")
    if "iptm" in confidence_metrics:
        logging.info(f"ipTM score: {confidence_metrics['iptm']:.4f}")
    
    # Create protein object with pLDDT as B-factors
    plddt_b_factors = np.repeat(
        plddt[:, None],
        residue_constants.atom_type_num,
        axis=-1
    )
    
    unrelaxed_protein = protein.from_prediction(
        features=processed_features,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=True,
    )
    
    return unrelaxed_protein, prediction_result, mean_plddt, confidence_metrics


def run_amber_relaxation(unrelaxed_protein, use_gpu: bool = True):
    """Run AMBER relaxation on the predicted structure."""
    logging.info("Running AMBER relaxation...")
    
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=0,
        tolerance=2.39,
        stiffness=10.0,
        exclude_residues=[],
        max_outer_iterations=3,
        use_gpu=use_gpu,
    )
    
    relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
    
    return relaxed_pdb_str


def extract_confidence_metrics(prediction_result: dict, sequence_length: int) -> dict:
    """
    Extract all confidence metrics from prediction result.
    
    Args:
        prediction_result: Raw prediction result from model
        sequence_length: Length of the protein sequence
        
    Returns:
        Dictionary containing all confidence metrics with proper formatting
    """
    metrics = {}
    
    # 1. pLDDT (per-residue confidence)
    if "plddt" in prediction_result:
        plddt = prediction_result["plddt"]
        metrics["plddt"] = {
            "per_residue": plddt.tolist() if isinstance(plddt, np.ndarray) else plddt,
            "mean": float(np.mean(plddt)),
            "min": float(np.min(plddt)),
            "max": float(np.max(plddt)),
            "median": float(np.median(plddt)),
            "std": float(np.std(plddt)),
        }
        
        # Confidence level statistics
        high_conf = np.sum(plddt > 90)
        medium_conf = np.sum((plddt >= 70) & (plddt <= 90))
        low_conf = np.sum(plddt < 70)
        
        metrics["plddt"]["confidence_levels"] = {
            "high_confidence_residues": int(high_conf),
            "high_confidence_percentage": float(high_conf / len(plddt) * 100),
            "medium_confidence_residues": int(medium_conf),
            "medium_confidence_percentage": float(medium_conf / len(plddt) * 100),
            "low_confidence_residues": int(low_conf),
            "low_confidence_percentage": float(low_conf / len(plddt) * 100),
        }
    
    # 2. Predicted Aligned Error (PAE)
    if "predicted_aligned_error" in prediction_result:
        pae = prediction_result["predicted_aligned_error"]
        if isinstance(pae, np.ndarray):
            metrics["pae"] = {
                "matrix": pae.tolist(),
                "mean": float(np.mean(pae)),
                "min": float(np.min(pae)),
                "max": float(np.max(pae)),
                "shape": list(pae.shape),
            }
    
    # 3. Max Predicted Aligned Error
    if "max_predicted_aligned_error" in prediction_result:
        metrics["max_pae"] = float(prediction_result["max_predicted_aligned_error"])
    
    # 4. pTM (predicted TM-score)
    if "ptm" in prediction_result:
        metrics["ptm"] = float(prediction_result["ptm"])
    
    # 5. ipTM (interface pTM, for multimer)
    if "iptm" in prediction_result:
        metrics["iptm"] = float(prediction_result["iptm"])
    
    # 6. Ranking confidence
    if "ranking_confidence" in prediction_result:
        metrics["ranking_confidence"] = float(prediction_result["ranking_confidence"])
    
    # 7. Distogram (distance predictions)
    if "distogram" in prediction_result:
        distogram = prediction_result["distogram"]
        if "logits" in distogram:
            logits = distogram["logits"]
            # Get predicted distance matrix (argmax of bins)
            if isinstance(logits, np.ndarray) and len(logits.shape) >= 3:
                # Only save for the actual sequence length
                logits_crop = logits[:sequence_length, :sequence_length, :]
                pred_distances = np.argmax(logits_crop, axis=-1)
                
                metrics["distogram"] = {
                    "predicted_distance_matrix": pred_distances.tolist(),
                    "shape": list(pred_distances.shape),
                }
                
                if "bin_edges" in distogram:
                    metrics["distogram"]["bin_edges"] = distogram["bin_edges"].tolist() if isinstance(distogram["bin_edges"], np.ndarray) else distogram["bin_edges"]
    
    # 8. Experimentally resolved confidence (if available)
    if "experimentally_resolved" in prediction_result:
        exp_resolved = prediction_result["experimentally_resolved"]
        if isinstance(exp_resolved, np.ndarray):
            metrics["experimentally_resolved"] = {
                "per_residue": exp_resolved.tolist(),
                "mean": float(np.mean(exp_resolved)),
            }
    
    # 9. Masked MSA (optional, if present)
    if "masked_msa" in prediction_result:
        masked_msa = prediction_result["masked_msa"]
        if isinstance(masked_msa, np.ndarray):
            metrics["masked_msa"] = {
                "shape": list(masked_msa.shape),
            }
    
    return metrics


def save_confidence_metrics(
    all_model_metrics: dict,
    ranked_predictions: OrderedDict,
    output_dir: Path
) -> Path:
    """
    Save comprehensive confidence metrics for all models to a single JSON file.
    
    Args:
        all_model_metrics: Dict mapping model_name -> confidence metrics
        ranked_predictions: OrderedDict from rank_predictions_by_plddt
        output_dir: Output directory
        
    Returns:
        Path to the saved JSON file
    """
    # Get target name from output_dir (it's the directory name)
    target_name = output_dir.name
    
    # Create comprehensive confidence data structure
    confidence_data = {
        "metadata": {
            "target_name": target_name,
            "num_models": len(all_model_metrics),
            "prediction_date": None,  # Can be added if needed
        },
        "models": {}
    }
    
    # Add metrics for each model in ranked order
    for rank, (model_name, pred_data) in enumerate(ranked_predictions.items(), start=1):
        model_key = f"rank_{rank}_{model_name}"
        
        # Get the detailed metrics for this model
        if model_name in all_model_metrics:
            model_metrics = all_model_metrics[model_name].copy()
        else:
            model_metrics = {}
        
        # Add ranking information
        model_metrics["rank"] = rank
        model_metrics["model_name"] = model_name
        
        # Add mean pLDDT from prediction data
        if "plddt" not in model_metrics and "plddt_array" in pred_data:
            plddt_array = pred_data["plddt_array"]
            model_metrics["plddt"] = {
                "mean": float(pred_data["mean_plddt"]),
                "per_residue": plddt_array.tolist() if isinstance(plddt_array, np.ndarray) else plddt_array,
            }
        
        confidence_data["models"][model_key] = model_metrics
    
    # Add summary statistics across all models
    all_mean_plddts = [pred_data["mean_plddt"] for pred_data in ranked_predictions.values()]
    confidence_data["summary"] = {
        "best_model": {
            "rank": 1,
            "model_name": list(ranked_predictions.keys())[0],
            "mean_plddt": float(all_mean_plddts[0]),
        },
        "all_models": {
            "mean_plddt": {
                "mean": float(np.mean(all_mean_plddts)),
                "min": float(np.min(all_mean_plddts)),
                "max": float(np.max(all_mean_plddts)),
                "std": float(np.std(all_mean_plddts)),
            }
        }
    }
    
    # Save to JSON file
    json_path = output_dir / "confidence.json"
    with open(json_path, "w") as f:
        json.dump(confidence_data, f, indent=2)
    
    logging.info(f"Saved comprehensive confidence metrics to {json_path.name}")
    return json_path


def plot_plddts(model2plddts: OrderedDict, output_path: str):
    """Plot pLDDT scores for all models."""
    num_models = len(model2plddts)
    plt.figure(figsize=(3 * num_models, 2), dpi=100)
    
    for n, (model_name, plddt) in enumerate(model2plddts.items()):
        plt.subplot(1, num_models, n + 1)
        plt.title(model_name)
        plt.plot(plddt)
        plt.ylim(0, 100)
        plt.ylabel("Predicted LDDT")
        plt.xlabel("Residue")
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()
    logging.info(f"Saved pLDDT plot to {output_path}")


def rank_predictions_by_plddt(predictions: dict) -> OrderedDict:
    """
    Rank predictions by mean pLDDT score.
    
    Args:
        predictions: Dict mapping model_name -> {
            'mean_plddt': float,
            'plddt_array': np.array,
            'unrelaxed_pdb': str,
            'relaxed_pdb': str (optional),
            'results': dict (optional)
        }
    
    Returns:
        OrderedDict sorted by mean pLDDT (highest first)
    """
    # Sort by mean pLDDT score
    sorted_items = sorted(
        predictions.items(),
        key=lambda x: x[1]['mean_plddt'],
        reverse=True
    )
    
    return OrderedDict(sorted_items)


def save_ranked_pdbs(ranked_predictions: OrderedDict, output_dir: Path, relaxed: bool = True):
    """
    Save ranked PDB files with rank prefix.
    
    Args:
        ranked_predictions: OrderedDict from rank_predictions_by_plddt
        output_dir: Output directory
        relaxed: Whether to save relaxed or unrelaxed structures
    """
    pdb_type = "relaxed" if relaxed else "unrelaxed"
    pdb_key = f"{pdb_type}_pdb"
    
    for rank, (model_name, pred_data) in enumerate(ranked_predictions.items(), start=1):
        if pdb_key in pred_data and pred_data[pdb_key]:
            pdb_str = pred_data[pdb_key]
            output_path = output_dir / f"rank_{rank}_{model_name}_{pdb_type}.pdb"
            
            with open(output_path, "w") as f:
                f.write(pdb_str)
            
            logging.info(f"Rank {rank}: {model_name} (pLDDT: {pred_data['mean_plddt']:.2f}) -> {output_path.name}")


def save_ranking_summary(ranked_predictions: OrderedDict, output_dir: Path):
    """Save ranking summary as JSON."""
    summary = []
    for rank, (model_name, pred_data) in enumerate(ranked_predictions.items(), start=1):
        summary.append({
            "rank": rank,
            "model_name": model_name,
            "mean_plddt": float(pred_data['mean_plddt']),
            "min_plddt": float(np.min(pred_data['plddt_array'])),
            "max_plddt": float(np.max(pred_data['plddt_array'])),
        })
    
    summary_path = output_dir / "ranking_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Saved ranking summary to {summary_path.name}")
    return summary


def save_detailed_plddt(ranked_predictions: OrderedDict, output_dir: Path, target_name: str):
    """
    Save detailed per-residue pLDDT scores for all models.
    
    Creates two files:
    1. JSON file with pLDDT arrays for all models
    2. CSV file with per-residue scores in tabular format
    """
    # Prepare data for JSON
    plddt_data = {}
    for rank, (model_name, pred_data) in enumerate(ranked_predictions.items(), start=1):
        plddt_data[f"rank_{rank}_{model_name}"] = {
            "rank": rank,
            "model_name": model_name,
            "mean_plddt": float(pred_data['mean_plddt']),
            "per_residue_plddt": pred_data['plddt_array'].tolist(),
        }
    
    # Save JSON file
    json_path = output_dir / f"{target_name}_plddt_detailed.json"
    with open(json_path, "w") as f:
        json.dump(plddt_data, f, indent=2)
    logging.info(f"Saved detailed pLDDT (JSON) to {json_path}")
    
    # Save CSV file for easier analysis
    csv_path = output_dir / f"{target_name}_plddt_per_residue.csv"
    with open(csv_path, "w") as f:
        # Header
        model_names = [f"rank_{rank}_{model_name}" 
                      for rank, (model_name, _) in enumerate(ranked_predictions.items(), start=1)]
        f.write("residue," + ",".join(model_names) + "\n")
        
        # Get the length from first model
        first_model_data = next(iter(ranked_predictions.values()))
        num_residues = len(first_model_data['plddt_array'])
        
        # Write per-residue scores
        for i in range(num_residues):
            scores = [str(pred_data['plddt_array'][i]) 
                     for _, pred_data in ranked_predictions.items()]
            f.write(f"{i+1}," + ",".join(scores) + "\n")
    
    logging.info(f"Saved per-residue pLDDT (CSV) to {csv_path}")
    
    return json_path, csv_path


def cleanup_unranked_pdbs(output_dir: Path, model_names: list):
    """
    Remove unranked PDB files (model_X_*.pdb) to avoid redundancy.
    Only keep the ranked files (rank_X_*.pdb).
    """
    removed_files = []
    
    for model_name in model_names:
        # Remove unrelaxed PDB
        unrelaxed_path = output_dir / f"{model_name}_unrelaxed.pdb"
        if unrelaxed_path.exists():
            unrelaxed_path.unlink()
            removed_files.append(unrelaxed_path.name)
        
        # Remove relaxed PDB
        relaxed_path = output_dir / f"{model_name}_relaxed.pdb"
        if relaxed_path.exists():
            relaxed_path.unlink()
            removed_files.append(relaxed_path.name)
    
    if removed_files:
        logging.info(f"Cleaned up {len(removed_files)} unranked PDB files")
        for fname in removed_files:
            logging.debug(f"  Removed: {fname}")
    
    return removed_files


def main():
    parser = argparse.ArgumentParser(
        description="AlphaFold2 Monomer Structure Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="Input protein sequence (single letter amino acid codes)"
    )
    parser.add_argument(
        "--a3m_path",
        type=str,
        required=True,
        help="Path to input MSA file in A3M format"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output files"
    )
    
    # Template search arguments
    parser.add_argument(
        "--use_templates",
        action="store_true",
        help="Whether to search and use structural templates"
    )
    parser.add_argument(
        "--pdb70_database_path",
        type=str,
        default="/data/alphafold/pdb70/pdb70",
        help="Path to PDB70 database for template search"
    )
    parser.add_argument(
        "--template_mmcif_dir",
        type=str,
        default="/data/alphafold/pdb_mmcif/mmcif_files",
        help="Directory containing mmCIF files for templates"
    )
    parser.add_argument(
        "--obsolete_pdbs_path",
        type=str,
        default="/data/alphafold/pdb_mmcif/obsolete.dat",
        help="Path to obsolete PDB entries file"
    )
    parser.add_argument(
        "--max_template_date",
        type=str,
        default="2022-12-31",
        help="Maximum template release date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--template_hits_file",
        type=str,
        default=None,
        help="Path to pre-computed template hits pickle file (optional)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="model_1_ptm",
        help="AlphaFold2 model name(s) to use. Can be a single model or comma-separated list (e.g., 'model_1_ptm' or 'model_1_ptm,model_2_ptm,model_3_ptm')"
    )
    parser.add_argument(
        "--params_dir",
        type=str,
        default="/data/alphafold",
        help="(Deprecated) Directory containing AlphaFold2 model parameters. Prefer --cache"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Single cache directory for all AlphaFold data (e.g. /data/alphafold).\nDerived subpaths: params=<cache>/params, pdb70=<cache>/pdb70, pdb_mmcif=<cache>/pdb_mmcif"
    )
    parser.add_argument(
        "--no_download",
        action="store_true",
        help="If set, do not attempt to auto-download missing data (fail instead)"
    )
    
    # Prediction parameters
    parser.add_argument(
        "--target_name",
        type=str,
        default="target",
        help="Name of the target protein"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_ensemble",
        type=int,
        default=1,
        help="Number of ensemble predictions"
    )
    parser.add_argument(
        "--max_recycles",
        type=int,
        default=3,
        help="Maximum number of recycling iterations"
    )
    parser.add_argument(
        "--max_msa_clusters",
        type=int,
        default=512,
        help="Maximum number of MSA clusters"
    )
    parser.add_argument(
        "--max_extra_msa",
        type=int,
        default=5120,
        help="Maximum number of extra MSA sequences"
    )
    
    # Output options
    parser.add_argument(
        "--no_relax",
        action="store_true",
        help="Skip AMBER relaxation step"
    )
    parser.add_argument(
        "--save_features",
        action="store_true",
        help="Save processed features as pickle file"
    )
    parser.add_argument(
        "--save_all_outputs",
        action="store_true",
        help="Save all prediction outputs (including representations)"
    )
    
    args = parser.parse_args()

    # Resolve cache directory and derive standard data paths.
    # Priority: --cache if provided, else fall back to --params_dir for backward compatibility.
    if args.cache:
        cache_dir = Path(args.cache)
    else:
        cache_dir = Path(args.params_dir)

    # Derived standard locations inside the cache
    derived_params_dir = str(cache_dir)
    derived_pdb70_db = str(cache_dir / "pdb70" / "pdb70")
    derived_mmcif_dir = str(cache_dir / "pdb_mmcif" / "mmcif_files")
    derived_obsolete = str(cache_dir / "pdb_mmcif" / "obsolete.dat")

    # Override args to use derived paths (but keep original if explicitly provided)
    # If user explicitly supplied params_dir/pdb70/template dirs, prefer those.
    if not args.params_dir or args.params_dir == "/data/alphafold":
        args.params_dir = derived_params_dir
    if not args.pdb70_database_path or args.pdb70_database_path == "/data/alphafold/pdb70/pdb70":
        args.pdb70_database_path = derived_pdb70_db
    if not args.template_mmcif_dir or args.template_mmcif_dir == "/data/alphafold/pdb_mmcif/mmcif_files":
        args.template_mmcif_dir = derived_mmcif_dir
    if not args.obsolete_pdbs_path or args.obsolete_pdbs_path == "/data/alphafold/pdb_mmcif/obsolete.dat":
        args.obsolete_pdbs_path = derived_obsolete

    # Before any heavy processing, check and optionally download missing data.
    if not args.no_download:
        check_and_download_data(
            params_dir=args.params_dir,
            use_templates=args.use_templates,
            pdb70_database_path=args.pdb70_database_path,
            template_mmcif_dir=args.template_mmcif_dir,
        )
    else:
        logging.info("--no_download set: skipping automatic data downloads. Ensure required data exists in cache.")

    # Create output directory with target_name subdirectory
    base_output_dir = Path(args.output_dir)
    target_name = args.target_name
    output_dir = base_output_dir / target_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output paths (without target_name prefix since files are in target_name directory)
    template_hits_path = output_dir / "template_hits.pkl"
    
    # Validate inputs
    if not Path(args.a3m_path).exists():
        logging.error(f"MSA file not found: {args.a3m_path}")
        sys.exit(1)
    
    sequence = args.sequence.upper()
    sequence_length = len(sequence)
    logging.info(f"Target: {target_name}")
    logging.info(f"Sequence length: {sequence_length}")
    logging.info(f"Models: {args.model_name}")
    
    # Step 1: Handle templates
    template_features = None
    
    if args.use_templates:
        # Check if pre-computed template hits are provided
        if args.template_hits_file and Path(args.template_hits_file).exists():
            logging.info(f"Loading pre-computed template hits from {args.template_hits_file}")
            with open(args.template_hits_file, "rb") as f:
                template_hits = pkl.load(f)
        else:
            # Search for templates
            template_hits = search_template(
                sequence=sequence,
                msa_path=args.a3m_path,
                pdb70_database_path=args.pdb70_database_path,
            )
            
            # Save template hits
            with open(template_hits_path, "wb") as f:
                pkl.dump(template_hits, f)
            logging.info(f"Saved template hits to {template_hits_path}")
        
        # Create template features
        if template_hits:
            template_features = make_template_features(
                sequence=sequence,
                template_hits=template_hits,
                template_mmcif_dir=args.template_mmcif_dir,
                max_template_date=args.max_template_date,
                obsolete_pdbs_path=args.obsolete_pdbs_path,
            )
        else:
            logging.warning("No template hits found, using empty template features")
            template_features = create_empty_template_features(sequence_length)
    else:
        logging.info("Not using templates")
        template_features = create_empty_template_features(sequence_length)
    
    # Parse model names (support comma-separated list)
    model_names = [m.strip() for m in args.model_name.split(',')]
    logging.info(f"Will predict with {len(model_names)} model(s): {', '.join(model_names)}")
    
    # Validate model names
    valid_models = ["model_1", "model_2", "model_3", "model_4", "model_5",
                   "model_1_ptm", "model_2_ptm", "model_3_ptm", "model_4_ptm", "model_5_ptm"]
    for model_name in model_names:
        if model_name not in valid_models:
            logging.error(f"Invalid model name: {model_name}")
            logging.error(f"Valid models: {', '.join(valid_models)}")
            sys.exit(1)
    
    # Dictionary to store all predictions
    all_predictions = {}
    
    # Dictionary to store all confidence metrics
    all_confidence_metrics = {}
    
    # Step 2 & 3: Run predictions for each model
    for model_idx, model_name in enumerate(model_names, start=1):
        logging.info("=" * 60)
        logging.info(f"Running prediction {model_idx}/{len(model_names)}: {model_name}")
        logging.info("=" * 60)
        
        # Process MSA and templates to features
        processed_features, model_config = process_msa_to_features(
            sequence=sequence,
            target_name=target_name,
            msa_path=args.a3m_path,
            template_features=template_features,
            model_name=model_name,
            random_seed=args.random_seed,
            num_ensemble=args.num_ensemble,
            max_recycles=args.max_recycles,
            max_msa_clusters=args.max_msa_clusters,
            max_extra_msa=args.max_extra_msa,
        )
        
        # Save features for first model only
        if model_idx == 1 and args.save_features:
            features_path = output_dir / "features.pkl"
            with open(features_path, "wb") as f:
                pkl.dump(processed_features, f)
            logging.info(f"Saved features to {features_path.name}")
        
        # Run structure prediction
        unrelaxed_protein, prediction_result, mean_plddt, confidence_metrics = predict_structure(
            target_name=target_name,
            processed_features=processed_features,
            model_config=model_config,
            model_name=model_name,
            params_dir=args.params_dir,
            random_seed=args.random_seed,
        )
        
        # Store confidence metrics for this model
        all_confidence_metrics[model_name] = confidence_metrics
        
        # Get pLDDT array
        plddt_array = unrelaxed_protein.b_factors[:, 0]
        
        # Save unrelaxed structure
        unrelaxed_pdb_str = protein.to_pdb(unrelaxed_protein)
        unrelaxed_pdb_path = output_dir / f"{model_name}_unrelaxed.pdb"
        with open(unrelaxed_pdb_path, "w") as f:
            f.write(unrelaxed_pdb_str)
        logging.info(f"Saved unrelaxed structure to {unrelaxed_pdb_path}")
        
        # Run AMBER relaxation (optional)
        relaxed_pdb_str = None
        if not args.no_relax:
            relaxed_pdb_str = run_amber_relaxation(unrelaxed_protein)
            relaxed_pdb_path = output_dir / f"{model_name}_relaxed.pdb"
            with open(relaxed_pdb_path, "w") as f:
                f.write(relaxed_pdb_str)
            logging.info(f"Saved relaxed structure to {relaxed_pdb_path}")
        
        # Save prediction results (optional)
        if args.save_all_outputs:
            results_path = output_dir / f"{model_name}_results.pkl"
            with open(results_path, "wb") as f:
                pkl.dump(prediction_result, f)
            logging.info(f"Saved prediction results to {results_path}")
        
        # Store prediction data
        all_predictions[model_name] = {
            'mean_plddt': mean_plddt,
            'plddt_array': plddt_array,
            'unrelaxed_pdb': unrelaxed_pdb_str,
            'relaxed_pdb': relaxed_pdb_str,
            'results': prediction_result if args.save_all_outputs else None,
        }
        
        logging.info(f"Model {model_name}: Mean pLDDT = {mean_plddt:.2f}")
    
    # Step 4: Rank predictions by pLDDT
    logging.info("=" * 60)
    logging.info("Ranking predictions by pLDDT...")
    logging.info("=" * 60)
    
    ranked_predictions = rank_predictions_by_plddt(all_predictions)
    
    # Save ranked PDB files
    if not args.no_relax:
        save_ranked_pdbs(ranked_predictions, output_dir, relaxed=True)
    save_ranked_pdbs(ranked_predictions, output_dir, relaxed=False)
    
    # Save ranking summary
    summary = save_ranking_summary(ranked_predictions, output_dir)
    
    # Save comprehensive confidence metrics (includes all pLDDT info, so no need for separate files)
    save_confidence_metrics(all_confidence_metrics, ranked_predictions, output_dir)
    
    # Clean up unranked PDB files (keep only ranked ones)
    if len(model_names) > 1:
        cleanup_unranked_pdbs(output_dir, model_names)
    
    # Step 5: Generate pLDDT plot
    if len(model_names) > 1:
        # Extract pLDDT arrays in ranked order
        model2plddts = OrderedDict([
            (model_name, pred_data['plddt_array'])
            for model_name, pred_data in ranked_predictions.items()
        ])
        
        plddt_plot_path = output_dir / "plddt_plot.png"
        plot_plddts(model2plddts, str(plddt_plot_path))
    
    # Print summary
    logging.info("=" * 60)
    logging.info("Prediction completed successfully!")
    logging.info("=" * 60)
    logging.info(f"Target: {target_name}")
    logging.info(f"Sequence length: {sequence_length}")
    logging.info(f"Number of models: {len(model_names)}")
    logging.info("")
    logging.info("Ranking (by mean pLDDT):")
    for rank, item in enumerate(summary, start=1):
        logging.info(f"  Rank {rank}: {item['model_name']} - pLDDT: {item['mean_plddt']:.2f}")
    logging.info("")
    logging.info(f"Output directory: {output_dir}")
    if not args.no_relax:
        logging.info(f"Best model (relaxed): rank_1_*_relaxed.pdb")
    logging.info(f"Best model (unrelaxed): rank_1_*_unrelaxed.pdb")
    if len(model_names) > 1:
        logging.info(f"pLDDT plot: plddt_plot.png")
    logging.info(f"Ranking summary: ranking_summary.json")
    logging.info(f"Confidence metrics: confidence.json")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
