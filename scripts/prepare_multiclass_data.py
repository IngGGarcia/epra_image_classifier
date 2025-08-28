#!/usr/bin/env python3
"""
Prepare multiclass data from binary model predictions.

This script:
1. Runs inference on all images using the trained binary model
2. Maps violence scores to 5 violence levels (0-4)  
3. Organizes images into level-based directories for multiclass training
4. Generates summary statistics and reports

Usage:
    python scripts/prepare_multiclass_data.py --binary-model outputs/binary/best_model.pth
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any


def score_to_level(violence_prob: float, thresholds: list[float] = [0.2, 0.4, 0.6, 0.8]) -> int:
    """
    Convert violence probability to violence level.
    
    Args:
        violence_prob: Violence probability from binary model (0.0-1.0)
        thresholds: Thresholds for levels [0.2, 0.4, 0.6, 0.8]
        
    Returns:
        Violence level (0-4):
        - Level 0 (0.0-0.2): No Violence  
        - Level 1 (0.2-0.4): Mild Violence
        - Level 2 (0.4-0.6): Moderate Violence
        - Level 3 (0.6-0.8): High Violence
        - Level 4 (0.8-1.0): Extreme Violence
    """
    if violence_prob < thresholds[0]:
        return 0
    elif violence_prob < thresholds[1]:
        return 1
    elif violence_prob < thresholds[2]:
        return 2
    elif violence_prob < thresholds[3]:
        return 3
    else:
        return 4


def get_level_directory_name(level: int) -> str:
    """Get semantic directory name for violence level."""
    level_names = [
        "level_0_no_violence",
        "level_1_mild_violence", 
        "level_2_moderate_violence",
        "level_3_high_violence",
        "level_4_extreme_violence",
    ]
    return level_names[level]


def run_binary_inference(binary_model_path: str, source_dir: str, output_file: str, config_path: str = "configs/binary_config.yaml") -> bool:
    """
    Run binary model inference on all images.
    
    Args:
        binary_model_path: Path to trained binary model
        source_dir: Directory containing source images  
        output_file: Output JSON file for predictions
        config_path: Configuration file path
        
    Returns:
        True if successful, False otherwise
    """
    print(f"🔍 Running binary model inference...")
    print(f"   Model: {binary_model_path}")
    print(f"   Source: {source_dir}")
    print(f"   Output: {output_file}")
    
    cmd = [
        sys.executable, "scripts/predict.py",
        "--config", config_path,
        "--checkpoint", binary_model_path,
        "--input", source_dir,
        "--output", output_file,
        "--return-probabilities",
        "--batch-size", "32"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Binary inference completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Binary inference failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def organize_images_by_levels(predictions_file: str, target_dir: str, copy_mode: bool = True, thresholds: list[float] = [0.2, 0.4, 0.6, 0.8]) -> Dict[str, Any]:
    """
    Organize images into level directories based on violence scores.
    
    Args:
        predictions_file: JSON file with predictions
        target_dir: Target directory for organized images
        copy_mode: If True copy files, if False move files
        thresholds: Violence level thresholds
        
    Returns:
        Dictionary with organization statistics
    """
    print(f"📁 Organizing images into multiclass directories...")
    
    # Load predictions
    with open(predictions_file, "r") as f:
        data = json.load(f)
    
    predictions = data["predictions"]
    print(f"   Processing {len(predictions)} predictions")
    
    # Create target directories
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    level_dirs = {}
    for level in range(5):
        level_name = get_level_directory_name(level)
        level_path = target_path / level_name
        level_path.mkdir(exist_ok=True)
        level_dirs[level] = level_path
    
    # Process predictions and organize files
    stats = {i: 0 for i in range(5)}
    errors = 0
    
    print("   Copying images to level directories...")
    for i, pred in enumerate(predictions):
        if i % 1000 == 0:
            print(f"   Progress: {i}/{len(predictions)} ({i/len(predictions)*100:.1f}%)")
            
        try:
            # Get violence probability and level
            violence_prob = pred.get("violence_probability", 0.0)
            level = score_to_level(violence_prob, thresholds)
            
            # Source and target paths
            source_path = Path(pred["image_path"])
            target_path = level_dirs[level] / source_path.name
            
            # Handle filename conflicts
            counter = 1
            original_target = target_path
            while target_path.exists():
                stem = original_target.stem
                suffix = original_target.suffix
                target_path = level_dirs[level] / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # Copy or move file
            if copy_mode:
                shutil.copy2(source_path, target_path)
            else:
                shutil.move(source_path, target_path)
            
            stats[level] += 1
            
        except Exception as e:
            print(f"   Error processing {pred.get('image_path', 'unknown')}: {e}")
            errors += 1
    
    return {
        "total_files": len(predictions),
        "files_organized": sum(stats.values()),
        "errors": errors,
        "level_distribution": stats,
        "thresholds": thresholds
    }


def print_organization_summary(stats: Dict[str, Any], target_dir: str):
    """Print detailed organization summary."""
    print("\n" + "="*60)
    print("MULTICLASS DATA PREPARATION COMPLETED")
    print("="*60)
    
    total_files = stats["files_organized"]
    errors = stats["errors"]
    level_distribution = stats["level_distribution"]
    thresholds = stats["thresholds"]
    
    print(f"📊 Total files processed: {total_files:,}")
    if errors > 0:
        print(f"❌ Errors: {errors}")
    
    print(f"📁 Output directory: {target_dir}")
    print()
    
    print("📈 Violence Level Distribution:")
    print("-" * 50)
    
    level_names = [
        "Level 0 (No Violence)",
        "Level 1 (Mild Violence)",
        "Level 2 (Moderate Violence)", 
        "Level 3 (High Violence)",
        "Level 4 (Extreme Violence)",
    ]
    
    threshold_ranges = [
        f"0.0 - {thresholds[0]:.1f}",
        f"{thresholds[0]:.1f} - {thresholds[1]:.1f}",
        f"{thresholds[1]:.1f} - {thresholds[2]:.1f}",
        f"{thresholds[2]:.1f} - {thresholds[3]:.1f}",
        f"{thresholds[3]:.1f} - 1.0",
    ]
    
    for i in range(5):
        count = level_distribution[i]
        percentage = (count / total_files * 100) if total_files > 0 else 0
        print(f"{level_names[i]:25} ({threshold_ranges[i]:>9}): {count:5,d} files ({percentage:5.1f}%)")
    
    print("-" * 50)
    
    # Violence vs Non-violence summary
    violence_files = sum(level_distribution[i] for i in range(1, 5))
    non_violence_files = level_distribution[0]
    violence_rate = (violence_files / total_files * 100) if total_files > 0 else 0
    
    print(f"🔴 Violence files (Levels 1-4): {violence_files:,} ({violence_rate:.1f}%)")
    print(f"🟢 Non-violence files (Level 0): {non_violence_files:,} ({100-violence_rate:.1f}%)")
    
    print("="*60)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare multiclass data from binary model predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--binary-model", 
        "-m",
        type=str, 
        required=True,
        help="Path to trained binary model checkpoint"
    )
    
    parser.add_argument(
        "--source-dir",
        "-s", 
        type=str, 
        default="data/raw",
        help="Source directory containing images"
    )
    
    parser.add_argument(
        "--target-dir",
        "-t",
        type=str,
        default="data/multiclass_data", 
        help="Target directory for organized multiclass data"
    )
    
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/binary_config.yaml",
        help="Binary model configuration file"
    )
    
    parser.add_argument(
        "--predictions-file",
        "-p", 
        type=str,
        default="predictions_for_multiclass.json",
        help="Temporary file to store predictions"
    )
    
    parser.add_argument(
        "--move-files",
        action="store_true",
        help="Move files instead of copying them"
    )
    
    parser.add_argument(
        "--thresholds",
        nargs=4,
        type=float,
        default=[0.2, 0.4, 0.6, 0.8],
        help="Violence level thresholds [level1, level2, level3, level4]"
    )
    
    parser.add_argument(
        "--keep-predictions",
        action="store_true", 
        help="Keep the predictions JSON file after processing"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate paths
    binary_model_path = Path(args.binary_model)
    source_dir = Path(args.source_dir)
    config_path = Path(args.config)
    
    if not binary_model_path.exists():
        print(f"❌ Binary model not found: {binary_model_path}")
        return 1
    
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return 1
        
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return 1
    
    print("🚀 Starting multiclass data preparation...")
    print(f"   Binary model: {binary_model_path}")
    print(f"   Source directory: {source_dir}")
    print(f"   Target directory: {args.target_dir}")
    print(f"   Violence thresholds: {args.thresholds}")
    print()
    
    try:
        # Step 1: Run binary inference
        success = run_binary_inference(
            str(binary_model_path),
            str(source_dir),
            args.predictions_file,
            str(config_path)
        )
        
        if not success:
            return 1
        
        # Step 2: Organize images by violence levels
        stats = organize_images_by_levels(
            args.predictions_file,
            args.target_dir,
            copy_mode=not args.move_files,
            thresholds=args.thresholds
        )
        
        # Step 3: Print summary
        print_organization_summary(stats, args.target_dir)
        
        # Step 4: Save summary to file
        summary_file = f"{args.target_dir}/preparation_summary.json"
        with open(summary_file, "w") as f:
            json.dump({
                "preparation_settings": {
                    "binary_model": str(binary_model_path),
                    "source_directory": str(source_dir),
                    "target_directory": args.target_dir,
                    "config_file": str(config_path),
                    "thresholds": args.thresholds,
                    "copy_mode": not args.move_files
                },
                "statistics": stats
            }, f, indent=2)
        
        print(f"📋 Detailed summary saved to: {summary_file}")
        
        # Step 5: Cleanup predictions file (unless requested to keep)
        if not args.keep_predictions:
            Path(args.predictions_file).unlink(missing_ok=True)
            print(f"🧹 Cleaned up temporary predictions file")
        else:
            print(f"💾 Predictions saved in: {args.predictions_file}")
        
        print("\n✅ Multiclass data preparation completed successfully!")
        print(f"🎯 Ready for multiclass training with: data in {args.target_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Preparation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
