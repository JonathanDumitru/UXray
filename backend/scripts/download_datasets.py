#!/usr/bin/env python3
"""
Script to download and prepare academic datasets for saliency prediction.

Downloads MIT1003, SALICON, and OSIE datasets and organizes them
for use with the UXray heatmap generation system.
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, Any
import json


class DatasetDownloader:
    """Download and prepare academic datasets for saliency prediction."""
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize dataset downloader.
        
        Args:
            data_dir: Directory to store downloaded datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset URLs and information
        self.datasets = {
            "mit1003": {
                "name": "MIT1003",
                "description": "MIT Saliency Benchmark dataset",
                "url": "https://github.com/klauscc/mit-saliency-benchmark/archive/refs/heads/master.zip",
                "size": "~50MB",
                "license": "Academic use",
                "reference": "Judd, T., et al. (2009). Learning to predict where humans look. ICCV."
            },
            "salicon": {
                "name": "SALICON",
                "description": "Saliency in Context dataset",
                "url": "https://github.com/CLIP-Art/SALICON/archive/refs/heads/master.zip",
                "size": "~2GB",
                "license": "Creative Commons Attribution 4.0",
                "reference": "Jiang, M., et al. (2015). SALICON: Saliency in Context. CVPR."
            },
            "osie": {
                "name": "OSIE",
                "description": "Object and Semantic Images and Eye-tracking dataset",
                "url": "https://github.com/OSIE-dataset/OSIE/archive/refs/heads/master.zip",
                "size": "~100MB",
                "license": "Academic research",
                "reference": "Xu, P., et al. (2014). Predicting human gaze beyond pixels. JAIR."
            }
        }
    
    def download_file(self, url: str, filepath: Path) -> bool:
        """
        Download file from URL.
        
        Args:
            url: URL to download from
            filepath: Path to save file
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            print(f"Downloading {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_dir: Path) -> bool:
        """
        Extract archive file.
        
        Args:
            archive_path: Path to archive file
            extract_dir: Directory to extract to
            
        Returns:
            True if extraction successful, False otherwise
        """
        try:
            extract_dir.mkdir(exist_ok=True)
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif archive_path.suffix in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
            else:
                print(f"Unsupported archive format: {archive_path.suffix}")
                return False
            
            print(f"Extracted to {extract_dir}")
            return True
            
        except Exception as e:
            print(f"Error extracting {archive_path}: {e}")
            return False
    
    def organize_dataset(self, dataset_name: str, extract_dir: Path) -> bool:
        """
        Organize extracted dataset into standard structure.
        
        Args:
            dataset_name: Name of the dataset
            extract_dir: Directory containing extracted files
            
        Returns:
            True if organization successful, False otherwise
        """
        try:
            dataset_dir = self.data_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            # Create standard directory structure
            (dataset_dir / "images").mkdir(exist_ok=True)
            (dataset_dir / "fixations").mkdir(exist_ok=True)
            (dataset_dir / "saliency").mkdir(exist_ok=True)
            
            # Find and organize files
            image_files = list(extract_dir.rglob("*.jpg")) + list(extract_dir.rglob("*.png"))
            fixation_files = list(extract_dir.rglob("*.mat")) + list(extract_dir.rglob("*fixation*"))
            
            # Copy image files
            for img_file in image_files:
                if "image" in img_file.name.lower() or img_file.parent.name.lower() in ["images", "img"]:
                    dest = dataset_dir / "images" / img_file.name
                    dest.write_bytes(img_file.read_bytes())
            
            # Copy fixation files
            for fix_file in fixation_files:
                dest = dataset_dir / "fixations" / fix_file.name
                dest.write_bytes(fix_file.read_bytes())
            
            print(f"Organized {dataset_name} dataset")
            return True
            
        except Exception as e:
            print(f"Error organizing {dataset_name}: {e}")
            return False
    
    def download_dataset(self, dataset_name: str) -> bool:
        """
        Download and prepare a specific dataset.
        
        Args:
            dataset_name: Name of the dataset to download
            
        Returns:
            True if successful, False otherwise
        """
        if dataset_name not in self.datasets:
            print(f"Unknown dataset: {dataset_name}")
            return False
        
        dataset_info = self.datasets[dataset_name]
        print(f"\nDownloading {dataset_info['name']} dataset...")
        print(f"Description: {dataset_info['description']}")
        print(f"Size: {dataset_info['size']}")
        print(f"License: {dataset_info['license']}")
        print(f"Reference: {dataset_info['reference']}")
        
        # Create temporary directory
        temp_dir = self.data_dir / "temp" / dataset_name
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        archive_path = temp_dir / f"{dataset_name}.zip"
        if not self.download_file(dataset_info['url'], archive_path):
            return False
        
        # Extract dataset
        extract_dir = temp_dir / "extracted"
        if not self.extract_archive(archive_path, extract_dir):
            return False
        
        # Organize dataset
        if not self.organize_dataset(dataset_name, extract_dir):
            return False
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"Successfully prepared {dataset_name} dataset")
        return True
    
    def download_all_datasets(self) -> Dict[str, bool]:
        """
        Download all available datasets.
        
        Returns:
            Dictionary mapping dataset names to success status
        """
        results = {}
        
        print("Starting download of academic datasets for UXray heatmap generation...")
        print("=" * 60)
        
        for dataset_name in self.datasets:
            print(f"\nProcessing {dataset_name.upper()} dataset...")
            results[dataset_name] = self.download_dataset(dataset_name)
        
        # Create dataset info file
        self._create_dataset_info()
        
        return results
    
    def _create_dataset_info(self):
        """Create a JSON file with dataset information."""
        info = {
            "datasets": self.datasets,
            "data_directory": str(self.data_dir),
            "total_datasets": len(self.datasets)
        }
        
        info_file = self.data_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\nDataset information saved to {info_file}")
    
    def list_datasets(self):
        """List available datasets and their information."""
        print("Available academic datasets for saliency prediction:")
        print("=" * 60)
        
        for name, info in self.datasets.items():
            print(f"\n{name.upper()}:")
            print(f"  Name: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size']}")
            print(f"  License: {info['license']}")
            print(f"  Reference: {info['reference']}")
    
    def check_downloaded_datasets(self) -> Dict[str, bool]:
        """
        Check which datasets are already downloaded.
        
        Returns:
            Dictionary mapping dataset names to availability status
        """
        status = {}
        
        for dataset_name in self.datasets:
            dataset_dir = self.data_dir / dataset_name
            images_dir = dataset_dir / "images"
            fixations_dir = dataset_dir / "fixations"
            
            status[dataset_name] = (
                dataset_dir.exists() and 
                images_dir.exists() and 
                fixations_dir.exists() and
                len(list(images_dir.glob("*"))) > 0
            )
        
        return status


def main():
    """Main function to run dataset download."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download academic datasets for saliency prediction")
    parser.add_argument("--data-dir", default="./data", help="Directory to store datasets")
    parser.add_argument("--dataset", choices=["mit1003", "salicon", "osie", "all"], 
                       default="all", help="Dataset to download")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--check", action="store_true", help="Check downloaded datasets")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.list:
        downloader.list_datasets()
        return
    
    if args.check:
        status = downloader.check_downloaded_datasets()
        print("\nDataset availability:")
        for name, available in status.items():
            print(f"  {name}: {'✓' if available else '✗'}")
        return
    
    if args.dataset == "all":
        results = downloader.download_all_datasets()
        print("\nDownload Summary:")
        for name, success in results.items():
            print(f"  {name}: {'✓' if success else '✗'}")
    else:
        success = downloader.download_dataset(args.dataset)
        print(f"\nDownload {'✓' if success else '✗'}")


if __name__ == "__main__":
    main() 