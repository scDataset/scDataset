"""
Label encoder for multi-task classification on Tahoe-100M dataset.

This module provides label encoding for all 4 classification tasks:
- Cell line classification
- Drug classification
- Mechanism of action (broad labels)
- Mechanism of action (fine labels)
"""

import os
import pickle
from typing import Dict, List, Tuple


class LabelEncoder:
    """
    Label encoder for multi-task classification using pickle files.

    This encoder loads pre-computed mapping dictionaries that map string labels
    to integer IDs for each of the 4 classification tasks.

    Attributes
    ----------
    cell_line_to_id : dict
        Mapping from cell line names to integer IDs
    drug_to_id : dict
        Mapping from drug names to integer IDs
    drug_to_moa_broad : dict
        Mapping from drug ID to broad MOA category ID
    drug_to_moa_fine : dict
        Mapping from drug ID to fine-grained MOA category ID
    num_cell_lines : int
        Number of unique cell lines
    num_drugs : int
        Number of unique drugs
    num_moa_broad : int
        Number of broad MOA categories
    num_moa_fine : int
        Number of fine-grained MOA categories
    """

    def __init__(self, data_dir: str = "/home/kidara/raid/volume/scdataset_private"):
        """
        Initialize label encoder with pickle files.

        Parameters
        ----------
        data_dir : str
            Directory containing pickle files with label mappings
        """
        self.data_dir = data_dir

        # Load mapping dictionaries
        self.cell_line_to_id = self._load_pickle("cell_line_code_map.pkl")
        self.drug_to_id = self._load_pickle("drug_code_map.pkl")
        self.drug_to_moa_broad = self._load_pickle("drug_to_moa_broad.pkl")
        self.drug_to_moa_fine = self._load_pickle("drug_to_moa_fine.pkl")

        # Compute task dimensions
        self.num_cell_lines = len(self.cell_line_to_id)
        self.num_drugs = len(self.drug_to_id)
        self.num_moa_broad = len(set(self.drug_to_moa_broad.values()))
        self.num_moa_fine = len(set(self.drug_to_moa_fine.values()))

    def _load_pickle(self, filename: str) -> Dict:
        """Load pickle file."""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def encode_labels(
        self, cell_lines: List[str], drugs: List[str]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Encode all labels for a batch of samples.

        Parameters
        ----------
        cell_lines : list of str
            List of cell line identifiers
        drugs : list of str
            List of drug names

        Returns
        -------
        list of tuple
            Each tuple is (cell_line_id, drug_id, moa_broad_id, moa_fine_id)
        """
        if len(cell_lines) != len(drugs):
            raise ValueError("cell_lines and drugs must have the same length")

        labels = []
        for cell_line, drug in zip(cell_lines, drugs):
            cell_line_id = self.cell_line_to_id[cell_line]
            drug_id = self.drug_to_id[drug]
            moa_broad_id = self.drug_to_moa_broad[drug_id]
            moa_fine_id = self.drug_to_moa_fine[drug_id]
            labels.append((cell_line_id, drug_id, moa_broad_id, moa_fine_id))
        return labels

    def get_task_dims(self) -> Dict[str, int]:
        """
        Get number of classes for each task.

        Returns
        -------
        dict
            Dictionary with task names as keys and number of classes as values
        """
        return {
            "cell_line": self.num_cell_lines,
            "drug": self.num_drugs,
            "moa_broad": self.num_moa_broad,
            "moa_fine": self.num_moa_fine,
        }

    def __repr__(self) -> str:
        return (
            f"LabelEncoder(cell_lines={self.num_cell_lines}, "
            f"drugs={self.num_drugs}, moa_broad={self.num_moa_broad}, "
            f"moa_fine={self.num_moa_fine})"
        )
