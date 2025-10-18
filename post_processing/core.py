"""
Core Post-Processing Module
==========================

Implements the main residue-to-pocket conversion pipeline following the default recipe:
1. Surface filtering (RSA ≥ 0.2)
2. Graph construction (Cα-Cα ≤ 8Å)
3. Connected components clustering
4. Cluster filtering (size ≥ 5, ∑p ≥ 2.0)
5. Pocket scoring and ranking

Author: PockNet Team
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict
import logging
from dataclasses import dataclass
from sklearn.neighbors import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Residue:
    """Represents a single residue with all necessary information for pocket prediction."""
    chain: str
    res_id: int
    xyz: np.ndarray  # 3D coordinates (Cβ/Cα)
    rsa: float       # Relative surface accessibility
    prob: float      # Prediction probability
    res_name: Optional[str] = None
    atom_type: Optional[str] = "CA"  # Default to Cα


@dataclass
class Pocket:
    """Represents a predicted binding pocket."""
    members: List[Tuple[str, int]]  # [(chain, res_id), ...]
    size: int
    center: np.ndarray              # 3D center coordinates
    score: float                    # Pocket score (∑p × surface_fraction)
    sump: float                     # Sum of probabilities
    surface_fraction: float         # Fraction of surface residues
    confidence: Optional[float] = None


class PocketPostProcessor:
    """
    Main post-processing class implementing the default recipe for 
    converting residue predictions to pocket predictions.
    """
    
    def __init__(self, 
                 rsa_min: float = 0.2,
                 graph_radius: float = 8.0,
                 knn_fallback: int = 16,
                 min_cluster_size: int = 5,
                 min_sump: float = 2.0,
                 max_pockets: int = 5):
        """
        Initialize post-processor with default parameters.
        
        Args:
            rsa_min: Minimum RSA for surface filtering (default: 0.2)
            graph_radius: Maximum Cα-Cα distance for graph edges (default: 8.0 Å)
            knn_fallback: k-NN fallback if radius graph is too sparse (default: 16)
            min_cluster_size: Minimum cluster size to keep (default: 5)
            min_sump: Minimum sum of probabilities to keep cluster (default: 2.0)
            max_pockets: Maximum number of top pockets to return (default: 5)
        """
        self.rsa_min = rsa_min
        self.graph_radius = graph_radius
        self.knn_fallback = knn_fallback
        self.min_cluster_size = min_cluster_size
        self.min_sump = min_sump
        self.max_pockets = max_pockets
        
        logger.info(f"Initialized PocketPostProcessor with parameters:")
        logger.info(f"  RSA threshold: {rsa_min}")
        logger.info(f"  Graph radius: {graph_radius} Å")
        logger.info(f"  Min cluster size: {min_cluster_size}")
        logger.info(f"  Min ∑p: {min_sump}")
    
    def process(self, 
                residues: List[Union[Residue, Dict]], 
                threshold: Optional[float] = None,
                adaptive_strategy: str = "percentile95") -> List[Pocket]:
        """
        Main processing function to convert residues to pockets.
        
        Args:
            residues: List of Residue objects or dictionaries
            threshold: Fixed threshold for positives (if None, use adaptive)
            adaptive_strategy: "percentile95", "prevalence", or "global_opt"
            
        Returns:
            List of Pocket objects sorted by score (descending)
        """
        logger.info(f"Processing {len(residues)} residues")
        
        # Convert to standard format if needed
        if residues and isinstance(residues[0], dict):
            residues = [self._dict_to_residue(r) for r in residues]
        
        # Step 1: Surface filtering
        surface_mask = np.array([r.rsa >= self.rsa_min for r in residues])
        probs = np.array([r.prob for r in residues])
        coords = np.stack([r.xyz for r in residues])
        
        logger.info(f"Surface residues: {surface_mask.sum()}/{len(residues)} "
                   f"({100*surface_mask.mean():.1f}%)")
        
        # Step 2: Adaptive thresholding
        if threshold is None:
            threshold = self._compute_adaptive_threshold(probs, adaptive_strategy)
        
        logger.info(f"Using threshold: {threshold:.4f}")
        
        # Step 3: Select positive residues
        positive_mask = (surface_mask) & (probs >= threshold)
        positive_indices = np.where(positive_mask)[0]
        
        if len(positive_indices) == 0:
            logger.warning("No positive residues found after filtering")
            return []
        
        logger.info(f"Positive residues: {len(positive_indices)} "
                   f"({100*len(positive_indices)/len(residues):.1f}%)")
        
        # Step 4: Build graph on all residues
        graph = self._make_graph(coords, self.graph_radius, self.knn_fallback)
        
        # Step 5: Find connected components on positive subgraph
        pockets = self._find_pocket_clusters(
            residues, positive_indices, graph, probs, surface_mask
        )
        
        # Step 6: Sort and limit
        pockets.sort(key=lambda p: p.score, reverse=True)
        final_pockets = pockets[:self.max_pockets]
        
        logger.info(f"Found {len(pockets)} clusters, returning top {len(final_pockets)}")
        
        return final_pockets
    
    def _dict_to_residue(self, res_dict: Dict) -> Residue:
        """Convert dictionary to Residue object."""
        return Residue(
            chain=res_dict['chain'],
            res_id=res_dict['res_id'],
            xyz=np.array(res_dict['xyz']),
            rsa=res_dict['rsa'],
            prob=res_dict['prob'],
            res_name=res_dict.get('res_name'),
            atom_type=res_dict.get('atom_type', 'CA')
        )
    
    def _compute_adaptive_threshold(self, probs: np.ndarray, strategy: str) -> float:
        """Compute adaptive threshold based on strategy."""
        if strategy == "percentile95":
            return float(np.percentile(probs, 95.0))
        elif strategy == "prevalence":
            # Assume typical pocket prevalence of 3%
            return float(np.percentile(probs, 97.0))
        elif strategy == "global_opt":
            # Would need validation data - fallback to percentile
            logger.warning("global_opt strategy needs validation data, using percentile95")
            return float(np.percentile(probs, 95.0))
        else:
            raise ValueError(f"Unknown adaptive strategy: {strategy}")
    
    def _make_graph(self, coords: np.ndarray, radius: float, knn: int) -> csr_matrix:
        """
        Build residue graph based on spatial proximity.
        
        Args:
            coords: (N, 3) coordinate array
            radius: Maximum distance for edges
            knn: k-NN fallback
            
        Returns:
            Sparse adjacency matrix
        """
        n_residues = len(coords)
        tree = KDTree(coords)
        
        # Use radius neighbors with k-NN fallback
        k_actual = min(knn, n_residues)
        indices = tree.query(coords, k=k_actual)[1]
        
        rows, cols = [], []
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if i == j:
                    continue
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist <= radius:
                    rows.append(i)
                    cols.append(j)
        
        # Create symmetric adjacency matrix
        data = np.ones(len(rows), dtype=np.uint8)
        graph = csr_matrix((data, (rows, cols)), shape=(n_residues, n_residues))
        graph = graph.maximum(graph.transpose())
        
        logger.info(f"Graph: {n_residues} nodes, {graph.nnz//2} edges "
                   f"(avg degree: {graph.nnz/n_residues:.1f})")
        
        return graph
    
    def _find_pocket_clusters(self, 
                             residues: List[Residue],
                             positive_indices: np.ndarray,
                             graph: csr_matrix,
                             probs: np.ndarray,
                             surface_mask: np.ndarray) -> List[Pocket]:
        """Find pocket clusters using connected components."""
        
        # Extract subgraph for positive residues
        pos_mapping = {idx: i for i, idx in enumerate(positive_indices)}
        
        # Build subgraph adjacency
        sub_rows, sub_cols = [], []
        for i, idx_i in enumerate(positive_indices):
            neighbors = graph[idx_i].indices
            for idx_j in neighbors:
                if idx_j in pos_mapping:
                    j = pos_mapping[idx_j]
                    sub_rows.append(i)
                    sub_cols.append(j)
        
        n_pos = len(positive_indices)
        sub_data = np.ones(len(sub_rows), dtype=np.uint8)
        sub_graph = csr_matrix((sub_data, (sub_rows, sub_cols)), shape=(n_pos, n_pos))
        
        # Find connected components
        n_components, labels = connected_components(sub_graph, directed=False)
        
        logger.info(f"Found {n_components} connected components")
        
        # Process each component
        pockets = []
        for comp_id in range(n_components):
            comp_local_indices = np.where(labels == comp_id)[0]
            comp_global_indices = positive_indices[comp_local_indices]
            
            # Apply filters
            cluster_size = len(comp_global_indices)
            cluster_sump = float(probs[comp_global_indices].sum())
            
            if cluster_size < self.min_cluster_size:
                continue
            if cluster_sump < self.min_sump:
                continue
            
            # Create pocket
            pocket = self._create_pocket(
                residues, comp_global_indices, probs, surface_mask
            )
            pockets.append(pocket)
        
        return pockets
    
    def _create_pocket(self,
                      residues: List[Residue],
                      indices: np.ndarray,
                      probs: np.ndarray,
                      surface_mask: np.ndarray) -> Pocket:
        """Create a Pocket object from cluster indices."""
        
        cluster_probs = probs[indices]
        cluster_coords = np.stack([residues[i].xyz for i in indices])
        cluster_surface = surface_mask[indices]
        
        # Probability-weighted centroid
        total_prob = cluster_probs.sum()
        center = (cluster_probs[:, None] * cluster_coords).sum(axis=0) / (total_prob + 1e-8)
        
        # Surface fraction
        surface_fraction = float(cluster_surface.mean())
        
        # Pocket score
        score = total_prob * surface_fraction
        
        # Member residues
        members = [(residues[i].chain, residues[i].res_id) for i in indices]
        
        return Pocket(
            members=members,
            size=len(indices),
            center=center,
            score=score,
            sump=float(total_prob),
            surface_fraction=surface_fraction
        )


def residue_to_pockets(residues: List[Union[Residue, Dict]], 
                      threshold: Optional[float] = None,
                      **kwargs) -> List[Pocket]:
    """
    Convenience function for residue-to-pocket conversion.
    
    Args:
        residues: List of residues (Residue objects or dicts)
        threshold: Probability threshold (if None, use adaptive)
        **kwargs: Additional parameters for PocketPostProcessor
        
    Returns:
        List of predicted pockets
    """
    processor = PocketPostProcessor(**kwargs)
    return processor.process(residues, threshold)


def make_residue_graph(coords: np.ndarray, 
                      radius: float = 8.0, 
                      knn: int = 16) -> csr_matrix:
    """
    Build residue connectivity graph.
    
    Args:
        coords: (N, 3) residue coordinates
        radius: Maximum edge distance
        knn: k-NN fallback
        
    Returns:
        Sparse adjacency matrix
    """
    processor = PocketPostProcessor()
    return processor._make_graph(coords, radius, knn)


def postprocess_residues(residues: List[Dict],
                        rsa_min: float = 0.2,
                        graph_radius: float = 8.0,
                        min_size: int = 5,
                        sump_min: float = 2.0,
                        threshold: Optional[float] = None) -> List[Dict]:
    """
    Minimal implementation matching the provided sketch.
    
    Args:
        residues: List of dicts with keys: chain, res_id, xyz, rsa, prob
        rsa_min: RSA threshold for surface filtering
        graph_radius: Graph connectivity radius
        min_size: Minimum cluster size
        sump_min: Minimum sum of probabilities
        threshold: Probability threshold (if None, use 95th percentile)
        
    Returns:
        List of pocket dictionaries
    """
    # Surface mask
    mask = np.array([r['rsa'] >= rsa_min for r in residues])
    probs = np.array([r['prob'] for r in residues])
    coords = np.stack([r['xyz'] for r in residues])
    
    # Adaptive threshold
    if threshold is None:
        threshold = float(np.percentile(probs, 95.0))
    
    # Select positive residues
    keep = np.where((mask) & (probs >= threshold))[0]
    if len(keep) == 0:
        return []
    
    # Build graph
    processor = PocketPostProcessor()
    graph = processor._make_graph(coords, graph_radius, 16)
    
    # Connected components on positive subgraph
    sub_idx = keep
    mapping = {idx: i for i, idx in enumerate(sub_idx)}
    rows, cols = [], []
    
    for idx_i in sub_idx:
        neighbors = graph[idx_i].indices
        for idx_j in neighbors:
            if idx_j in mapping:
                i, j = mapping[idx_i], mapping[idx_j]
                rows.append(i)
                cols.append(j)
    
    n_sub = len(sub_idx)
    if len(rows) == 0:
        # No connections - each residue is its own component
        components_data = [(i, [i]) for i in range(n_sub)]
    else:
        pos_graph = csr_matrix((np.ones_like(rows), (rows, cols)), shape=(n_sub, n_sub))
        n_comp, labels = connected_components(pos_graph, directed=False)
        components_data = [(c, np.where(labels == c)[0]) for c in range(n_comp)]
    
    # Process components
    pockets = []
    for comp_id, comp_local in components_data:
        comp_global = sub_idx[comp_local]
        
        if len(comp_global) < min_size:
            continue
        
        sump = float(probs[comp_global].sum())
        if sump < sump_min:
            continue
        
        # Compute center and surface fraction
        comp_coords = coords[comp_global]
        comp_probs = probs[comp_global]
        center = (comp_probs[:, None] * comp_coords).sum(axis=0) / (sump + 1e-8)
        
        surf_frac = float((np.array([residues[i]['rsa'] for i in comp_global]) >= rsa_min).mean())
        score = sump * surf_frac
        
        pockets.append({
            'members': [(residues[i]['chain'], residues[i]['res_id']) for i in comp_global],
            'size': len(comp_global),
            'sump': sump,
            'surface_fraction': surf_frac,
            'score': score,
            'center': center.tolist()
        })
    
    # Sort by score
    pockets.sort(key=lambda x: x['score'], reverse=True)
    return pockets


# Example usage and testing
if __name__ == "__main__":
    # Create some test data
    np.random.seed(42)
    n_residues = 100
    
    test_residues = []
    for i in range(n_residues):
        residue = {
            'chain': 'A',
            'res_id': i + 1,
            'xyz': np.random.randn(3) * 10,
            'rsa': np.random.rand(),
            'prob': np.random.rand()
        }
        test_residues.append(residue)
    
    # Test the post-processing
    pockets = postprocess_residues(test_residues)
    print(f"Found {len(pockets)} pockets")
    
    for i, pocket in enumerate(pockets[:3]):
        print(f"Pocket {i+1}: size={pocket['size']}, score={pocket['score']:.3f}")