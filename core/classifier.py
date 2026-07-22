import logging
import numpy as np
import os
import glob

logger = logging.getLogger(__name__)

class AnomalyClassifier:
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.centroid = None
        self.references_matrix = None
        self.references = [] 
        logger.info("Initialized AnomalyClassifier (core).")

    def load_references(self, embeddings_dir):
        """
        Loads the reference embeddings matrix for anomaly detection.
        It looks for the first .npy file in the directory (e.g., 'san_pham_tot.npy').
        """
        self.references_matrix = None
        try:
            if not os.path.isdir(embeddings_dir):
                logger.error(f"Embeddings directory not found: {embeddings_dir}")
                return

            found_files = [f for f in glob.glob(os.path.join(embeddings_dir, "*.npy"))]
            if not found_files:
                logger.warning(f"No .npy embedding files found in {embeddings_dir}.")
                return

            # Load the first .npy file found as the matrix
            centroid_path = found_files[0]
            emb_array = np.load(centroid_path)
            
            # Handle if the old file is a 1D vector or new file is 2D matrix
            if len(emb_array.shape) == 1:
                emb_array = emb_array.reshape(1, -1)
                
            # Normalize all embeddings in the matrix
            norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
            norms[norms == 0] = 1 # prevent div by zero
            self.references_matrix = emb_array / norms
            logger.info(f"Loaded reference matrix of shape {self.references_matrix.shape} from: {os.path.basename(centroid_path)}")

        except Exception as e:
            logger.error(f"Error loading reference matrix: {e}", exc_info=True)

    def classify(self, embedding):
        """
        Classifies if an embedding is normal (OK) or anomaly (NG)
        by finding the maximum cosine similarity against all references (rotation-invariant).
        """
        if embedding is None:
            return {'result': 'NG', 'similarity': 0.0, 'threshold': self.threshold}
            
        if self.references_matrix is None:
            logger.warning("No reference matrix available. Defaulting to NG.")
            return {'result': 'NG', 'similarity': 0.0, 'threshold': self.threshold}
            
        # Normalize the input embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Calculate cosine similarity against ALL references in the matrix
        similarities = np.dot(self.references_matrix, embedding)
        max_similarity = float(np.max(similarities))
                
        result = 'OK' if max_similarity >= self.threshold else 'NG'
        
        # Log the score for debugging
        logger.info(f"DEBUG (core): Max Similarity Score = {max_similarity:.4f} | Threshold = {self.threshold} -> {result}")
        
        return {
            'result': result,
            'similarity': max_similarity,
            'threshold': self.threshold
        }
