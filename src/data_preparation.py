"""
Data preparation for fine-tuning embedding model.
Prepares positive pairs, hard negatives, and creates training dataset.
"""

import json
import random
from typing import List, Dict, Tuple

from datasets import Dataset
from dotenv import load_dotenv
import time

try:
    from .vector_store import vectorstore
except ImportError:
    from vector_store import vectorstore

load_dotenv()


class LegalDataPreparator:
    """Prepare training data for embedding fine-tuning"""
    
    def __init__(self, test_set_path: str = "test_set.json",k_hard_negatives: int=5):
        self.test_set_path= test_set_path
        self.k_hard_negatives= k_hard_negatives
        self.triplets=[]
        self.positive_pairs=[]  # (query, positive, negative)
    

    def load_test_cases(self) -> List[Dict]:
        """Load test cases from JSON file"""
        try:
            with open(self.test_set_path, "r", encoding="utf-8") as f:
                test_cases = json.load(f)
                return test_cases
        except Exception as e:
            print(f"Error loading test cases: {e}")
            return []
            

    def mine_hard_negatives(self, query: str, positive_doc_content: str, k: int = 5) -> List[str]:
        """
        Mine hard negatives: retrieve documents similar to query but NOT the ground truth
        Using FAISS retriever to get top-k similar chunks
        """
        try:
            # Retrieve similar documents
            docs = vectorstore.similarity_search(query, k=k*2) # Get more to filter
            hard_negatives =[]
            
            for doc in docs: 
                 # Filter out the positive document
                if positive_doc_content.strip()[:100] not in doc.page_content[:100]:
                    hard_negatives.append(doc.page_content)
                    if len(hard_negatives)>=k:
                        break
            return hard_negatives
        except Exception as e:
            print(f"Error mining hard negatives: {e}")
            return []


    def prepare_positive_pairs(self, test_cases: List[Dict]) -> List[Tuple[str, str]]:
        """
        Create (query, relevant_document) pairs from test set
        ground_truth format: list of relevant document contents
        """
        for i, case in enumerate(test_cases):
            query = case["question"]
            ground_truth = case.get("ground_truth", [])

            if isinstance(ground_truth, list) and len(ground_truth) > 0:
                for gt in ground_truth:
                    if isinstance(gt, str) and len(gt) > 10:
                        self.positive_pairs.append((query, gt))
            elif isinstance(ground_truth, str) and len(ground_truth) > 10:
                self.positive_pairs.append((query, ground_truth))
        
        return self.positive_pairs
    
    def prepare_triplets(self, test_cases: List[Dict], use_mining: bool = True) -> List[Tuple[str, str, str]]:
        """
        Create triplets: (query, positive_doc, negative_doc)
        For contrastive learning and triplet loss
        
        Args:
            test_cases: List of test cases with questions and ground_truth
            use_mining: If True, mine hard negatives from FAISS; else use random negatives
        """
        print(f"Preparing triplets from {len(test_cases)} test cases...")
        
        for i, case in enumerate(test_cases):
            query = case["question"]
            ground_truth = case.get("ground_truth", [])
            
            # Ensure ground_truth is a list
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth]
            
            if not ground_truth or not any(isinstance(gt, str) and len(gt) > 10 for gt in ground_truth):
                continue
            
            # Get first valid positive
            positive = next((gt for gt in ground_truth if isinstance(gt, str) and len(gt) > 10), None)
            if not positive:
                continue
            
            # Mine or create negatives
            if use_mining:
                negatives = self.mine_hard_negatives(query, positive, k=self.k_hard_negatives)
            else:
                # Random negatives from other queries' ground truths
                negatives = self._get_random_negatives(query, positive, test_cases, k=self.k_hard_negatives)
            
            # Create triplets with each negative
            for negative in negatives:
                if negative.strip():
                    self.triplets.append((query, positive, negative))
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(test_cases)} cases, {len(self.triplets)} triplets so far")
            
            time.sleep(0.1)  # Rate limiting
        
        return self.triplets
    
    def _get_random_negatives(self, query: str, positive: str, test_cases: List[Dict], k: int = 5) -> List[str]:
        """Get random negatives from other test cases"""
        negatives = []
        other_ground_truths = []
        
        for case in test_cases:
            if case["question"] != query:
                gt = case.get("ground_truth", [])
                if isinstance(gt, list):
                    other_ground_truths.extend([g for g in gt if isinstance(g, str) and len(g) > 10])
                elif isinstance(gt, str) and len(gt) > 10:
                    other_ground_truths.append(gt)
        
        if other_ground_truths:
            negatives = random.sample(other_ground_truths, min(k, len(other_ground_truths)))
        
        return negatives
    
    def create_dataset(self) -> Dataset:
        """
        Create HuggingFace Dataset from triplets
        Format suitable for ContrastiveHardNegativesLoss or TripletLoss
        """
        if not self.triplets:
            print("No triplets prepared! Call prepare_triplets() first")
            return None
        
        dataset_dict = {
            "anchor": [t[0] for t in self.triplets],      # queries
            "positive": [t[1] for t in self.triplets],    # positive documents
            "negative": [t[2] for t in self.triplets],    # negative documents
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        print(f"Created dataset with {len(dataset)} triplets")
        
        return dataset
    
    def save_dataset(self, dataset: Dataset, output_path: str = "legal_triplets_dataset"):
        """Save dataset to disk"""
        if dataset is None:
            print("No dataset to save")
            return
        dataset.save_to_disk(output_path)
        print(f"Dataset saved to {output_path}")
    
    def prepare_all(self, save_to_disk: bool = True) -> Dataset:
        """One-shot preparation: load -> prepare positive pairs -> mine hard negatives -> create dataset"""
        test_cases = self.load_test_cases()
        
        print(f"Loading {len(test_cases)} test cases...")
        self.prepare_positive_pairs(test_cases)
        print(f"✓ Prepared {len(self.positive_pairs)} positive pairs")
        
        print("\nMining hard negatives...")
        self.prepare_triplets(test_cases, use_mining=True)
        print(f"✓ Created {len(self.triplets)} triplets")
        
        dataset = self.create_dataset()
        
        if save_to_disk:
            self.save_dataset(dataset)
        
        return dataset


if __name__ == "__main__":
    # Example usage
    preparator = LegalDataPreparator(k_hard_negatives=3)
    dataset = preparator.prepare_all(save_to_disk=True)
    
    print("\n=== Dataset Sample ===")
    if dataset and len(dataset) > 0:
        sample = dataset[0]
        print(f"Query: {sample['anchor'][:100]}...")
        print(f"Positive: {sample['positive'][:100]}...")
        print(f"Negative: {sample['negative'][:100]}...")
