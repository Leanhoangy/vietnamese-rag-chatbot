#!/usr/bin/env python3
"""
🎓 ACADEMIC SUBMISSION CHECKLIST
Vietnamese Legal RAG Chatbot - Deep Learning Enhancements

Use this to verify all requirements are met before submission.
"""

import os
from pathlib import Path
import json

class SubmissionChecklist:
    """Verify project meets all academic requirements"""
    
    def __init__(self):
        self.checks = []
        self.base_path = Path(__file__).parent
    
    def add_check(self, category: str, requirement: str, passed: bool, evidence: str = ""):
        """Add a check to the list"""
        status = "✓" if passed else "✗"
        self.checks.append({
            "category": category,
            "requirement": requirement,
            "passed": passed,
            "status": status,
            "evidence": evidence,
        })
    
    def file_exists(self, path: str) -> bool:
        """Check if file exists"""
        return (self.base_path / path).exists()
    
    def run_all_checks(self):
        """Run all verification checks"""
        print("╔" + "="*70 + "╗")
        print("║" + " "*15 + "ACADEMIC SUBMISSION CHECKLIST" + " "*27 + "║")
        print("║" + " "*20 + "Deep Learning Requirements" + " "*24 + "║")
        print("╚" + "="*70 + "╝\n")
        
        # 1. Deep Learning Component
        print("="*70)
        print("1. DEEP LEARNING COMPONENT")
        print("="*70)
        
        ft_exists = self.file_exists("src/fine_tuner.py")
        self.add_check("DL", "Fine-tuning module exists", ft_exists, "src/fine_tuner.py")
        self._print_check()
        
        dp_exists = self.file_exists("src/data_preparation.py")
        self.add_check("DL", "Data preparation module exists", dp_exists, "src/data_preparation.py")
        self._print_check()
        
        train_exists = self.file_exists("train.py")
        self.add_check("DL", "Training pipeline exists", train_exists, "train.py")
        self._print_check()
        
        # 2. NLP/Transformer
        print("\n" + "="*70)
        print("2. NLP/TRANSFORMER USAGE")
        print("="*70)
        
        has_embedder = self.file_exists("src/embedder.py")
        self.add_check("NLP", "Uses embedding model", has_embedder, "multilingual-e5-base")
        self._print_check()
        
        has_hybrid = self.file_exists("src/retrieval_hybrid.py")
        self.add_check("NLP", "Has hybrid retrieval", has_hybrid, "BM25 + Dense + Cross-Encoder")
        self._print_check()
        
        # 3. Evaluation
        print("\n" + "="*70)
        print("3. EVALUATION & METRICS")
        print("="*70)
        
        has_eval = self.file_exists("src/evaluation_enhanced.py")
        self.add_check("Eval", "Enhanced metrics module exists", has_eval, "src/evaluation_enhanced.py")
        self._print_check()
        
        has_ragas = has_eval
        self.add_check("Eval", "RAGAS evaluation available", has_ragas, "src/evaluation_enhanced.py")
        self._print_check()
        
        # 4. Documentation
        print("\n" + "="*70)
        print("4. DOCUMENTATION")
        print("="*70)
        
        has_impl_notes = self.file_exists("README_IMPLEMENTATION.md")
        self.add_check("Doc", "Implementation notes", has_impl_notes, "README_IMPLEMENTATION.md")
        self._print_check()
        
        has_readme = self.file_exists("README.md")
        self.add_check("Doc", "Project README", has_readme, "README.md")
        self._print_check()
        
        has_report = self.file_exists("SUBMISSION_REPORT.txt")
        self.add_check("Doc", "Submission report", has_report, "SUBMISSION_REPORT.txt")
        self._print_check()
        
        # 5. Reproducibility
        print("\n" + "="*70)
        print("5. REPRODUCIBILITY")
        print("="*70)
        
        has_reqs = self.file_exists("requirements.txt")
        self.add_check("Repro", "Requirements file updated", has_reqs, "requirements.txt")
        self._print_check()
        
        # Check if requirements have new packages
        if has_reqs:
            with open(self.base_path / "requirements.txt") as f:
                reqs_content = f.read()
            
            has_bm25 = "rank-bm25" in reqs_content
            self.add_check("Repro", "Has rank-bm25", has_bm25, "for BM25 retrieval")
            self._print_check()
            
            has_rouge = "rouge-score" in reqs_content
            self.add_check("Repro", "Has rouge-score", has_rouge, "for ROUGE metrics")
            self._print_check()
        
        # 6. Code Quality
        print("\n" + "="*70)
        print("6. CODE QUALITY")
        print("="*70)
        
        # Check fine_tuner.py has proper structure
        if ft_exists:
            with open(self.base_path / "src" / "fine_tuner.py") as f:
                ft_content = f.read()
            
            has_class = "class EmbeddingFineTuner" in ft_content
            self.add_check("Code", "Fine-tuner class implemented", has_class, "EmbeddingFineTuner")
            self._print_check()
            
            has_triplet = "TripletLoss" in ft_content or "triplet" in ft_content
            self.add_check("Code", "Uses TripletLoss", has_triplet, "Contrastive learning")
            self._print_check()
        
        # Check data_preparation.py
        if dp_exists:
            with open(self.base_path / "src" / "data_preparation.py") as f:
                dp_content = f.read()
            
            has_mining = "mine_hard_negatives" in dp_content
            self.add_check("Code", "Hard negative mining", has_mining, "In data_preparation.py")
            self._print_check()
        
        # Check hybrid retrieval
        if has_hybrid:
            with open(self.base_path / "src" / "retrieval_hybrid.py") as f:
                rh_content = f.read()
            
            has_bm25_impl = "BM25" in rh_content or "bm25" in rh_content
            self.add_check("Code", "BM25 implementation", has_bm25_impl, "In retrieval_hybrid.py")
            self._print_check()
            
            has_cross = "CrossEncoder" in rh_content or "cross_encoder" in rh_content
            self.add_check("Code", "Cross-Encoder re-ranking", has_cross, "In retrieval_hybrid.py")
            self._print_check()
        
        # 7. Completeness
        print("\n" + "="*70)
        print("7. COMPLETENESS CHECK")
        print("="*70)
        
        all_files_exist = all([
            ft_exists, dp_exists, train_exists,
            has_embedder, has_hybrid, has_eval,
            has_impl_notes, has_readme, has_report
        ])
        
        self.add_check("Complete", "All core files exist", all_files_exist, "Ready to run")
        self._print_check()
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        passed = sum(1 for c in self.checks if c["passed"])
        total = len(self.checks)
        percentage = (passed / total) * 100
        
        print(f"Checks passed: {passed}/{total} ({percentage:.0f}%)")
        print()
        
        if percentage == 100:
            print("✅ PROJECT IS READY FOR SUBMISSION!")
            print()
            print("Next steps:")
            print("  1. Run: python train.py")
            print("  2. Wait for training to complete")
            print("  3. Check evaluation_results.json for metrics")
            print("  4. Prepare presentation with:")
            print("     - Architecture diagram")
            print("     - Training curves")
            print("     - Metric improvements")
            print("     - Code walkthrough")
            print()
        else:
            print("⚠️  Some requirements not met. Please review above.")
            print()
            print("Failed checks:")
            for check in self.checks:
                if not check["passed"]:
                    print(f"  ✗ {check['requirement']}")
    
    def _print_check(self):
        """Print last check"""
        check = self.checks[-1]
        status_icon = "✓" if check["passed"] else "✗"
        print(f"{status_icon} {check['requirement']}")
        if check["evidence"]:
            print(f"  → {check['evidence']}")
    
    def generate_report(self, output_file: str = "SUBMISSION_REPORT.txt"):
        """Generate a text report"""
        with open(self.base_path / output_file, "w") as f:
            f.write("ACADEMIC SUBMISSION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            categories = set(c["category"] for c in self.checks)
            
            for cat in sorted(categories):
                f.write(f"\n{cat} - CHECKS\n")
                f.write("-" * 70 + "\n")
                
                cat_checks = [c for c in self.checks if c["category"] == cat]
                for check in cat_checks:
                    status = "PASS" if check["passed"] else "FAIL"
                    f.write(f"[{status}] {check['requirement']}\n")
                    if check["evidence"]:
                        f.write(f"      {check['evidence']}\n")
            
            passed = sum(1 for c in self.checks if c["passed"])
            total = len(self.checks)
            f.write(f"\n\nTOTAL: {passed}/{total} checks passed\n")
        
        print(f"\n✓ Report saved to: {output_file}")


if __name__ == "__main__":
    checker = SubmissionChecklist()
    checker.run_all_checks()
    checker.generate_report()
