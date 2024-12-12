import matplotlib.pyplot as plt
from Bio import SeqIO, pairwise2
from Bio.pairwise2 import format_alignment
import pandas as pd

genome_1 = "ATGCGTACGTAGCTAGCTAGCTAGCGTACG"
genome_2 = "ATGCGTACGTAGCTAGCTGCTAGCGTAGCA"

alignments = pairwise2.align.globalxx(genome_1, genome_2)
print("Best Alignment:")
print(format_alignment(*alignments[0]))

alignment_1, alignment_2, score, start, end = alignments[0]
similarity = sum(1 for a, b in zip(alignment_1, alignment_2) if a == b) / len(alignment_1) * 100
print(f"Similarity: {similarity:.2f}%")

matches = [1 if a == b else 0 for a, b in zip(alignment_1, alignment_2)]
plt.figure(figsize=(10, 4))
plt.plot(range(len(matches)), matches, label="Match (1 = Match, 0 = Mismatch)", color="blue")

for idx, match in enumerate(matches):
    if match == 0:
        plt.text(idx, match, alignment_1[idx], color="red", fontsize=8, ha="center")
        plt.text(idx, match - 0.1, alignment_2[idx], color="green", fontsize=8, ha="center")


plt.title("Genome Strand Comparison")
plt.xlabel("Position")
plt.ylabel("Match (1=Match, 0=Mismatch)")
plt.ylim(-0.5, 1.5)
plt.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.7)
plt.legend()
plt.tight_layout()
plt.show()