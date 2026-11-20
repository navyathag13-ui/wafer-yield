"""What each WM-811K pattern looks like and what published papers say it may point to.

Every cause below is quoted or closely paraphrased from a paper whose text I read, and the
papers hedge ("might indicate", "for a specific technology node"). They are leads for an
engineer to check against tool, chamber and lot history, not diagnoses. Patterns with no
cause in those papers say so instead of guessing.
"""
from __future__ import annotations

SOURCES = {
    "geng2023": ("Geng, Sun, Chen, Xu, Ho and Yu, 'Mixed-Type Wafer Failure Pattern Recognition' "
                 "(invited paper), ASP-DAC 2023",
                 "http://www.cse.cuhk.edu.hk/~byu/papers/C154-ASPDAC2023-Mixwafer.pdf"),
    "hu2021": ("Hu, He and Li, 'Semi-supervised Wafer Map Pattern Recognition using Domain-Specific Data "
               "Augmentation and Contrastive Learning', IEEE International Test Conference 2021",
               "https://doi.org/10.1109/ITC50571.2021.00019"),
    "wu2015": ("Wu, Jang and Chen, 'Wafer Map Failure Pattern Recognition and Similarity Ranking for "
               "Large-Scale Data Sets', IEEE Transactions on Semiconductor Manufacturing 28, 2015 (WM-811K)",
               "https://doi.org/10.1109/TSM.2014.2364237"),
}

GUIDE = {
    "Center": {"looks": "Fails clustered in a disc around the wafer centre.",
               "causes": [("Abnormal liquid flow or liquid pressure", "geng2023")]},
    "Donut": {"looks": "A ring of fails part-way out, with a cleaner centre and edge.",
              "causes": [("At one technology node: under-etched metal causing resistive vias in the ring", "hu2021")]},
    "Edge-Loc": {"looks": "A cluster of fails at one spot on the edge.",
                 "causes": [("Contamination (pollution) during fabrication", "geng2023")]},
    "Edge-Ring": {"looks": "Fails all the way around the outer edge.",
                  "causes": [("At one technology node: poly critical dimension (CD) running short near the edge", "hu2021")]},
    "Loc": {"looks": "One local cluster of fails away from the edge.", "causes": []},
    "Near-full": {"looks": "Almost every die fails.", "causes": []},
    "Random": {"looks": "Fails scattered across the wafer with no cluster.",
               "causes": [("Particles such as dust scattered over the surface", "geng2023")]},
    "Scratch": {"looks": "A thin line or arc of fails.",
                "causes": [("Sharp edges touching and slipping on the wafer surface during manufacturing or wafer test", "hu2021")]},
    "none": {"looks": "No defect pattern: only a few scattered fails.", "causes": []},
}
