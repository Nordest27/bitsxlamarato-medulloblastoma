# Medulloblastoma G3/G4 Classification Documentation

Welcome to the documentation for the Medulloblastoma G3/G4 Classification project! This AI-powered tool assists in classifying medulloblastoma subtypes from gene expression data.

## What is Medulloblastoma?

Medulloblastoma is the most common malignant brain tumor in children, accounting for about 20% of all pediatric brain tumors. The World Health Organization (WHO) recognizes four main molecular subgroups:

- **WNT** (Wingless) - Best prognosis
- **SHH** (Sonic Hedgehog) - Intermediate prognosis  
- **Group 3** - Poorest prognosis
- **Group 4** - Intermediate prognosis

This project focuses specifically on distinguishing between **Group 3** and **Group 4** subtypes, which is crucial for treatment planning and prognosis.

## Project Highlights

### 🧬 Large-Scale Dataset
- Uses GSE85217 dataset with 763 medulloblastoma patients
- Comprehensive gene expression profiles from Affymetrix microarrays
- ~54,000 gene probes for detailed molecular characterization

### 🔬 Advanced ML Pipeline
- Optimized preprocessing with multiple outlier detection methods
- Interactive UMAP visualizations for data exploration
- Ready-to-implement CVAE architecture for classification

### ⚡ High Performance
- Optimized algorithms reduce processing time from hours to minutes
- Automatic method selection based on dataset size
- Memory-efficient implementations for large datasets

### 📊 Explainable AI
- Gene-level analysis without dimensionality reduction
- Identification of key discriminative genes
- Interactive visualizations for result interpretation

## Quick Links

- [Installation Guide](getting-started.md) - Get started in 5 minutes
- [API Reference](api/dataset.md) - Detailed function documentation
- [Examples](examples/basic-usage.md) - Code examples and tutorials
- [GitHub Repository](https://github.com/bsc-health/bitsxlamarato-medulloblastoma) - Source code

## Scientific Impact

This project was developed during the BitsxLaMarató hackathon, supporting pediatric cancer research funded by La Marató de TV3. The tools and methods developed here can:

- Assist clinicians in accurate subtype classification
- Accelerate research into medulloblastoma biology
- Support development of personalized treatment strategies
- Provide a foundation for further AI applications in pediatric oncology

## Getting Help

If you encounter issues or have questions:

1. Check the [Getting Started](getting-started.md) guide
2. Browse the [Examples](examples/basic-usage.md)
3. Search existing [GitHub Issues](https://github.com/bsc-health/bitsxlamarato-medulloblastoma/issues)
4. Contact the development team

---

*Developed with ❤️ for advancing pediatric cancer research*
