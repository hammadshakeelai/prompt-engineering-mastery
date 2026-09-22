# Model Merging: SLERP, TIES, and DARE

- **SLERP**: Spherical Linear Interpolation blends weights along high-dimensional arcs, preserving vector orientation and magnitude.
- **TIES**: Trims low-magnitude delta weights, elects sign consensus, and averages disjoint agreeing parameters.
- **DARE**: Randomly drops 90-99% of delta parameters and rescales remaining weights by $1/(1-p)$ prior to merging, removing parameter conflict.