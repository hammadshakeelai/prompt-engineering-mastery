# AlphaCode 2: Filtering, Clustering & Selection Strategy

1. **Filtering**: Executes candidate solutions against public problem tests, eliminating ~95% failing candidates.
2. **Clustering**: Executes surviving candidates on synthetically generated test inputs, grouping programs with identical output behavior into clusters.
3. **Selection**: Samples candidates from largest behavioral consensus clusters and rates quality via scoring models to submit top 10 final solutions.