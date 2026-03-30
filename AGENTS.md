This is a testing repo for a segmentation architecture I'm working on with these characteristics:
- Clustering based: objects are detected based on clusters. Each embedding cluster corresponds to an object.
- No hungarian matching for training: assignations are done in a differentiable way using cosine distance between detection queries and GT-encoded embeddings.

Tests can be done using the `pytorch200` conda environment.