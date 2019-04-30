# Bipartite Embedding

This application implements a number of methods for representing, embedding, and evaluating hypergraphs.

WARNING: This application is a work-in-progress.

A hypergraph is a structure where multiple "nodes" can be present in multiple "communities" (typically referred to as edges).
A typical graph is the special case wherein each community contains exactly two nodes.

The goal of this project is to embed nodes and edges such that a downstream classifier can accurately learn community groupings.
As opposed to typical embedding methods, we do not enforce that nodes and edges be in the same logical space.
We evaluate this through three classifier-based link-prediction tasks.



