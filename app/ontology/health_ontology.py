"""
Health ontology using NetworkX graph.
Represents relationships between health concepts.
"""

import networkx as nx
from typing import List, Dict, Set
from app.utils.logger import logger


class HealthOntology:
    """Health domain ontology as a directed graph."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_ontology()
    
    def _build_ontology(self):
        """Build basic health ontology with relationships."""
        
        # Nodes: health concepts
        concepts = [
            "hydration", "fatigue", "energy", "sleep", "exercise",
            "nutrition", "stress", "mood", "heart_health", "weight",
            "blood_pressure", "recovery", "endurance", "focus",
            "immune_system", "inflammation", "mental_clarity"
        ]
        
        self.graph.add_nodes_from(concepts)
        
        # Edges: relationships (concept -> affects)
        relationships = [
            ("hydration", "fatigue"),
            ("hydration", "energy"),
            ("hydration", "focus"),
            ("sleep", "energy"),
            ("sleep", "mood"),
            ("sleep", "recovery"),
            ("sleep", "immune_system"),
            ("exercise", "energy"),
            ("exercise", "mood"),
            ("exercise", "heart_health"),
            ("exercise", "sleep"),
            ("nutrition", "energy"),
            ("nutrition", "immune_system"),
            ("nutrition", "weight"),
            ("stress", "sleep"),
            ("stress", "mood"),
            ("stress", "heart_health"),
            ("weight", "heart_health"),
            ("weight", "blood_pressure"),
            ("recovery", "endurance"),
            ("recovery", "energy"),
            ("inflammation", "recovery"),
            ("hydration", "mental_clarity"),
            ("exercise", "stress"),
            ("sleep", "mental_clarity")
        ]
        
        self.graph.add_edges_from(relationships)
        logger.info(f"Health ontology built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def get_related_concepts(self, concept: str, depth: int = 2) -> Set[str]:
        """Get concepts related to given concept within depth."""
        if concept not in self.graph:
            return set()
        
        related = set()
        
        # Get descendants (concepts affected by this one)
        try:
            descendants = nx.descendants(self.graph, concept)
            related.update(descendants)
        except nx.NetworkXError:
            pass
        
        # Get ancestors (concepts that affect this one)
        try:
            ancestors = nx.ancestors(self.graph, concept)
            related.update(ancestors)
        except nx.NetworkXError:
            pass
        
        return related
    
    def get_path_between(self, source: str, target: str) -> List[str]:
        """Get shortest path between two concepts."""
        if source not in self.graph or target not in self.graph:
            return []
        
        try:
            path = nx.shortest_path(self.graph.to_undirected(), source, target)
            return path
        except nx.NetworkXNoPath:
            return []
    
    def get_influence_concepts(self, concept: str) -> List[str]:
        """Get concepts directly influenced by given concept."""
        if concept not in self.graph:
            return []
        return list(self.graph.successors(concept))
    
    def get_influencing_concepts(self, concept: str) -> List[str]:
        """Get concepts that directly influence given concept."""
        if concept not in self.graph:
            return []
        return list(self.graph.predecessors(concept))
    
    def query_ontology(self, query_terms: List[str]) -> Dict[str, List[str]]:
        """Query ontology with multiple terms and return related concepts."""
        results = {}
        for term in query_terms:
            term_lower = term.lower()
            if term_lower in self.graph:
                results[term_lower] = {
                    "influences": self.get_influence_concepts(term_lower),
                    "influenced_by": self.get_influencing_concepts(term_lower),
                    "related": list(self.get_related_concepts(term_lower, depth=1))
                }
        return results