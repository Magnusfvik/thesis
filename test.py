def jaccard_distance(set_A, set_B):
    """Compute Jaccard distance between two sets"""
    if not set_A or not set_B:
        return 1.0
    
    intersection = len(set_A & set_B)
    union = len(set_A | set_B)
    
    if union == 0:
        return 1.0
    
    similarity = intersection / union
    return 1.0 - similarity

# Test jaccard_distance function
set1 = {'Rock', 'Pop'}
set2 = {'Rock', 'Pop'}
print(f"Same sets: {jaccard_distance(set1, set2)}")  # Should be 0.0

set3 = {'Rock', 'Pop'}
set4 = {'Jazz', 'Classical'}
print(f"Completely different: {jaccard_distance(set3, set4)}")  # Should be 1.0

set5 = {'Rock', 'Pop'}
set6 = {'Rock', 'Jazz'}
print(f"50% overlap: {jaccard_distance(set5, set6)}") 