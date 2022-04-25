from exrecsys.metrics import SystemMetrics

knowledge_concepts = {
    0: 'a', 
    1: 'b',
    2: 'c', 
    3: 'd',
    4: 'e'
}

rel = [
    {
        'content_id': 10685,
        'concepts': ['a', 'b', 'e'],
        'difficulty': 0.7
    }, 
    {
        'content_id': 471,
        'concepts': ['b', 'd'],
        'difficulty': 0.6
    }, 
]

delta = [0.1680097,  0.09412533]

# accuracy = SystemMetrics().accuracy(rel=rel, delta=delta)
# print(f"\n- accuracy = {accuracy}")

# novelty = SystemMetrics().novelty(rel=rel, correctly_answered=[1, 1])

# print(f"\n- novelty = {novelty}")

# diversity = SystemMetrics().diversity(rel=rel, knowledge_concepts=knowledge_concepts)

# print(f"\n- diversity = {diversity}")
