from exrecsys.filtering_layers.rel_generator import RELGenerator

exercise_set = [
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
    {
        'content_id': 1056,
        'concepts': ['a', 'd'],
        'difficulty': 0.2
    }, 
    {
        'content_id': 592,
        'concepts': ['e', 'c'],
        'difficulty': 0.4
    },
    {
        'content_id': 4271,
        'concepts': ['c', 'd'],
        'difficulty': 0.6
    }, 
    {
        'content_id': 10256,
        'concepts': ['f', 'c', 'd'],
        'difficulty': 0.2
    }, 
    {
        'content_id': 5922,
        'concepts': ['e'],
        'difficulty': 0.4
    },
    {
        'content_id': 140256,
        'concepts': ['a', 'e'],
        'difficulty': 0.2
    }, 
    {
        'content_id': 25932,
        'concepts': ['d', 'f'],
        'difficulty': 0.4
    },
]

knowledge_concepts = {
    0: 'a', 
    1: 'b',
    2: 'c', 
    3: 'd',
    4: 'e',
    5: 'f'
}


generator = RELGenerator(
    exercise_set=exercise_set,
    temperature=10,
    n_iterations=100,
    knowledge_concepts=knowledge_concepts
)

best_exercises, score, scores = generator.generate(n_samples=3)

print('')
print('-'*10)
print(best_exercises)
print(score)
print(scores)
print('-'*10)

try:
    generator.plot(scores)
except:
    pass