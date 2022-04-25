## Exercises Recommend System (ExRecSYS)

Get started in 30 Seconds

### I. Introduction

Recommendation module is an important module in Adaptive learning systems. In order to consolidate the learning effect of students at certain stages, corresponding exercises are often provided to them appropriately. A good recommendation for exercises can effectively help to point the students/learners in the right direction, and potentially empower their learning interest. It’s a challenge to recommend exercises with suitable difficulty levels for students as they have different learning status; variety of types; exercises bank is very large; knowledge concepts contained therein meet the requirements of the learning progress. The Output of this module is a list of recommended exercises (REL)


### II. How to use?

Ez to use ! 

#### Step 1: Training the Knowledge Tracing (KT) module

```js

make train-kt

```

#### Step 2: Training the Knowledge Coverage Concpets Prediction (KCCP) module

```js

make train-kccp

```
#### Step 3: Running

```js

python run.py --mode test --user_id 1984659

```