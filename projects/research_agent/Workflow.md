# Research Workflow

In order to automate research, we first need to understand our workflow to define what we can automate in order to make the path more effective. 

## Open Questions 

Tactical:
- Should this be claude skills/agents or some other slash command? 
- How can I use hooks so that in step 4: running experiments, claude can ping me when it needs feedback, has a new report, etc? 
- How do we go from previous experiment to next => particularly on the code/data side of things to avoid re-inventing the wheel?

## High-Level

The goal of research is to refine our "tastes" into tactical exploration that definitively answers a hypothesis. The most exciting breakthroughs occur when we are able to prove mechanically that our intuition has a physical, reproducable property. As a result, it changes how practioners execute. Commonly, the core insight is distilled and bottled and foundational element which spreads prolifically.  

### Summary of Artifacts
- Research Report:
  - Ideation: Includes initial idea in human-language and relevant sources
  - Hypothesis: Specific, testable hypothesis
  - Experiment Design: High-level experiment design, ablation studies, hyper-parameters
  - Overview of design
  - Core Results
  - Supporting Evidence
  - Future Work
- Experimental Code:
  - Source files (core experimental framework)
  - Tools (e.g. data analytics)
- Experimental Results
  - Database of results (indexed)
  - Model snapshots
  - Reports (summaries of smaller studies) 
- Packaged, Polished code
  - Source code, refined
  - Curated documentation: step-by-step of how to run the most insightful/impactful studies
  - Supporting tools to analyze/view data


## Step 1: Exploration and Ideation

When we go to pursue finding a new hypothesis to test - we are never starting from zero. We have precognitions and need to understand how we can start to steer and shape our intuitions. At this stage, we are primarily reading papers, putting concepts together, and writing and reflecting on how things work to build our own understanding. 

How can AI help here? 
- Find relevant work
- Validate your understanding of the problem
- Work as a sounding board 

How AI can subvert the process?
- Overtaking your learning
- Disconnecting your from the core concepts, principle, and depth of the field

Desired Outcomes: Documentation (Research Report: Idea)
- Articulated ideas of pieces which fit together, but often lacking a formal hypothesis
- Assessment of novelty
- Potential implications: What does this change if our intuitions are right? 


## Step 2: Hypothesis formation

Desired Outcomes: Documentation (Research Report: Hypothesis)
- Articulated hypothesis, which is provable or disaprovable (with reasonable certainty/feasibility constraints)
- Scoping of the problem: what is in bounds of testing, what is out? 
- Baseline and comparison: how do we know this is working (better that the precedent)? 

How AI can help:
- Refinement of the problem statement/scope/baseline
- Generating the final "polished 

Where to be careful:
- AI changes the core hypothesis or human no longer appreciates it's intent
- Scope is not practical given realistic constraints


## Step 3: Experiment Design

Desired Outcomes (Documentation: Codified plan, typically markdown)
- Hyper-parameters defined
- Initial set of ablation studies to consider
- Scope/scale of data, model, and duration of experiments
- Core metrics to track and produce

How AI Can help:
- Author code to make it testable
- Craft/refine "report templates"
- Build and refine tooling for experiments

AI failure modes:
- Humans don't understand the ablation studies: why are we doing this?
- By default there is combinatorial complexity, need to ensure "intelligent" experimentation
- Insufficient architecture / under-appreciation of complexity needed to manage (AI allows us to "over-engineer" the experiment infra, but the challenge is that it may create challenges *between* experiments)

### Step 4: The Experimental Feedback Loop

Desired Outcomes (Code, Data, Graphs, Documentation: Custom Reports):
- Experimental results which prove/disprove hypothesis
- Supporting evidence: why? how do we know? 
- Sensitivity analysis: Under what conditions is it successful? Where does it break? 
- Lots of graphs, tools, and sketches of results

How AI can help:
- Automated running of experiments and ablation studies
- Report generation to produce key results in understandable format (think "research factory")
- Code generation / execution of experiments / debugging
- Continuous feedback loop (AI runs experiment => generates report => analyzes => designs/runs next experiment)

Where Humans must stay in the loop:
- Trajectory alignment: are we still answering our core hypothesis?
- Validation of the core sensitivity analysis: did we validate the right things? Did we answer the key questions?
- Correctness/Alignment: is the code actually doing what it says it's doing?

### Step 5: Refinment

Desired Outcomes:
- "Paper" Style report
- Idea -> Background -> Core Insight -> Testing Framework -> Core Result -> Supporting Evidence -> Future Work
- Curates graphs along "core axis"
- Clean, reproducable "package" 

How AI can help:
- Aggregation of existing elements
- Consolidation of produced results (e.g. data refinement)
- Co-author of final report

How AI can be dangerous:
- Data quality issues (e.g. improper aggregation of results) => reproducability is key here

