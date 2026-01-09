# SQL Query Patterns for Experiments

Common DuckDB queries for the experiment database.

## Basic Queries

### List All Experiments
```sql
SELECT experiment_name, MIN(perplexity) as best_perplexity
FROM experiments
GROUP BY experiment_name
ORDER BY best_perplexity
```

### Best by Metric
```sql
SELECT *
FROM experiments
WHERE perplexity = (SELECT MIN(perplexity) FROM experiments)
```

### Recent Experiments
```sql
SELECT *
FROM experiments
WHERE saved_at > NOW() - INTERVAL '7 days'
ORDER BY saved_at DESC
```

## Filtering

### By Project
```sql
SELECT *
FROM experiments
WHERE experiment_name LIKE 'custom_transformer%'
```

### By Metric Threshold
```sql
SELECT experiment_name, perplexity, loss
FROM experiments
WHERE perplexity < 20
ORDER BY perplexity
```

### By Epoch
```sql
SELECT *
FROM experiments
WHERE epoch >= 5
```

## Aggregations

### Best Per Project
```sql
SELECT
    SPLIT_PART(experiment_name, '_', 1) as project,
    MIN(perplexity) as best_perplexity,
    COUNT(*) as num_experiments
FROM experiments
GROUP BY project
ORDER BY best_perplexity
```

### Training Progress
```sql
SELECT epoch, AVG(perplexity) as avg_perplexity
FROM experiments
WHERE experiment_name = 'exp_001'
GROUP BY epoch
ORDER BY epoch
```

## Comparisons

### Side-by-Side
```sql
SELECT
    e1.experiment_name as exp1,
    e2.experiment_name as exp2,
    e1.perplexity as ppl1,
    e2.perplexity as ppl2,
    e1.perplexity - e2.perplexity as diff
FROM experiments e1
CROSS JOIN experiments e2
WHERE e1.experiment_name = 'exp_001'
  AND e2.experiment_name = 'exp_002'
  AND e1.epoch = e2.epoch
```

## Natural Language Mappings

| User Says | Maps To |
|-----------|---------|
| "best perplexity" | `ORDER BY perplexity LIMIT 5` |
| "worst loss" | `ORDER BY loss DESC LIMIT 5` |
| "last week" | `WHERE saved_at > NOW() - INTERVAL '7 days'` |
| "today" | `WHERE DATE(saved_at) = CURRENT_DATE` |
| "transformer experiments" | `WHERE experiment_name LIKE '%transformer%'` |
