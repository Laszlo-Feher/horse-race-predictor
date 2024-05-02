# horse-race-predictor

Classification methods:
- TODO: remove ids
- TODO: doesn't break races
- TODO: throw median check number


- classification_with_bulk_fvs:
  + train - test data: 80 - 20%
  + distribution of rankings: all racer
  - missing data: filled median
  - kept in races: no
- classification_with_equal_results:
  + train - test data: 80 - 20%
  + distribution of rankings: 1 vs rest - 50-50%
  - missing data: filled median
  - kept in races: no
- classification_with_individual_results:
  + train - test data: 80 - 20%
  + distribution of rankings: all racer
  - missing data: filled median
  + kept in races: yes
- split_to_first_3_and_the_rest:
  - train - test data: 80 - 20%
  - number of models: 2
  - distribution of rankings: 1-3 vs rest - 50-50%, then 1 vs 2-3 33-67%
  - missing data: filled median
  - kept in races: yes
