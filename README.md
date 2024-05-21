# horse-race-predictor

#### Követelmények telepítése: 
1. python 3.10 telepítése és konfigurálása
2. pip install -r requirements.txt futtatása a csomagok telepítéséhez

#### Szoftver használata:
1. Futtatás előtt csomagoljuk ki a demo_data.zip fájlt.
2. Állítsd át a paramétereket a kötvetkezőre:
   - copy_files = False
   - create_feature_vectors = False
   - run_learning = True
   - amounts_of_files = 100
   - amounts_of_races = [805]
   - algorythm = "all"
   - median = "drop"
3. Futtasd a main.py main metódusát.
4. A reports mappában megtekinthetőek az eredmények.

#### Átparaméterezése lehetőségek:
1. algorythm - futtatja a kiválasztott algoritmust
   - all - az összeset lefutattja
   - classification_with_bulk_fvs
   - classification_with_individual_results
   - classification_with_equal_results
   - split_to_first_3_and_the_rest
   - classify_by_race_without_conversion
   - pairwise_learn_to_rank_pairwise
   - pairwise_learn_to_rank_ndcg
2. amounts_of_races - a versenyek mennyisége
   - több tanítási ciklus futtatására is van lehetőség, ha több értéket adunk meg neki, mint pl.: [100, 300, 800]


#### Reportok felépítése:
- date_time - (folder)
  - classifications - (sub_folder)
    - classification_with_bulk_fvs - (txt)
    - classification_with_equal_results - (txt)
    - classification_with_individual_results - (txt)
    - classify_by_race_without_conversion (txt)
    - split_to_first_3_and_the_rest - (txt)
  - learn_to_rank - (sub_folder)
    - pairwise_learn_to_rank_ndcg - (txt)
    - pairwise_learn_to_rank_pairwise - (txt)
  


Classification methods:
- classification_with_bulk_fvs:
  + train - test data: 80 - 20%
  + distribution of rankings: all racer
  + missing data: drop
  - kept in races: no
- classification_with_equal_results:
  + train - test data: 80 - 20%
  + distribution of rankings: 1 vs rest - 50-50%
  + missing data: drop
  - kept in races: no
- classification_with_individual_results:
  + train - test data: 80 - 20%
  + distribution of rankings: all racer
  + missing data: drop
  + kept in races: yes
- split_to_first_3_and_the_rest:
  + train - test data: 80 - 20%
  + train data: used equal 1 - 0
  + test data: kept original ratio
  + number of models: 2
  + distribution of rankings: 1-3 vs rest - 50-50%, then 1 vs 2-3 33-67%
  + missing data: drop
  + kept in races: yes
- classify_by_race_without_conversion:
  + train - test data: 80 - 20%
  + train data: keep original results
  + distribution of rankings:  all racer
  + missing data: drop
  + kept in races: yes