# Notes pour la présentation

## PowerPoint livré
Version Canva importée (25 pages) : https://www.canva.com/d/Dwj9fqn-6V5plag . Identifiant : `DAHYCMiUKI4`. Texte des pages 1, 8, 17, 18 et 20 contrôlé après conversion, notamment paramètres, scores et durées. Le fichier PowerPoint reste la source locale.

Version visuelle recommandée : `presentation/Comparaison_SimpleRNN_LSTM_GRU_Design_Final.pptx`. Refonte des 25 diapositives avec fond nuit et pages claires, palette stable par architecture, grands chiffres et graphiques natifs. Les notes françaises et les sources restent dans le PowerPoint. Couverture conceptuelle créée avec ImageGen, sans valeur de mesure météorologique.

`presentation/Comparaison_SimpleRNN_LSTM_GRU.pptx` : 25 diapositives en français, dont 23 pour l'exposé et les questions, puis deux annexes. Durée indicative 29,5 minutes. Les notes du présentateur accompagnent chaque diapositive et citent les sources pertinentes. Les résultats réels, leurs faibles gains face à la persistance et les limites de la comparaison constituent le cœur de l'exposé.

## Expérience réelle terminée — expérience principale du diaporama
Le Notebook `02_comparaison_jena_climate.ipynb` a été exécuté par l'étudiant jusqu'aux exports. Expérience `4614d659431c` : 72 températures horaires pour prévoir une cible à +24 h, erreurs en °C, références par persistance et hebdomadaire. Fichiers de scores, six figures et trois modèles sauvegardés vérifiés.

Résultats de test (MAE / RMSE, en °C) : persistance 2,4958 / 3,2477 ; référence hebdomadaire 4,6262 / 5,7685 ; SimpleRNN 2,5623 / 3,2510 ; LSTM 2,4666 / 3,1585 ; GRU 2,4621 / 3,1502. Réduction de MAE face à la persistance : SimpleRNN -2,6648 %, LSTM +1,1679 %, GRU +1,3506 %.

Interprétation : la persistance constitue une référence forte. Les améliorations de LSTM et GRU sont modestes (environ 0,03 °C de MAE) ; SimpleRNN ne bat pas la persistance sur ce protocole. GRU et LSTM sont très proches (écart de MAE d'environ 0,0046 °C). Ne pas présenter une supériorité statistiquement établie à partir d'une seule graine. Les courbes de validation de SimpleRNN oscillent davantage ; cela suggère une sensibilité de l'optimisation dans cette configuration, pas une preuve isolée de disparition du gradient ni une incapacité générale des RNN.

Temps de fit : SimpleRNN 100,04 s, LSTM 92,29 s, GRU 82,92 s. Médianes des époques après la première : 4,481 s, 4,359 s, 3,980 s. GRU est la plus rapide dans cette exécution ; le classement dépend du matériel, des implémentations, du démarrage et de l'ordre d'exécution. Ne pas attribuer la vitesse au seul nombre de paramètres.

Le test a été consulté ; ne pas ajuster les modèles pour améliorer ce même test. Les résultats constituent une comparaison expérimentale valable, même si les gains sur la persistance sont petits.

Les notes et figures sont sauvegardées sous `resultats/jena/4614d659431c/notes_presentation_jena.md` et `figures/jena/4614d659431c/`. Réorganiser le diaporama : courte introduction avec la sinusoïde, puis données réelles, protocole, résultats et limites. Ne pas comparer directement les valeurs d'erreur entre les deux expériences, dont les unités et horizons diffèrent. La structure de 30 minutes ci-dessous concerne l'expérience synthétique et sera adaptée aux résultats réels.

## État actuel
- Sujet proposé : prévision d'un signal périodique synthétique bruité.
- Versions vérifiées : Python 3.12.3, TensorFlow 2.21.0, Keras 3.15.1.
- Environnement et détection d'un GPU confirmés par les sorties de l'étudiant.
- Génération du signal réussie : 2 000 mesures, forme `(2000,)`, six cycles visibles sur les 300 premières mesures. Figure `figures/01_signal_synthetique.png` sauvegardée selon la sortie partagée.
- Fenêtre et cible exécutées par l'étudiant : entrée de forme `(40,)`, cible d'indice 40 et de valeur arrondie -0,877. Figure `figures/02_fenetre_et_cible.png` sauvegardée selon la sortie partagée.
- Construction des fenêtres confirmée par les sorties de l'étudiant : X `(1960, 40, 1)`, y `(1960, 1)`. Correspondance du premier exemple vérifiée.
- Séparation chronologique confirmée par les sorties de l'étudiant : 1 372 / 294 / 294 exemples. Indices des cibles : 40–1411, 1412–1705 et 1706–1999. Protocole : prévision glissante à un pas à partir de mesures passées observées, avec cibles distinctes entre les jeux.
- Référence par persistance exécutée sur la validation : MAE = 0,1425 et RMSE = 0,1765 (valeurs arrondies affichées par l'étudiant). Graphique `figures/04_reference_naive_validation.png` et tableau `resultats/reference_validation.csv` enregistrés selon la sortie partagée. Le décalage d'un pas illustre la règle de persistance.
- Construction confirmée : SimpleRNN 1 121 paramètres, LSTM 4 385, GRU 3 393. Tableau `resultats/architectures.csv` enregistré selon la sortie partagée. Configuration commune : 32 unités et Dense(1), Adam à 0,001, 20 époques, lots de 32. Comparaison à nombre d'unités égal, pas à nombre de paramètres égal.
- SimpleRNN entraîné par l'étudiant : 20 époques fixes, modèle final conservé, sans arrêt anticipé, `shuffle=False`, validation après chaque époque. Durée fit affichée : 12,21 s ; première époque : 4,18 s. MSE validation : 0,01853 à l'époque 1, 0,01196 à l'époque 20 ; MSE entraînement finale : 0,01072. Valeurs arrondies issues de la sortie partagée. Modèle, historique et durées sauvegardés. Le démarrage inclut les coûts initiaux éventuels ; ne pas assimiler la durée totale à la seule vitesse récurrente.
- SimpleRNN évalué par l'étudiant sur la validation : MAE = 0,0866, RMSE = 0,1094 ; réduction relative de MAE face à la persistance = 39,3 % (valeurs arrondies affichées). Figure `figures/05_SimpleRNN_validation.png` et CSV de métriques/prévisions sauvegardés selon la sortie partagée. Les courbes baissent puis se stabilisent, sans hausse durable visible de la MSE de validation. Aucun score de test calculé.
- LSTM entraîné par l'étudiant avec le même protocole : durée fit 13,73 s, première époque 5,65 s ; MSE entraînement finale 0,01086, MSE validation finale 0,01183. Valeurs arrondies affichées. Évaluation MAE/RMSE et figure LSTM préparées, à exécuter.
- La différence de MSE validation finale entre LSTM (0,01183) et SimpleRNN (0,01196) est faible sur cette exécution ; ne pas en déduire une supériorité générale. Les durées totales incluent le démarrage ; comparer aussi les durées des époques suivantes.
- Évaluation LSTM confirmée par la sortie de l'étudiant : MAE validation 0,0865, RMSE validation 0,1088, réduction relative de MAE face à la persistance 39,3 % (valeurs arrondies). Figure `figures/06_LSTM_validation.png` et CSV sauvegardés selon la sortie. Scores très proches de SimpleRNN ; aucune conclusion de supériorité statistique.
- Entraînement GRU confirmé par la sortie de l'étudiant : durée fit 9,38 s ; première époque 1,15 s ; MSE entraînement finale 0,01155 ; MSE validation finale 0,01277, encore en baisse à l'époque 20. Modèle et mesures sauvegardés selon la sortie partagée. Évaluation MAE/RMSE et figure préparées, à exécuter.
- GRU présente la plus petite durée totale observée dans cette session, avec un démarrage moins coûteux ; les époques suivantes semblent du même ordre de durée que celles des autres modèles. Attendre la comparaison des médianes avant toute interprétation sur la vitesse.
- Évaluation GRU confirmée par la sortie de l'étudiant : MAE validation 0,0899, RMSE 0,1130, réduction relative de MAE face à la persistance 36,9 % (valeurs arrondies). Figure `figures/07_GRU_validation.png` et CSV sauvegardés selon la sortie partagée.
- Évaluation du test exécutée par l'étudiant, modèles finaux et réglages gelés. Résultats arrondis : persistance MAE 0,1361 / RMSE 0,1683 ; SimpleRNN 0,0819 / 0,1035 ; LSTM 0,0809 / 0,1021 ; GRU 0,0850 / 0,1067. Réductions relatives de MAE face à la persistance : 39,8263 %, 40,5511 % et 37,5679 % respectivement. CSV et figures `08_erreurs_test.png`, `09_predictions_test.png` sauvegardés selon la sortie partagée.
- Durées médianes des époques 2 à 20 : SimpleRNN 0,4192 s, LSTM 0,4157 s, GRU 0,4314 s. Durées proches ; la plus petite durée totale de GRU reflète notamment un démarrage plus court. Une mesure unique ne permet pas un classement robuste de vitesse.
- Interprétation à présenter : les trois réseaux améliorent la persistance ; LSTM a les plus faibles erreurs observées, mais SimpleRNN reste très proche avec moins de paramètres. Limites : signal synthétique simple, prévision à un pas, une seule graine de modèle, pas de test de mémoire à longue portée ni de généralisation à des données réelles.
- Figures comparatives du chapitre 15 exécutées par l'étudiant : `figures/10_courbes_comparatives.png` et `figures/11_parametres_et_durees.png` enregistrées selon la sortie partagée. La phase expérimentale de base est terminée.

## Organisation proposée du PowerPoint (~30 minutes)
1. Titre et question de comparaison — 1 min.
2. Pourquoi des données synthétiques ? Signal, bruit et limites — 2 min. Figure 01.
3. Fenêtre de 40 mesures et cible suivante — 2 min. Figure 02.
4. SimpleRNN : état caché et traitement séquentiel — 3 min.
5. LSTM et GRU : rôle des portes et différences de mémoire — 4 min.
6. Protocole commun et séparation chronologique — 3 min. Figure 03.
7. Persistance, MAE et RMSE — 2 min. Figure 04.
8. Apprentissage des trois réseaux — 3 min. Figure 10.
9. Résultats du test et prévisions — 4 min. Figures 08 et 09.
10. Paramètres, démarrage et durée des époques — 2 min. Figure 11.
11. Limites et conclusion fondée sur cette expérience — 2 min.
12. Questions — 2 min.

Conserver les figures 05 à 07 en annexe si nécessaire. Éviter de répéter toutes les courbes individuelles dans le corps du diaporama. Le PowerPoint sera créé sur demande de l'étudiant, avec les résultats mesurés et un contenu en français.
- Aucun entraînement ni résultat expérimental à ce stade.

## Figures à produire progressivement
1. Signal et séparation chronologique des données.
2. Exemple d'une fenêtre de 40 valeurs et de sa cible.
3. Courbes d'apprentissage des trois modèles.
4. Prédictions comparées aux valeurs réelles sur le test.
5. Comparaison des erreurs, paramètres et durées mesurées.

## Observations à compléter après chaque étape
Consigner les résultats obtenus, leur interprétation, les difficultés et les limites. Ne pas présumer du modèle gagnant.

## Nouvelle création native Canva
Design créé depuis zéro : DAHYEdIavgA, 25 pages en français. Lien : https://canva.link/z9b31oqdf46dpul . Cette création ne réutilise pas le PowerPoint importé. Contenu textuel et valeurs du test vérifiés. Les outils connectés ne permettent pas de régler ou de confirmer les transitions et animations.
