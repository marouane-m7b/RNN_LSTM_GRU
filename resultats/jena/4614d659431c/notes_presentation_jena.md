# Expérience réelle : Jena Climate

Expérience : 4614d659431c
Prévision directe à +24 h à partir de 72 températures horaires.
Une exécution par architecture, 20 époques fixes ; erreurs en °C.

## Scores de test mesurés

- Persistance (24 h) : MAE 2.4958 °C ; RMSE 3.2477 °C.
- Référence hebdomadaire : MAE 4.6262 °C ; RMSE 5.7685 °C.
- SimpleRNN : MAE 2.5623 °C ; RMSE 3.2510 °C.
- LSTM : MAE 2.4666 °C ; RMSE 3.1585 °C.
- GRU : MAE 2.4621 °C ; RMSE 3.1502 °C.

## Questions à discuter après exécution

- Les réseaux améliorent-ils les deux références ?
- Les écarts entre réseaux sont-ils grands par rapport aux erreurs ?
- Les courbes montrent-elles un surapprentissage ou un apprentissage encore en cours ?
- Quel compromis entre erreur, paramètres et durées mesurées ?

## Limites à mentionner

- Une station, une variable, un horizon et une seule graine de modèle.
- Fenêtres recouvrantes ; pas de test de significativité ni de répétitions.
- Découpage chronologique : les saisons et années des jeux diffèrent.
- Moyennes horaires, remplissage causal limité à 3 h ; fenêtres incomplètes et cibles initialement absentes exclues.
- Températures issues du passé réel à chaque instant de prévision ; pas de simulation récursive de toute la période.
- Les résultats de la sinusoïde et de Jena ne sont pas directement comparables.
- Ne pas généraliser le classement à tous les problèmes temporels.

## Figures

- 01_donnees_et_periodes.png
- 02_fenetre_horizon.png
- 03_apprentissage.png
- 04_erreurs_test.png
- 05_predictions_test.png
- 06_parametres_durees.png