# Comparaison pédagogique : SimpleRNN, LSTM et GRU

## Objectif
Prédire la prochaine valeur d'un signal périodique synthétique bruité à partir de 40 observations. Construire et comprendre l'expérience une cellule à la fois.

## Progression prévue
1. Vérifier l'environnement et le GPU.
2. Générer et visualiser le signal avec une graine fixée.
3. Créer les fenêtres temporelles et expliquer les dimensions.
4. Réaliser une séparation chronologique entraînement/validation/test, sans fuite de données. Ajuster toute normalisation uniquement sur l'entraînement.
5. Comparer une référence naïve (dernière valeur observée), SimpleRNN, LSTM et GRU.
6. Utiliser les mêmes données, fenêtres, divisions, unités récurrentes, couche de sortie, optimiseur, taille de lot et budget d'époques. Documenter les conditions GPU, les paramètres et la durée d'entraînement ; les nombres de paramètres ne seront pas identiques.
7. Comparer MAE et RMSE sur le même test, courbes d'apprentissage et prédictions. Répéter avec plusieurs graines si le temps le permet.

## Livrables à construire progressivement
- `01_comparaison_rnn.ipynb` : explications et expérience en français.
- `figures/` : figures PNG en français, sauvegardées au fur et à mesure.
- `resultats/` : mesures et conditions expérimentales au format CSV.
- `notes_presentation.md` : observations réelles et idées de diapositives.

## Ouverture du Notebook

### JupyterLab dans le navigateur (méthode recommandée)
Après chaque redémarrage, ouvrir PowerShell et exécuter :

```powershell
cd C:\Users\marou\Desktop\ENSET\IA_Avance\RNN_LSTM_GRU
.\run_wsl.ps1 -m jupyterlab --no-browser --ServerApp.ip=127.0.0.1 --ServerApp.port=8888
```

Garder cette fenêtre ouverte : le serveur s'exécute au premier plan. Copier dans le navigateur le lien complet affiché, y compris `?token=...`. Utiliser le lien de la session actuelle ; le jeton peut changer au redémarrage. Si le port 8888 est occupé, utiliser le port indiqué dans les messages.

Ouvrir `01_comparaison_rnn.ipynb` et choisir **Python (RNN-LSTM-GRU | WSL GPU)**. Exécuter les cellules avec Maj+Entrée.

Pour terminer : enregistrer le Notebook avec Ctrl+S, puis revenir dans PowerShell, appuyer sur Ctrl+C et confirmer avec `y` si demandé. Fermer l'onglet du navigateur ne suffit pas à arrêter le serveur.

Après un redémarrage, les cellules et sorties enregistrées restent dans le fichier, mais les variables en mémoire sont perdues. Réexécuter les cellules nécessaires dans l'ordre ; ne pas relancer automatiquement les entraînements longs.

### Autre possibilité : VS Code dans WSL
Dans VS Code connecté à **WSL: Ubuntu-24.04**, ouvrir ce dossier puis le Notebook. Choisir la noyau **Python (RNN-LSTM-GRU | WSL GPU)** via le sélecteur de noyau. Cette noyau utilise le Python de `.venv/bin/` et `run_gpu.py` pour préparer les bibliothèques GPU. Si nécessaire, choisir cette noyau dans la liste des noyaux Jupyter.

Ne pas utiliser un interpréteur Python Windows. Réutiliser l'environnement existant ; toute modification de dépendances passe par uv. Voir `ENVIRONMENT.md`.

## Présentation finale (~30 minutes)
Préparer environ 10 à 12 diapositives : objectif et données (4 min), séquences et dimensions (4 min), principes des trois architectures (6 min), protocole équitable (4 min), résultats et figures (8 min), limites et conclusion (4 min).

Le PowerPoint sera créé à la demande après les expériences, à partir des résultats réellement obtenus.

## État de l'expérience

### Extension sur données réelles : Jena Climate
Le Notebook `02_comparaison_jena_climate.ipynb` est préparé intégralement, avec 17 étapes expliquées en français. Il reste à exécuter : aucun résultat réel n'est présumé. Ouvrir le fichier dans JupyterLab et utiliser la noyau **Python (RNN-LSTM-GRU | WSL GPU)**. Exécuter dans l'ordre ; examiner les contrôles de qualité et les dimensions avant les trois cellules d'entraînement (étapes 10–12).

Objectif : prévoir une température horaire à +24 h à partir des 72 températures horaires précédentes. Archive Jena 2009–2016 utilisée par le tutoriel officiel TensorFlow : https://www.tensorflow.org/tutorials/structured_data/time_series . Le téléchargement est lancé par la cellule 2 uniquement lors de l'exécution ; réutilisation locale ensuite. Aucun changement de dépendances nécessaire.

Protocole : séparation chronologique 70/15/15, délai de 24 h avant les premières cibles de validation/test, normalisation ajustée uniquement sur l'entraînement, cibles horaires absentes exclues, 32 unités, 20 époques, lots de 128 et Adam à 0,001. Références : persistance à l'instant de prévision et température une semaine avant la cible. À +24 h, la référence « même heure hier » est identique à la persistance.

Les sorties sont dans `figures/jena/<identifiant>/`, `resultats/jena/<identifiant>/` et `modeles/jena/<identifiant>/`. `resultats/jena/derniere_experience.json` indique les chemins de la dernière préparation. L'identifiant dépend du protocole et de l'archive. Les expériences synthétiques restent séparées.

Après redémarrage, réexécuter les cellules dans l'ordre avec `REENTRAINER = False` : les entraînements complets sauvegardés seront rechargés. Une exécution interrompue recommence depuis zéro. `REENTRAINER = True` force le remplacement des entraînements correspondants. Une fois le test examiné, ne pas ajuster les modèles pour améliorer le même test.

Le Notebook a été contrôlé structurellement et syntaxiquement, avec un contrôle miniature de l'indexation des fenêtres. Après le premier contrôle de qualité de l'étudiant, la préparation a été corrigée pour conserver les longues lacunes sur la grille horaire et exclure les fenêtres incomplètes, plutôt que de les remplir artificiellement. La normalisation ignore les valeurs absentes de la période d'entraînement ; les cibles initialement absentes sont exclues. Les entraînements restent à exécuter dans Jupyter.

**Mise à jour : expérience réelle exécutée par l'étudiant**, identifiant `4614d659431c`. Les scores finaux sont dans `resultats/jena/4614d659431c/comparaison_test.csv`. MAE test : persistance 2,4958 °C ; SimpleRNN 2,5623 °C ; LSTM 2,4666 °C ; GRU 2,4621 °C. Gains modestes pour LSTM/GRU, pas de gain pour SimpleRNN. Les six figures et trois modèles sont sauvegardés. Le test ayant été examiné, ne pas modifier les réglages pour améliorer ce même test. Voir `notes_presentation.md` pour l'interprétation.

### Expérience synthétique terminée
Les trois modèles ont été entraînés par l'étudiant pendant 20 époques et évalués sur le test réservé. Les figures comparatives sont produites. Les modèles finaux sont dans `modeles/`, les mesures et prévisions dans `resultats/` et les images dans `figures/`.

Scores de test arrondis (MAE / RMSE) : persistance 0,1361 / 0,1683 ; SimpleRNN 0,0819 / 0,1035 ; LSTM 0,0809 / 0,1021 ; GRU 0,0850 / 0,1067. Une seule exécution par architecture : les résultats concernent ce signal synthétique et ce protocole, sans conclusion générale sur le meilleur modèle.

La préparation du PowerPoint peut maintenant s'appuyer sur `notes_presentation.md` et les fichiers sauvegardés. Enregistrer le Notebook dans JupyterLab avec Ctrl+S avant de fermer la session.
