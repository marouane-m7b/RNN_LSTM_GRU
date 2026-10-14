"""Génère le Notebook pédagogique ; n'exécute aucune expérience."""
import ast
import sys
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def section(texte, code=None):
    cells.append(nbf.v4.new_markdown_cell(texte.strip()))
    if code is not None:
        ast.parse(code)
        cells.append(nbf.v4.new_code_cell(code.strip()))

section('''
# Comparaison de SimpleRNN, LSTM et GRU sur des températures réelles

**Question :** prévoir une température horaire à Jena 24 heures après la dernière observation, à partir des 72 dernières températures horaires.

Ce Notebook complète l'expérience synthétique sans la remplacer. Il contient toutes les étapes, mais **aucune sortie n'est préremplie**. Exécuter les cellules dans l'ordre et examiner chaque résultat. Le téléchargement nécessite Internet une première fois ; les données sont ensuite réutilisées localement.

**Source :** Jena Climate, mesures du Max Planck Institute for Biogeochemistry, archive 2009–2016 utilisée par le [tutoriel officiel TensorFlow](https://www.tensorflow.org/tutorials/structured_data/time_series). Notre préparation utilise des moyennes horaires, plutôt que la sélection d'une mesure sur six du tutoriel.

Les scores ne seront connus qu'après exécution. Aucune architecture n'est supposée gagnante. Les trois entraînements sont dans trois cellules séparées. Par défaut, un entraînement complet déjà sauvegardé est rechargé au lieu d'être recommencé.
''')

section('''
## 1. Vérifier l'environnement et fixer les réglages

Choisir **Python (RNN-LSTM-GRU | WSL GPU)**. La fenêtre contient 72 heures ; la cible est à +24 heures de sa dernière observation. Le modèle émet une seule valeur, pas les 24 valeurs intermédiaires.

Protocole fixé avant les scores : 32 unités, Adam à 0,001, MSE, lots de 128, 20 époques fixes, graine 42, sans arrêt anticipé ni recherche d'hyperparamètres. Le nombre de paramètres diffère entre architectures. Une seule graine ne permet pas d'établir une supériorité statistique.
''', '''
from pathlib import Path
import hashlib
import json
import platform
import time
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from IPython.display import display

RACINE = Path.cwd()
assert (RACINE / 'run_gpu.py').exists(), 'Ouvrir Jupyter depuis le dossier du projet.'
CONFIG = {
    'fenetre_heures': 72, 'horizon_heures': 24,
    'unites': 32, 'lot': 128, 'epoques': 20,
    'taux_apprentissage': 0.001, 'graine': 42,
    'fraction_train': 0.70, 'fraction_validation': 0.15,
    'clipnorm': 1.0, 'melange': False,
    'preparation': 'moyenne_horaire_ffill_3h_exclusion_fenetres_v2',
}
REENTRAINER = False  # True force le remplacement des entraînements de cette expérience.
keras.utils.set_random_seed(CONFIG['graine'])
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print('Python :', platform.python_version())
print('TensorFlow :', tf.__version__, '| Keras :', keras.__version__)
print('GPU visibles :', gpus)
if not gpus:
    print('Attention : aucun GPU visible ; vérifier la noyau avant les entraînements.')
display(pd.Series(CONFIG, name='Configuration'))
''')

section('''
## 2. Télécharger et inspecter les mesures brutes

L'archive officielle est mise en cache dans `donnees/jena/`. Le CSV est lu dans l'archive, sans extraction des fichiers. Son empreinte SHA-256 permet de documenter précisément la version utilisée ; ce calcul n'est pas une vérification contre une empreinte publiée.

Résultat attendu : plusieurs centaines de milliers de mesures à dix minutes, une colonne de date et des variables météorologiques. Nous n'utiliserons que **T (degC)** ; ajouter plus de variables changerait la question expérimentale.
''', '''
URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
DOSSIER_DONNEES = RACINE / 'donnees' / 'jena'
DOSSIER_DONNEES.mkdir(parents=True, exist_ok=True)
archive = Path(keras.utils.get_file(
    fname='jena_climate_2009_2016.csv.zip', origin=URL,
    cache_dir=str(DOSSIER_DONNEES), cache_subdir='', extract=False))
empreinte_archive = hashlib.sha256(archive.read_bytes()).hexdigest()
with zipfile.ZipFile(archive) as fichier_zip:
    candidats = [nom for nom in fichier_zip.namelist()
                 if Path(nom).name == 'jena_climate_2009_2016.csv']
    assert len(candidats) == 1, 'CSV attendu absent ou ambigu dans l’archive.'
    with fichier_zip.open(candidats[0]) as fichier_csv:
        donnees_brutes = pd.read_csv(fichier_csv)
print('Nombre de lignes brutes :', len(donnees_brutes))
print('SHA-256 de l’archive :', empreinte_archive)
display(donnees_brutes.head())
''')

section('''
## 3. Construire une série horaire et contrôler la qualité

Nous convertissons les dates, trions les observations et calculons la **moyenne de chaque heure**. Une moyenne horodatée à 14 h représente ici l'intervalle [13 h, 14 h) : elle est disponible à 14 h. Cela évite de dater une moyenne avant la fin de sa collecte.

Les valeurs de température hors de [-90, 60] °C sont signalées puis traitées comme absentes, selon une règle fixe de contrôle. Les heures manquantes sont comblées seulement avec le passé, au maximum sur trois heures consécutives. Pas d'interpolation utilisant le futur. Les cibles initialement absentes seront exclues de l'évaluation ; une cible imputée ne doit pas être considérée comme une vérité mesurée.

Les trous plus longs restent absents. Nous conservons la grille horaire et exclurons les fenêtres qui nécessitent ces valeurs, plutôt que de supprimer des heures et rapprocher artificiellement deux instants éloignés. Les contrôles et périodes manquantes sont affichés puis enregistrés.
''', '''
dates = pd.to_datetime(donnees_brutes['Date Time'], format='%d.%m.%Y %H:%M:%S', errors='raise')
temperature_brute = pd.to_numeric(donnees_brutes['T (degC)'], errors='coerce')
invalides = temperature_brute.notna() & ~temperature_brute.between(-90, 60)
serie = pd.Series(temperature_brute.mask(invalides).to_numpy(), index=dates,
                  name='Temperature_C').sort_index()
doublons = int(serie.index.duplicated().sum())
# Les doublons temporels éventuels sont moyennés par le regroupement horaire.
horaire_observee = serie.resample('1h', closed='left', label='right').mean()
cible_observee = horaire_observee.notna().to_numpy()
horaire = horaire_observee.ffill(limit=3)
qualite = {
    'lignes_brutes': len(serie), 'dates_dupliquees': doublons,
    'temperatures_hors_limites': int(invalides.sum()),
    'valeurs_brutes_absentes': int(temperature_brute.isna().sum()),
    'heures': len(horaire),
    'heures_absentes_avant_remplissage': int(horaire_observee.isna().sum()),
    'heures_absentes_apres_remplissage': int(horaire.isna().sum()),
}
display(pd.Series(qualite, name='Contrôle qualité'))
manquantes = horaire_observee.isna()
groupes = manquantes.ne(manquantes.shift()).cumsum()
periodes_manquantes = pd.DataFrame([
    {'Début': str(groupe.index[0]), 'Fin': str(groupe.index[-1]), 'Heures': len(groupe)}
    for _, groupe in horaire_observee[manquantes].groupby(groupes[manquantes])
])
display(periodes_manquantes)
print('Les trous longs restent absents ; leurs fenêtres seront exclues à l’étape 6.')
assert (horaire.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
print('Période horaire :', horaire.index.min(), '→', horaire.index.max())
display(horaire.describe().to_frame())
''')

section('''
## 4. Séparation, standardisation et traçabilité

Les premières 70 % des heures définissent la période d'entraînement, les 15 % suivantes la validation et les dernières 15 % le test. La moyenne et l'écart-type sont calculés **uniquement sur l'entraînement** : z = (T − moyenne) / écart-type.

Pour éviter qu'un modèle ait appris des cibles encore inconnues au moment d'une prévision de validation ou de test, nous réservons un délai de 24 heures au début de chaque période d'évaluation. Les cibles d'évaluation commencent à frontière +24 h ; la dernière observation de leur première fenêtre est donc à la frontière. Des observations antérieures peuvent figurer dans les fenêtres : elles sont disponibles au moment de prévoir.

Les dossiers de sortie sont propres à cette expérience. Une empreinte du protocole, des données et des versions évite de recharger silencieusement un modèle associé à d'autres réglages.
''', '''
n_heures = len(horaire)
frontiere_train = int(CONFIG['fraction_train'] * n_heures)
frontiere_val = frontiere_train + int(CONFIG['fraction_validation'] * n_heures)
valeurs_c = horaire.to_numpy(dtype=np.float32)
moyenne_train = float(np.nanmean(valeurs_c[:frontiere_train], dtype=np.float64))
ecart_type_train = float(np.nanstd(valeurs_c[:frontiere_train], dtype=np.float64))
assert ecart_type_train > 0
valeurs_z = ((valeurs_c - moyenne_train) / ecart_type_train).astype(np.float32)
protocole = {
    'config': CONFIG, 'url': URL, 'sha256_archive': empreinte_archive,
    'qualite': qualite, 'frontiere_train': frontiere_train,
    'frontiere_validation': frontiere_val,
    'moyenne_train_C': moyenne_train, 'ecart_type_train_C': ecart_type_train,
    'python': platform.python_version(), 'tensorflow': tf.__version__,
    'keras': keras.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
    'gpu': [tf.config.experimental.get_device_details(g).get('device_name', g.name) for g in gpus],
}
ID_EXPERIENCE = hashlib.sha256(json.dumps(protocole, sort_keys=True).encode()).hexdigest()[:12]
FIGURES = RACINE / 'figures' / 'jena' / ID_EXPERIENCE
RESULTATS = RACINE / 'resultats' / 'jena' / ID_EXPERIENCE
MODELES = RACINE / 'modeles' / 'jena' / ID_EXPERIENCE
for dossier in [FIGURES, RESULTATS, MODELES]:
    dossier.mkdir(parents=True, exist_ok=True)
(RESULTATS / 'protocole.json').write_text(json.dumps(protocole, ensure_ascii=False, indent=2), encoding='utf-8')
periodes_manquantes.to_csv(RESULTATS / 'periodes_manquantes.csv', index=False)
(RESULTATS.parent / 'derniere_experience.json').write_text(
    json.dumps({'id': ID_EXPERIENCE, 'resultats': str(RESULTATS), 'figures': str(FIGURES),
                'modeles': str(MODELES)}, ensure_ascii=False, indent=2), encoding='utf-8')
print('Expérience :', ID_EXPERIENCE)
print('Moyenne entraînement :', round(moyenne_train, 3), '°C')
print('Écart-type entraînement :', round(ecart_type_train, 3), '°C')
print('Dossier des résultats :', RESULTATS)
''')

section('''
## 5. Observer les températures et les périodes

Le graphique global utilise des moyennes journalières pour rester lisible ; les modèles recevront bien les températures **horaires**. Le second graphique montre deux semaines du début de l'entraînement. Nous n'utilisons pas la forme du test pour ajuster nos modèles.
''', '''
fig, axes = plt.subplots(2, 1, figsize=(12, 7))
horaire.iloc[:frontiere_train].resample('1D').mean().plot(ax=axes[0], label='Entraînement')
horaire.iloc[frontiere_train:frontiere_val].resample('1D').mean().plot(ax=axes[0], label='Validation')
# Montrer la période du test sans inspecter sa courbe avant l'évaluation finale.
axes[0].axvspan(horaire.index[frontiere_val], horaire.index[-1], color='gray', alpha=0.15,
               label='Test réservé (valeurs masquées)')
axes[0].set_title('Jena : températures moyennes journalières et périodes chronologiques')
horaire.iloc[:24 * 14].plot(ax=axes[1], color='tab:blue')
axes[1].set_title('Deux semaines de températures horaires : début de l’entraînement')
for ax in axes:
    ax.set_ylabel('Température (°C)')
    ax.set_xlabel('Date')
    ax.grid(alpha=0.3)
axes[0].legend()
fig.tight_layout()
fig.savefig(FIGURES / '01_donnees_et_periodes.png', dpi=180, bbox_inches='tight')
plt.show()
''')

section('''
## 6. Construire les fenêtres et les cibles à +24 h

Pour une cible à l'indice j, la dernière observation est j−24. Les 72 observations sont donc celles des indices j−95 à j−24, inclus. Une fenêtre commençant à l'indice 0 contient les indices 0 à 71 et sa cible est à l'indice 95.

Nous démarrons les cibles à l'indice 168 pour rendre également disponible une référence hebdomadaire j−168. Toutes les méthodes sont évaluées sur les mêmes cibles mesurées. Nous excluons les cibles initialement absentes, les fenêtres contenant encore une valeur absente et les exemples dont la référence hebdomadaire est absente. Le temps n'est jamais comprimé pour sauter une lacune.

Forme de X : `(exemples, 72, 1)` ; forme de y : `(exemples, 1)`. Les fenêtres voisines se recouvrent, donc les exemples ne sont pas indépendants. La mémoire des couches n'est pas conservée entre fenêtres (`stateful=False`).
''', '''
FENETRE = CONFIG['fenetre_heures']
HORIZON = CONFIG['horizon_heures']
premiere_cible = max(FENETRE - 1 + HORIZON, 168)
indices = np.arange(premiere_cible, n_heures)
# Compter les valeurs absentes dans chaque fenêtre sans créer tous les tableaux.
fins = indices - HORIZON
debuts = fins - FENETRE + 1
cumul_absences = np.concatenate(([0], np.cumsum(~np.isfinite(valeurs_z))))
fenetre_complete = (cumul_absences[fins + 1] - cumul_absences[debuts]) == 0
cible_valide = cible_observee[indices]
reference_valide = np.isfinite(valeurs_c[indices - 168])
garder = fenetre_complete & cible_valide & reference_valide
exclusions = {'candidats': len(indices), 'cibles_absentes': int((~cible_valide).sum()),
              'fenetres_incompletes': int((~fenetre_complete).sum()),
              'references_absentes': int((~reference_valide).sum()),
              'exemples_exclus_union': int((~garder).sum())}
display(pd.Series(exclusions, name='Exclusions avant séparation (motifs non exclusifs)'))
(RESULTATS / 'exclusions.json').write_text(json.dumps(exclusions, indent=2), encoding='utf-8')
indices = indices[garder]
indices_jeux = {
    'train': indices[indices < frontiere_train],
    'validation': indices[(indices >= frontiere_train + HORIZON) & (indices < frontiere_val)],
    'test': indices[indices >= frontiere_val + HORIZON],
}
def creer_fenetres(indices_cibles):
    fins = indices_cibles - HORIZON
    positions = fins[:, None] - np.arange(FENETRE - 1, -1, -1)[None, :]
    X_jeu = valeurs_z[positions, None]
    y_jeu = valeurs_z[indices_cibles, None]
    return X_jeu, y_jeu

jeux = {nom: creer_fenetres(idx) for nom, idx in indices_jeux.items()}
description_jeux = []
for nom, (entrees, cibles) in jeux.items():
    idx = indices_jeux[nom]
    assert len(idx) > 0 and entrees.shape[1:] == (FENETRE, 1)
    assert np.isfinite(entrees).all() and np.isfinite(cibles).all()
    description_jeux.append({'Jeu': nom, 'Exemples': len(idx), 'Forme X': str(entrees.shape),
                             'Première cible': str(horaire.index[idx[0]]),
                             'Dernière cible': str(horaire.index[idx[-1]])})
assert indices_jeux['train'][-1] < indices_jeux['validation'][0] - HORIZON
assert indices_jeux['validation'][-1] < indices_jeux['test'][0] - HORIZON
assert np.isclose(jeux['train'][0][0, -1, 0], valeurs_z[indices_jeux['train'][0] - HORIZON])
pd.DataFrame(description_jeux).to_csv(RESULTATS / 'divisions.csv', index=False)
display(pd.DataFrame(description_jeux))
''')

section('''
## 7. Visualiser exactement une fenêtre et son horizon

Les 72 points bleus sont connus au moment de la prévision. La cible rouge est 24 heures après le dernier point bleu. Les températures intermédiaires ne sont pas fournies au modèle. La cible montrée est issue de l'entraînement.
''', '''
j = int(indices_jeux['train'][0])
fin = j - HORIZON
debut = fin - FENETRE + 1
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(horaire.index[debut:fin + 1], valeurs_c[debut:fin + 1], 'o-', markersize=3,
        label='Entrée : 72 températures horaires')
ax.scatter(horaire.index[j], valeurs_c[j], color='red', s=90, label='Cible à +24 h', zorder=3)
ax.axvspan(horaire.index[fin], horaire.index[j], alpha=0.15, color='gray', label='Horizon de 24 h')
ax.set_title('Prévision directe : trois jours observés pour une cible à +24 heures')
ax.set_ylabel('Température (°C)')
ax.set_xlabel('Date et heure')
ax.legend()
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIGURES / '02_fenetre_horizon.png', dpi=180, bbox_inches='tight')
plt.show()
''')

section('''
## 8. Définir les métriques et les références sans apprentissage

Les erreurs seront toujours mesurées **en °C après retour à l'échelle d'origine**. MAE = moyenne des erreurs absolues ; RMSE = racine de la moyenne des erreurs au carré.

Références :
- Persistance : température à l'instant de prévision, soit cible−24 h. À cet horizon précis, « même heure de la veille » et persistance sont la même référence ; nous ne les comptons pas deux fois.
- Référence hebdomadaire : température mesurée à cible−168 h. Cette observation est déjà disponible à l'instant de prévision, même si elle est hors de la fenêtre de 72 h.

Nous n'examinons ici que la validation. Une méthode simple peut battre un réseau : cela reste un résultat valable.
''', '''
def vers_celsius(valeurs):
    return np.asarray(valeurs).reshape(-1) * ecart_type_train + moyenne_train

def metriques(reel, predit):
    reel = np.asarray(reel, dtype=np.float64).reshape(-1)
    predit = np.asarray(predit, dtype=np.float64).reshape(-1)
    assert reel.shape == predit.shape and np.isfinite(reel).all() and np.isfinite(predit).all()
    erreur = predit - reel
    return {'MAE (°C)': float(np.mean(np.abs(erreur))),
            'RMSE (°C)': float(np.sqrt(np.mean(erreur ** 2)))}

def predictions_references(nom_jeu):
    idx = indices_jeux[nom_jeu]
    return {'Persistance (24 h)': valeurs_c[idx - HORIZON],
            'Référence hebdomadaire': valeurs_c[idx - 168]}

reel_val = valeurs_c[indices_jeux['validation']]
predictions_val = predictions_references('validation')
table_references = pd.DataFrame([
    {'Modèle': nom, 'Jeu': 'Validation', **metriques(reel_val, pred)}
    for nom, pred in predictions_val.items()])
display(table_references.round(4))
table_references.to_csv(RESULTATS / 'references_validation.csv', index=False)
''')

section('''
## 9. Définir les architectures et la procédure d'entraînement

Entrée `(72, 1)` → couche récurrente de 32 unités → Dense(1). Chaque réseau possède son propre Adam ; même graine et même ordre des exemples, mais les poids ne sont pas identiques car leurs formes diffèrent. Le gradient est limité avec `clipnorm=1` pour les trois architectures.

Les cellules suivantes entraînent séparément chaque modèle. Aucun entraînement ne démarre ici. La première époque peut être beaucoup plus longue (initialisation/compilation GPU). Nous enregistrons temps total de fit, première époque et médiane des suivantes. Ce ne sont pas des mesures répétées de performance matérielle.

**Reprise après redémarrage :** exécuter les cellules depuis le début. Si le modèle, l'historique, les temps et le marqueur de fin existent pour ce protocole, ils seront rechargés. Un entraînement interrompu sans marqueur de fin recommence depuis zéro ; il n'est pas repris à mi-époque. `REENTRAINER=True` force le réentraînement et remplace les fichiers correspondants.
''', '''
NOMS = ['SimpleRNN', 'LSTM', 'GRU']
def construire(nom):
    keras.utils.set_random_seed(CONFIG['graine'])
    classes = {'SimpleRNN': keras.layers.SimpleRNN, 'LSTM': keras.layers.LSTM, 'GRU': keras.layers.GRU}
    recurrente = classes[nom](CONFIG['unites'], return_sequences=False)
    modele = keras.Sequential([keras.Input(shape=(FENETRE, 1)), recurrente, keras.layers.Dense(1)], name=nom)
    modele.compile(optimizer=keras.optimizers.Adam(
        learning_rate=CONFIG['taux_apprentissage'], clipnorm=CONFIG['clipnorm']), loss='mse')
    return modele

class Suivi(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.durees = []
    def on_epoch_begin(self, epoch, logs=None):
        self.debut = time.perf_counter()
    def on_epoch_end(self, epoch, logs=None):
        self.durees.append(time.perf_counter() - self.debut)
        print(f'Époque {epoch + 1:02d}/{CONFIG["epoques"]} | MSE normalisée : {logs["loss"]:.5f}'
              f' | Validation : {logs["val_loss"]:.5f} | {self.durees[-1]:.2f} s', flush=True)

def entrainer_ou_charger(nom):
    fichier_modele = MODELES / f'{nom}.keras'
    fichier_historique = RESULTATS / f'historique_{nom}.csv'
    fichier_temps = RESULTATS / f'temps_{nom}.json'
    marqueur = RESULTATS / f'termine_{nom}.json'
    if not REENTRAINER and all(p.exists() for p in [fichier_modele, fichier_historique, fichier_temps, marqueur]):
        assert json.loads(marqueur.read_text())['experience'] == ID_EXPERIENCE
        print(nom, ': entraînement sauvegardé rechargé ; aucun nouvel entraînement.')
        return (keras.models.load_model(fichier_modele), pd.read_csv(fichier_historique),
                json.loads(fichier_temps.read_text()))
    marqueur.unlink(missing_ok=True)
    modele = construire(nom)
    suivi = Suivi()
    print(nom, ': début de l’entraînement ; première époque potentiellement longue.', flush=True)
    debut = time.perf_counter()
    historique = modele.fit(*jeux['train'], validation_data=jeux['validation'],
        epochs=CONFIG['epoques'], batch_size=CONFIG['lot'], shuffle=False,
        callbacks=[suivi, keras.callbacks.TerminateOnNaN()], verbose=0)
    duree = time.perf_counter() - debut
    tableau = pd.DataFrame(historique.history)
    assert len(tableau) == CONFIG['epoques'], 'Entraînement incomplet : ne pas comparer ce modèle.'
    assert np.isfinite(tableau[['loss', 'val_loss']].to_numpy()).all(), 'Perte non finie.'
    tableau.insert(0, 'epoque', np.arange(1, len(tableau) + 1))
    tableau['duree_secondes'] = suivi.durees
    temps = {'total_fit_s': duree, 'premiere_epoque_s': suivi.durees[0],
             'mediane_epoques_suivantes_s': float(np.median(suivi.durees[1:]))}
    modele.save(fichier_modele)
    tableau.to_csv(fichier_historique, index=False)
    fichier_temps.write_text(json.dumps(temps, indent=2), encoding='utf-8')
    marqueur.write_text(json.dumps({'experience': ID_EXPERIENCE, 'epoques': len(tableau)}), encoding='utf-8')
    print(f'{nom} terminé et sauvegardé : {duree:.2f} s de fit.')
    return modele, tableau, temps

architectures = pd.DataFrame([{'Modèle': nom, 'Unités': CONFIG['unites'],
    'Paramètres': construire(nom).count_params()} for nom in NOMS])
architectures.to_csv(RESULTATS / 'architectures.csv', index=False)
display(architectures)
modeles, historiques, durees = {}, {}, {}
''')

for numero, nom in enumerate(['SimpleRNN', 'LSTM', 'GRU'], 10):
    section(f'''
## {numero}. Entraîner ou recharger {nom}

Cette cellule lance au maximum 20 époques pour **{nom}**, ou recharge son entraînement complet déjà enregistré. Laisser la cellule terminer avant de passer à la suivante. Les pertes affichées sont des MSE sur températures standardisées, pas des erreurs en °C. Aucune donnée de test n'est utilisée.
''', f'''
modeles['{nom}'], historiques['{nom}'], durees['{nom}'] = entrainer_ou_charger('{nom}')
''')

section('''
## 13. Comparer sur la validation

Les trois modèles sont évalués après les mêmes 20 époques. Les prédictions sont converties en °C. Nous affichons MAE et RMSE et les comparons aux deux références. Aucun choix d'hyperparamètres supplémentaire n'est automatisé.

Les courbes permettent de repérer un écart croissant entre entraînement et validation. Si un réseau ne bat pas une référence, ne pas masquer ce résultat. Ne pas comparer directement ces scores à ceux de la sinusoïde : les unités, les données et l'horizon diffèrent.
''', '''
assert set(modeles) == set(NOMS), 'Exécuter les trois cellules précédentes.'
for nom in NOMS:
    predictions_val[nom] = vers_celsius(modeles[nom].predict(
        jeux['validation'][0], batch_size=CONFIG['lot'], verbose=0))
comparaison_val = pd.DataFrame([{'Modèle': nom, 'Jeu': 'Validation', **metriques(reel_val, pred)}
                               for nom, pred in predictions_val.items()])
display(comparaison_val.round(4))
comparaison_val.to_csv(RESULTATS / 'comparaison_validation.csv', index=False)
pd.DataFrame({'date_cible': horaire.index[indices_jeux['validation']], 'Réel (°C)': reel_val,
              **predictions_val}).to_csv(RESULTATS / 'predictions_validation.csv', index=False)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for nom in NOMS:
    h = historiques[nom]
    axes[0].plot(h['epoque'], h['loss'], label=nom)
    axes[1].plot(h['epoque'], h['val_loss'], label=nom)
for ax, titre in zip(axes, ['Entraînement', 'Validation']):
    ax.set_title(titre)
    ax.set_xlabel('Époque')
    ax.set_ylabel('MSE normalisée')
    ax.legend()
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / '03_apprentissage.png', dpi=180, bbox_inches='tight')
plt.show()
''')

section('''
## 14. Évaluer le test après avoir figé les modèles

**Avant d'exécuter :** vérifier que les entraînements et la validation sont complets. Les modèles finaux et le protocole restent fixes. Nous évaluons tous les modèles, sans en réentraîner un pour améliorer ce test.

L'évaluation avance dans le temps : pour chaque cible à +24 h, les observations disponibles à l'instant de prévision sont utilisées. Il ne s'agit pas de prévoir toute la période de test depuis une seule date initiale.

La réduction relative de MAE est calculée par rapport à la persistance ; une valeur négative signifie une dégradation. Ce pourcentage n'est pas une précision de classification. Les références ne nécessitent pas d'entraînement ; leurs temps d'entraînement sont non applicables.
''', '''
reel_test = valeurs_c[indices_jeux['test']]
predictions_test = predictions_references('test')
for nom in NOMS:
    predictions_test[nom] = vers_celsius(modeles[nom].predict(
        jeux['test'][0], batch_size=CONFIG['lot'], verbose=0))
mae_reference = metriques(reel_test, predictions_test['Persistance (24 h)'])['MAE (°C)']
lignes = []
for nom, pred in predictions_test.items():
    score = metriques(reel_test, pred)
    lignes.append({'Modèle': nom, 'Jeu': 'Test', **score,
        'Réduction MAE / persistance (%)': 100 * (1 - score['MAE (°C)'] / mae_reference),
        'Paramètres': modeles[nom].count_params() if nom in modeles else 0,
        'Durée fit (s)': durees[nom]['total_fit_s'] if nom in durees else np.nan,
        'Première époque (s)': durees[nom]['premiere_epoque_s'] if nom in durees else np.nan,
        'Médiane époques suivantes (s)': durees[nom]['mediane_epoques_suivantes_s'] if nom in durees else np.nan})
comparaison_test = pd.DataFrame(lignes)
display(comparaison_test.round(4))
comparaison_test.to_csv(RESULTATS / 'comparaison_test.csv', index=False)
pd.DataFrame({'date_cible': horaire.index[indices_jeux['test']], 'Réel (°C)': reel_test,
              **predictions_test}).to_csv(RESULTATS / 'predictions_test.csv', index=False)
print('Évaluation finale enregistrée. Ne pas ajuster les modèles à partir du test.')
''')

section('''
## 15. Produire les figures de test pour le PowerPoint

Les barres utilisent tout le test. Le graphique temporel montre sa première semaine, choisie à l'avance pour éviter de sélectionner une période particulièrement favorable. L'axe horizontal représente la date de la **cible**, dont la prévision a été émise 24 heures auparavant.

Les prévisions des réseaux sont affichées ensemble ; les références sont dans un panneau séparé pour conserver la lisibilité. Une figure affiche des températures prévues, pas seulement une tendance apprise par définition.
''', '''
couleurs = ['gray', 'tab:purple', 'tab:blue', 'tab:orange', 'tab:green']
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, mesure in zip(axes, ['MAE (°C)', 'RMSE (°C)']):
    barres = ax.bar(comparaison_test['Modèle'], comparaison_test[mesure], color=couleurs)
    ax.bar_label(barres, fmt='%.3f', padding=3)
    ax.set_ylim(0, comparaison_test[mesure].max() * 1.2)
    ax.set_ylabel(mesure)
    ax.set_title(f'Test à +24 h : {mesure}')
    ax.tick_params(axis='x', rotation=25)
    ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / '04_erreurs_test.png', dpi=180, bbox_inches='tight')
plt.show()

dates_test = horaire.index[indices_jeux['test']]
masque_semaine = dates_test < dates_test[0] + pd.Timedelta(days=7)
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
for ax in axes:
    ax.plot(dates_test[masque_semaine], reel_test[masque_semaine], color='black', label='Réel', linewidth=2)
for nom, couleur in zip(NOMS, couleurs[2:]):
    axes[0].plot(dates_test[masque_semaine], predictions_test[nom][masque_semaine], label=nom, color=couleur)
for nom, couleur in zip(list(predictions_test)[:2], couleurs[:2]):
    axes[1].plot(dates_test[masque_semaine], predictions_test[nom][masque_semaine], label=nom, color=couleur, linestyle='--')
axes[0].set_title('Première semaine du test : prévisions des réseaux à +24 h')
axes[1].set_title('Même semaine : références sans apprentissage')
for ax in axes:
    ax.set_ylabel('Température (°C)')
    ax.legend()
    ax.grid(alpha=0.3)
axes[1].set_xlabel('Date de la cible')
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIGURES / '05_predictions_test.png', dpi=180, bbox_inches='tight')
plt.show()
''')

section('''
## 16. Comparer taille et temps mesurés

Comparer séparément les paramètres, le temps total de fit (démarrage inclus) et la médiane des époques après la première. Les modèles ont le même nombre d'unités mais pas le même nombre de paramètres. L'ordre d'exécution, les bibliothèques optimisées et la compilation peuvent influencer les temps. Une seule exécution ne suffit pas pour conclure qu'une architecture est toujours plus rapide.
''', '''
bilan = comparaison_test.set_index('Modèle').loc[NOMS]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, colonne, titre, unite, fmt in zip(axes,
    ['Paramètres', 'Durée fit (s)', 'Médiane époques suivantes (s)'],
    ['Taille des réseaux', 'Temps de fit, démarrage inclus', 'Époque après démarrage (médiane)'],
    ['Paramètres', 'Secondes', 'Secondes'], ['%.0f', '%.1f', '%.3f']):
    valeurs = bilan[colonne]
    barres = ax.bar(NOMS, valeurs, color=couleurs[2:])
    ax.bar_label(barres, fmt=fmt, padding=3)
    ax.set_ylim(0, valeurs.max() * 1.2)
    ax.set_title(titre, fontsize=10)
    ax.set_ylabel(unite)
    ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / '06_parametres_durees.png', dpi=180, bbox_inches='tight')
plt.show()
''')

section('''
## 17. Exporter les observations pour préparer le diaporama

Cette cellule exporte les scores réellement obtenus et une liste de questions d'interprétation. Elle ne rédige pas de conclusion gagnante à l'avance. Compléter l'analyse après lecture des tableaux et des courbes.

Le diaporama pourra consacrer une courte introduction à la sinusoïde puis se concentrer sur Jena, le protocole, les résultats en °C et leurs limites. Ne pas mélanger les scores des deux problèmes sur un même axe.
''', '''
lignes_notes = [
    '# Expérience réelle : Jena Climate', '',
    f'Expérience : {ID_EXPERIENCE}',
    'Prévision directe à +24 h à partir de 72 températures horaires.',
    'Une exécution par architecture, 20 époques fixes ; erreurs en °C.', '',
    '## Scores de test mesurés', '',
]
for _, ligne in comparaison_test.iterrows():
    lignes_notes.append(f'- {ligne["Modèle"]} : MAE {ligne["MAE (°C)"]:.4f} °C ; RMSE {ligne["RMSE (°C)"]:.4f} °C.')
lignes_notes += [
    '', '## Questions à discuter après exécution', '',
    '- Les réseaux améliorent-ils les deux références ?',
    '- Les écarts entre réseaux sont-ils grands par rapport aux erreurs ?',
    '- Les courbes montrent-elles un surapprentissage ou un apprentissage encore en cours ?',
    '- Quel compromis entre erreur, paramètres et durées mesurées ?',
    '', '## Limites à mentionner', '',
    '- Une station, une variable, un horizon et une seule graine de modèle.',
    '- Fenêtres recouvrantes ; pas de test de significativité ni de répétitions.',
    '- Découpage chronologique : les saisons et années des jeux diffèrent.',
    '- Moyennes horaires, remplissage causal limité à 3 h ; fenêtres incomplètes et cibles initialement absentes exclues.',
    '- Températures issues du passé réel à chaque instant de prévision ; pas de simulation récursive de toute la période.',
    '- Les résultats de la sinusoïde et de Jena ne sont pas directement comparables.',
    '- Ne pas généraliser le classement à tous les problèmes temporels.',
    '', '## Figures', '',
] + [f'- {p.name}' for p in sorted(FIGURES.glob('*.png'))]
(RESULTATS / 'notes_presentation_jena.md').write_text('\\n'.join(lignes_notes), encoding='utf-8')
print('Notes enregistrées :', RESULTATS / 'notes_presentation_jena.md')
print('Figures :', FIGURES)
print('Modèles :', MODELES)
print('Enregistrer le Notebook avec Ctrl+S après vérification des sorties.')
''')

nb.cells = cells
nb.metadata = {'kernelspec': {'display_name': 'Python (RNN-LSTM-GRU | WSL GPU)',
                              'language': 'python', 'name': 'rnn-lstm-gru'},
               'language_info': {'name': 'python', 'version': '3.12.3'}}
nbf.validate(nb)
destination = Path(sys.argv[1] if len(sys.argv) > 1 else '02_comparaison_jena_climate.ipynb')
if destination.exists():
    raise SystemExit('Notebook déjà présent : ne pas écraser le travail enregistré.')
nbf.write(nb, destination)
print(f'{destination} : {len(cells)} cellules ; validation structurelle et syntaxique réussie.')
