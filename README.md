# Atelier Projet 2 — Apprentissage automatique pour la génération de mots de passe

> UE Innovation · Master ILSEN · Avignon Université · Encadrants : Mohamed Morchid, Majed Haddad

Un réseau de neurones récurrent (RNN / GRU / LSTM, en PyTorch) apprend à **générer des séquences caractère par caractère**. Le projet se fait en trois temps : on apprend la méthode sur du texte, on prépare son adaptation aux mots de passe, puis on l'implémente et on l'évalue. C'est un projet de recherche en sécurité : il sert à mesurer à quel point des mots de passe choisis par des humains sont prévisibles.

---

## Sommaire

1. [Le sujet](#1-le-sujet)
2. [Suivi d'avancement](#2-suivi-davancement)
3. [Ce semestre : ce qu'il faut rendre](#3-ce-semestre--ce-quil-faut-rendre)
4. [Du texte aux mots de passe](#4-du-texte-aux-mots-de-passe)
5. [Documentation : quoi lire, dans quel ordre](#5-documentation--quoi-lire-dans-quel-ordre)
6. [Utiliser le code](#6-utiliser-le-code)
7. [Organisation du dépôt](#7-organisation-du-dépôt)

---

## 1. Le sujet

D'après [`docs/sujet/sujet_defi_generation_mdp.pdf`](docs/sujet/sujet_defi_generation_mdp.pdf).

### Objectif

Proposer un système qui génère des mots de passe :

| Critère | Ce que ça veut dire |
|---|---|
| **Diversité** | Retrouver le plus possible de mots de passe d'une base de test **commune à tous les étudiants** |
| **Efficacité** | Être appris dans le temps le plus court possible |
| **Sobriété** | Utiliser le moins de données d'apprentissage possible |

**Métrique finale :** le **taux de recouvrement** (nombre de mots de passe du test retrouvés ÷ nombre de mots de passe dans le test), avec le temps de calcul.

### Les étapes du défi

1. Comprendre la modélisation neuronale de séquences
2. Choisir la stratégie d'apprentissage : le modèle (RNN, LSTM…) et les données (corpus, train / validation / test)
3. Prendre en main PyTorch
4. Implémenter le système
5. Analyser les résultats

### Ce qui est attendu (slides 39 à 42)

| Bloc | Attendus |
|---|---|
| **Analyse des données** | Choix du corpus (d'autres corpus sont possibles s'ils sont justifiés, le test reste le même), stratégie de gestion des données, statistiques (au minimum des cooccurrences ; une PCA pour la variabilité) |
| **Réflexion sur le modèle** | Analyse des modèles existants de génération de séquences ; stratégie d'apprentissage (dropout, normalisation, nombre de couches, hyperparamètres) |
| **Implémentation** | Prise en main de PyTorch ; implémentation du ou des systèmes de génération |
| **Analyse des résultats** | Choix des hyperparamètres sur la validation ; évaluation sur le test commun (recouvrement + temps) ; analyse et pistes d'amélioration |

---

## 2. Suivi d'avancement

Légende : ✅ fait · 🔄 en cours · ⬜ à faire

### Premier semestre : apprendre la méthode sur du texte (Shakespeare)

| Étape du défi | Tâche | État |
|---|---|---|
| 1. Comprendre | Neurone, loss, rétropropagation, RNN, BPTT, vanishing gradient, GRU, LSTM | ✅ |
| 3. PyTorch | Faire tourner `articleGeneration.py` (entraînement + génération) | ✅ |
| 2. Stratégie | Plan d'expériences : taille cachée, nombre de couches, nombre d'itérations | 🔄 Résultats à reporter ci-dessous |
| 2. Stratégie | Améliorations : LSTM, dropout, LayerNorm, clipping, validation | 🔄 |

**Résultats du S1** (à compléter avec vos chiffres, voir le [guide TP §4.5](docs/cours/2_guide_TP.md#45)) :

| Config | Couches | Hidden | Itérations | Loss finale | Texte généré depuis `Wh` |
|---|---|---|---|---|---|
| Référence | 2 | 128 | 10 000 | `[__]` | `[__]` |
| … | | | | | |

### Ce semestre : préparer la génération de mots de passe

Aucun mot de passe n'est généré ce semestre.

| Tâche | État |
|---|---|
| Compléter le tableau de résultats du S1 | ⬜ |
| Rédiger le cahier des charges (checklist du [§3](#le-cahier-des-charges)) | ⬜ |
| Préparer les slides de l'oral | ⬜ |
| Réviser le QCM | ⬜ |
| Répéter l'oral en entier, chronométré | ⬜ |

### Semestre prochain : implémenter et évaluer

| Étape du défi | Tâche | État |
|---|---|---|
| 2. Stratégie | Analyse statistique du corpus de mots de passe | ⬜ |
| 4. Implémentation | Adapter le code (voir [§4](#4-du-texte-aux-mots-de-passe)) | ⬜ |
| 5. Analyse | Choisir les hyperparamètres sur la validation | ⬜ |
| 5. Analyse | Évaluer sur le test commun : recouvrement + temps | ⬜ |
| 5. Analyse | Analyser les résultats, proposer des améliorations | ⬜ |

---

## 3. Ce semestre : ce qu'il faut rendre

### Consignes données en cours

Notes prises telles quelles :

> Pas de rapport, mais prévoir une partie cahier des charges pendant l'oral.
>
> **Présentation :**
> - 50 % cahier des charges, 50 % oral
> - ce qu'on a fait au premier semestre et ce qui est prévu pour le semestre prochain
> - comment exploiter ce qu'on a fait au premier semestre pour l'adapter à la génération de mots de passe
> - ce semestre : préparation pour la génération de mots de passe (mais aucun mot de passe généré ce semestre)
> - 15 minutes de présentation, 5 minutes de questions
>
> **Coefficients :** 20 % QCM · 40 % cahier des charges (oral) · 40 % oral en lui-même

### Plan de l'oral proposé (15 min)

| # | Partie | Durée | Contenu |
|---|---|---|---|
| 1 | Contexte & objectif | 1 min 30 | Le sujet, les 3 critères, la métrique |
| 2 | Bilan du S1 | 4 min | Principe du générateur, architecture, expériences, résultats, GRU vs LSTM |
| 3 | Adaptation aux mots de passe | 2 min | Le tableau du [§4](#4-du-texte-aux-mots-de-passe) |
| 4 | **Cahier des charges** | 6 min | La checklist ci-dessous |
| 5 | Planning du semestre prochain & risques | 1 min 30 | Jalons, répartition, points de blocage possibles |

### Le cahier des charges

Construit sur les attendus du sujet ([§1](#ce-qui-est-attendu-slides-39-à-42)) :

- [ ] **Besoin et objectifs** : les 3 critères, le taux de recouvrement, le temps de calcul
- [ ] **Données** : corpus retenu (et corpus supplémentaires justifiés), découpe train / validation / test, nettoyage (doublons, longueurs, caractères hors vocabulaire), analyses statistiques prévues (longueurs, fréquence des caractères, cooccurrences, PCA)
- [ ] **Modèle** : état de l'art (RNN, GRU, LSTM, éventuellement Transformer), modèle retenu et pourquoi (en s'appuyant sur le S1), stratégie d'apprentissage (dropout, normalisation, couches, hyperparamètres à explorer)
- [ ] **Implémentation** : PyTorch, modifications du code du S1, environnement (GPU / CPU)
- [ ] **Évaluation** : hyperparamètres choisis sur la validation, recouvrement et temps sur le test, comparaison avec l'outil de référence indiqué dans le sujet
- [ ] **Contraintes** : temps de calcul, matériel, cadre éthique
- [ ] **Planning du semestre prochain** et livrables
- [ ] **Risques** et parades : sur-apprentissage, doublons générés, temps d'entraînement

### Réviser le QCM

Tout est dans le [cours de théorie](docs/cours/1_theorie_RNN_LSTM.md), et sa [fiche récapitulative (§10)](docs/cours/1_theorie_RNN_LSTM.md#10) + ses [questions d'oral (§11)](docs/cours/1_theorie_RNN_LSTM.md#oral) servent aussi pour les 5 minutes de questions.

---

## 4. Du texte aux mots de passe

Le modèle reste un générateur de caractères. C'est **ce qui l'entoure** qui change.

| | S1 : Shakespeare | Mots de passe |
|---|---|---|
| **Données** | Un long texte continu | Une liste de séquences courtes et indépendantes |
| **Découpage** | Morceaux aléatoires de 13 caractères | Un mot de passe = une séquence, avec des marqueurs **début** et **fin** |
| **Arrêt de la génération** | Longueur fixe (`--length`) | Le modèle décide quand s'arrêter (marqueur de fin) |
| **Vocabulaire** | `string.printable` | Les caractères réellement présents dans le corpus |
| **Objectif** | Un texte lisible | Beaucoup de candidats **différents** et **plausibles** |
| **Évaluation** | Loss + lecture | Loss de validation + **taux de recouvrement** |
| **Point d'attention** | — | Doublons, compromis entre température et diversité |

**Modifications du code à prévoir** (semestre prochain) :
- charger le corpus ligne par ligne au lieu de `random_chunk`, avec des lots de séquences ;
- ajouter les marqueurs de début et de fin au vocabulaire ;
- réécrire `evaluate` pour produire N candidats uniques qui s'arrêtent au marqueur de fin ;
- ajouter le calcul du taux de recouvrement sur le test.

> **Cadre éthique :** on travaille uniquement sur les jeux de données fournis par les encadrants, dans un but de recherche (évaluer la robustesse des mots de passe). Les données ne sont pas redistribuées et ne sont jamais utilisées contre des comptes réels.

---

## 5. Documentation : quoi lire, dans quel ordre

| Ordre | Fichier | Contenu | Quand le lire |
|---|---|---|---|
| 1 | [`docs/sujet/sujet_defi_generation_mdp.pdf`](docs/sujet/sujet_defi_generation_mdp.pdf) | Le sujet officiel : objectifs, théorie, attendus, évaluation | Pour savoir ce qu'on attend de nous |
| 2 | [`docs/cours/1_theorie_RNN_LSTM.md`](docs/cours/1_theorie_RNN_LSTM.md) | Toute la théorie, du neurone au LSTM, avec exemples chiffrés + questions d'oral | Pour comprendre, réviser le QCM, préparer les questions |
| 3 | [`docs/cours/2_guide_TP.md`](docs/cours/2_guide_TP.md) | La pratique : données, code ligne par ligne, commandes, expériences, améliorations | Pour faire tourner le code et les expériences |
| — | [`docs/sujet/cours_securite_mots_de_passe.pdf`](docs/sujet/cours_securite_mots_de_passe.pdf) | Cours d'introduction à la sécurité et à l'authentification | Pour le contexte de l'oral |

Les deux cours ne se répètent pas : le guide TP renvoie vers la théorie au lieu de la réexpliquer.

> Les formules sont en LaTeX : ouvrir les `.md` dans VS Code (`Ctrl + Shift + V`) ou sur GitHub.

---

## 6. Utiliser le code

```bash
pip install torch unidecode matplotlib
```

À lancer **depuis la racine du dépôt** (les chemins par défaut sont `data/` et `models/`).

```bash
# Entraîner (défaut : 2 couches, hidden = 128, 10 000 itérations, corpus Shakespeare)
python articleGeneration.py --num_layers 2 --hidden_size 128 --max_epochs 10000 --run exp

# Générer 300 caractères avec ce modèle (mêmes --run, --num_layers, --hidden_size)
python articleGeneration.py -te eval --num_layers 2 --hidden_size 128 --run exp --length 300
```

Le modèle est sauvegardé dans `models/<run>_<num_layers>_<hidden_size>.pt`.

> ⚠️ Avec PyTorch ≥ 2.6, le mode `eval` plante (`weights_only`). La correction est dans le [guide TP §4.4](docs/cours/2_guide_TP.md#44).

| Argument | Défaut | Rôle |
|---|---|---|
| `-d`, `--trainingData` | `data/shakespeare.txt` | Corpus d'entraînement |
| `-te`, `--trainEval` | `train` | `train` ou `eval` |
| `-r`, `--run` | `rnnGeneration` | Nom de l'expérience |
| `-m`, `--model` | `models` | Dossier des modèles |
| `--length` | `100` | Caractères générés en `eval` |
| `--num_layers` | `2` | Couches GRU |
| `--hidden_size` | `128` | Taille de l'état caché |
| `--max_epochs` | `10000` | Itérations d'entraînement |

---

## 7. Organisation du dépôt

```
AtelierProjet2/
├── README.md                  ← ce fichier : sujet, suivi, consignes, utilisation
├── articleGeneration.py       ← générateur caractère par caractère (GRU, PyTorch)
├── data/
│   └── shakespeare.txt        ← corpus du premier semestre
├── models/                    ← modèles entraînés (.pt)
└── docs/
    ├── cours/
    │   ├── 1_theorie_RNN_LSTM.md      ← théorie + questions d'oral
    │   └── 2_guide_TP.md              ← pratique : code, expériences, améliorations
    └── sujet/
        ├── sujet_defi_generation_mdp.pdf    ← sujet officiel du défi
        └── cours_securite_mots_de_passe.pdf ← cours d'introduction à la sécurité
```

`articleGeneration.py`, `data/` et `models/` restent à la racine : le script les cherche à ces chemins.
