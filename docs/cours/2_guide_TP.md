# TP d'initiation : générer du texte avec un réseau de neurones récurrent (PyTorch)

> Atelier Projet 2, Master ILSEN. Sujet de M. Morchid et M. Haddad.
>
> **Pour voir les formules correctement**, ouvre ce fichier dans **VS Code** (`Ctrl + Shift + V`) ou sur GitHub. Les formules sont écrites en LaTeX entre `$ ... $`.

> **À lire en second.** Ce guide est la **partie pratique** : les données, le code ligne par ligne, les commandes, le plan d'expériences et les améliorations de la partie III. La théorie (neurone, softmax, loss, RNN, BPTT, GRU, LSTM) est dans [`1_theorie_RNN_LSTM.md`](1_theorie_RNN_LSTM.md) : les sections du guide qui la répétaient renvoient maintenant vers elle.

Les sorties de terminal montrées ici viennent de vrais lancements du programme sur ton PC (CPU, PyTorch 2.x).

---

## Sommaire

- [0. Le TP en une page](#0)
- [1. Les bases indispensables](#1)
  - [1.1 Vecteur et matrice](#11)
  - [1.2 Le neurone](#12)
  - [1.3 Les fonctions d'activation](#13)
  - [1.4 Softmax : des scores aux probabilités](#14)
  - [1.5 La loss : l'entropie croisée](#15)
  - [1.6 Gradient et descente de gradient](#16)
  - [1.7 La rétropropagation](#17)
  - [1.8 Hyperparamètres, train / validation, sur-apprentissage](#18)
- [2. Partie I du TP : récupérer et préparer les données](#2)
  - [2.1 Un corpus, c'est quoi ?](#21)
  - [2.2 Organiser le dossier](#22)
  - [2.3 Pourquoi un fichier de moins de 10 Mo ?](#23)
  - [2.4 Du texte aux nombres : unidecode, string.printable, index](#24)
  - [2.5 L'embedding](#25)
- [3. Les réseaux récurrents (RNN)](#3)
  - [3.1 Le RNN simple](#31)
  - [3.2 Prédire le caractère suivant](#32)
  - [3.3 Générer du texte et la température](#33)
- [4. Partie II du TP : l'apprentissage](#4)
  - [4.1 Le code, fonction par fonction](#41)
  - [4.2 Les arguments de la ligne de commande](#42)
  - [4.3 Lire la sortie d'entraînement](#43)
  - [4.4 Le bug du mode eval (PyTorch 2.6 et plus)](#44)
  - [4.5 Plan d'expériences](#45)
- [5. Partie III du TP : améliorer l'IA](#5)
  - [5.1 Gradient qui disparaît ou qui explose](#51)
  - [5.2 GRU et LSTM](#52)
  - [5.3 Remplacer le GRU par un LSTM](#53)
  - [5.4 Ajouter du dropout](#54)
  - [5.5 Ajouter une LayerNorm](#55)
  - [5.6 Le gradient clipping](#56)
  - [5.7 Un learning rate variable](#57)
  - [5.8 Mesurer honnêtement : validation et meilleur modèle](#58)
  - [5.9 Augmenter ou diversifier le corpus](#59)
- [6. Fiche récapitulative](#6)
- [7. Checklist de fin de TP](#7)

---

<a id="0"></a>
# 0. Le TP en une page

**Le but :** apprendre à une machine à écrire « comme Shakespeare », **lettre par lettre**.

L'idée est simple. On montre au programme des millions de fois un bout de texte et on lui demande : *« quel est le caractère suivant ? »*. Au début il répond au hasard. À force de corrections, il apprend que après `th` vient souvent `e`, qu'après un point vient une majuscule, qu'un nom de personnage est suivi de `:` et d'un retour à la ligne, etc.

Une fois entraîné, on lui donne un début (par exemple `Wh`) et il invente la suite, caractère par caractère.

```
   Texte d'entraînement : "To be, or not to be"

   Entrée  :  T   o   ␣   b   e   ,
               \   \   \   \   \   \
   Cible   :   o   ␣   b   e   ,   ␣      (le caractère suivant, à chaque fois)
```

Les 3 parties du TP :

| Partie | Ce que tu fais | Section du cours |
|---|---|---|
| I. Données | Créer `AtelierProjet2/data/` et y mettre `shakespeare.txt` | [2](#2) |
| II. Apprentissage | Lancer `articleGeneration.py`, tester plusieurs tailles de réseau | [3](#3) et [4](#4) |
| III. Amélioration | Remplacer le GRU par un LSTM, ajouter dropout, normalisation, etc. | [5](#5) |

---

<a id="1"></a>
# 1. Les bases indispensables

<a id="11"></a>
## 1.1 Vecteur et matrice

> 📖 Théorie : voir [théorie §1.1 à §1.4](1_theorie_RNN_LSTM.md). Cette section a été retirée ici pour ne pas faire doublon.

<a id="12"></a>
## 1.2 Le neurone

> 📖 Théorie : voir [théorie §2.1](1_theorie_RNN_LSTM.md). Cette section a été retirée ici pour ne pas faire doublon.

<a id="13"></a>
## 1.3 Les fonctions d'activation

> 📖 Théorie : voir [théorie §2.2 à §2.6](1_theorie_RNN_LSTM.md). Cette section a été retirée ici pour ne pas faire doublon.

<a id="14"></a>
## 1.4 Softmax : des scores aux probabilités

> 📖 Théorie : voir [théorie §3.1](1_theorie_RNN_LSTM.md). Cette section a été retirée ici pour ne pas faire doublon.

<a id="15"></a>
## 1.5 La loss : l'entropie croisée

> 📖 Théorie : voir [théorie §3.2](1_theorie_RNN_LSTM.md). Cette section a été retirée ici pour ne pas faire doublon.

<a id="16"></a>
## 1.6 Gradient et descente de gradient

> 📖 Théorie : voir [théorie §3.3](1_theorie_RNN_LSTM.md). Cette section a été retirée ici pour ne pas faire doublon.

<a id="17"></a>
## 1.7 La rétropropagation

> 📖 Théorie : voir [théorie §1.6 et §3.4](1_theorie_RNN_LSTM.md). Cette section a été retirée ici pour ne pas faire doublon.

<a id="18"></a>
## 1.8 Hyperparamètres, train / validation, sur-apprentissage

**Paramètres vs hyperparamètres :**

| | Paramètres | Hyperparamètres |
|---|---|---|
| Qui les choisit ? | Le réseau, pendant l'apprentissage | **Toi**, avant de lancer |
| Exemples | Les poids $W$, les biais $b$, l'embedding | `hidden_size`, `num_layers`, `max_epochs`, `lr`, `chunk_len`, température |
| Combien ? | Des centaines de milliers | Une dizaine |

**Train / validation :** on coupe le texte en deux morceaux.
- Le **train** (≈ 90 %) sert à apprendre.
- La **validation** (≈ 10 %) n'est **jamais** utilisée pour apprendre. Elle sert à vérifier que le réseau sait prédire du texte **qu'il n'a pas vu**.

**Analogie :** un élève qui révise avec les annales. Si l'examen reprend exactement les annales, un élève qui a tout appris par cœur aura 20/20 sans rien comprendre. Pour savoir s'il a compris, il faut l'interroger sur des exercices nouveaux : c'est la validation.

**Sur-apprentissage (*overfitting*) :** le réseau apprend le texte d'entraînement par cœur au lieu d'en apprendre les règles. Signe typique :

```
   loss
    │ ╲
    │  ╲___                     ← validation : descend puis REMONTE
    │   ╲  ╲____      ____/
    │    ╲      ╲____/
    │     ╲
    │      ╲___________         ← train : continue de descendre
    │                  ╲______
    └──────────────────────────► itérations
                  ▲
          meilleur modèle : là où la validation est au plus bas
```

⚠️ Le code fourni **n'a pas** de validation : il affiche seulement la loss d'entraînement. On l'ajoute en [5.8](#58).

---

<a id="2"></a>
# 2. Partie I du TP : récupérer et préparer les données

<a id="21"></a>
## 2.1 Un corpus, c'est quoi ?

Un **corpus** est la collection de textes sur laquelle le réseau apprend. C'est son **unique source de connaissance** : un réseau entraîné sur Shakespeare écrira du faux Shakespeare, un réseau entraîné sur Wikipédia écrira du faux Wikipédia, un réseau entraîné sur une liste de mots de passe générera de faux mots de passe (c'est la suite de l'atelier).

`shakespeare.txt` : 1 115 394 caractères (≈ 1,1 Mo), des pièces de théâtre au format :

```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.
```

<a id="22"></a>
## 2.2 Organiser le dossier

```
AtelierProjet2/
├── articleGeneration.py      ← le programme fourni
├── data/
│   └── shakespeare.txt       ← le corpus (téléchargé depuis la plateforme de cours)
├── models/                   ← créé automatiquement : les modèles entraînés (.pt)
└── docs/                     ← les cours et les PDF de l'énoncé
```

Sur la machine GPU (Linux) :

```bash
mkdir -p ~/AtelierProjet2/data
cd ~/AtelierProjet2/data
# copier shakespeare.txt ici, par exemple depuis ton PC :
#   scp shakespeare.txt login@machine-gpu:~/AtelierProjet2/data/
ls -lh           # vérifier la taille (doit être < 10 Mo)
head -20 shakespeare.txt
wc -c shakespeare.txt   # nombre d'octets (≈ nombre de caractères)
```

Sur ton PC Windows (PowerShell) :

```powershell
New-Item -ItemType Directory -Force data
Get-Item data\shakespeare.txt | Select-Object Name, Length
```

> **Ne t'étonne pas de l'écart :** `wc -c` ou `Length` affichent **1 155 394 octets**, alors que le programme annonce `file_len = 1115394` caractères. La différence (40 000) vient des 40 000 fins de ligne : le fichier utilise des fins de ligne Windows `\r\n` (2 octets), que Python, en mode texte, lit comme un seul caractère `\n`.

⚠️ Le programme se lance **depuis la racine** `AtelierProjet2` (le chemin par défaut est `data/shakespeare.txt`, relatif au dossier courant).

<a id="23"></a>
## 2.3 Pourquoi un fichier de moins de 10 Mo ?

1. **Tout le fichier est chargé d'un coup en mémoire** (`open(repData).read()`), puis passé dans `unidecode`. Avec un très gros fichier, c'est lent et ça consomme beaucoup de RAM sur une machine partagée.
2. **Le réseau ne verrait de toute façon pas tout.** Chaque itération lit un seul morceau de 13 caractères. Avec 10 000 itérations, le réseau voit $10\,000 \times 13 = 130\,000$ caractères, soit environ **12 %** de Shakespeare. Un corpus de 100 Mo serait vu à 0,1 % : grossir le fichier sans augmenter le nombre d'itérations ne sert à rien.
3. Le TP est fait pour que les entraînements durent quelques minutes, pour que tu puisses **comparer plusieurs configurations**.

<a id="24"></a>
## 2.4 Du texte aux nombres : unidecode, string.printable, index

Un réseau ne manipule que des nombres. Il faut donc transformer chaque caractère en nombre.

### Étape 1 : unidecode

```python
file = unidecode.unidecode(open(repData).read())
```

`unidecode` remplace chaque caractère non-ASCII par son équivalent ASCII le plus proche : `é → e`, `ç → c`, `œ → oe`, `« → <<`, `’ → '`.

**Pourquoi ?** Parce que le vocabulaire du programme ne contient que des caractères ASCII (voir étape 2). Sans ça, un `é` ferait planter le programme.

### Étape 2 : le vocabulaire `string.printable`

```python
all_characters = string.printable      # 100 caractères
n_characters = len(all_characters)     # 100
```

`string.printable` contient, **dans cet ordre** :

| Index | Caractères |
|---|---|
| 0 à 9 | `0123456789` |
| 10 à 35 | `abcdefghijklmnopqrstuvwxyz` |
| 36 à 61 | `ABCDEFGHIJKLMNOPQRSTUVWXYZ` |
| 62 à 93 | la ponctuation : ``!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`` |
| 94 à 99 | les espaces : espace, tabulation `\t`, retour ligne `\n`, `\r`, `\x0b`, `\x0c` |

C'est pour ça que $K = 100$ et que la loss de départ vaut $\ln 100$.

### Étape 3 : caractère → index

```python
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)
```

**Exemple :** `char_tensor("To be")` donne `tensor([55, 24, 94, 11, 14])`
(`T` = 36 + 19 = 55, `o` = 10 + 14 = 24, espace = 94, `b` = 11, `e` = 14).

⚠️ Si ton corpus contient un caractère qui n'est **pas** dans `string.printable` même après `unidecode` (un caractère de contrôle, par exemple), `.index()` lève une `ValueError`. Solution : filtrer le texte après l'avoir lu :

```python
file = ''.join(c for c in file if c in all_characters)
```

⚠️ **Sous Windows**, `open()` lit par défaut en `cp1252`. Pour un fichier UTF-8 avec des accents, ajoute l'encodage :

```python
file = unidecode.unidecode(open(repData, encoding='utf-8').read())
```

<a id="25"></a>
## 2.5 L'embedding

Un index (55 pour `T`) n'est pas une bonne description : ça ne veut pas dire que `T` est « plus grand » que `o` (24). On associe donc à chaque caractère un **vecteur** de taille $H$ (`hidden_size`) que le réseau **apprend**. C'est l'**embedding**.

$$
\mathbf{e}_t = E[i_t], \qquad E \in \mathbb{R}^{100 \times H}
$$

**En français :** $E$ est une table de 100 lignes (une par caractère) et $H$ colonnes. Pour le caractère d'index $i_t$, on va chercher sa ligne. Au départ les nombres sont aléatoires ; pendant l'apprentissage, les caractères qui jouent des rôles proches (les voyelles, par exemple) finissent avec des vecteurs proches.

**Exemple** avec $H = 4$ (en vrai $H = 128$ par défaut) :

| Index | Caractère | Vecteur d'embedding (appris) |
|---|---|---|
| 10 | `a` | (0,21 ; -0,80 ; 0,05 ; 1,10) |
| 14 | `e` | (0,25 ; -0,75 ; 0,12 ; 0,98) ← proche de `a` |
| 94 | espace | (-1,30 ; 0,40 ; 0,90 ; -0,20) |

Dans le code : `self.encoder = nn.Embedding(input_size, hidden_size)`, c'est-à-dire `nn.Embedding(100, 128)` : 12 800 paramètres.

> Équivalence utile : l'embedding revient à multiplier un vecteur **one-hot** (que des 0 et un seul 1 à la position de l'index) par la matrice $E$. C'est juste beaucoup plus rapide de lire directement la ligne.

---

<a id="3"></a>
# 3. Les réseaux récurrents (RNN)

<a id="31"></a>
## 3.1 Le RNN simple

> 📖 Théorie : voir [théorie §4.2 et §4.3](1_theorie_RNN_LSTM.md). Cette section a été retirée ici pour ne pas faire doublon.

<a id="32"></a>
## 3.2 Prédire le caractère suivant

> 📖 Théorie : voir [théorie §4.4](1_theorie_RNN_LSTM.md). Cette section a été retirée ici pour ne pas faire doublon.

<a id="33"></a>
## 3.3 Générer du texte et la température

Une fois entraîné, on génère ainsi :

1. On donne un début, par exemple `Wh`, pour « chauffer » la mémoire.
2. Le réseau donne des probabilités pour le caractère suivant.
3. On **tire au sort** un caractère selon ces probabilités (pas forcément le plus probable, sinon le texte tourne en boucle).
4. Ce caractère devient l'entrée suivante. On recommence.

```
 "W" ─► RNN ─► "h" ─► RNN ─► tirage: "a" ─► RNN ─► tirage: "t" ─► ...
```

La **température** $T$ règle l'audace du tirage :

$$
p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

**En français :** on divise tous les scores par $T$ avant la softmax. $T < 1$ accentue les écarts (le réseau devient prudent, il choisit presque toujours le favori). $T > 1$ aplatit les écarts (le réseau prend des risques, plus de fautes et d'originalité).

**Exemple** avec les scores $(2, 1, 0)$ :

| $T$ | Scores divisés | Probabilités (`e`, `a`, `z`) | Effet |
|---|---|---|---|
| 0,5 | (4 ; 2 ; 0) | (0,867 ; 0,117 ; 0,016) | Prudent, répétitif |
| 1 | (2 ; 1 ; 0) | (0,665 ; 0,245 ; 0,090) | Normal |
| 2 | (1 ; 0,5 ; 0) | (0,506 ; 0,307 ; 0,186) | Audacieux, plus de fautes |

Le programme utilise $T = 0{,}8$ (codé en dur dans `evaluating`).

---

<a id="4"></a>
# 4. Partie II du TP : l'apprentissage

<a id="41"></a>
## 4.1 Le code, fonction par fonction

### Vue d'ensemble

```
 main
  ├─ lit les arguments, lit le fichier, le passe dans unidecode
  ├─ crée le modèle RNN, l'optimiseur Adam, la loss CrossEntropy
  ├─ si --trainEval train :
  │     training(n_epochs, file)
  │        └─ répète n_epochs fois : train(*random_training_set(file))
  │               random_training_set ─► random_chunk ─► char_tensor
  │     torch.save(modèle)
  └─ si --trainEval eval :
        torch.load(modèle)
        evaluating ─► boucle : tu tapes un début ─► evaluate(...) génère la suite
```

### Les réglages en haut du fichier

```python
all_characters = string.printable   # vocabulaire de 100 caractères (section 2.4)
n_characters = len(all_characters)  # K = 100
chunk_len = 13                      # longueur des séquences d'entraînement
lr = 0.005                          # learning rate η (section 1.6)
```

Les variables `n_epochs`, `hidden_size`, `n_layers`, `print_every` écrites juste à côté sont **écrasées** plus loin par les arguments de la ligne de commande : ne les modifie pas, elles ne servent à rien. En revanche `chunk_len` et `lr` ne sont **pas** des arguments : pour les changer, il faut éditer le fichier.

Le bloc `if torch.cuda.is_available()` choisit le GPU s'il y en a un (`CUDA AVAILABLE`), sinon le CPU (`ONLY CPU AVAILABLE`). Chaque `.to(device)` envoie les données ou le modèle au bon endroit.

### `random_chunk(file)`

```python
def random_chunk(file):
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]
```

Tire une position au hasard dans le texte et renvoie **14** caractères (`chunk_len + 1`). Le « +1 » sert à créer le décalage entrée/cible (13 entrées, 13 cibles).

Cas limite : `randint` inclut la borne haute. Si le tirage tombe sur la toute dernière position possible, le morceau ne fait que 13 caractères (donc 12 entrées). Cela arrive environ une fois sur un million, sans conséquence.

**Exemple réel** (le programme en affiche un au démarrage) : `":\nHow now, sir"`.

### `char_tensor(string)`

Transforme une chaîne en tenseur d'index (section [2.4](#24)). `Variable(...)` est une vieille syntaxe PyTorch, sans effet aujourd'hui.

### `random_training_set(file)`

```python
def random_training_set(file):
    chunk = random_chunk(file)
    inp = char_tensor(chunk[:-1]).to(device)   # tout sauf le dernier
    target = char_tensor(chunk[1:]).to(device) # tout sauf le premier
    return inp, target
```

Fabrique **un** exemple d'entraînement : l'entrée et la cible décalée d'un cran (section [3.2](#32)).

### La classe `RNN`

```python
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        ...
        self.encoder = nn.Embedding(input_size, hidden_size)    # table E  (100 × H)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)   # la mémoire (n_layers couches)
        self.decoder = nn.Linear(hidden_size, output_size)      # W_o, b_o (H → 100 scores)

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))                 # index → vecteur e_t
        output, hidden = self.gru(input.view(1, 1, -1), hidden) # h_{t-1} → h_t
        output = self.decoder(output.view(1, -1))               # h_t → z_t (100 scores)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device))
```

| Ligne | Formule du cours |
|---|---|
| `self.encoder(...)` | $\mathbf{e}_t = E[i_t]$ ([2.5](#25)) |
| `self.gru(...)` | $\mathbf{h}_t = \text{GRU}(\mathbf{e}_t, \mathbf{h}_{t-1})$ (version améliorée du RNN, [5.2](#52)) |
| `self.decoder(...)` | $\mathbf{z}_t = W_o \mathbf{h}_t + \mathbf{b}_o$ ([3.1](#31)) |
| `init_hidden()` | $\mathbf{h}_0 = \mathbf{0}$ : mémoire vide au début de chaque séquence |

Remarques :
- Le nom `decoder` pour le modèle entier (dans le `main`) et `self.decoder` pour la dernière couche prête à confusion : ce sont deux choses différentes.
- La forme `view(1, 1, -1)` veut dire (longueur de séquence = 1, taille du lot = 1, taille du vecteur). Le réseau traite **un seul caractère à la fois**.
- `forward` ne contient **pas** la softmax : `nn.CrossEntropyLoss` l'applique elle-même.
- L'état caché a la forme `(n_layers, 1, hidden_size)` : une mémoire par couche.

**Combien de paramètres ?** Une couche GRU contient $3 \times (2H^2 + 2H)$ paramètres (3 blocs, voir [5.2](#52)).

| Config | Embedding $100H$ | GRU | Linear $100H + 100$ | **Total** | Taille du `.pt` |
|---|---|---|---|---|---|
| 2 couches, $H = 128$ | 12 800 | 198 144 | 12 900 | **≈ 224 000** | ≈ 0,9 Mo |
| 3 couches, $H = 500$ | 50 000 | 4 509 000 | 50 100 | **≈ 4,6 millions** | ≈ 18 Mo |

(Chaque paramètre prend 4 octets : $224\,000 \times 4 \approx 0{,}9$ Mo. Ça correspond aux fichiers de ton dossier `models/`.)

### `train(inp, target)` : une étape d'apprentissage

```python
def train(inp, target):
    hidden = decoder.init_hidden()          # h_0 = 0
    decoder.zero_grad()                     # efface les anciens gradients
    loss = 0
    for c in range(inp.size(0)):            # pour chacun des 13 caractères
        output, hidden = decoder(inp[c], hidden)                 # prédiction z_t
        loss += criterion(output, target[c].unsqueeze(0))        # + (-ln p_cible)
    loss.backward()                         # rétropropagation (1.7)
    decoder_optimizer.step()                # θ ← θ − η∇L (1.6)
    return loss.item() / chunk_len          # loss moyenne par caractère
```

| Ligne | Formule |
|---|---|
| `loss += criterion(...)` | $\sum_t -\ln p(c_{t+1})$ |
| `/ chunk_len` | la moyenne $\frac{1}{N}\sum$ |
| `loss.backward()` | calcul de $\nabla_\theta \mathcal{L}$ en remontant les 13 pas de temps |
| `step()` | $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$ |

`train` utilise `decoder`, `criterion` et `decoder_optimizer` sans les recevoir en paramètres : ce sont des **variables globales** créées dans le `main`. Ça marche parce que le `main` les crée avant d'appeler `training`. Si tu déplaces ce code, il faudra les passer en paramètres.

Une « epoch » dans ce programme = **une seule** séquence de 13 caractères (ce n'est pas le sens habituel du mot, qui désigne normalement un passage complet sur tout le corpus ; ici c'est plutôt une **itération**).

### `training(n_epochs, file)` : la boucle d'apprentissage

```python
print_every = n_epochs / 100
for epoch in range(1, n_epochs + 1):
    loss = train(*random_training_set(file))
    loss_avg += loss
    if epoch % print_every == 0:
        print(...)                           # affiche tous les 1 %
    if best_loss > (loss_avg / epoch):
        best_loss = loss_avg / epoch
        print(...)                           # affiche aussi quand la moyenne s'améliore
```

Elle appelle `train` `n_epochs` fois, et affiche une ligne tous les 1 % **et** à chaque fois que la moyenne s'améliore (d'où les lignes en double, voir [4.3](#43)). La variable `best_loss` sert seulement à afficher : le programme **ne garde pas** le meilleur modèle, il sauvegarde le **dernier**.

### `evaluate(decoder, prime_str, predict_len, temperature)` : la génération

```python
hidden = decoder.init_hidden()
for p in range(len(prime_str) - 1):              # 1) on "chauffe" la mémoire avec le début
    _, hidden = decoder(prime_input[p], hidden)
inp = prime_input[-1]
for p in range(predict_len):                     # 2) on génère predict_len caractères
    output, hidden = decoder(inp, hidden)
    output_dist = output.data.view(-1).div(temperature).exp()   # e^{z_i / T}
    top_i = torch.multinomial(output_dist, 1)[0]                # tirage au sort
    predicted_char = all_characters[top_i]
    predicted += predicted_char
    inp = char_tensor(predicted_char).to(device) # 3) le caractère tiré devient l'entrée
```

La ligne `.div(temperature).exp()` calcule $e^{z_i/T}$ : c'est la formule de la [température](#33) **sans** la division par la somme. Pas besoin : `torch.multinomial` normalise tout seul.

### `evaluating(decoder, length)`

Boucle infinie : elle te demande un début de texte, affiche `length` caractères générés, recommence. **`Ctrl + C`** pour quitter.

### Le `main`

1. Lit les arguments ([4.2](#42)).
2. Lit le fichier et le passe dans `unidecode`.
3. Crée le modèle `RNN(100, hidden_size, 100, num_layers)`, l'optimiseur Adam et la loss `nn.CrossEntropyLoss()` (= softmax + entropie croisée).
4. Construit le nom du fichier du modèle : `{run}_{num_layers}_{hidden_size}.pt`, par exemple `rnnGeneration_2_128.pt`.
5. `train` : entraîne puis sauvegarde dans `models/`. `eval` : charge et génère. (En mode `eval`, le modèle et l'optimiseur créés à l'étape 3 ne servent à rien : `torch.load` remplace le modèle par celui du fichier.)

<a id="42"></a>
## 4.2 Les arguments de la ligne de commande

| Argument | Court | Défaut | Rôle |
|---|---|---|---|
| `--trainingData` | `-d` | `data/shakespeare.txt` | Le corpus à lire |
| `--trainEval` | `-te` | `train` | `train` : apprendre et sauvegarder. `eval` : charger et générer |
| `--run` | `-r` | `rnnGeneration` | Début du nom du fichier modèle (pratique pour distinguer tes expériences) |
| `--model` | `-m` | `models` | Dossier où sauvegarder (train) ou charger (eval) le modèle |
| `--length` | | 100 | Nombre de caractères à générer en mode eval |
| `--num_layers` | | 2 | Nombre de couches GRU empilées |
| `--hidden_size` | | 128 | Taille $H$ de l'embedding et de la mémoire |
| `--max_epochs` | | 10000 | Nombre d'itérations d'entraînement |

⚠️ **Règle importante :** en mode `eval`, il faut redonner **les mêmes** `--run`, `--num_layers` et `--hidden_size` qu'à l'entraînement, sinon le programme cherche un fichier qui n'existe pas.

### Exemples de commandes

```bash
# Entraînement avec les valeurs par défaut (2 couches, H=128, 10 000 itérations)
python articleGeneration.py

# Un test rapide pour vérifier que tout marche (quelques secondes)
python articleGeneration.py --max_epochs 500 --run test

# Un réseau plus gros, plus longtemps, avec un nom d'expérience
python articleGeneration.py --num_layers 3 --hidden_size 256 --max_epochs 20000 --run exp_h256

# Génération avec le modèle précédent, 300 caractères
python articleGeneration.py -te eval --num_layers 3 --hidden_size 256 --run exp_h256 --length 300

# Un autre corpus
python articleGeneration.py -d data/wikipedia_fr.txt --run wiki
```

En mode eval, le programme affiche `Enter a starting two or tree charachters` : tape par exemple `Wh`, puis Entrée.

**Sur la machine GPU**, pour qu'un long entraînement continue si tu fermes le terminal, et garder une trace :

```bash
nohup python articleGeneration.py --hidden_size 512 --num_layers 3 --max_epochs 50000 --run big > log_big.txt 2>&1 &
tail -f log_big.txt     # suivre la progression (Ctrl+C pour arrêter de suivre, l'entraînement continue)
nvidia-smi              # vérifier que le GPU est utilisé
```

<a id="43"></a>
## 4.3 Lire la sortie d'entraînement

Début d'une vraie sortie :

```
ONLY CPU AVAILABLE
:
How now, sir
Training file_len = 1115394

[0m 0s (1 0%) 4.6021 (4.6021)]
[0m 0s (2 0%) 4.5851 (4.5681)]
[0m 0s (3 1%) 4.5395 (4.4482)]
[0m 0s (3 1%) 4.5395 (4.4482)]
...
```

| Ligne | Signification |
|---|---|
| `ONLY CPU AVAILABLE` | Pas de GPU : ça marche, mais plus lentement |
| `:` / `How now, sir` | Un morceau aléatoire du texte (test d'affichage) |
| `Training file_len = 1115394` | Nombre de caractères du corpus |

Décomposition d'une ligne, par exemple `[0m 1s (60 3%) 3.3660 (3.2612)]` :

```
 [ 0m 1s   ( 60   3% )   3.3660        ( 3.2612 ) ]
   │         │    │       │               │
   │         │    │       │               └─ loss de la DERNIÈRE séquence (13 caractères)
   │         │    │       └─ loss MOYENNE depuis le début de l'entraînement
   │         │    └─ avancement : 60 / max_epochs
   │         └─ numéro de l'itération
   └─ temps écoulé
```

**Comment l'interpréter :**
- La **loss moyenne** (premier nombre) est stable mais **en retard** : elle traîne les grosses erreurs du début (4,6). Elle sous-estime les progrès récents.
- La **loss instantanée** (entre parenthèses) est **bruitée** : elle ne porte que sur 13 caractères, un morceau facile ou difficile la fait sauter de ±0,5.
- Pour juger une configuration, regarde la **tendance** et la **valeur finale**, et idéalement une loss de validation ([5.8](#58)).
- $3{,}366$ signifie une perplexité de $e^{3{,}366} \approx 29$ : le réseau hésite encore entre environ 29 caractères. Il a déjà compris quelles lettres sont fréquentes, pas encore les mots.

**Pourquoi certaines lignes sont en double ?** Le programme affiche une ligne à chaque 1 % (`epoch % print_every == 0`) **et** à chaque fois que la moyenne bat son record (`best_loss > ...`). Quand les deux arrivent à la même itération, la ligne apparaît deux fois. Au début la moyenne baisse presque à chaque itération, donc ça affiche beaucoup.

**Ordres de grandeur** sur Shakespeare (à vérifier avec tes propres mesures) :

| Loss moyenne | Ce que le texte généré ressemble |
|---|---|
| ≈ 4,6 | Caractères au hasard : `x#Q]7t&m` |
| ≈ 3 | Des lettres fréquentes et des espaces : `e ot the e hae` |
| ≈ 2,2 | Des pseudo-mots, de la ponctuation, des noms en majuscules |
| ≈ 1,6 et moins | Des vrais mots, la structure des répliques (`NOM:` puis texte) |

<a id="44"></a>
## 4.4 Le bug du mode eval (PyTorch 2.6 et plus)

Avec PyTorch ≥ 2.6 (ton PC a la 2.14), cette commande plante :

```bash
python articleGeneration.py -te eval
```

```
_pickle.UnpicklingError: Weights only load failed ...
Unsupported global: GLOBAL __main__.RNN was not an allowed global by default.
```

**Pourquoi ?** Le programme sauvegarde **l'objet Python complet** (`torch.save(decoder, ...)`). Depuis la version 2.6, `torch.load` utilise par défaut `weights_only=True` : il refuse de recréer des objets Python arbitraires, car un fichier `.pt` piégé pourrait exécuter du code malveillant au chargement.

**Correction rapide** (acceptable ici, car c'est toi qui as créé le fichier) :

```python
decoder = torch.load(join(args.model, modelFile), weights_only=False)
```

**Correction propre (recommandée) :** sauvegarder uniquement les poids, le `state_dict` (un dictionnaire « nom de la couche → tenseur de poids »).

```python
# à l'entraînement, à la place de torch.save(decoder, ...)
torch.save(decoder.state_dict(), join(args.model, modelFile))

# en eval, à la place de decoder = torch.load(...)
# (le modèle a déjà été créé plus haut avec RNN(n_characters, args.hidden_size, ...))
decoder.load_state_dict(torch.load(join(args.model, modelFile), map_location=device))
```

Avantages du `state_dict` : pas d'avertissement de sécurité, fichier plus robuste si tu renommes ou déplaces des fichiers, et `map_location=device` permet de charger sur CPU un modèle entraîné sur GPU. Seule contrainte : il faut recréer **exactement la même architecture** avant de charger (mêmes `hidden_size`, `num_layers`, même type de cellule).

⚠️ Les anciens `.pt` (objet complet) ne se chargent pas avec `load_state_dict`, et inversement. Après avoir changé de méthode, **réentraîne** tes modèles.

<a id="45"></a>
## 4.5 Plan d'expériences

**Règle d'or : ne change qu'un seul hyperparamètre à la fois**, sinon tu ne sais pas lequel a causé la différence. Pour que la comparaison soit juste, fixe aussi le hasard en ajoutant au début du `main` :

```python
random.seed(0)
torch.manual_seed(0)
```

### Série A : taille de la couche cachée (2 couches, 10 000 itérations)

```bash
for H in 32 64 128 256 512; do
  python articleGeneration.py --hidden_size $H --num_layers 2 --max_epochs 10000 --run expA > logA_$H.txt
done
```

(Sous PowerShell : `foreach ($H in 32,64,128,256,512) { python articleGeneration.py --hidden_size $H --num_layers 2 --max_epochs 10000 --run expA > logA_$H.txt }`)

### Série B : nombre de couches ($H$ = 128, 10 000 itérations)

`--num_layers 1`, `2`, `3`, `4`.

### Série C : nombre d'itérations (2 couches, $H$ = 128)

`--max_epochs 2000`, `10000`, `30000`, `100000`.

### Tableau à remplir

| # | `num_layers` | `hidden_size` | `max_epochs` | Nb paramètres | Durée | Loss moy. finale | Loss val. (si 5.8) | Qualité du texte (1 à 5) | Exemple généré depuis `Wh` |
|---|---|---|---|---|---|---|---|---|---|
| A1 | 2 | 32 | 10 000 | | | | | | |
| A2 | 2 | 64 | 10 000 | | | | | | |
| A3 | 2 | 128 | 10 000 | ≈ 224 000 | | | | | |
| A4 | 2 | 256 | 10 000 | | | | | | |
| A5 | 2 | 512 | 10 000 | | | | | | |
| B1 | 1 | 128 | 10 000 | | | | | | |
| B3 | 3 | 128 | 10 000 | | | | | | |
| B4 | 4 | 128 | 10 000 | | | | | | |
| C1 | 2 | 128 | 2 000 | | | | | | |
| C3 | 2 | 128 | 30 000 | | | | | | |
| C4 | 2 | 128 | 100 000 | | | | | | |

Pour récupérer la dernière ligne de chaque log : `tail -n 1 logA_*.txt` (Linux) ou `Get-Content logA_128.txt -Tail 1` (PowerShell).

### Ce qu'on s'attend à observer

| Série | Attendu | Pourquoi |
|---|---|---|
| A : `hidden_size` ↑ | La loss baisse de 32 à 256 environ, puis le gain diminue. Chaque itération est plus lente. Sur CPU, 512 peut être **pire** à nombre d'itérations égal. | Plus de mémoire = plus de « pouvoir d'abstraction », mais plus de poids à régler, donc il faut plus d'itérations pour en profiter. |
| B : `num_layers` ↑ | 1 → 2 aide souvent ; 3 et 4 apportent peu, voire dégradent, avec 10 000 itérations. | Plus profond = plus dur à entraîner (le gradient traverse plus de couches) ; sans dropout, risque de mémorisation. |
| C : `max_epochs` ↑ | La loss baisse nettement, puis de plus en plus lentement (courbe en « coude »). Le texte s'améliore visiblement. | 10 000 itérations = seulement 12 % du corpus vu. Avec 100 000 on voit tout le corpus environ une fois. |

**À noter dans ton rapport :** un gros modèle entraîné trop peu peut faire moins bien qu'un petit ; la taille du modèle et le nombre d'itérations vont ensemble. Et sans validation, tu ne peux pas savoir si une loss d'entraînement très basse vient d'un vrai apprentissage ou d'une mémorisation.

---

<a id="5"></a>
# 5. Partie III du TP : améliorer l'IA

<a id="51"></a>
## 5.1 Gradient qui disparaît ou qui explose

> 📖 Théorie : voir [théorie §5](1_theorie_RNN_LSTM.md). Cette section a été retirée ici pour ne pas faire doublon.

<a id="52"></a>
## 5.2 GRU et LSTM

> 📖 Théorie : voir [théorie §6, §7 et §8 (équations, exemples, comparatif, nombre de paramètres)](1_theorie_RNN_LSTM.md). Cette section a été retirée ici pour ne pas faire doublon.

<a id="53"></a>
## 5.3 Remplacer le GRU par un LSTM

La difficulté : `nn.LSTM` attend et renvoie un état caché qui est un **couple** `(h, c)`, alors que `nn.GRU` utilise un seul tenseur `h`. Il faut donc modifier `init_hidden`.

**Avant (GRU) :**

```python
self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
...
output, hidden = self.gru(input.view(1, 1, -1), hidden)
...
def init_hidden(self):
    return Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device))
```

**Après (LSTM) :**

```python
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)     # ← GRU remplacé
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(input.view(1, 1, -1), hidden)    # hidden = (h, c)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        h = torch.zeros(self.n_layers, 1, self.hidden_size, device=device)
        c = torch.zeros(self.n_layers, 1, self.hidden_size, device=device)
        return (h, c)                                               # ← un couple
```

Rien d'autre à changer : `train` et `evaluate` se contentent de faire circuler `hidden` sans regarder dedans, donc ça marche que ce soit un tenseur ou un couple.

**Version avec un choix en ligne de commande** (pratique pour comparer) :

```python
# dans le main, avec les autres arguments
parser.add_argument('--cell', default='gru', choices=['gru', 'lstm', 'rnn'])
...
decoder = RNN(n_characters, args.hidden_size, n_characters, args.num_layers, cell=args.cell).to(device)
modelFile = args.run + "_" + args.cell + "_" + str(args.num_layers) + "_" + str(args.hidden_size) + ".pt"
```

```python
# dans la classe RNN
def __init__(self, input_size, hidden_size, output_size, n_layers=1, cell='gru'):
    ...
    self.cell = cell
    rnn_class = {'gru': nn.GRU, 'lstm': nn.LSTM, 'rnn': nn.RNN}[cell]
    self.rnn = rnn_class(hidden_size, hidden_size, n_layers)

def init_hidden(self):
    h = torch.zeros(self.n_layers, 1, self.hidden_size, device=device)
    if self.cell == 'lstm':
        return (h, torch.zeros_like(h))
    return h
```

Ajouter `nn.RNN` (le RNN simple) dans la comparaison est intéressant : il devrait être le moins bon des trois, ce qui illustre la section 5.1.

**Les pièges :**
- utilise un `--run` (ou un `--cell`) différent pour les modèles LSTM, sinon tu écrases tes modèles GRU ;
- avec la sauvegarde « objet complet » (`torch.save(decoder, ...)`), un ancien `.pt` GRU ne se recharge plus une fois la classe modifiée : raison de plus pour passer au `state_dict` ([4.4](#44)) ;
- en mode `eval`, redonne les **mêmes** `--num_layers`, `--hidden_size` (et `--cell`) qu'à l'entraînement ;
- pour comparer honnêtement, il faut une **loss de validation** et le **meilleur** modèle sauvegardé ([5.8](#58)), avec la même graine (`random.seed(0)`, `torch.manual_seed(0)`).

<a id="54"></a>
## 5.4 Ajouter du dropout

Le **dropout** met à zéro, **pendant l'entraînement seulement**, une proportion $p$ des neurones tirés au hasard, et multiplie les autres par $\frac{1}{1-p}$ pour garder la même moyenne :

$$
\tilde{x}_i = \frac{m_i}{1 - p}\, x_i, \qquad m_i \sim \text{Bernoulli}(1 - p)
$$

**En français :** chaque neurone a une probabilité $p$ d'être éteint à cette itération. Le réseau ne peut plus compter sur un seul neurone « par cœur » : il doit répartir l'information. C'est un remède contre le sur-apprentissage.

**Exemple :** $p = 0{,}5$, $\mathbf{x} = (2, 4, 6, 8)$, masque tiré $\mathbf{m} = (1, 0, 1, 0)$ :
$\tilde{\mathbf{x}} = (2 \times 2,\ 0,\ 6 \times 2,\ 0) = (4, 0, 12, 0)$.

**Analogie :** une équipe de foot qui s'entraîne en faisant sortir des joueurs au hasard : chacun doit savoir jouer à plusieurs postes.

Dans le code, deux endroits :

```python
# 1) entre les couches du LSTM/GRU (ne fait rien si n_layers == 1)
self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=0.2 if n_layers > 1 else 0)

# 2) avant la couche de sortie
self.drop = nn.Dropout(0.2)
...
output = self.drop(output.view(1, -1))
output = self.decoder(output)
```

⚠️ Le dropout doit être **désactivé** pendant la génération et la validation : c'est le rôle de `decoder.eval()` (déjà appelé dans le `main` en mode eval). Pendant l'entraînement, `decoder.train()` le réactive.

Valeurs usuelles : $p$ entre 0,1 et 0,5. Utile surtout pour les gros modèles ($H \ge 256$, plusieurs couches) et les longs entraînements.

<a id="55"></a>
## 5.5 Ajouter une LayerNorm

La **normalisation de couche** recentre et remet à l'échelle un vecteur pour que ses valeurs aient une moyenne 0 et un écart-type 1, puis applique une échelle $\gamma$ et un décalage $\beta$ appris :

$$
\text{LN}(\mathbf{x})_i = \gamma_i \, \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta_i, \qquad \mu = \frac{1}{H}\sum_{j} x_j, \quad \sigma^2 = \frac{1}{H}\sum_j (x_j - \mu)^2
$$

**En français :** on retire la moyenne du vecteur et on divise par son écart-type, pour que les valeurs restent dans une zone « raisonnable ». Ça stabilise et accélère souvent l'apprentissage. $\varepsilon$ est un tout petit nombre (10⁻⁵) pour ne jamais diviser par zéro.

**Exemple :** $\mathbf{x} = (1, 2, 3, 6)$, avec $\gamma = 1$, $\beta = 0$.
$\mu = 3$, $\sigma^2 = \frac{4 + 1 + 0 + 9}{4} = 3{,}5$, $\sigma \approx 1{,}871$.
$\text{LN}(\mathbf{x}) \approx (-1{,}07 ;\ -0{,}53 ;\ 0 ;\ 1{,}60)$.

Dans le code, entre la sortie du LSTM et la couche linéaire :

```python
self.norm = nn.LayerNorm(hidden_size)
...
output, hidden = self.rnn(input.view(1, 1, -1), hidden)
output = self.norm(output.view(1, -1))
output = self.decoder(output)
```

> Pourquoi pas `BatchNorm` ? Elle calcule des moyennes sur un **lot** d'exemples. Ici le lot fait 1 exemple : BatchNorm ne marche pas. LayerNorm calcule sur le vecteur lui-même, elle fonctionne quelle que soit la taille du lot.

### La classe complète avec LSTM + dropout + LayerNorm (testée)

```python
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout=0.2):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers,
                           dropout=dropout if n_layers > 1 else 0)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(input.view(1, 1, -1), hidden)
        output = self.drop(self.norm(output.view(1, -1)))
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self):
        h = torch.zeros(self.n_layers, 1, self.hidden_size, device=device)
        c = torch.zeros(self.n_layers, 1, self.hidden_size, device=device)
        return (h, c)
```

Pour ajouter **une couche cachée linéaire** supplémentaire (piste de l'énoncé), on peut remplacer `self.decoder` par un petit MLP :

```python
self.decoder = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size),
)
```

<a id="56"></a>
## 5.6 Le gradient clipping

Contre le gradient qui explose ([5.1](#51)) : si le gradient est trop grand, on le **raccourcit** en gardant sa direction.

$$
\text{si } \|\mathbf{g}\| > c : \quad \mathbf{g} \leftarrow \mathbf{g} \cdot \frac{c}{\|\mathbf{g}\|}
$$

**En français :** on mesure la longueur totale du gradient. Si elle dépasse le seuil $c$, on le ramène à la longueur $c$, sans changer la direction. On descend toujours dans le bon sens, mais jamais d'un pas démesuré.

**Exemple :** $\mathbf{g} = (30, 40)$, donc $\|\mathbf{g}\| = \sqrt{30^2 + 40^2} = 50$. Seuil $c = 5$ : $\mathbf{g} \leftarrow (30, 40) \times \frac{5}{50} = (3, 4)$, de longueur 5.

Dans `train`, **entre** `backward()` et `step()` :

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
decoder_optimizer.step()
```

Valeurs usuelles : entre 1 et 5. Coût quasi nul, bénéfice surtout visible avec un learning rate élevé, un RNN simple, ou des séquences plus longues (`chunk_len` ↑).

<a id="57"></a>
## 5.7 Un learning rate variable

Au début on veut de grands pas (on est loin de la vallée), à la fin de petits pas (pour se poser au fond sans rebondir). Le plus simple : diviser le learning rate régulièrement.

$$
\eta_k = \eta_0 \cdot \gamma^{\lfloor k / s \rfloor}
$$

**En français :** toutes les $s$ itérations, on multiplie le learning rate par $\gamma$ (par exemple 0,5 : on le divise par 2). $\lfloor \cdot \rfloor$ veut dire « partie entière ».

**Exemple :** $\eta_0 = 0{,}005$, $\gamma = 0{,}5$, $s = 2\,000$. À l'itération $k = 5\,000$ : $\lfloor 5000 / 2000 \rfloor = 2$, donc $\eta = 0{,}005 \times 0{,}5^2 = 0{,}00125$.

Dans le `main`, après l'optimiseur :

```python
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=2000, gamma=0.5)
```

Dans `train`, après `decoder_optimizer.step()` :

```python
scheduler.step()
```

Adapte `step_size` à `max_epochs` : avec 10 000 itérations et `step_size = 2000`, le learning rate finit à $0{,}005 / 32$. Pour afficher sa valeur actuelle : `decoder_optimizer.param_groups[0]['lr']`.

Variante plus intelligente, si tu as une loss de validation : `ReduceLROnPlateau` divise le learning rate seulement quand la validation arrête de progresser (`scheduler.step(val_loss)`).

<a id="58"></a>
## 5.8 Mesurer honnêtement : validation et meilleur modèle

Sans ça, tu ne peux pas affirmer qu'une modification est une amélioration : la loss d'entraînement affichée est bruitée, moyennée depuis le début, et ne détecte pas le sur-apprentissage.

### Étape 1 : couper le corpus (dans le `main`, après la lecture)

```python
file = unidecode.unidecode(open(repData, encoding='utf-8').read())
file = ''.join(c for c in file if c in all_characters)
split = int(0.9 * len(file))
train_text = file[:split]      # 90 % pour apprendre
val_text = file[split:]        # 10 % jamais vus pendant l'apprentissage
file_len = len(train_text)     # random_chunk utilise file_len : il doit correspondre au train
```

On coupe en **deux blocs** (et pas en tirant des caractères au hasard) pour que la validation contienne des passages vraiment nouveaux.

### Étape 2 : une fonction de loss de validation

```python
def validation_loss(text, n_chunks=200):
    decoder.eval()                        # coupe le dropout
    total = 0
    with torch.no_grad():                 # pas de gradient : plus rapide, rien n'est appris
        for _ in range(n_chunks):
            start = random.randint(0, len(text) - chunk_len - 1)
            chunk = text[start:start + chunk_len + 1]
            inp = char_tensor(chunk[:-1]).to(device)
            target = char_tensor(chunk[1:]).to(device)
            hidden = decoder.init_hidden()
            loss = 0
            for c in range(inp.size(0)):
                output, hidden = decoder(inp[c], hidden)
                loss += criterion(output, target[c].unsqueeze(0))
            total += loss.item() / chunk_len
    decoder.train()                       # réactive le dropout
    return total / n_chunks
```

On moyenne sur 200 morceaux pour que la mesure soit **stable** (contrairement à la loss instantanée sur 13 caractères). Pour une comparaison encore plus juste entre modèles, fixe la graine juste avant (`random.seed(123)`) pour qu'ils soient tous évalués sur les **mêmes** morceaux.

### Étape 3 : garder le meilleur modèle (dans `training`)

```python
best_val = float('inf')
for epoch in range(1, n_epochs + 1):
    loss = train(*random_training_set(train_text))
    loss_avg += loss
    if epoch % print_every == 0:
        val = validation_loss(val_text)
        print('[%s (%d %d%%) train %.4f | val %.4f]' % (time_since(start), epoch,
              epoch / n_epochs * 100, loss_avg / epoch, val))
        if val < best_val:
            best_val = val
            torch.save(decoder.state_dict(), join(args.model, modelFile))   # meilleur modèle
```

Et **supprime** le `torch.save` qui suit `training(...)` dans le `main`, sinon le dernier modèle écrase le meilleur.

Le principe (vérifié sur ton PC, 300 itérations, LSTM 2 couches $H = 128$) donne une sortie de ce type :

```
itération   loss train   loss val   learning rate
   100        3.183        2.892      0.0025
   200        2.583        2.692      0.00125
   300        2.384        2.612      0.000625
```

### Étape 4 : tracer les courbes (pour ton rapport)

```python
train_hist, val_hist = [], []
# dans le if epoch % print_every == 0 : train_hist.append(loss_avg / epoch); val_hist.append(val)
# après la boucle :
plt.plot(train_hist, label='train')
plt.plot(val_hist, label='validation')
plt.xlabel('itérations (× print_every)'); plt.ylabel('loss'); plt.legend()
plt.savefig('courbe_' + modelFile.replace('.pt', '.png'))
```

(`matplotlib` est déjà importé en haut du fichier.)

### Comment conclure qu'une amélioration est réelle

1. Même corpus, même découpage, même nombre d'itérations, même graine.
2. On compare la **loss de validation** du **meilleur** modèle de chaque configuration.
3. Une différence de 0,01 est du bruit ; plusieurs centièmes répétés sur 2 ou 3 graines différentes, c'est une vraie amélioration.
4. On regarde **aussi** le texte généré, avec le même début (`Wh`, `ROMEO:`) : la loss ne dit pas tout.

Tableau conseillé pour la partie III :

| Modèle | Paramètres | Loss val. meilleure | Itération du meilleur | Durée | Exemple généré |
|---|---|---|---|---|---|
| GRU (référence) 2 × 128 | ≈ 224 000 | | | | |
| RNN simple 2 × 128 | | | | | |
| LSTM 2 × 128 | | | | | |
| LSTM + dropout 0,2 | | | | | |
| LSTM + dropout + LayerNorm | | | | | |
| … + clipping + StepLR | | | | | |

<a id="59"></a>
## 5.9 Augmenter ou diversifier le corpus

| Idée | Comment | Attention à |
|---|---|---|
| **Plus de texte du même style** | Autres œuvres de Shakespeare, autres dramaturges anglais (Projet Gutenberg, domaine public) | Rester < 10 Mo ; enlever les en-têtes et licences Gutenberg (sinon le réseau apprend à écrire des licences) |
| **Autre style** | Articles Wikipédia (dump ou copier-coller d'articles), romans, recettes, code source | Un corpus mélangé donne un texte mélangé : c'est voulu ou pas ? |
| **Français** | Hugo, Molière, Zola (Gutenberg) | `unidecode` enlève les accents : le texte généré sera sans accents. Sinon, remplacer `string.printable` par la liste des caractères du corpus : `all_characters = ''.join(sorted(set(file)))` (et alors $K$ change, et la loss de départ vaut $\ln K$) |
| **Concaténer plusieurs fichiers** | `cat data/*.txt > data/corpus.txt` (Linux) ou `Get-Content data\*.txt \| Set-Content data\corpus.txt -Encoding utf8` | Vérifier la taille finale |
| **Nettoyer** | Supprimer les lignes vides en série, les numéros de page, les caractères bizarres | Garder la structure utile (retours à la ligne des répliques) |

**Important :** plus de données n'aide que si le réseau les **voit**. Avec un corpus 5 fois plus grand, augmente aussi `--max_epochs`. Et un corpus plus varié augmente souvent un peu la loss (le texte est moins prévisible) tout en améliorant la **généralisation** : ne compare les loss que sur un **même** jeu de validation.

**Autres pistes** à citer si tu veux aller plus loin : augmenter `chunk_len` (13 caractères, c'est une mémoire très courte : essayer 50 à 200) ; entraîner par **lots** (*batch*) de 32 à 64 séquences à la fois pour aller beaucoup plus vite sur GPU ; donner une séquence entière à `nn.LSTM` au lieu d'une boucle caractère par caractère.

---

<a id="6"></a>
# 6. Fiche récapitulative

| Notion | En une phrase | Formule / code |
|---|---|---|
| Vecteur, matrice | Liste / tableau de nombres ; une matrice transforme un vecteur | $W\mathbf{x}$ |
| Neurone | Somme pondérée + biais, puis activation | $y = f(\mathbf{w}\cdot\mathbf{x} + b)$ |
| Activation | Ajoute de la non-linéarité | ReLU, $\sigma$, tanh |
| Softmax | Scores → probabilités qui somment à 1 | $p_i = e^{z_i} / \sum_j e^{z_j}$ |
| Entropie croisée | Pénalise une faible probabilité sur la bonne réponse | $\mathcal{L} = -\ln p_{\text{cible}}$ |
| Loss de départ | Hasard parmi 100 caractères | $\ln 100 \approx 4{,}6$ |
| Perplexité | Nombre de caractères entre lesquels on hésite | $e^{\mathcal{L}}$ |
| Gradient | Pente de la loss par rapport à chaque poids | $\nabla_\theta \mathcal{L}$ |
| Descente de gradient | Petit pas dans le sens qui descend | $\theta \leftarrow \theta - \eta\nabla_\theta\mathcal{L}$ |
| Rétropropagation | Calcul des gradients par la règle de la chaîne | `loss.backward()` |
| Embedding | Table apprise : caractère → vecteur | `nn.Embedding(100, H)` |
| RNN | Réseau avec une mémoire mise à jour à chaque caractère | $\mathbf{h}_t = \tanh(W_x\mathbf{e}_t + W_h\mathbf{h}_{t-1} + \mathbf{b})$ |
| Vanishing / exploding | Produit de nombreux facteurs → 0 ou → ∞ | $0{,}5^{20} \approx 10^{-6}$, $1{,}5^{20} \approx 3\,300$ |
| GRU | 2 portes, une mémoire | `nn.GRU` |
| LSTM | 3 portes, deux mémoires $(h, c)$ | `nn.LSTM`, `init_hidden` renvoie `(h, c)` |
| Température | Règle l'audace de la génération | $e^{z_i/T}$ ; $T = 0{,}8$ |
| Hyperparamètre | Réglage choisi par toi | `hidden_size`, `num_layers`, `max_epochs`, `lr` |
| Train / validation | Apprendre sur l'un, juger sur l'autre | 90 % / 10 % |
| Sur-apprentissage | Par cœur : train ↓, validation ↑ | Surveiller la loss de validation |
| Dropout | Éteindre des neurones au hasard à l'entraînement | `nn.Dropout(0.2)`, `dropout=` dans LSTM |
| LayerNorm | Recentrer / réduire un vecteur | `nn.LayerNorm(H)` |
| Gradient clipping | Limiter la longueur du gradient | `clip_grad_norm_(..., 5.0)` |
| LR variable | Grands pas puis petits pas | `StepLR`, `ReduceLROnPlateau` |
| `state_dict` | Sauvegarder seulement les poids | `torch.save(m.state_dict(), f)` / `m.load_state_dict(torch.load(f))` |

**Commandes à retenir :**

```bash
python articleGeneration.py --num_layers 2 --hidden_size 128 --max_epochs 10000 --run exp
python articleGeneration.py -te eval --num_layers 2 --hidden_size 128 --run exp --length 300
```

---

<a id="7"></a>
# 7. Checklist de fin de TP

L'énoncé ne donne pas de liste de rendu précise : voici ce qu'il est raisonnable de pouvoir **montrer**. Vérifie avec tes enseignants le format attendu (rapport, démo, code).

### Partie I : données
- [ ] Le dossier `AtelierProjet2/data/` existe et contient `shakespeare.txt` (taille vérifiée < 10 Mo).
- [ ] Tu sais expliquer comment un caractère devient un nombre (unidecode → index dans `string.printable` → embedding).
- [ ] Bonus : un deuxième corpus (autre style ou langue), nettoyé, < 10 Mo.

### Partie II : apprentissage
- [ ] Au moins un entraînement complet avec les valeurs par défaut, et sa sortie (log).
- [ ] Le bug `weights_only` corrigé, et une génération réussie en mode `eval`.
- [ ] Le tableau d'expériences rempli (séries A, B, C), avec la loss finale, la durée et un exemple de texte généré pour chacune.
- [ ] Une analyse : effet de `hidden_size`, de `num_layers`, de `max_epochs`. Explication de la loss de départ ≈ 4,6 = ln 100.
- [ ] Bonus : essai de plusieurs températures sur un même modèle.

### Partie III : amélioration
- [ ] Le GRU remplacé par un LSTM (`init_hidden` qui renvoie `(h, c)`), le code montré.
- [ ] Au moins une amélioration supplémentaire : dropout, LayerNorm, gradient clipping, learning rate variable, couche linéaire en plus.
- [ ] Un découpage train / validation, la loss de validation affichée, et le **meilleur** modèle sauvegardé (`state_dict`).
- [ ] Un tableau comparatif GRU / LSTM / variantes sur la **loss de validation**, avec les mêmes réglages.
- [ ] Les courbes de loss (train et validation) enregistrées en image.
- [ ] Des exemples de textes générés **avant / après** amélioration, avec le même début.

### À savoir expliquer à l'oral
- [ ] Ce que fait chaque fonction du programme.
- [ ] Pourquoi un RNN simple oublie vite (gradient qui disparaît) et comment le GRU/LSTM y remédie.
- [ ] La différence entre loss d'entraînement et loss de validation, et ce qu'est le sur-apprentissage.
- [ ] Le rôle de la température pendant la génération.
- [ ] Le lien avec la suite de l'atelier : le même programme, entraîné sur une liste de mots de passe, génère des mots de passe plausibles.
