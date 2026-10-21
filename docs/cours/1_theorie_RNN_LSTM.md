# Du neurone au LSTM : le cours complet de la première partie

> Atelier Projet 2, Master ILSEN. Sujet de M. Morchid et M. Haddad.
>
> **Pour voir les formules**, ouvre ce fichier dans **VS Code** (`Ctrl + Shift + V`) ou sur GitHub.
>
> **À lire en premier.** Ce cours contient **toute la théorie** : les outils mathématiques, le neurone, l'apprentissage, le RNN, son grand défaut (le gradient qui disparaît ou explose) et les deux solutions, GRU et LSTM. Il couvre les slides 5 à 38 du sujet et se termine par les questions d'oral (§11), utiles pour le QCM.
>
> **La pratique** (données, code, commandes, expériences, améliorations) est dans [`2_guide_TP.md`](2_guide_TP.md).

Chaque notion est suivie :
- d'une phrase **« En français »** ;
- d'un **exemple avec de vrais nombres** (tous vérifiés en Python) ;
- quand c'est utile, de **« Dans le code »** : où la notion apparaît dans `articleGeneration.py`.

---

## Sommaire

- [0. La carte du cours](#0)
- [1. Les outils mathématiques](#1)
  - [1.1 Vecteur et matrice](#11)
  - [1.2 Le produit scalaire](#12)
  - [1.3 Matrice × vecteur : deux façons de calculer](#13)
  - [1.4 La règle des tailles](#14)
  - [1.5 Exponentielle et logarithme (ln)](#15)
  - [1.6 Dérivée et règle de la chaîne](#16)
- [2. Le neurone et les fonctions d'activation](#2)
  - [2.1 Le neurone](#21)
  - [2.2 Pourquoi une fonction d'activation ?](#22)
  - [2.3 Les trois fonctions à connaître](#23)
  - [2.4 Pourquoi ces intervalles ? Tout dépend du rôle](#24)
  - [2.5 La pente des activations (elle comptera pour le vanishing)](#25)
  - [2.6 Où sont les activations dans le code ?](#26)
- [3. Comment un réseau apprend](#3)
  - [3.1 Softmax : des scores aux probabilités](#31)
  - [3.2 La loss : l'entropie croisée](#32)
  - [3.3 La descente de gradient](#33)
  - [3.4 Rétropropagation et Adam : deux étapes différentes](#34)
- [4. Le réseau récurrent (RNN)](#4)
  - [4.1 Le problème du MLP : une fenêtre fixe](#41)
  - [4.2 Le RNN : un réseau avec une mémoire](#42)
  - [4.3 Plusieurs couches](#43)
  - [4.4 Apprendre à prédire le caractère suivant](#44)
- [5. Pourquoi le RNN simple oublie : la BPTT](#5)
  - [5.1 W est utilisée à chaque pas](#51)
  - [5.2 Les deux sommes](#52)
  - [5.3 Le terme qui pose problème : un produit](#53)
  - [5.4 Exemple chiffré sur 13 caractères](#54)
  - [5.5 Avec des matrices : les valeurs propres](#55)
  - [5.6 Vanishing, exploding et leurs remèdes](#56)
- [6. Le GRU](#6)
- [7. Le LSTM](#7)
- [8. RNN, GRU, LSTM : le comparatif](#8)
- [9. Dans le code : passer du GRU au LSTM](#9)
- [10. Fiche récapitulative](#10)
- [11. Les questions d'oral et leurs réponses](#oral)

---

<a id="0"></a>
# 0. La carte du cours

Chaque étape corrige un défaut de la précédente :

```
 Neurone          une somme pondérée + une activation
    │
    ▼
 Couche / MLP     beaucoup de neurones d'un coup = une matrice
    │             défaut : il ne voit qu'une fenêtre fixe (3 caractères)
    ▼
 RNN              ajoute une MÉMOIRE h qui circule d'un caractère à l'autre
    │             défaut : il oublie vite (gradient qui disparaît) ou devient instable (gradient qui explose)
    ▼
 GRU / LSTM       ajoutent des PORTES qui décident quoi garder : la mémoire traverse le temps
```

| Étape | Slides du sujet | Mise en pratique ([guide TP](2_guide_TP.md)) | Dans le code |
|---|---|---|---|
| Neurone, MLP | 5, 6 | §2.5 (embedding) | `nn.Linear`, `nn.Embedding` |
| RNN | 7, 38 | §3.3 (génération), §4 (le code) | `nn.RNN` (pas utilisé dans le code fourni) |
| BPTT, vanishing, exploding | 8 à 32 | §5.6 (gradient clipping) | `loss.backward()` |
| GRU, LSTM | 33 | §5.3 (passer au LSTM) | `nn.GRU` (code fourni), `nn.LSTM` (partie III) |

---

<a id="1"></a>
# 1. Les outils mathématiques

**Pourquoi des maths ici ?** Un réseau de neurones ne comprend **que des nombres**. Tout ce qu'il **manipule** (un caractère, sa mémoire, ses prédictions) est une **liste de nombres** : un **vecteur**. Tout ce qu'il **apprend** est rangé dans des **tableaux de nombres** : des **matrices**. Les 223 844 paramètres du modèle par défaut ne sont rien d'autre que les cases de quelques matrices.

Le travail d'un réseau se résume à : **prendre un vecteur, le multiplier par une matrice, appliquer une petite fonction, obtenir un nouveau vecteur**, et recommencer. **Apprendre**, c'est modifier les nombres des matrices jusqu'à ce que le vecteur final soit le bon.

<a id="11"></a>
## 1.1 Vecteur et matrice

Un **vecteur** est une liste de nombres où **chaque position a un sens fixe**.

$$
\mathbf{x} = \begin{pmatrix} 2 \\ 1 \end{pmatrix}
$$

**Exemple concret :** on peut décrire un mot de passe par (longueur ; nombre de chiffres ; nombre de majuscules). `Marie2004` devient $(9 ; 4 ; 1)$ et `azerty` devient $(6 ; 0 ; 0)$.

Une **matrice** est un tableau de nombres avec $m$ lignes et $n$ colonnes (« taille $m \times n$ ») :

$$
W = \begin{pmatrix} 1 & 2 \\ 0 & -1 \end{pmatrix} \qquad (2 \times 2)
$$

> **« Une matrice, c'est un vecteur de vecteurs ? »** Oui : c'est une **pile de vecteurs tous de la même taille**. En Python, c'est littéralement une liste de listes : `[[1, 2], [0, -1]]`. On peut la lire comme une pile de **lignes** ou comme une pile de **colonnes**, et les deux lectures servent (section 1.3).

> En PyTorch, un vecteur, une matrice ou un tableau à 3 dimensions s'appelle un **tenseur** (`torch.Tensor`). Par exemple, la mémoire du GRU a la forme `(n_layers, 1, hidden_size)`.

<a id="12"></a>
## 1.2 Le produit scalaire

On prend **deux vecteurs de même taille**, on multiplie **case par case**, puis on **additionne** tout :

$$
\begin{pmatrix} 1 \\ 2 \end{pmatrix} \cdot \begin{pmatrix} 2 \\ 1 \end{pmatrix} = 1 \times 2 + 2 \times 1 = 4
$$

**En français :** le résultat est **un seul nombre**, qui mesure « à quel point le vecteur correspond au motif décrit par les poids ». Il est grand quand les grandes valeurs tombent en face de grands poids positifs.

**Exemple :** avec les poids $(0 ; 1 ; 0)$, on ne regarde que le nombre de chiffres. `Marie2004` donne $0 \times 9 + 1 \times 4 + 0 \times 1 = 4$ et `azerty` donne $0$. Ce calcul est un petit **« détecteur de chiffres »**. Un réseau, c'est des milliers de détecteurs comme celui-là, dont **les poids sont appris** au lieu d'être choisis à la main.

<a id="13"></a>
## 1.3 Matrice × vecteur : deux façons de calculer

On calcule $\begin{pmatrix} 1 & 2 \\ 0 & -1 \end{pmatrix} \times \begin{pmatrix} 1 \\ 2 \end{pmatrix}$.

### Façon 1 : par lignes (la vision « neurones »)

On « couche » le vecteur sur **chaque ligne** et on fait un produit scalaire. On obtient **un nombre par ligne** :

```
ligne 1 :  ( 1 ,  2 ) · ( 1 , 2 )  =  1×1 +    2×2   =   5
ligne 2 :  ( 0 , -1 ) · ( 1 , 2 )  =  0×1 + (-1)×2   =  -2
```

### Façon 2 : par colonnes (la vision « contributions »)

Chaque nombre du vecteur multiplie **une colonne**, puis on additionne les colonnes :

```
     1 × [ 1 ]   +   2 × [  2 ]   =   [ 1 ]  +  [  4 ]   =   [  5 ]
         [ 0 ]           [ -1 ]       [ 0 ]     [ -2 ]       [ -2 ]
       colonne 1       colonne 2
```

**Même résultat, $(5 ; -2)$** : ce sont exactement les mêmes multiplications, dans un ordre différent.

> **Piège classique :** ce ne sont **jamais les lignes** qu'on multiplie par 1 puis par 2. Le vecteur a 2 nombres et chaque ligne a 2 cases : le 1ᵉʳ nombre du vecteur va toujours avec la 1ʳᵉ case de chaque ligne, c'est-à-dire avec la **1ʳᵉ colonne**.

**À quoi sert chaque vision en IA :**

| Vision | Lecture | Où on s'en sert |
|---|---|---|
| **Par lignes** | chaque ligne = **un neurone**, qui calcule son score à partir de toute l'entrée | pour lire une couche `nn.Linear` : multiplier par la matrice = calculer **tous les neurones de la couche en une seule opération** |
| **Par colonnes** | chaque colonne = **ce qu'apporte** l'entrée n° 1, n° 2… | pour comprendre l'**embedding** : avec un vecteur **one-hot** comme $(0, 0, 1, 0)$, toutes les colonnes sont multipliées par 0 sauf la 3ᵉ. Le résultat est donc **la 3ᵉ colonne** : l'embedding « sélectionne » le vecteur du caractère |

<a id="14"></a>
## 1.4 La règle des tailles

$$
\underbrace{W}_{m \times n} \; \underbrace{\mathbf{x}}_{n} = \underbrace{\mathbf{y}}_{m}
$$

- **nombre de colonnes** de $W$ = **taille de l'entrée** (chaque ligne doit se « coller » au vecteur case par case) ;
- **nombre de lignes** de $W$ = **taille de la sortie** (un nombre par ligne).

**Exemple :** une matrice $2 \times 3$ fois un vecteur de 3 donne un vecteur de 2. Une matrice $2 \times 3$ fois un vecteur de 2 est **impossible** : PyTorch lève l'erreur `mat1 and mat2 shapes cannot be multiplied`.

**Dans le code** (avec $H$ = `hidden_size` = 128) :

| Couche | Ce qu'elle contient | Entrée → sortie |
|---|---|---|
| `nn.Embedding(100, 128)` (`self.encoder`) | une table $100 \times 128$ | numéro de caractère → une ligne de 128 nombres |
| `nn.GRU(128, 128, n_layers)` (`self.gru`) | plusieurs matrices $128 \times 128$ par couche | (caractère, ancienne mémoire) → nouvelle mémoire de 128 |
| `nn.Linear(128, 100)` (`self.decoder`) | une matrice $100 \times 128$ + 100 biais | mémoire de 128 → **100 scores**, un par caractère |

**Pourquoi des matrices plutôt que des boucles ?** C'est la même chose mathématiquement, mais c'est plus court à écrire (une couche entière = $W\mathbf{x} + \mathbf{b}$) et **beaucoup plus rapide** : processeurs et cartes graphiques font des milliers de multiplications-additions en parallèle.

<a id="15"></a>
## 1.5 Exponentielle et logarithme (ln)

$e \approx 2{,}718$. L'**exponentielle** $e^z$ :
- est **toujours strictement positive**, même pour $z$ négatif ;
- **croît très vite** : $e^0 = 1$, $e^1 \approx 2{,}72$, $e^5 \approx 148$.

**ln** (le **logarithme népérien**, noté aussi **log** dans les slides) est la **fonction inverse** : il répond à la question « $e$ puissance combien ? ».

| Exponentielle | ln (l'inverse) |
|---|---|
| $e^0 = 1$ | $\ln(1) = 0$ |
| $e^1 \approx 2{,}718$ | $\ln(2{,}718) = 1$ |
| $e^2 \approx 7{,}39$ | $\ln(7{,}39) = 2$ |
| $e^{4{,}6} \approx 100$ | $\ln(100) \approx 4{,}6$ |

**En français :** $\ln(100) \approx 4{,}6$ veut dire « il faut élever $e$ à la puissance 4,6 pour obtenir 100 ». C'est le même principe que la racine carrée, qui « défait » le carré ($\sqrt{9} = 3$ parce que $3^2 = 9$).

**Ce qu'il faut retenir pour la suite :**
- $\ln(1) = 0$ ;
- le ln d'un nombre **entre 0 et 1 est négatif**, d'où le « moins » devant dans $-\ln(p)$ ;
- plus $p$ est proche de 0, plus $-\ln(p)$ est grand.

| $p$ | 1 | 0,5 | 0,1 | 0,01 | proche de 0 |
|---|---|---|---|---|---|
| $-\ln(p)$ | 0 | 0,69 | 2,30 | 4,61 | tend vers $+\infty$ |

Tu n'as jamais à le calculer à la main : `math.log()` ou PyTorch le font.

<a id="16"></a>
## 1.6 Dérivée et règle de la chaîne

La **dérivée** d'une fonction mesure **sa pente** : si j'augmente un tout petit peu l'entrée, de combien change la sortie ?

**Exemple :** $f(w) = w^2$, dérivée $f'(w) = 2w$. En $w = 3$ la pente vaut 6 : si $w$ passe de 3 à 3,01, $f$ passe de 9 à environ $9 + 6 \times 0{,}01 = 9{,}06$ (en vrai $3{,}01^2 = 9{,}0601$).

On note aussi la dérivée $\dfrac{\partial f}{\partial w}$ (« d rond f sur d rond w ») quand $f$ dépend de **plusieurs** variables et qu'on dérive par rapport à **une seule**. Le **gradient**, c'est la liste de **toutes** ces dérivées : une par poids.

**La règle de la chaîne :** si $w$ influence $y$, qui influence $\mathcal{L}$, alors

$$
\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial y} \times \frac{\partial y}{\partial w}
$$

**En français :** les effets se **multiplient** le long d'une chaîne.

**Exemple :** une heure de travail en plus te rapporte 12 €, et chaque euro te permet d'acheter 2 bonbons. Une heure de plus = $12 \times 2 = 24$ bonbons de plus.

> **Retiens bien ce « se multiplient ».** Dans un RNN, la chaîne est aussi longue que la phrase : on multiplie autant de facteurs qu'il y a de caractères. C'est **toute** l'origine du problème de la section 5.

---

<a id="2"></a>
# 2. Le neurone et les fonctions d'activation

<a id="21"></a>
## 2.1 Le neurone

Un neurone fait un **produit scalaire** entre ses poids et ses entrées, ajoute un **biais**, puis applique une **fonction d'activation** $f$ **à la fin** :

$$
y = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(\mathbf{w}\cdot\mathbf{x} + b)
$$

**Exemple :** $\mathbf{x} = (2, 1)$, $\mathbf{w} = (0{,}5 ; -1)$, $b = 0{,}3$ et $f$ = ReLU :

$$
0{,}5 \times 2 + (-1)\times 1 + 0{,}3 = 0{,}3 \quad\Rightarrow\quad y = \text{ReLU}(0{,}3) = 0{,}3
$$

Une **couche** de neurones s'écrit d'un coup avec une matrice (une ligne par neurone, section 1.3) :

$$
\mathbf{y} = f(W\mathbf{x} + \mathbf{b})
$$

- un poids positif **excite** le neurone, un poids négatif l'**inhibe** (slide 5) ;
- le biais règle **à partir de quand** le neurone s'active ;
- **apprendre, c'est trouver les bonnes valeurs de tous les $W$ et $b$**.

<a id="22"></a>
## 2.2 Pourquoi une fonction d'activation ?

### Sans activation, empiler des couches ne sert à rien

Deux neurones en chaîne, sans activation : le 1ᵉʳ calcule $2x$, le 2ᵉ calcule $3 \times (2x)$. Au total $6x$ : **un seul neurone aurait suffi**.

Avec des matrices, c'est pareil : $W_3 \, W_2 \, W_1 \, \mathbf{x}$ se multiplie d'avance en **une seule matrice**. 50 couches sans activation = 1 couche.

Or une couche seule ne sait faire que du **proportionnel** (« plus il y a de chiffres, plus le score monte », en ligne droite). Elle ne sait pas faire de **seuil** ou de **condition** (« *si* le mot finit par une année, *alors*… »).

### L'activation ajoute un « coude »

ReLU, par exemple, dit : **« je ne réagis que si mon score dépasse 0, sinon je me tais »**. C'est un petit `if`. En empilant des milliers de petits `if` réglables, le réseau peut apprendre des règles compliquées.

### Pourquoi on ne peut plus simplifier

Un réseau de 3 couches, ce n'est **pas** « 3 fois $x$, puis 3 fois la sigmoïde ». Entre deux activations, il y a **à chaque fois une matrice différente** :

$$
\sigma\Big( W_3 \, \sigma\big( W_2 \, \sigma( W_1 \, \mathbf{x} ) \big) \Big)
$$

C'est une **composition de fonctions** (notée $f \circ g$ en maths) : le réseau entier est bien **une seule grosse fonction**. Mais contrairement au cas sans activation, **on ne peut pas la simplifier** en une formule plus courte. C'est exactement ce qui donne sa puissance à chaque couche ajoutée.

> **Et si on appliquait la sigmoïde 50 fois de suite, sans matrices ?** $\sigma(\sigma(\sigma(\dots(x))))$ converge vers **0,659** quel que soit $x$ au départ (vérifié en partant de 5 et de −5). **Toute l'information sur $x$ est perdue.** Ce sont les matrices intercalées, apprises, qui permettent de garder et de transformer l'information d'une couche à l'autre.

<a id="23"></a>
## 2.3 Les trois fonctions à connaître

| Nom | Formule | Sortie | Exemples |
|---|---|---|---|
| **Sigmoïde** (ou « logistique ») | $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ | entre 0 et 1 | $\sigma(0) = 0{,}5$ ; $\sigma(5) \approx 0{,}993$ ; $\sigma(-5) \approx 0{,}007$ |
| **tanh** (tangente hyperbolique) | $\tanh(z) = \dfrac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$ | entre −1 et 1 | $\tanh(0) = 0$ ; $\tanh(1) \approx 0{,}76$ |
| **ReLU** | $\text{ReLU}(z) = \max(0, z)$ | 0 ou plus | $\text{ReLU}(-2) = 0$ ; $\text{ReLU}(3) = 3$ |

> Dans les slides, $\sigma(\cdot)$ désigne « une fonction en forme de S » (sigmoïde ou tanh). À partir de la slide 22, la même fonction est notée $\alpha$.

Il en existe beaucoup d'autres : **Leaky ReLU** (un ReLU qui laisse passer un peu de négatif), **ELU**, **GELU** (celle des Transformers, comme ChatGPT), **SiLU / Swish**… Et la **softmax** (section 3.1) joue le rôle d'activation de la **dernière** couche.

<a id="24"></a>
## 2.4 Pourquoi ces intervalles ? Tout dépend du rôle

**L'intervalle de sortie est choisi en fonction de ce qu'on fait du nombre juste après.** Le meilleur exemple est la formule de mise à jour du GRU (détaillée en section 6) :

$$
\text{nouvelle mémoire} = (1 - z) \times \text{proposition} + z \times \text{ancienne mémoire}
$$

Ici $z$ doit être un **pourcentage** : « la part d'ancienne mémoire qu'on garde ». Regardons ce qui se passe selon la fonction qui calcule $z$ :

| $z$ calculé avec… | Valeur possible | Résultat | Verdict |
|---|---|---|---|
| **Sigmoïde** | $z = 0{,}9$ | $0{,}1 \times$ proposition $+ 0{,}9 \times$ ancienne : on garde 90 % | Un vrai mélange, ça a du sens |
| ReLU | $z = 3$ | $-2 \times$ proposition $+ 3 \times$ ancienne : la mémoire est **triplée**. Au caractère suivant, encore triplée… Après 13 caractères : $3^{13} \approx 1{,}6$ million | La mémoire **explose** |
| tanh | $z = -0{,}5$ | « garder −50 % de la mémoire » | **Ne veut rien dire** |

D'où le tableau des rôles :

| Fonction | Sortie | Rôle | Pourquoi cet intervalle |
|---|---|---|---|
| **Sigmoïde** | entre 0 et 1 | les **portes** du GRU et du LSTM | le nombre est ensuite **multiplié** par une information : 0 = « je bloque », 1 = « je laisse tout passer ». Une valeur négative l'inverserait, une valeur au-dessus de 1 l'amplifierait |
| **tanh** | entre −1 et 1 | le **contenu** de la mémoire | une information peut aller dans les deux sens (+ ou −), donc il faut des négatifs. Et comme la mémoire est **réinjectée à chaque pas**, elle doit rester **bornée**, sinon elle grossirait sans fin |
| **ReLU** | 0 ou plus | les couches cachées « normales » des réseaux profonds | très rapide à calculer et, surtout, **ne freine pas le gradient** (section 2.5) |

> **« Pourquoi ReLU ne se limite pas à [0 ; 1] pour tout faire d'un coup ? »** Parce qu'elle deviendrait une sigmoïde et perdrait son avantage : toute fonction « écrasée » dans un intervalle a forcément une **pente proche de 0 sur les bords**, et c'est cette pente qui multiplie le gradient (section suivante).

<a id="25"></a>
## 2.5 La pente des activations (elle comptera pour le vanishing)

Pendant la rétropropagation, chaque activation traversée **multiplie** le gradient par **sa pente** (règle de la chaîne, section 1.6).

| Fonction | Pente (dérivée) | Pente maximale | Où elle est faible |
|---|---|---|---|
| Sigmoïde | $\sigma'(z) = \sigma(z)\,(1 - \sigma(z))$ | **0,25** (en $z = 0$) | presque 0 aux extrémités, là où la courbe est « écrasée » |
| tanh | $\tanh'(z) = 1 - \tanh^2(z)$ | **1** (en $z = 0$) | tombe vite dès qu'on s'éloigne de 0 |
| ReLU | 1 si $z > 0$, sinon 0 | **1** | vaut **exactement 1** pour tout score positif |

**Exemple :** traverser 10 sigmoïdes multiplie le gradient au mieux par $0{,}25^{10} \approx 0{,}000001$. Le gradient **disparaît** : c'est le même phénomène que le vanishing des slides BPTT (section 5), mais dans la profondeur au lieu du temps. ReLU l'évite parce qu'elle n'est pas écrasée vers le haut.

<a id="26"></a>
## 2.6 Où sont les activations dans le code ?

Tu n'en verras **aucune écrite en toutes lettres** dans `articleGeneration.py` :

- **sigmoïde et tanh sont cachées dans `nn.GRU`** : sigmoïde pour les portes $r$ et $z$, tanh pour la mémoire candidate $n$ (section 6). Elles tournent **à chaque caractère lu** ;
- **en sortie**, pas d'activation mais une **softmax**, appliquée à l'intérieur de `nn.CrossEntropyLoss` pendant l'entraînement ;
- **ReLU n'est pas dans le code fourni**. Elle n'apparaît que dans une piste facultative du TP (section 5.5 : ajouter une couche linéaire en plus).

**Pour la partie III du TP :**
- **dans le GRU ou le LSTM**, on n'y touche pas : sigmoïde et tanh y sont pour les raisons de la section 2.4, et il faudrait réécrire la cellule à la main ;
- **dans la couche ajoutée** (TP 5.5), on peut comparer ReLU, GELU et tanh : une ligne à changer, effet probablement faible ;
- **avec `nn.RNN`**, il existe une vraie option : `nn.RNN(..., nonlinearity='relu')` au lieu de `'tanh'`. Intéressant à tester, mais la mémoire n'est plus bornée : le risque d'explosion augmente (d'où le gradient clipping, section 5.6).

Ce n'est pas la priorité : LSTM vs GRU, dropout et validation rapportent plus.

---

<a id="3"></a>
# 3. Comment un réseau apprend

**L'analogie de la radio :** une radio avec 200 000 boutons sans étiquettes, qui doit jouer du Shakespeare.
1. Tu écoutes : c'est du bruit (**forward**, la prédiction).
2. Tu mesures « à quel point c'est faux » : c'est la **loss**.
3. Pour chaque bouton, tu calcules dans quel sens le tourner pour que ce soit moins faux : c'est le **gradient**, calculé par la **rétropropagation**.
4. Tu tournes chaque bouton un tout petit peu : c'est la **mise à jour** (Adam).
5. Tu recommences des milliers de fois.

<a id="31"></a>
## 3.1 Softmax : des scores aux probabilités

La dernière couche donne un **score** quelconque (un *logit*) pour chacun des 100 caractères. La **softmax** les transforme en probabilités :

$$
p_i = \text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**En français :** l'exponentielle rend chaque score positif (et amplifie les écarts), puis on divise par le total pour que la somme fasse 1.

**Exemple** avec 3 caractères et les scores $(2, 1, 0)$ :

| Caractère | Score $z_i$ | $e^{z_i}$ | $p_i = e^{z_i} / 11{,}107$ |
|---|---|---|---|
| `e` | 2 | 7,389 | **0,665** |
| `a` | 1 | 2,718 | 0,245 |
| `z` | 0 | 1,000 | 0,090 |
| **Total** | | 11,107 | 1,000 |

**Dans le code :** pas de softmax dans le modèle. `nn.CrossEntropyLoss` l'applique pendant l'entraînement, et `evaluate()` calcule directement $e^{z_i / T}$ pendant la génération (la **température** $T$, voir TP 3.3).

<a id="32"></a>
## 3.2 La loss : l'entropie croisée

$$
\boxed{\;\mathcal{L} = -\ln\left(p_{\text{cible}}\right)\;}
$$

**En français :** « l'erreur vaut moins le ln de la probabilité que le réseau a donnée à la **bonne** réponse ». Grâce au comportement de $-\ln$ (section 1.5) :
- si le réseau était sûr et avait raison ($p = 1$), l'erreur vaut **0** ;
- s'il a donné presque aucune chance à ce qui s'est réellement passé ($p$ proche de 0), l'erreur est **énorme**.

| Probabilité donnée à la bonne réponse | Loss $-\ln(p)$ | Lecture |
|---|---|---|
| 0,665 (l'exemple softmax, cible `e`) | 0,41 | Bien |
| 0,5 | 0,69 | Moyen |
| 0,09 (cible `z`) | 2,41 | Mauvais |
| 0,01 = 1/100 | **4,61** | Réponse au hasard parmi 100 |

Sur une séquence, on **moyenne** les erreurs de chaque position :

$$
\mathcal{L} = -\frac{1}{N}\sum_{t=1}^{N} \ln p\left(c_{t+1} \mid c_1, \dots, c_t\right)
$$

**Pourquoi la loss commence à 4,6 :** au départ, le réseau donne la même chance $\frac{1}{100}$ à chacun des 100 caractères. Sa loss vaut donc $-\ln\left(\frac{1}{100}\right) = \ln(100) \approx 4{,}605$. C'est la toute première valeur affichée par le programme : **la preuve qu'au début il tire au hasard**.

**La perplexité** $e^{\mathcal{L}}$ se lit comme « le nombre de caractères entre lesquels le réseau hésite » : $e^{4{,}6} \approx 100$, $e^{1{,}5} \approx 4{,}5$.

> **Coquille de la slide 6 :** la formule y est écrite sans le signe « moins » et avec les bornes des deux sommes inversées. La bonne forme est $\mathcal{L} = -\sum_{t=1}^{L-1}\sum_{i=1}^{V} x_{t+1,i}\,\ln(y_{t,i})$, qui se réduit à $-\ln(p_{\text{cible}})$ à chaque position parce que $x_{t+1}$ est un one-hot.

<a id="33"></a>
## 3.3 La descente de gradient

Tu es dans le brouillard en montagne et tu veux descendre. Tu sens la pente sous tes pieds, tu fais un petit pas dans le sens qui descend le plus, et tu recommences.

- l'**altitude**, c'est la loss ;
- ta **position**, ce sont les poids ;
- la **pente**, c'est le **gradient** $\dfrac{\partial \mathcal{L}}{\partial W}$ ;
- la **taille du pas**, c'est le **learning rate** $\alpha$ (`lr = 0.005` dans le code).

$$
W \leftarrow W - \alpha \, \frac{\partial \mathcal{L}}{\partial W}
$$

**En français :** « nouveau poids = ancien poids − (taille du pas × pente) ». Le « moins » parce que le gradient indique le sens où l'erreur **monte**, et qu'on veut aller dans l'autre sens.

**Exemple à la main :** $\mathcal{L}(w) = (w - 3)^2$, gradient $2(w - 3)$, départ $w = 0$, $\alpha = 0{,}1$.

| Étape | $w$ | Gradient | Nouveau $w$ | Loss |
|---|---|---|---|---|
| 0 | 0 | −6 | 0,6 | 9 |
| 1 | 0,6 | −4,8 | 1,08 | 5,76 |
| 2 | 1,08 | −3,84 | 1,464 | 3,69 |
| … | … | … | → 3 | → 0 |

Trop petit $\alpha$ : on descend très lentement. Trop grand : on saute par-dessus la vallée et la loss oscille, voire explose.

<a id="34"></a>
## 3.4 Rétropropagation et Adam : deux étapes différentes

**La rétropropagation n'est pas ce que fait Adam.** Ce sont deux étapes qui se suivent :

| Étape | Qui la fait | Ce qu'elle fait |
|---|---|---|
| **1. Rétropropagation** | `loss.backward()` | **calcule** le gradient de chaque poids, en remontant le réseau depuis la loss grâce à la règle de la chaîne. Elle ne modifie **aucun** poids |
| **2. Mise à jour** | `decoder_optimizer.step()`, c'est-à-dire **Adam** | **utilise** ces gradients pour **modifier** réellement les poids |

**Avec la montagne :** la rétropropagation, c'est **sentir la pente** (on mesure, on ne bouge pas). Adam, c'est **décider du pas et le faire**.

```
Forward  (calcul) :     x ──► couche 1 ──► couche 2 ──► sortie ──► loss
Backward (gradients) :  x ◄── couche 1 ◄── couche 2 ◄── sortie ◄── loss
```

**Dans le code, la fonction `train` :**

```python
decoder.zero_grad()          # 0. efface les gradients de l'étape précédente
...                          #    forward : prédictions + calcul de la loss
loss.backward()              # 1. RÉTROPROPAGATION : calcule les gradients
decoder_optimizer.step()     # 2. ADAM : modifie les poids avec ces gradients
```

Entre les deux, les gradients restent stockés à côté de chaque poids, dans `.grad` : `backward()` les remplit, `step()` vient les lire. `zero_grad()` est nécessaire parce que PyTorch **additionne** les nouveaux gradients aux anciens si on ne les efface pas.

**Ce qu'Adam ajoute à la formule de base :**
- il **se souvient des gradients précédents** et les moyenne : il prend de l'élan, comme une bille qui roule, au lieu de suivre chaque secousse ;
- il **adapte la taille du pas pour chaque poids** : petits pas pour les poids dont le gradient varie beaucoup, plus grands pour les autres.

**La preuve que ce sont deux choses séparées :** si tu remplaces Adam par `torch.optim.SGD` (la descente de gradient simple), `loss.backward()` ne change pas d'une virgule. Comparer Adam et SGD, ou plusieurs `lr`, peut faire une petite expérience de partie III.

---

<a id="4"></a>
# 4. Le réseau récurrent (RNN)

<a id="41"></a>
## 4.1 Le problème du MLP : une fenêtre fixe

Le but (slide 6) : prédire le caractère suivant sachant tout ce qui précède.

$$
y_t = p\left(x_{t+1} \mid x_t, x_{t-1}, \dots, x_1\right)
$$

Le **MLP** (perceptron multicouche) de la slide 6 ne regarde que les **3 derniers** caractères :

$$
\begin{aligned}
e_t &= W^{ex} x_t && \text{(embedding)} \\
h_t &= \sigma\left(W^{he}\,[e_t\,;\,e_{t-1}\,;\,e_{t-2}] + b^h\right) && \text{(on colle les 3 embeddings bout à bout)} \\
y_t &= \text{softmax}\left(W^{yh} h_t + b^y\right) && \text{(probabilités du caractère suivant)}
\end{aligned}
$$

> **Convention de nommage :** $W^{he}$ se lit « la matrice qui va de $e$ vers $h$ » (l'exposant indique d'abord l'arrivée, puis le départ).

**Sa limite :** dans `motdepasse2004`, pour prédire le `4` final, il ne voit que `200`. Tout le début est oublié. Il faudrait un réseau qui lit **un caractère à la fois** et **se souvient** de tout ce qu'il a lu.

<a id="42"></a>
## 4.2 Le RNN : un réseau avec une mémoire

Le RNN (slide 7) garde un vecteur de **mémoire**, l'**état caché** $h_t$ (128 nombres par défaut). Seule la couche cachée change par rapport au MLP, avec **le terme en rouge** :

$$
\begin{aligned}
e_t &= W^{ex} x_t \\
h_t &= \sigma\left(W^{he} e_t + \color{red}{W^{hh} h_{t-1}} + b^h\right) \\
y_t &= \text{softmax}\left(W^{yh} h_t + b^y\right)
\end{aligned}
$$

**En français :** la nouvelle mémoire est un mélange de **ce qu'on lit maintenant** ($e_t$) et de **l'ancienne mémoire** ($h_{t-1}$), écrasé entre −1 et 1 par tanh. Comme $h_t$ dépend de $h_{t-1}$, qui dépend de $h_{t-2}$…, $h_t$ contient en principe un résumé de **toute** la séquence lue.

**Exemple** avec des nombres seuls ($H = 1$) : $w_x = 0{,}5$, $w_h = 0{,}8$, $b = 0$, ancienne mémoire $h_{t-1} = 0{,}2$, entrée $e_t = 1$ :

$$
h_t = \tanh(0{,}5 \times 1 + 0{,}8 \times 0{,}2) = \tanh(0{,}66) \approx 0{,}578
$$

**Le RNN « déplié » dans le temps :**

```
          y1          y2          y3          y4        ← prédictions
           ▲           ▲           ▲           ▲
  h0 ───► [h1] ─W──► [h2] ─W──► [h3] ─W──► [h4]         ← la mémoire circule
           ▲           ▲           ▲           ▲
          "a"         "z"         "e"         "r"       ← caractères lus
```

Ce n'est pas 4 réseaux différents : c'est **le même réseau**, avec **les mêmes matrices**, appliqué à chaque caractère. Le mot « récurrent » vient de là : la sortie $h$ est réinjectée en entrée au pas suivant.

> **Point crucial pour la suite :** la matrice $W^{hh}$ (notée $W$ ou $W_h$ dans les slides 8 à 32) est **la même à chaque flèche**. On dit que les poids sont **partagés dans le temps**. C'est sa force (peu de paramètres, séquences de n'importe quelle longueur) et, on va le voir, la source de son gros problème.

**Dans PyTorch** (slide 38, code corrigé) :

```python
self.rnn = nn.RNN(x_size, h_dim, n_layers)   # h_t = tanh(W_x e_t + W_h h_{t-1} + b)
self.fc  = nn.Linear(h_dim, y_size)           # scores de sortie
```

> Le code de la slide 38 contient des erreurs (`hidden_dim`, `out` et `hidden` n'existent pas, `init_hidden` n'est pas défini) : c'est une illustration, pas du code prêt à exécuter.

<a id="43"></a>
## 4.3 Plusieurs couches

Avec `num_layers = 2`, on empile deux RNN : la sortie $h_t$ de la couche 1 devient l'entrée de la couche 2. **Chaque couche a sa propre mémoire.** La 1ʳᵉ apprend des motifs simples (lettres), la 2ᵉ des motifs plus abstraits (syllabes, mots).

```
  couche 2 :  h0' ──► [RNN2] ──► [RNN2] ──► ...
                        ▲          ▲
  couche 1 :  h0  ──► [RNN1] ──► [RNN1] ──► ...
                        ▲          ▲
                       "T"        "o"
```

C'est pour ça que la mémoire a la forme `(n_layers, 1, hidden_size)` dans le code : une mémoire par couche.

<a id="44"></a>
## 4.4 Apprendre à prédire le caractère suivant

On prend un morceau de 14 caractères et on l'aligne avec lui-même, décalé d'un cran :

```
 entrée (inp)    : H  o  w  ␣  n  o  w  ,  ␣  s  i  r  .
 cible  (target) : o  w  ␣  n  o  w  ,  ␣  s  i  r  .  ⏎
```

À chaque position, le réseau lit le caractère d'entrée, met à jour sa mémoire et doit prédire la cible. Le texte **se supervise lui-même** : la bonne réponse est gratuite.

**Dans le code, `train` fait exactement le RNN déplié :**

```python
hidden = decoder.init_hidden()                            # h_0 = 0
for c in range(inp.size(0)):                              # 13 pas de temps
    output, hidden = decoder(inp[c], hidden)              # h_{t-1} → h_t, puis scores
    loss += criterion(output, target[c].unsqueeze(0))     # + (-ln p_cible)
loss.backward()                                           # BPTT : remonte les 13 pas
decoder_optimizer.step()                                  # Adam
```

La variable `hidden` qui passe d'un tour de boucle au suivant, **c'est** la flèche horizontale du schéma déplié.

---

<a id="5"></a>
# 5. Pourquoi le RNN simple oublie : la BPTT

> **Le fil conducteur des slides 8 à 32 :** on veut calculer comment modifier $W_h$. Parce que $W_h$ sert à chaque pas de temps, le calcul fait apparaître **$W_h$ multipliée par elle-même** autant de fois qu'il y a de pas. Et multiplier beaucoup de fois par le même nombre donne soit **presque 0**, soit **un nombre gigantesque**.

<a id="51"></a>
## 5.1 W est utilisée à chaque pas

On applique la rétropropagation (section 3.4) au RNN **déplié** : c'est la **BPTT** (*Back-Propagation Through Time*). On veut faire (slide 8) :

$$
W \leftarrow W - \alpha\, \frac{\partial \mathcal{L}}{\partial W}
$$

**Le problème :** dans un réseau classique, chaque poids sert une seule fois. Ici, $W_h$ sert **à chaque transition** $h_{t-1} \to h_t$. Elle influence donc l'erreur par **plusieurs chemins** (slides 9 à 14) :

| Erreur | Chemins depuis $W_h$ | Nombre |
|---|---|---|
| $\mathcal{L}_1$ | via $h_0 \to h_1$ | 1 |
| $\mathcal{L}_2$ | via $h_0 \to h_1$ et via $h_1 \to h_2$ | 2 |
| $\mathcal{L}_3$ | via $h_0 \to h_1$, $h_1 \to h_2$ et $h_2 \to h_3$ | 3 |
| $\mathcal{L}_j$ | … | $j$ |

<a id="52"></a>
## 5.2 Les deux sommes

**Première somme (slide 15), sur les erreurs de chaque pas.** La loss totale est la somme des losses de chaque pas, donc sa dérivée est la somme des dérivées ($T$ = longueur de la séquence, pas la température) :

$$
\frac{\partial \mathcal{L}}{\partial W_h} = \sum_{j=0}^{T-1} \frac{\partial \mathcal{L}_j}{\partial W_h}
$$

**Seconde somme (slide 16), sur les chemins.** Un terme par chemin, c'est-à-dire par état caché passé $h_k$ :

$$
\frac{\partial \mathcal{L}_j}{\partial W_h} = \sum_{k=1}^{j} \frac{\partial \mathcal{L}_j}{\partial h_k}\,\frac{\partial h_k}{\partial W_h}
$$

**En français :** « l'effet de $W_h$ sur l'erreur au temps $j$ = la somme, pour chaque instant passé $k$, de (l'effet de $W_h$ sur la mémoire $h_k$) × (l'effet de $h_k$ sur l'erreur $\mathcal{L}_j$) ».

<a id="53"></a>
## 5.3 Le terme qui pose problème : un produit

$\mathcal{L}_j$ ne dépend pas **directement** de $h_k$ (slide 17). Le chemin passe par la sortie $y_j$, par la mémoire $h_j$, puis remonte de $h_j$ jusqu'à $h_k$. Règle de la chaîne, maillon par maillon (slides 18 et 19) :

$$
\frac{\partial \mathcal{L}_j}{\partial W_h} = \sum_{k=1}^{j}
\underbrace{\frac{\partial \mathcal{L}_j}{\partial y_j}}_{\text{rose}}\;
\underbrace{\frac{\partial y_j}{\partial h_j}}_{\text{vert}}\;
\underbrace{\frac{\partial h_j}{\partial h_k}}_{\text{orange}}\;
\frac{\partial h_k}{\partial W_h}
$$

| Couleur | Terme | Sens | Difficulté |
|---|---|---|---|
| Rose | $\partial \mathcal{L}_j / \partial y_j$ | comment l'erreur change quand la prédiction change | facile |
| Vert | $\partial y_j / \partial h_j$ | comment la prédiction change quand la mémoire change | facile |
| **Orange** | $\partial h_j / \partial h_k$ | comment la mémoire au temps $j$ dépend de la mémoire au temps $k$ | **un long trajet dans le temps** |

Pour aller de $h_k$ à $h_j$, on passe par **chaque** étape, et les effets se **multiplient** (slide 20) :

$$
\frac{\partial h_j}{\partial h_k} = \prod_{m=k+1}^{j} \frac{\partial h_m}{\partial h_{m-1}}
$$

**Exemple :** $\dfrac{\partial h_4}{\partial h_1} = \dfrac{\partial h_4}{\partial h_3} \times \dfrac{\partial h_3}{\partial h_2} \times \dfrac{\partial h_2}{\partial h_1}$ : trois facteurs pour trois pas. Pour deux caractères éloignés de 20 positions, 20 facteurs.

**Un facteur du produit (slide 22).** Chaque transition s'écrit $h_m = \alpha\left(W_x x_m + W_h h_{m-1}\right)$, donc :

$$
\frac{\partial h_m}{\partial h_{m-1}} = W_h^{\top}\, \text{diag}\!\left(\alpha'\left(W_x x_m + W_h h_{m-1}\right)\right)
$$

**D'où ça vient :** encore la règle de la chaîne, avec deux maillons.
1. $h_{m-1}$ est multiplié par $W_h$ : la dérivée d'une multiplication par une matrice, c'est la matrice (le $^{\top}$, « transposée », est un détail d'écriture) ;
2. on applique $\alpha$ à chaque case : on multiplie par sa pente $\alpha'$ (section 2.5), rangée sur la diagonale d'une matrice avec $\text{diag}(\dots)$.

*En une dimension :* si $h_m = \tanh(w \cdot h_{m-1})$, alors $\dfrac{\partial h_m}{\partial h_{m-1}} = w \times \tanh'(\dots)$.

**Le produit final (slides 23 et 24) :**

$$
\frac{\partial h_j}{\partial h_k} = \prod_{m=k+1}^{j} \underbrace{W_h^{\top}}_{\text{matrice de poids}}\; \underbrace{\text{diag}\!\left(\alpha'(\dots)\right)}_{\text{pente de l'activation}}
$$

**Ce qu'il faut voir : $W_h$ est multipliée par elle-même $(j - k)$ fois.** Plus deux caractères sont éloignés, plus il y a de multiplications.

> Tu n'as jamais à calculer ça : `loss.backward()` le fait. Il faut seulement **comprendre pourquoi le produit $\prod$ pose problème**.

<a id="54"></a>
## 5.4 Exemple chiffré sur 13 caractères

Reprenons l'exemple de la section 4.2 ($w_h = 0{,}8$, $h_t = \tanh(0{,}66) \approx 0{,}578$). Un facteur du produit vaut :

$$
w_h \times \tanh'(0{,}66) = 0{,}8 \times (1 - 0{,}578^2) = 0{,}8 \times 0{,}665 \approx 0{,}53
$$

Le programme entraîne sur **13 caractères**. Si chaque pas donne à peu près ce facteur, le signal d'erreur qui remonte du 13ᵉ caractère jusqu'au 1ᵉʳ est multiplié par :

$$
0{,}53^{13} \approx 0{,}0003
$$

**En français :** l'erreur commise au 13ᵉ caractère n'envoie **quasiment aucun signal** vers le 1ᵉʳ. Le réseau **ne peut pas apprendre** que le 1ᵉʳ caractère aurait dû influencer le 13ᵉ. Il a une **mémoire courte**, même si en théorie $h_t$ résume toute la séquence.

Et si le facteur était un peu au-dessus de 1, c'est l'inverse :

| Facteur répété | ×13 | ×50 | Problème |
|---|---|---|---|
| 0,53 | 0,0003 | ≈ 0 | le gradient **disparaît** |
| 0,9 | 0,25 | 0,005 | le gradient disparaît (plus lentement) |
| 1 | 1 | 1 | idéal, mais n'arrive jamais par hasard |
| 1,1 | 3,5 | 117 | le gradient **explose** |

<a id="55"></a>
## 5.5 Avec des matrices : les valeurs propres

Pour une matrice, le rôle de « ce nombre qu'on répète » est joué par ses **valeurs propres**.

**Les hypothèses de la slide 26 :** l'activation $\alpha$ est l'identité (donc $\alpha' = 1$, le terme $\text{diag}$ disparaît) et $W_h$ est **diagonalisable**. On peut alors écrire :

$$
W_h = Q^{-1} A\, Q
$$

- $A$ (slide 27) est **diagonale** : des nombres sur la diagonale, des 0 ailleurs. Ce sont les **valeurs propres** de $W_h$ ;
- $Q$ (slide 28) contient les **vecteurs propres** de $W_h$.

**Sans refaire un cours d'algèbre :** une matrice étire ou écrase les vecteurs. Les **vecteurs propres** sont les directions où elle **ne fait qu'étirer ou écraser**, sans tourner. La **valeur propre** dit **de combien** (2 = double la longueur, 0,5 = divise par 2).

**La puissance d'une matrice diagonalisée (slide 29) :** les $Q$ et $Q^{-1}$ du milieu s'annulent.

$$
W_h^2 = Q^{-1} A \underbrace{Q\, Q^{-1}}_{= I} A\, Q = Q^{-1} A^2 Q
\qquad\Longrightarrow\qquad
W_h^{n} = Q^{-1} A^{n} Q
$$

Et élever une matrice diagonale à la puissance $n$, c'est **élever chaque nombre de la diagonale à la puissance $n$**. Le comportement de $W_h^n$ est donc entièrement dicté par les **valeurs propres à la puissance $n$**.

**L'exemple des slides 30 à 32 :**

$$
A = \begin{pmatrix} -0{,}71 & 0 \\ 0 & 1{,}68 \end{pmatrix}
\qquad\Longrightarrow\qquad
A^{10} = \begin{pmatrix} (-0{,}71)^{10} & 0 \\ 0 & 1{,}68^{10} \end{pmatrix} = \begin{pmatrix} 0{,}03 & 0 \\ 0 & 179{,}1 \end{pmatrix}
$$

> Le « b » qui traîne dans la matrice de la slide 30 est une coquille.

<a id="56"></a>
## 5.6 Vanishing, exploding et leurs remèdes

| | Valeur propre | Après 10 pas | Nom | Conséquence |
|---|---|---|---|---|
| Slide 31 | $\lvert -0{,}71 \rvert < 1$ | 0,03 | **disparition du gradient** (*vanishing*) | le réseau n'apprend pas les dépendances longues : il **oublie** le début de la séquence |
| Slide 32 | $1{,}68 > 1$ | 179 | **explosion du gradient** (*exploding*) | les mises à jour deviennent énormes, la loss fait des bonds absurdes ou devient `nan` |

> Avec une vraie activation comme tanh, la pente $\alpha'$ est toujours $\le 1$ (section 2.5) : elle ajoute des facteurs plus petits que 1 et **aggrave** encore la disparition.

**Les remèdes :**
- **contre l'explosion : le gradient clipping.** Si la longueur du gradient dépasse un seuil $c$, on la ramène à $c$ sans changer sa direction. Exemple : $\mathbf{g} = (30, 40)$, longueur 50, seuil 5 → $\mathbf{g} = (3, 4)$. Dans `train`, entre `backward()` et `step()` : `torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)`. Ça règle l'explosion, **pas** la disparition ;
- **contre la disparition : changer d'architecture.** La slide 33 cite le **CEC** (*Constant Error Carousel*), le **LSTM** et le **GRU**. L'idée commune : créer un chemin où la mémoire passe d'un pas à l'autre **sans être multipliée par $W_h$**.

---

<a id="6"></a>
# 6. Le GRU (Gated Recurrent Unit, Cho et al. 2014)

C'est **le modèle du code fourni** (`nn.GRU`).

## L'idée : des portes

Dans un RNN simple, la mémoire est **entièrement recalculée** à chaque pas, en passant par $W_h$. Le GRU ajoute des **portes** : des vecteurs de nombres **entre 0 et 1** (grâce à la sigmoïde, section 2.4) qui agissent comme des **robinets**. 0 = « je bloque », 1 = « je laisse passer ». Le réseau **apprend** quand garder la mémoire et quand la remplacer.

**Analogie :** un carnet de notes. Au lieu de réécrire tout le carnet à chaque mot lu (RNN simple), tu décides quoi **garder** et quoi **réécrire**.

## Les équations (celles de `nn.GRU`)

Le symbole $\odot$ désigne la multiplication **case par case** de deux vecteurs.

$$
\begin{aligned}
r_t &= \sigma(W_r e_t + U_r h_{t-1} + b_r) && \text{porte de reset} \\
z_t &= \sigma(W_z e_t + U_z h_{t-1} + b_z) && \text{porte de mise à jour} \\
n_t &= \tanh\left(W_n e_t + r_t \odot (U_n h_{t-1}) + b_n\right) && \text{mémoire candidate (la proposition)} \\
h_t &= (1 - z_t) \odot n_t + z_t \odot h_{t-1} && \text{nouvelle mémoire}
\end{aligned}
$$

**En français :**
- $r_t$ (sigmoïde) décide **quelle partie du passé utiliser** pour fabriquer la proposition ;
- $z_t$ (sigmoïde) décide, case par case, s'il faut **garder l'ancienne mémoire** ($z \approx 1$) ou **la remplacer** par la proposition ($z \approx 0$) ;
- $n_t$ (tanh) est la **proposition**, calculée comme dans un RNN classique, entre −1 et 1.

**Exemple** (une seule case) : $z = 0{,}9$, ancienne mémoire $h_{t-1} = 0{,}5$, proposition $n = -0{,}2$ :

$$
h_t = 0{,}1 \times (-0{,}2) + 0{,}9 \times 0{,}5 = -0{,}02 + 0{,}45 = 0{,}43
$$

La mémoire a très peu changé.

## Pourquoi ça règle le vanishing

Dans la dernière équation, $h_{t-1}$ passe vers $h_t$ par un **chemin direct**, multiplié seulement par $z_t$, **sans passer par la matrice $W_h$ ni par une activation**. Quand $z_t \approx 1$, l'information et le gradient traversent le temps presque intacts : c'est une **autoroute**.

**Comparaison sur 13 caractères :**

| Chemin | Facteur par pas | Signal après 13 pas |
|---|---|---|
| RNN simple (section 5.4) | ≈ 0,53 | ≈ **0,0003** |
| GRU avec $z = 0{,}9$ | 0,9 | ≈ **0,25** |
| GRU avec $z = 0{,}99$ | 0,99 | ≈ **0,88** |

Et c'est le **réseau qui apprend** la valeur de $z$ : il peut garder $z \approx 1$ pour une information importante (« le mot de passe commence par une majuscule ») et $z \approx 0$ pour ce qui peut être oublié.

**Exemple pour les mots de passe :** dans `Marie2004!`, le réseau peut garder en mémoire « j'ai commencé par une majuscule et un prénom » pendant qu'il écrit l'année, puis s'en servir pour décider d'ajouter un `!`.

**Paramètres :** 3 blocs de poids ($r$, $z$, $n$), donc environ **3 fois** plus qu'un RNN simple de même taille.

---

<a id="7"></a>
# 7. Le LSTM (Long Short-Term Memory, Hochreiter et Schmidhuber 1997)

Plus ancien que le GRU et un peu plus complexe. C'est **le modèle à mettre à la place du GRU** dans la partie III du TP.

## Deux mémoires

- $c_t$, la **cellule** : la mémoire à **long terme**. C'est elle qui joue le rôle du CEC de la slide 33 ;
- $h_t$ : la **sortie**, la mémoire de travail, celle qui sert à prédire.

## Trois portes (toutes des sigmoïdes)

- la porte d'**oubli** $f_t$ : que faut-il **effacer** de $c_{t-1}$ ?
- la porte d'**entrée** $i_t$ : que faut-il **écrire** de nouveau ?
- la porte de **sortie** $o_t$ : que faut-il **montrer** dans $h_t$ ?

## Les équations

$$
\begin{aligned}
f_t &= \sigma(W_f e_t + U_f h_{t-1} + b_f) && \text{porte d'oubli} \\
i_t &= \sigma(W_i e_t + U_i h_{t-1} + b_i) && \text{porte d'entrée} \\
g_t &= \tanh(W_g e_t + U_g h_{t-1} + b_g) && \text{contenu candidat} \\
o_t &= \sigma(W_o e_t + U_o h_{t-1} + b_o) && \text{porte de sortie} \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t && \text{nouvelle cellule} \\
h_t &= o_t \odot \tanh(c_t) && \text{nouvelle sortie}
\end{aligned}
$$

On retrouve la règle de la section 2.4 : **sigmoïde pour les portes** (des pourcentages), **tanh pour les contenus** (signés et bornés).

**En français :** la cellule $c$ est un **tapis roulant**. À chaque pas, on **efface** une partie (porte $f$), on **écrit** du nouveau (porte $i$), puis on **lit** une partie pour produire $h$ (porte $o$).

```
          c_{t-1} ──►(×f)──►(+)──────────────────► c_t
                              ▲            │
                            (×i)◄── g      tanh
                                           │
                                         (×o) ──► h_t
```

**Exemple** (une case) : $f = 0{,}9$, $c_{t-1} = 2$, $i = 0{,}3$, $g = 0{,}5$, $o = 0{,}8$ :

$$
c_t = 0{,}9 \times 2 + 0{,}3 \times 0{,}5 = 1{,}95 \qquad h_t = 0{,}8 \times \tanh(1{,}95) = 0{,}8 \times 0{,}960 \approx 0{,}768
$$

## Pourquoi ça règle le vanishing

L'équation clé est la mise à jour de la cellule : c'est une **addition**, pas une multiplication par une matrice. Sur le tapis roulant, la dérivée d'un pas au suivant vaut simplement :

$$
\frac{\partial c_t}{\partial c_{t-1}} = f_t
$$

**Pas de $W_h$, pas de pente d'activation** : si la porte d'oubli $f_t \approx 1$, la mémoire **et le gradient** traversent le temps intacts. C'est exactement l'idée du **CEC** : un chemin où l'erreur est multipliée par (presque) exactement 1 à chaque pas, et « tourne en rond » sans s'atténuer.

**Paramètres :** 4 blocs de poids ($f$, $i$, $g$, $o$), donc environ **4 fois** plus qu'un RNN simple, et **33 % de plus** qu'un GRU de même taille.

---

<a id="8"></a>
# 8. RNN, GRU, LSTM : le comparatif

| | RNN simple | GRU | LSTM |
|---|---|---|---|
| Mémoires | $h$ | $h$ | $h$ et $c$ |
| Portes | 0 | 2 ($r$, $z$) | 3 ($f$, $i$, $o$) |
| Blocs de poids | 1 | 3 | 4 |
| Mémoire longue | Mauvaise (vanishing) | Bonne | Très bonne |
| Vitesse | Rapide | Moyenne | Plus lente |
| PyTorch | `nn.RNN` | `nn.GRU` | `nn.LSTM` |
| État caché | un tenseur `h` | un tenseur `h` | **un couple** `(h, c)` |

**Nombre de paramètres** (vérifiés avec PyTorch ; total = embedding + cellule + couche de sortie) :

| Modèle | Cellule seule | **Total** |
|---|---|---|
| RNN simple 2 × 128 | 66 048 | **91 748** |
| GRU 2 × 128 (le modèle par défaut) | 198 144 | **223 844** |
| LSTM 2 × 128 | 264 192 | **289 892** |
| GRU 2 × 148 | 264 624 | **294 324** |

Une couche de cellule contient $k \times (2H^2 + 2H)$ paramètres avec $k$ = 1 (RNN), 3 (GRU) ou 4 (LSTM), quand l'entrée a la même taille $H$ que la mémoire.

> **Pour une comparaison équitable :** à taille égale, le LSTM a plus de paramètres. Compare aussi à nombre de paramètres proche, par exemple **LSTM 2 × 128 (289 892) contre GRU 2 × 148 (294 324)**, ou signale-le dans ton rapport.

**Pour les mots de passe,** qui sont courts (souvent moins de 16 caractères), un GRU suffit généralement. **Comparer RNN, GRU et LSTM** est une bonne expérience : le RNN simple devrait être le moins bon, ce qui illustre la section 5.

---

<a id="9"></a>
# 9. Dans le code : passer du GRU au LSTM

> 🛠️ Le code complet (classe LSTM, option `--cell`, pièges) est dans le [guide du TP §5.3](2_guide_TP.md). Les autres améliorations (dropout, LayerNorm, clipping, learning rate variable, validation) sont dans les §5.4 à §5.8 du même guide.

---

<a id="10"></a>
# 10. Fiche récapitulative

| Notion | En une phrase | Formule / code |
|---|---|---|
| Vecteur / matrice | Liste de nombres / pile de vecteurs de même taille | $W\mathbf{x}$ |
| Produit scalaire | Multiplier case par case, additionner | $(1,2)\cdot(2,1) = 4$ |
| Matrice × vecteur | Un produit scalaire par ligne = combinaison des colonnes | $(m \times n)(n) \to (m)$ |
| ln | Inverse de l'exponentielle | $\ln(100) \approx 4{,}6$ |
| Règle de la chaîne | Les effets se multiplient le long d'une chaîne | $\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial y}\frac{\partial y}{\partial w}$ |
| Neurone | Produit scalaire + biais, puis activation | $y = f(\mathbf{w}\cdot\mathbf{x} + b)$ |
| Activation | Empêche les couches de se réduire à une seule | ReLU, $\sigma$, tanh |
| Sigmoïde | Entre 0 et 1 : un pourcentage, pour les **portes** | pente max 0,25 |
| tanh | Entre −1 et 1 : signé et borné, pour la **mémoire** | pente max 1 |
| Softmax | Scores → probabilités qui somment à 1 | $e^{z_i} / \sum_j e^{z_j}$ |
| Entropie croisée | $-\ln$(probabilité de la bonne réponse) | départ : $\ln 100 \approx 4{,}6$ |
| Descente de gradient | Petit pas dans le sens qui descend | $W \leftarrow W - \alpha \frac{\partial \mathcal{L}}{\partial W}$ |
| Rétropropagation | **Calcule** les gradients | `loss.backward()` |
| Adam | **Applique** les gradients, avec élan et pas adaptatif | `optimizer.step()` |
| RNN | Une mémoire mise à jour à chaque caractère, poids partagés | $h_t = \tanh(W_x e_t + W_h h_{t-1} + b)$ |
| BPTT | La rétropropagation sur le RNN déplié | $\frac{\partial h_j}{\partial h_k} = \prod W_h^{\top}\text{diag}(\alpha')$ |
| Vanishing / exploding | $W_h$ multipliée par elle-même : valeurs propres $<1$ → 0, $>1$ → ∞ | $0{,}71^{10} \approx 0{,}03$ ; $1{,}68^{10} \approx 179$ |
| Gradient clipping | Plafonner la longueur du gradient (contre l'explosion) | `clip_grad_norm_(..., 5.0)` |
| GRU | 2 portes, 1 mémoire, chemin direct $z_t \odot h_{t-1}$ | `nn.GRU` |
| LSTM | 3 portes, 2 mémoires, cellule mise à jour par **addition** | `nn.LSTM`, `init_hidden` → `(h, c)` |

---

<a id="oral"></a>
# 11. Les questions d'oral et leurs réponses

**Pourquoi utilise-t-on des matrices en IA ?**
Parce qu'une couche de neurones, c'est un produit scalaire par neurone : les ranger en lignes d'une matrice permet de calculer toute la couche en une seule opération, très rapide sur GPU. Les poids appris sont les cases de ces matrices.

**À quoi sert une fonction d'activation ?**
Sans elle, empiler des couches revient à une seule couche ($W_3 W_2 W_1$ = une matrice). Elle ajoute la non-linéarité (des seuils, des conditions) qui permet d'apprendre des règles complexes.

**Pourquoi une porte utilise la sigmoïde et pas ReLU ?**
Une porte est un pourcentage qui multiplie une information : il faut une valeur entre 0 et 1. Avec ReLU, elle pourrait valoir 3 et tripler la mémoire à chaque pas ($3^{13} \approx 1{,}6$ million après 13 caractères).

**Pourquoi la mémoire utilise tanh ?**
Parce qu'elle doit pouvoir être négative (une information va dans les deux sens) et rester bornée, puisqu'elle est réinjectée à chaque pas.

**Pourquoi la loss commence à 4,6 ?**
Au départ, le réseau donne $\frac{1}{100}$ à chacun des 100 caractères. La loss vaut $-\ln\frac{1}{100} = \ln 100 \approx 4{,}6$ : il tire au hasard.

**La rétropropagation, c'est Adam ?**
Non. `loss.backward()` **calcule** les gradients (règle de la chaîne, de la loss vers l'entrée). Adam (`step()`) **utilise** ces gradients pour modifier les poids. On peut changer d'optimiseur sans toucher à la rétropropagation.

**Qu'est-ce qui distingue un RNN d'un MLP ?**
Le terme $W^{hh} h_{t-1}$ : la mémoire du pas précédent est réinjectée. Le RNN lit un caractère à la fois et se souvient de tout ce qui précède, au lieu d'une fenêtre fixe de 3 caractères.

**Pourquoi un RNN simple oublie-t-il vite ?**
En BPTT, le gradient qui remonte de $k$ pas est un produit de $k$ facteurs contenant $W_h$ et la pente de l'activation. Si ces facteurs sont un peu en dessous de 1 (valeurs propres $< 1$), le produit tend vers 0 : $0{,}53^{13} \approx 0{,}0003$. Le réseau ne reçoit plus de signal pour apprendre les dépendances longues.

**Et l'exploding ?**
Même produit, mais avec des valeurs propres $> 1$ : $1{,}68^{10} \approx 179$. Les mises à jour deviennent énormes et l'entraînement diverge. Remède : le gradient clipping.

**Pourquoi la diagonalisation dans les slides ?**
Parce que $W_h^n = Q^{-1} A^n Q$ : la puissance d'une matrice se ramène aux puissances de ses valeurs propres. Ça montre que tout dépend de si elles sont plus petites ou plus grandes que 1.

**Comment le GRU règle-t-il le problème ?**
Par sa mise à jour $h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}$ : l'ancienne mémoire passe par un chemin direct, multiplié seulement par la porte $z_t$, sans $W_h$. Si $z_t \approx 1$, le signal traverse le temps ($0{,}99^{13} \approx 0{,}88$).

**Comment le LSTM règle-t-il le problème ?**
La cellule est mise à jour par addition, $c_t = f_t \odot c_{t-1} + i_t \odot g_t$, donc $\partial c_t / \partial c_{t-1} = f_t$. Si la porte d'oubli vaut presque 1, mémoire et gradient passent intacts : c'est le Constant Error Carousel.

**GRU ou LSTM ?**
Le GRU a 2 portes et une mémoire : plus léger (3 blocs de poids contre 4) et souvent aussi bon. Le LSTM a 3 portes et deux mémoires : un peu meilleur sur les dépendances très longues, plus lent. Les mots de passe étant courts, le GRU suffit souvent, mais la comparaison fait partie de la partie III.

**Qu'est-ce qui change dans le code pour passer au LSTM ?**
`nn.GRU` devient `nn.LSTM`, et `init_hidden` renvoie un couple `(h, c)` au lieu d'un seul tenseur. `train` et `evaluate` ne changent pas, car ils font seulement circuler `hidden`.
