# BétonFiabilité

Application Streamlit pédagogique pour étudier la probabilité qu'une résistance individuelle à la compression du béton soit inférieure à un seuil caractéristique `fck`.

## Contenu du dossier

- `app.py` : interface, calculs statistiques, distributions, simulation Monte-Carlo, exports.
- `requirements.txt` : dépendances Python.

## Prérequis

- Python 3.10 ou plus récent (Python 3.11 recommandé).
- Windows 10/11, macOS ou Linux.
- Connexion Internet pour installer les bibliothèques la première fois.

## Installation sous Windows avec VS Code

1. Installer Python depuis https://www.python.org/downloads/ et cocher **Add Python to PATH**.
2. Installer Visual Studio Code.
3. Décompresser le dossier `BetonFiabilite`.
4. Dans VS Code, choisir **File > Open Folder** et ouvrir le dossier contenant `app.py`.
5. Ouvrir le terminal intégré : **Terminal > New Terminal**.
6. Créer un environnement virtuel :

   ```powershell
   py -m venv .venv
   ```

7. L'activer :

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   Si PowerShell bloque l'activation, utiliser temporairement :

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

8. Installer les bibliothèques :

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

9. Lancer l'application :

   ```powershell
   python -m streamlit run app.py
   ```

10. Le navigateur s'ouvrira normalement sur `http://localhost:8501`.

## Utilisation rapide

1. Dans le menu latéral, cliquer sur **Charger l'exemple fictif** pour découvrir les fonctionnalités.
2. Dans **Données expérimentales**, saisir les résistances (MPa) séparées par `;` ou importer un CSV/Excel.
3. Définir le seuil `fck` et les informations du béton.
4. Consulter **Analyse statistique**.
5. Dans **Probabilité de non-atteinte**, choisir le modèle normal, lognormal ou empirique.
6. Dans **Simulation Monte-Carlo**, régler `μ`, `σ`, `fck`, le nombre de simulations et la graine.
7. Comparer plusieurs formulations dans **Comparaison des bétons**.
8. Générer les rapports PDF, CSV et Excel dans **Rapports et exportations**.

## Calculs effectués

### Résistance à la compression

Pour une charge de rupture `F` exprimée en kN et une section chargée `A` en mm² :

`fc = (F × 1000) / A`

Le résultat est en MPa, car `1 MPa = 1 N/mm²`.

Pour un cylindre de diamètre `d` : `A = π d² / 4`.

Pour un cube de côté `a` : `A = a²`.

### Statistiques

- Moyenne arithmétique.
- Variance et écart-type d'échantillon (division par `n − 1`).
- Coefficient de variation `CV = s / moyenne × 100 %`.
- Minimum, maximum, médiane, quartiles.
- Proportion empirique des résultats strictement inférieurs à `fck`.

### Probabilité

Pour une loi normale `X ~ N(μ, σ²)` :

`P(X < fck) = Φ((fck − μ) / σ)`.

Pour la loi lognormale, l'application convertit la moyenne et l'écart-type arithmétiques en paramètres logarithmiques.

La méthode empirique calcule le nombre de résultats observés strictement inférieurs au seuil divisé par l'effectif. Un intervalle de Wilson à 95 % est présenté.

### Monte-Carlo

La fréquence simulée est :

`p_chapeau = nombre de simulations avec X < fck / nombre total de simulations`.

L'erreur-type Monte-Carlo est `sqrt(p_chapeau × (1 − p_chapeau) / N)`. La graine rend la simulation reproductible.

## Stockage et protection des données

Les analyses enregistrées localement peuvent être rechargées ou supprimées depuis la rubrique Rapports et exportations. Pour modifier une analyse chargée, modifiez les données puis enregistrez un nouvel état. Elles sont conservées hors du dossier du projet, dans :

`%USERPROFILE%\.betonfiabilite\analyses.json`

Sur macOS/Linux, le dossier est `~/.betonfiabilite/`.

Les données de la session Streamlit ne sont pas une sauvegarde durable. Utiliser **Télécharger la sauvegarde JSON** pour conserver une copie exportable. Les données fictives sont identifiées et ne doivent pas être confondues avec les essais réels.

Sur un hébergement cloud, le disque local peut être temporaire : ne pas considérer ce stockage comme une base de données durable.

## Déploiement Streamlit Community Cloud

Streamlit Community Cloud attend le nom standard `requirements.txt`. Le projet fourni utilise déjà ce nom.

1. Créer un dépôt GitHub.
2. Ajouter `app.py`, `requirements.txt` et `README.md`.
3. Pousser les fichiers sur GitHub.
4. Dans Streamlit Community Cloud, créer une nouvelle application.
5. Sélectionner le dépôt, la branche et `app.py`.
6. Déployer puis consulter les journaux si une erreur survient.

Commandes Git usuelles :

```bash
git status
git add .
git commit -m "Initialisation de BetonFiabilite"
git push
```

Aucune URL publique n'est fournie tant qu'un déploiement réel n'a pas été effectué.

## Limites scientifiques importantes

- `P(X < fck)` décrit une résistance individuelle sous le modèle statistique choisi.
- Cette probabilité n'est pas automatiquement la probabilité de rejet d'une gâchée, d'un lot ou d'un ouvrage.
- La résistance caractéristique est généralement associée à un fractile inférieur de 5 % dans les référentiels concernés, mais sa définition opérationnelle dépend du référentiel et des conditions d'évaluation.
- L'indice `β = (μ − fck) / σ` est ici un indicateur statistique lié au seuil, pas un indice réglementaire de sécurité structurale.
- La loi normale peut attribuer une probabilité à des résistances négatives ; cette limite est signalée.
- Une simulation Monte-Carlo réduit l'erreur numérique de simulation, mais ne corrige pas un échantillon biaisé ou un modèle inadapté.
- Aucune norme camerounaise ni règle d'acceptation de lot n'est présumée dans cette version. Vérifier le référentiel applicable, le protocole d'essai, l'âge et la géométrie des éprouvettes avant toute conclusion technique.

## Vérifications réalisées

Le fichier `app.py` est contrôlé syntaxiquement avant livraison. Les fonctions de calcul principales sont vérifiées sur des cas numériques de référence dans le processus de génération. Pour une utilisation réelle, il reste nécessaire de tester l'interface dans un environnement où les dépendances sont installées et de confronter les résultats à des données de laboratoire et au référentiel applicable.
