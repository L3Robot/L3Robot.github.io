Title: Sur l'utilisation de la moyenne de buts.
Date: 2021-02-08 21:29
Category: hockey

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<style>
.center {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 50%;
}
.no_after::after {
    content: none !important;
}
</style>

Quand je vois ce genre de tweet, ça me fait sursauter.

<a class="no_after" style="display: block;" href="https://twitter.com/mirtle/status/1358236697876697089"><img class="center" src="{static}/images/naive_avg_goals/james_mirtle_claim.png" alt="James Mirtle Claim"></a>

Bon, ne soyons pas de mauvaise fois, James Mirtle a le droit de s'emporter après une excellente performance de son joueur vedette, mais est-ce que cette prédiction se tient?

# Prédiction naïve: par la moyenne

Commençons tout d'abord par expliquer comment certains commentateurs sportifs arrivent à faire cette projection. Celle-ci se base sur la moyenne de buts d'un joueur depuis le début de la saison \\(\bar{g}\\). On estime ensuite que le joueur continue avec cette même cadence jusqu'à la fin de la saison. Pour obtenir la prédiction finale, on multiplie la moyenne de buts par le nombre de matchs restant à la saison \\(r\\) et on ajoute le nombre de buts comptés jusqu'à maintenant \\(g\\):

$$p_{naive}=g + \bar g \cdot r $$

Le 6 février, Auston Matthews avait compté \\(g=10\\) buts en 11 matchs, ce qui est impressionnant et ce qui équivaut à une moyenne de \\(\bar g=0.909\\) buts par matchs. À cette date, il restait \\(r=45\\) matchs à jouer dans la saison, ce qui donne une prédiction de 51.905 buts. Puisqu'il est, dieu soit loué, impossible de compter des fractions de buts, on arrondi à la baisse le nombre de buts et on arrive à 51, exactement la prédiction de Mirtle.

# Problème avec la moyenne

Le problème avec la moyenne, c'est qu'elle est fortement influencée par les débuts de saison exceptionnelle du joueur. Regardons en premier la distribution du nombre de buts par match de Auston Matthews pour ses 4 dernières saisons:

<img class="center" style="width: 100%" src="{static}/images/naive_avg_goals/auston_matthews_past_season_goal_dist.png" alt="Auston Matthews past goals distribution">

S'il est vrai qu'on y voit une certaine amélioration dans sa production, son nombre de matchs à vide diminue avec le temps, sa production reste relativement stable. Maintenant, si on regarde la distribution du nombre de buts pour les 11 premiers matchs lors de la saison 2020-2021:

<img class="center" style="width: 80%" src="{static}/images/naive_avg_goals/auston_matthews_11games_goal_dist.png" alt="Auston Matthews 11 games goals distribution">

On voit tout de suite une surreprésentation de matchs avec 1 but comparativement aux années passées. Si Matthews conservait cette distribution tout au long de la saison, cela représenterait une amélioration extraordinaire comparée aux années passées. D'ailleurs, la dernière fois que Matthews avait autant de buts projetés en début de saison, c'était à la saison 2018-2019, voilà ce qui est arrivé:

<img class="center" style="width: 90%" src="{static}/images/naive_avg_goals/auston_matthews_naive_prediction.png" alt="Auston Matthews 11 games goals distribution">

Non seulement la prédiction par la moyenne surestime le nombre de buts total de Matthews tout au long de la saison, mais il faut attendre mi-janvier pour que la moyenne se stabilise.

Bref, il faut toujours avoir des réserves quand on utilise la moyenne pour faire une projection. En effet, il faut beaucoup de données pour que celle-ci soit significative. Mais est-ce qu'il est possible de faire mieux?

# Prédiction bayésienne

Il serait intéressant et plus réaliste d'incorporer l'information disponible a priori sur Matthews pour calculer une projection de buts pour cette année. Il est tout aussi important de mettre à jour le modèle avec les nouvelles informations disponibles. C'est ce qui permet de faire la prédiction bayésienne. Le [théorème de Bayes](https://www.wikiwand.com/fr/Th%C3%A9or%C3%A8me_de_Bayes) permet de raffiner l'information disponible a priori avec les nouvelles informations:

$$P(A|B) \propto P(B|A)P(A)$$

Dans le cas étudié ici, il est considéré que \\(P(B|A)\\) représente la probabilité que Matthews compte \\(B\\) buts dans un match sachant que cette distribution suit l'histogramme A (plus exactement, la [loi multinoulli](https://www.wikiwand.com/en/Categorical_distribution) A) et \\(P(A)\\) représente la distribution a priori sur l'histogramme A modélisé par une [loi de Dirichlet](https://www.wikiwand.com/fr/Loi_de_Dirichlet).

La loi de Dirichlet est paramétrisé par autant de \\(\alpha\\) qu'il y a de catégories dans la loi multinoulli correspondante. Il est posé arbitrairement que le maximum théorique soit de 5 buts par match, ce qui fait 6 catégories et 6 \\(\alpha\\) avec la possibilité de ne compter aucun but. La loi de Dirichlet \\(\mathcal{D}\\) a la propriété intéressante de représenter l'incertitude la distribution des catégories lorsque les \\(\alpha\\) représentent les comptes de chaque catégorie. De plus, la probabilité a posteriori \\(P(A|B)\\) suit tout simplement:

$$P(A|B)\sim\mathcal{D}(\alpha_1+\beta_1, ..., \alpha_n+\beta_n)$$

Où \\(\beta\\) représente les nouveaux comptes des nouvelles observations. Il est donc trivial d'obtenir la nouvelle distribution sur les catégories.

## Résultats

Si l'on reproduit la dernière figure, mais cette fois-ci avec la moyenne de la loi de Dirichlet mis à jour, on obtient le résultat suivant:

<img class="center" style="width: 90%" src="{static}/images/naive_avg_goals/auston_matthews_bayesian_prediction.png" alt="Auston Matthews 11 games goals distribution bayesian">

Même si le modèle surestime toujours le nombre de buts, la prédiction est beaucoup plus conservatrice et l'erreur est beaucoup plus petite: 7 buts comparativement à 24 avec la prédiction naïve. La prédiction est également beaucoup plus stable dans le temps puisqu'elle se base sur un a priori fort basé sur la dernière saison.

# Prédiction bayésienne avec Monte-Carlo

Finalement, puisque le modèle produit une distribution sur les histogrammes, il est possible d'obtenir une valeur de l'incertitude du modèle. À la place d'assumer que la moyenne de la Dirichlet représente la moyenne de but de Matthews jusqu'à la fin de la saison, il est possible de simuler plusieurs trajectoires possibles. Pour chaque trajectoire, un loi multinoulli A est d'abord tiré avec la Dirichlet et la saison est complétée en pigeant un nombre de buts par match B avec \\(P(B|A)\\) pour chaque match restant. La procédure est répétée 10000 fois. La figure suivante montre toutes les trajectoires et la moyenne de celle-ci en rouge.

<img class="center" style="width: 90%" src="{static}/images/naive_avg_goals/auston_matthews_monte_carlo_simulation.png" alt="Auston Matthews monte carlo simulations">

Il est possible d'y voir que l'incertitude augmente avec le temps et que la moyenne se situe autour de 40 buts. Finalement, on peut calculer la proportion de trajectoires qui se situe dans chaque intervalle de 10 buts pour fournir une probabilité sur la prédiction avec le modèle actuel:

<img class="center" style="width: 90%" src="{static}/images/naive_avg_goals/auston_matthews_monte_carlo_probs.png" alt="Auston Matthews monte carlo probs">

La figure montre que Matthews a plus de 50% de chance de compter entre \\([40, 50[\\) buts, mais seulement 10% de chance de dépasser 50 buts.

# Conclusion

Il est tentant d'utiliser la moyenne de buts d'un joueur pour projeter son nombre de buts total dans la saison, mais il faut faire preuve de prudence. Premièrement, ce modèle naïf ne prend pas en compte l'information déjà disponible sur le joueur. Deuxièmement, la faible quantité de données produit un estimateur avec une trop forte incertitude, incertitude qui est souvent absente lors de la publication de la projection. L'utilisation d'un modèle bayésien couplé de simulations Monte-Carlo offre une solution à ces deux problèmes et produit ainsi un meilleur estimateur.

Matthews est certainement un bon joueur, il va sûrement dépasser les 40 buts cette saison, ce qui est un exploit, mais 50 buts, c'est très peu probable.
