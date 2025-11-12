#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 00:16:59 2024

@author: fatineamezziane
"""

#AMEZZIANE FATINE ET SACI DOUAA L2 MATHS
#PROJET SUR UNE BASE DE DONNÉES COMPORTANT DES INFOS SUR LES SALARIÉS TRAVAILLANT DANS LA DATA SCIENCE

#IMPORTATION DE TOUTES LES LIBRAIRIES NÉCESSAIRES POUR LE PROJET ET VUES EN COURS              
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import  r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

#PARTIE I : ANALYSE DE DONNÉES

#1: Préparation des données
#Importation des données et infos de base afin de comprendre notre base de données
datascience = pd.read_csv('ds_salaries.csv')
# Affichage des premières lignes de la base de données
print('DATASCIENCE PAS ENCORE NETTOYÉ', datascience.head())
# Affichage des noms des colonnes
print(datascience.columns)
# Affichage des informations sur la base de données (nombre de lignes, de colonnes, types de données, etc.)
print(datascience.info())
# Affichage des statistiques descriptives pour les colonnes numériques
print(datascience.describe())
# Affichage du nombre de données dans la base de données
print(len(datascience))
# Affichage de la forme (shape) de la base de données (nombre de lignes, nombre de colonnes)
print(datascience.shape)
# Affichage de la taille (nombre total d'éléments) de la base de données
print(datascience.size)

# Nettoyage de la base de données
# Vérifier les valeurs manquantes du tableau
val_manquantes = {}
for colonne in datascience.columns:
    nb_manquantes = datascience[colonne].isnull().sum()       
    val_manquantes[colonne] = nb_manquantes
# Affichage du nombre de valeurs manquantes pour chaque colonne
print(val_manquantes)
# Condition dans le cas où le nombre de valeurs manquantes par colonne dépasse 80%
prb_colonnes = []
for colonne, nb_manquantes in val_manquantes.items():
    if nb_manquantes > 0.8 * len(datascience):
        prb_colonnes.append(colonne)
# Afficher les colonnes problématiques
print("Colonnes problématiques en raison de valeurs manquantes : ", prb_colonnes)


# On fait en sorte que la colonne 'work_year' soit au bon format
datascience['work_year'] = pd.to_datetime(datascience['work_year'], errors='coerce', format='%Y')
# (i) Remplacer les valeurs manquantes par la moyenne (exemple pour une colonne numérique)
datascience['salary'] = datascience['salary'].fillna(datascience['salary'].mean())
datascience['salary_in_usd'] = datascience['salary_in_usd'].fillna(datascience['salary_in_usd'].mean())
# (ii) Remplacer les valeurs manquantes par le maximum ou le minimum (pour les colonnes numériques)
datascience['remote_ratio'] = datascience['remote_ratio'].fillna(datascience['remote_ratio'].min())
# (iii) Remplacer les valeurs manquantes par le mot « NULL » (exemple pour une colonne de texte)
datascience['job_title'] = datascience['job_title'].fillna('NULL')
datascience['employment_type'] = datascience['employment_type'].fillna('NULL')
datascience['employee_residence'] = datascience['employee_residence'].fillna('NULL')
datascience['company_location'] = datascience['company_location'].fillna('NULL')
# (iv) Remplacer par la valeur qui apparaît le plus souvent
datascience['experience_level'] = datascience['experience_level'].fillna(datascience['experience_level'].mode()[0])
datascience['salary_currency'] = datascience['salary_currency'].fillna(datascience['salary_currency'].mode()[0])
datascience['company_size'] = datascience['company_size'].fillna(datascience['company_size'].mode()[0])

# Création et affichage du DataFrame nettoyé
datascience2 = datascience.drop(prb_colonnes, axis=1)
print(datascience2.head(10))

## 2 : ANALYSE DES DONNÉES
#INFOS DU DATAFRAME NETTOYÉ
print('datascience2 NETTOYÉ : colonnes' , datascience2.columns)
print(datascience2.info())
print(datascience2.describe())
print(len(datascience2))
print(datascience2.shape)
print(datascience2.size)

# Affichage des différentes possibilités de réponse pour chaque colonne 
# but : déterminer les boxplots et histogrammes les plus pertinents en fonction des réponses
for colonne in datascience.columns:
    print("Colonne :", colonne)
    print("Nombre de possibilités :", len(datascience2[colonne].unique()))
    print(datascience2[colonne].unique())   
    print()


# Boxplot : utile pour la colonne "salary_in_usd" --> salaires de tous pays exprimés dans une meme monnaie
# Création d'un boxplot pour la colonne "salary_in_usd"
plt.figure(figsize=(10, 7))
plt.boxplot(datascience2['salary_in_usd'])
plt.title('Boxplot : Salaires en dollars US des salariés en data science' )
plt.ylabel('Salaire (USD)')
plt.show()

# Calcul des moustaches haute et moustache basse
Q1 = datascience2['salary_in_usd'].quantile(0.25)
Q3 = datascience2['salary_in_usd'].quantile(0.75)
IQR = Q3 - Q1
moustachebasse = Q1 - 1.5 * IQR
moustachehaute = Q3 + 1.5 * IQR
print('Moustache basse : ', moustachebasse)
print('1er quartile : ',Q1)
print('3e quartile :', Q3)
print('IQR :', IQR)
print('Moustache haute : ', moustachehaute)


# Compter le nombre de valeurs aberrantes du boxplot
nombre_valeurs_aberrantes = 0
for valeur in datascience2['salary_in_usd']:
    if valeur < moustachebasse or valeur > moustachehaute:
        nombre_valeurs_aberrantes += 1

print("Nombre de valeurs aberrantes :", nombre_valeurs_aberrantes)
# 63 val aberrantes, nombre relaivement faible car il y a 3755 données par colonne

# Seule corrélation numérique "pour le moment" : celle entre "salary" et "salary_in_usd"
correlation1 = datascience2['salary'].corr(datascience2['salary_in_usd'])
print("Corrélation entre 'salary' et 'salary_in_usd' :", correlation1)
# Le coef affiché : -0.023675813981249263


# On aimerait calculer une "corrélation" entre "employee_residence" et "company_location" mais ce sont des valeurs non numériques
# Importation de la bibliothèque "LabelEncoder" présente dans la bibliothèque "sklearn" deja rencontrée auparavant
# Convertir les variables catégorielles en variables numériques
label_encoder = LabelEncoder()
datascience2['employee_residence_encoded'] = label_encoder.fit_transform(datascience2['employee_residence'])
datascience2['company_location_encoded'] = label_encoder.fit_transform(datascience2['company_location'])
correlation2 = np.corrcoef(datascience2['employee_residence_encoded'], datascience2['company_location_encoded'])[0, 1]
print("Approximation de la corrélation entre 'employee_residence' et 'company_location' :", correlation2)
# Corrélation : 0.9456758551105585 

corr1= -0.023675813981249263
corr2= 0.9456758551105585

########
# PLOT pour mettre en avant ces 2 corrélations calculées
fig, ax = plt.subplots(figsize=(10, 10))

plt.title('Calcul de corrélations pertinentes pour ce dataset', fontsize=30, color='red')
# Ajout des lignes de texte
text_lines = [
    "° Seule corrélation numérique calculable selon nous : celle entre 'salary' et 'salary_in_usd'",
    f"° Corrélation entre 'salary' et 'salary_in_usd' : {corr1:.2f}",
    "° INTERPRETATION : Le coef affiché est négatif et faible donc presque insignifiant",
    "° On a remarqué un fort lien entre le lieu de résidence d'un salarié et le lieu de son entreprise",
    f"° Corrélation entre 'employee_residence' et 'company_location' : {corr2:.2f}",
    "° Très proche de 1 donc corrélation très élevée entre les 2, lien très étroit entre ces 2.",
    "° INTERPRETATION : lorsqu'une personne réside dans un pays il est fort probable que l'entreprise où elle travaille soit dans ce meme pays",
    "° Néanmoins cette corrélation n'est pas égale à 1 donc il existe des employés ne travaillant pas dans le meme pays de résidence"
]

# Positionnement du texte
for i, line in enumerate(text_lines):
    ax.text(0.1, 0.9 - i * 0.1, line, ha='left', va='center', color='black', fontsize=16)

ax.axis('off')
plt.show()


## 3 : EXTRACTION DES CONNAISSANCES

# 1ere étude : création d'un PIE CHART en ayant observé le boxplot, nous souhaitons observer la répartition de la masse salariale totale entre les 4 quartiles
# On souhaite montrer par ex que le 4e quartile regroupe à lui seul une grande partie de la masse salariale totale
# Objectif : mettre en avant l'écart qu'il peut y avoir entre les différents métiers (et leur salaire associé) de cette base de données

Q1 = datascience2['salary_in_usd'].quantile(0.25)
median = datascience2['salary_in_usd'].quantile(0.5)
Q3 = datascience2['salary_in_usd'].quantile(0.75)
# Calcul des salaires dans chaque quartile
Q1_salaries = datascience2[datascience2['salary_in_usd'] <= Q1]['salary_in_usd'].sum()
Q2_salaries = datascience2[(datascience2['salary_in_usd'] > Q1) & (datascience2['salary_in_usd'] <= median)]['salary_in_usd'].sum()
Q3_salaries = datascience2[(datascience2['salary_in_usd'] > median) & (datascience2['salary_in_usd'] <= Q3)]['salary_in_usd'].sum()
Q4_salaries = datascience2[datascience2['salary_in_usd'] > Q3]['salary_in_usd'].sum()
# Présentation du PIE CHART pour la répartition de la masse salariale
labels = ['1er quartile', '2e quartile', '3e quartile', '4e quartile']
sizes = [Q1_salaries, Q2_salaries, Q3_salaries, Q4_salaries]
colors = ['orange','lightblue','lightgreen','red']
plt.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Répartition de la masse salariale totale dans les quartiles')
plt.axis('equal') #pour que le pie chart soit bien un cercle

# INTERPRETATION : le 4e quartile regroupe à lui seul presque 40% de la masse salariale totale
# On remarque très bien l'écart qu'il peut y avoir entre les différents métiers (et leur salaire associé) de cette base de données

# Il est intéressant de s'interesser maintenant aux métiers présent dans le 4e quartile
# Ce sont par def les gens les plus aisés et le 4e quartile comprend les 63 valeurs abérrantes
# Nous nous intéressons à présent à cela


#2eme étude
#Etape 1 : extraction des informations des 63 plus riches du tableau
# Trier le DataFrame par ordre décroissant de la colonne 'salary_in_usd'
datascience_sorted = datascience2.sort_values(by='salary_in_usd', ascending=False)
# Extraire les 63 premières lignes
top_63_rich = datascience_sorted.head(63)
# Afficher les lignes correspondantes
print(top_63_rich.head())

#Remarque : il y a 13 colonnes dans ce tableau du au calcul de la "corrélation"entre "company_location" et "employee_residence"

#Etape 2 : voir les métiers différents de ces 63 ainsi que leur occurence dans cette liste
# Compter le nombre de réponses différentes dans la colonne "job_title"
occurrences_par_metier = top_63_rich['job_title'].value_counts()  
print("Colonne : job_title")
print("Nombre de possibilités :", len(top_63_rich['job_title'].unique()))
print(top_63_rich['job_title'].value_counts())
print()

#Etape 3 : création d'un HISTOGRAMME pour bien illustrer tout ca

# Regroupement des métiers avec une seule occurrence sous la catégorie "Autres"
occurrences_par_metier['Autres'] = occurrences_par_metier[occurrences_par_metier == 1].sum()
occurrences_par_metier = occurrences_par_metier[occurrences_par_metier > 1]

plt.figure(figsize=(10, 6))
occurrences_par_metier.plot(kind='bar', color='turquoise')
plt.title('Occurrences des métiers chez les 63 plus riches (= les 63 valeurs aberrantes)')
plt.xlabel('Métier')
plt.ylabel('Occurrences')
plt.xticks(rotation=30, ha='right')
plt.show()

# On remarque donc que parmi les 63 plus aisés, 16 sont Data Engineer --> 25,4% donc part importante

#À présent on aimerait comparer le salaire moyen de chaque métier dans ce top 63 avec le salaire moyen de ce meme métier dans le tableau global hors 63 premiers
#Calcul du salaire moyen par métier dans les 63 premiers métiers
salaire_moyen_par_metier_63 = top_63_rich.groupby('job_title')['salary_in_usd'].mean()
# Création d'un nouveau tableau sans les 63 salaires les plus élevés
datascience2_sans_top63 = datascience2[~datascience2.index.isin(top_63_rich.index)]
# Calcul du salaire moyen global par métier dans le nouveau tableau
salaire_moyen_par_metier_nouveau = datascience2_sans_top63.groupby('job_title')['salary_in_usd'].mean()

# Comparaison des salaires moyens par métier
for metier in salaire_moyen_par_metier_63.index:
    salaire_moyen_63 = salaire_moyen_par_metier_63[metier]
    salaire_moyen_nouveau = salaire_moyen_par_metier_nouveau.get(metier, 0)
    if salaire_moyen_nouveau == 0:
        print(f"Le métier '{metier}' n'est pas présent dans le nouveau tableau.")
    else:
        print(f"Salaire moyen pour le métier '{metier}' dans les 63 premiers métiers : {salaire_moyen_63}")
        print(f"Salaire moyen global pour le métier '{metier}' dans le nouveau tableau : {salaire_moyen_nouveau}")
        print("Comparaison :")
        if salaire_moyen_63 > salaire_moyen_nouveau:
            print("Le salaire moyen dans les 63 premiers métiers est supérieur au salaire moyen global dans le nouveau tableau.")
        elif salaire_moyen_63 < salaire_moyen_nouveau:
            print("Le salaire moyen dans les 63 premiers métiers est inférieur au salaire moyen global dans le nouveau tableau.")
        else:
            print("Le salaire moyen dans les 63 premiers métiers est égal au salaire moyen global dans le nouveau tableau.")
        print()


# Les résultats sont affichés dans la console, on utilise ces résultats pour créer un histogramme pour l'esthetique
# Saisie longue des 20 résultats (car 20 métiers dans les 63)

metiers = ['AI Developer', 'AI Scientist', 'Applied Data Scientist', 'Applied Machine Learning Scientist',
           'Applied Scientist', 'Computer Vision Engineer', 'Data Analyst', 'Data Analytics Lead', 'Data Architect',
           'Data Engineer', 'Data Science Manager', 'Data Scientist', 'Director of Data Science', 'Head of Data',
           'Head of Data Science', 'Machine Learning Engineer', 'Machine Learning Software Engineer',
           'Principal Data Scientist', 'Research Scientist']
salaires_63 = [300000.0, 423834.0, 380000.0, 423000.0, 322933.3333333333, 342810.0, 407983.5, 405000.0, 360840.0,
               306656.25, 299257.14285714284, 324145.0, 339100.0, 329500.0, 314100.0, 317700.0, 375000.0, 416000.0,
               365000.0]
salaires_globaux = [120332.7, 89206.66666666667, 84140.33333333333, 80948.54545454546, 183028.0, 131814.11764705883,
                    107735.02295081968, 17509.0, 157691.0202020202, 140225.2578125, 176458.21568627452,
                    138661.63012048192, 163149.77777777778, 167675.0, 141403.125, 151234.6996466431,
                    172133.33333333334, 167052.7142857143, 150763.64102564103]

# Création de l'hiqtogramme
fig, ax = plt.subplots(figsize=(10, 8))
largeur = 0.35
indices = range(len(metiers))
rects1 = ax.barh(indices, salaires_63, largeur, label='Salaires moyens pour les 63 plus riches')
rects2 = ax.barh([i + largeur for i in indices], salaires_globaux, largeur, label='Salaires moyens sans les 63')
ax.set_ylabel('Métiers')
ax.set_xlabel('Salaire moyen en USD')
ax.set_title('Comparaison des salaires moyens par métier')
ax.set_yticks([i + largeur / 2 for i in indices])
ax.set_yticklabels(metiers)
ax.legend()
plt.show()

# On constate qu'il y a un énorme écart pour chaque métier, cela est presque anormal
# Les 63 plus riches gagnent nettement plus que la moyenne
#  Après une grande étude de ces valeurs aberrantes nous allons nous concentrer maintenant sur le nouveau tableau sans les 63 valeurs aberrantes

#PARTIE 2 : VISUALISATION DES DONNÉES

nouveau_tableau = datascience2_sans_top63
print('NOUVEAU TABLEAU SANS 63 VALEURS ABERRANTES' , nouveau_tableau.head())   # 3692 rows : le tableau est bien celui sans les 63 val ab

# Création d'un nouveau boxplot pour la colonne "salary_in_usd"
plt.figure(figsize=(10, 7))
plt.boxplot(nouveau_tableau['salary_in_usd'])
plt.title('Boxplot du salaire en USD sans les 63 plus riches')
plt.ylabel('Salaire (USD)')
plt.show()
# Plus aucune valeur aberrante dans le boxplot !
# Calcul Q1, Q2, Q3, Q4
Q1 = nouveau_tableau['salary_in_usd'].quantile(0.25)
Q3 = nouveau_tableau['salary_in_usd'].quantile(0.75)
Q2 = nouveau_tableau['salary_in_usd'].quantile(0.5)
Q4 = nouveau_tableau['salary_in_usd'].quantile(1.0)
print('1er quartile : ',Q1)   #1er quartile :  94916.25
print('3e quartile :', Q3)    #3e quartile : 174500.0
print('2e quartile :', Q2)    #2e quartile : 133916.0  --> le salaire median est donc de 133916 dollars par an
print('4e quartile :', Q4)    #4e quartile : 293000.0


# Sélectionner les données pour chaque expérience_level
SE_data = nouveau_tableau[nouveau_tableau['experience_level'] == 'SE']['salary_in_usd']
EX_data = nouveau_tableau[nouveau_tableau['experience_level'] == 'EX']['salary_in_usd']
MI_data = nouveau_tableau[nouveau_tableau['experience_level'] == 'MI']['salary_in_usd']
EN_data = nouveau_tableau[nouveau_tableau['experience_level'] == 'EN']['salary_in_usd']


data_experience_level = [SE_data, EX_data, MI_data, EN_data]    # liste de données pour les boxplots
fig = plt.figure(figsize=(10, 7))
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(data_experience_level, labels=['SENIOR', 'EXEXUTIVE(Cadres)', 'MID-LEVEL(Intermediaire)', 'ENTRY(Débutant)'])
plt.title('Boxplots des salaires en USD en fonction du niveau d\'experience')
plt.xlabel('Niveau d\'experience')
plt.ylabel('Salaires')
plt.show()

# Initialiser une liste pour stocker les quartiles de chaque expérience_level
quartiles = []
# Pour chaque experience_level, calculer les quartiles
for data in data_experience_level:
    Q1 = data.quantile(0.25)
    Q2 = data.quantile(0.5)
    Q3 = data.quantile(0.75)
    Q4 = data.quantile(1.0)
    quartiles.append((Q1, Q2, Q3, Q4))
# Afficher les quartiles pour chaque expérience_level
for i, (Q1, Q2, Q3, Q4) in enumerate(quartiles):
    print(f"Quartiles pour l'expérience_level {['SE', 'EX', 'MI', 'EN'][i]} :")
    print(f"Q1 : {Q1}")
    print(f"Q2 : {Q2}")
    print(f"Q3 : {Q3}")
    print(f"Q4 : {Q4}")
    print()

# Nous avons nos boxplots et les valeurs de nos quartiles
# EN=Entry(Débutant) MI=Mid(Intermediaire)  SE=Senior EX=Executive(Cadre sup ou de Direction)
# La boite des 'EX' est la plus haute et plus haute que les 'SE'
# La boite des 'EN' est la plus basse
# Q3_EX : 217000.0 > Q3_SE : 185000.0 > Q3_MI : 135000.0 > Q3_EN : 110000.0
# INTEPRETATION : les salaires sont plus hauts lorsqu'on est cadre supérieur ou de direction.
# Les salaires sont plus hauts aussi pour ceux qui ont plus d'expérience(longévité)
# Les seniors gagnent plus que les débutants en moyenne


#On aimerait connaitre le nombre de salariés dans chaque niveau d'experience et le représenter sous forme d'histogramme
occurrences_par_experience = nouveau_tableau['experience_level'].value_counts()
plt.figure(figsize=(10, 6))
occurrences_par_experience.plot(kind='bar', color='grey')
plt.title('Nombre de salariés de chaque niveau d\'expérience')
plt.xlabel('Niveau d\'expérience')
plt.ylabel('Effectif')
plt.yticks(np.arange(0, max(occurrences_par_experience)+100, 100)) #ajuster les graduations des y pour la précision
xticks_labels = ['SENIOR','MID-LEVEL(Intermediaire)', 'ENTRY(Débutant)', 'EXEXUTIVE(Cadres)' ]
plt.xticks(range(0, 4), xticks_labels, rotation=0)
plt.show()

# Seulement 100 salariés cadres, peu mais ce sont eux les plus riches
# On sait aussi que les cadres ont un role essentiel dans chaque entreprise
# Neanmoins on a vu dans les boxplots que les salariés à partir du Q3 sont tous >100k$ et moustaches hautes >200k$


# Création de 4 sous-tableaux pour faciliter l'étude et le subplot
tableauEX = nouveau_tableau[nouveau_tableau['experience_level'] == 'EX']
tableauSE = nouveau_tableau[nouveau_tableau['experience_level'] == 'SE']
tableauEN = nouveau_tableau[nouveau_tableau['experience_level'] == 'EN']
tableauMI = nouveau_tableau[nouveau_tableau['experience_level'] == 'MI']

# On souhaite créer un subplot présentant une répartition des salaires pour chaque niveau d'experience
# Or on veut s'axer sur le salaire moyen du "nouveau tableau" et diviser la repartition en 4 parties avec cette donnée
salaire_min = nouveau_tableau['salary_in_usd'].min()
salaire_max = nouveau_tableau['salary_in_usd'].max()
salaire_moyen = nouveau_tableau['salary_in_usd'].mean()
partie_basse = 0.5*(salaire_moyen)
partie_haute = 1.5*(salaire_moyen)
print('salaire min :', salaire_min)
print('salaire max :', salaire_max)
print('salaire moyen:', salaire_moyen)
print('Partie basse :', partie_basse)
print('Partie haute :', partie_haute)

# Création du subplot avec 4 histogrammes
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
bin_edges = [5132, 67131.49, 134262.99, 201394.49, 293000]   # correspondent aux vals des 5 calculs précédents
# Plot histogramme pour experience level 'EX'
ax = axes[0, 0]
ax.hist(tableauEX['salary_in_usd'], bins=bin_edges, color='skyblue', edgecolor='grey')
ax.set_title('Répartition des salaires chez les cadres("EX")')
ax.set_xlabel('Salaire (USD)')
ax.set_ylabel('Effectif')
ax.set_xticks([5132, 67131.49, 134262.99, 201394.49, 293000])  
# Plot histogramme pour experience level 'SE'
ax = axes[0, 1]
ax.hist(tableauSE['salary_in_usd'], bins=bin_edges, color='yellow', edgecolor='grey')
ax.set_title('Répartition des salaires chez les seniors("SE")')
ax.set_xlabel('Salaire (USD)')
ax.set_ylabel('Effectif')
ax.set_xticks([5132, 67131.49, 134262.99, 201394.49, 293000])
# Plot histogramme pour experience level 'MI'
ax = axes[1, 0]
ax.hist(tableauMI['salary_in_usd'], bins=bin_edges, color='red', edgecolor='grey')
ax.set_title('Répartition des salaires chez les intermediaires("MI")')
ax.set_xlabel('Salaire (USD)')
ax.set_ylabel('Effectif')
ax.set_xticks([5132, 67131.49, 134262.99, 201394.49, 293000])
# Plot histogramme pour experience level 'EN'
ax = axes[1, 1]
ax.hist(tableauEN['salary_in_usd'], bins=bin_edges, color='green', edgecolor='grey')
ax.set_title('Répartition des salaires chez les débutants ("EN")')
ax.set_xlabel('Salaire (USD)')
ax.set_ylabel('Effectif')
ax.set_xticks([5132, 67131.49, 134262.99, 201394.49, 293000])
plt.tight_layout()
plt.show()

# INTERPRETATION
#
#
#
#

# On veut à présent faire des subplots en fonction des 4 années
# Cela pour voir si au fil des années la proportion de "bons salaires" augmentent ou non
# INTERPRETATiON : augmentation --> jobs attractifs
# On ne le fait pas pour EX car comporte que 100 lignes, c'est peu

#Nouveau subplot, repartition des SE en fonction des années
SE2020 = tableauSE[tableauSE['work_year'] == '2020-01-01T00:00:00.000000000']
SE2021 = tableauSE[tableauSE['work_year'] == '2021-01-01T00:00:00.000000000']
SE2022 = tableauSE[tableauSE['work_year'] == '2022-01-01T00:00:00.000000000']
SE2023 = tableauSE[tableauSE['work_year'] == '2023-01-01T00:00:00.000000000']

# Création du subplot avec 4 histogrammes
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
bin_edges = [5132, 67131.49, 134262.99, 201394.49, 293000]  
# Plot histogramme pour experience level 'SE' en fonction des années
ax = axes[0, 0]
ax.hist(SE2020['salary_in_usd'], bins=bin_edges, color='Gold', edgecolor='grey')
ax.set_title('Répartition des salaires "SE" en 2020')
ax.set_xlabel('Salaire (USD)')
ax.set_ylabel('Effectif')
ax.set_xticks([5132, 67131.49, 134262.99, 201394.49, 293000])  

ax = axes[0, 1]
ax.hist(SE2021['salary_in_usd'], bins=bin_edges, color='Goldenrod', edgecolor='grey')
ax.set_title('Répartition des salaires "SE" en 2021')
ax.set_xlabel('Salaire (USD)')
ax.set_ylabel('Effectif')
ax.set_xticks([5132, 67131.49, 134262.99, 201394.49, 293000])  

ax = axes[1, 0]
ax.hist(SE2022['salary_in_usd'], bins=bin_edges, color='PaleGoldenrod', edgecolor='grey')
ax.set_title('Répartition des salaires "SE" en 2022')
ax.set_xlabel('Salaire (USD)')
ax.set_ylabel('Effectif')
ax.set_xticks([5132, 67131.49, 134262.99, 201394.49, 293000])  

ax = axes[1, 1]
ax.hist(SE2023['salary_in_usd'], bins=bin_edges, color='Orange', edgecolor='grey')
ax.set_title('Répartition des salaires "SE" en 2023')
ax.set_xlabel('Salaire (USD)')
ax.set_ylabel('Effectif')
ax.set_xticks([5132, 67131.49, 134262.99, 201394.49, 293000])  

plt.tight_layout()
plt.show()


# Nouveau subplot, repartition que des MI en fonction des années
MI2020 = tableauMI[tableauMI['work_year'] == '2020-01-01T00:00:00.000000000']
MI2021 = tableauMI[tableauMI['work_year'] == '2021-01-01T00:00:00.000000000']
MI2022 = tableauMI[tableauMI['work_year'] == '2022-01-01T00:00:00.000000000']
MI2023 = tableauMI[tableauMI['work_year'] == '2023-01-01T00:00:00.000000000']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
bin_edges = [5132, 67131.49, 134262.99, 201394.49, 293000]   

ax = axes[0, 0]
ax.hist(MI2020['salary_in_usd'], bins=bin_edges, color='Darkred', edgecolor='grey')
ax.set_title('Répartition des salaires "MI" en 2020')
ax.set_xlabel('Salaire (USD)')
ax.set_ylabel('Effectif')
ax.set_xticks([5132, 67131.49, 134262.99, 201394.49, 293000])

ax = axes[0, 1]
ax.hist(MI2021['salary_in_usd'], bins=bin_edges, color='tomato', edgecolor='grey')
ax.set_title('Répartition des salaires "MI" en 2021')
ax.set_xlabel('Salaire (USD)')
ax.set_ylabel('Effectif')
ax.set_xticks([5132, 67131.49, 134262.99, 201394.49, 293000])

ax = axes[1, 0]
ax.hist(MI2022['salary_in_usd'], bins=bin_edges, color='indianred', edgecolor='grey')
ax.set_title('Répartition des salaires "MI" en 2022')
ax.set_xlabel('Salaire (USD)')
ax.set_ylabel('Effectif')
ax.set_xticks([5132, 67131.49, 134262.99, 201394.49, 293000])

ax = axes[1, 1]
ax.hist(MI2023['salary_in_usd'], bins=bin_edges, color='firebrick', edgecolor='grey')
ax.set_title('Répartition des salaires "MI" en 2023')
ax.set_xlabel('Salaire (USD)')
ax.set_ylabel('Effectif')
ax.set_xticks([5132, 67131.49, 134262.99, 201394.49, 293000])

plt.tight_layout()
plt.show()


# Nouveau subplot, répartition uniquement pour 'EN' en fonction des années
EN2020 = tableauEN[tableauEN['work_year'] == '2020-01-01T00:00:00.000000000']
EN2021 = tableauEN[tableauEN['work_year'] == '2021-01-01T00:00:00.000000000']
EN2022 = tableauEN[tableauEN['work_year'] == '2022-01-01T00:00:00.000000000']
EN2023 = tableauEN[tableauEN['work_year'] == '2023-01-01T00:00:00.000000000']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
bin_edges = [5132, 67131.49, 134262.99, 201394.49, 293000]  

ax = axes[0, 0]
ax.hist(EN2020['salary_in_usd'], bins=bin_edges, color='lime', edgecolor='grey')
ax.set_title('Répartition des salaires "EN" en 2020')
ax.set_xlabel('Salaire (USD)')
ax.set_ylabel('Effectif')
ax.set_xticks([5132, 67131.49, 134262.99, 201394.49, 293000])

ax = axes[0, 1]
ax.hist(EN2021['salary_in_usd'], bins=bin_edges, color='greenyellow', edgecolor='grey')
ax.set_title('Répartition des salaires "EN" en 2021')
ax.set_xlabel('Salaire (USD)')
ax.set_ylabel('Effectif')
ax.set_xticks([5132, 67131.49, 134262.99, 201394.49, 293000])

ax = axes[1, 0]
ax.hist(EN2022['salary_in_usd'], bins=bin_edges, color='limegreen', edgecolor='grey')
ax.set_title('Répartition des salaires "EN" en 2022')
ax.set_xlabel('Salaire (USD)')
ax.set_ylabel('Effectif')
ax.set_xticks([5132, 67131.49, 134262.99, 201394.49, 293000])

ax = axes[1, 1]
ax.hist(EN2023['salary_in_usd'], bins=bin_edges, color='forestgreen', edgecolor='grey')
ax.set_title('Répartition des salaires "EN" en 2023')
ax.set_xlabel('Salaire (USD)')
ax.set_ylabel('Effectif')
ax.set_xticks([5132, 67131.49, 134262.99, 201394.49, 293000])

plt.tight_layout()
plt.show()

# INTERPRETATION
#
#
#
#

#Focus sur les cadres ('EX'), on veut en savoir + sur eux
tableauEX = nouveau_tableau[nouveau_tableau['experience_level'] == 'EX']

# Leur lieu d'habitation avec la colonne employee_residence : PIE CHART
# Compter le nombre d'occurrences de chaque lieu de résidence
occurrences_par_habitation = tableauEX['employee_residence'].value_counts()
# Regroupement des lieux de résidence avec une seule occurrence sous la catégorie "Autres"
occurrences_par_habitation['Autres'] = occurrences_par_habitation[occurrences_par_habitation == 1].sum()
occurrences_par_habitation = occurrences_par_habitation[occurrences_par_habitation > 1]
plt.figure(figsize=(8, 8))
plt.pie(occurrences_par_habitation, labels=occurrences_par_habitation.index, autopct='%1.1f%%', startangle=200)
plt.title('Répartition des lieux de résidence des cadres (= les salariés dans le tableauEX)')
plt.axis('equal')
plt.show()

# INTERPRETATION
#
#
#

# On souhaite savoir la taille des entreprises où les cadres US travaillent et les comparer avec les seniors, mids et jeunes
#SUBPLOT AVEC DES PIE CHART

cadres_USA = tableauEX[tableauEX['employee_residence'] == 'US'] #tableau avec uniquement les cadres des USA
seniors_USA = tableauSE[tableauSE['employee_residence'] == 'US']   #tableau avec uniquement les seniors des USA
mids_USA = tableauMI[tableauMI['employee_residence'] == 'US']   #tableau avec uniquement les intermediaires des USA
jeunes_USA = tableauEN[tableauEN['employee_residence'] == 'US']   #tableau avec uniquement les débnutants des USA

# Données pour les cadres américains
entreprise_cadres = cadres_USA['company_size'].value_counts()
entreprise_cadres['Autres'] = entreprise_cadres[entreprise_cadres == 1].sum()
entreprise_cadres = entreprise_cadres[entreprise_cadres > 1]
# Données pour les seniors américains
entreprise_seniors = seniors_USA['company_size'].value_counts()
entreprise_seniors['Autres'] = entreprise_seniors[entreprise_seniors == 1].sum()
entreprise_seniors = entreprise_seniors[entreprise_seniors > 1]
# Données pour les mids américains
entreprise_mids = mids_USA['company_size'].value_counts()
entreprise_mids['Autres'] = entreprise_mids[entreprise_mids == 1].sum()
entreprise_mids = entreprise_mids[entreprise_mids > 1]
# Données pour les jeunes américains
entreprise_jeunes = jeunes_USA['company_size'].value_counts()
entreprise_jeunes['Autres'] = entreprise_jeunes[entreprise_jeunes == 1].sum()
entreprise_jeunes = entreprise_jeunes[entreprise_jeunes > 1]

# Création du subplot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
# Pie Chart pour les cadres américains
axes[0, 0].pie(entreprise_cadres, labels=entreprise_cadres.index, autopct='%1.1f%%', startangle=200)
axes[0, 0].set_title('Taille des entreprises où sont les cadres US', color='blue', fontsize=20)
# Pie Chart pour les seniors américains
axes[0, 1].pie(entreprise_seniors, labels=entreprise_seniors.index, autopct='%1.1f%%', startangle=200)
axes[0, 1].set_title('Taille des entreprises où sont les seniors US', color='Goldenrod', fontsize=20)
# Pie Chart pour les mids américains
axes[1, 0].pie(entreprise_mids, labels=entreprise_mids.index, autopct='%1.1f%%', startangle=200)
axes[1, 0].set_title(' Taille des entreprises où sont les mids US', color='red', fontsize=20)
# Pie Chart pour les jeunes américains
axes[1, 1].pie(entreprise_jeunes, labels=entreprise_jeunes.index, autopct='%1.1f%%', startangle=200)
axes[1, 1].set_title('Taille des entreprises où sont les jeunes US', color='green', fontsize=20)

plt.tight_layout()
plt.show()


#INTERPRETATION
#
#
#
#

#Corrélation entre la taille de l'entreprise et la proportion de télétravail des salariés

# Création d'un LabelEncoder
label_encoder2 = LabelEncoder()
# Encodage des valeurs de "company_size" en valeurs numériques
cadres_USA.loc[:, 'company_size_encoded'] = label_encoder2.fit_transform(cadres_USA['company_size'])
# Calcul de la corrélation entre "remote_ratio" et "company_size_encoded"
correlation = cadres_USA['remote_ratio'].corr(cadres_USA['company_size_encoded'])
print("Corrélation entre remote_ratio et company_size_encoded:", correlation)
#0.1521289066099762



## PARTIE 2B : PROJECTION DE DONNÉES ET VISUALISATION

#1 : MATRICE DE CORRÉLATION & HEATMAP
datascience2['work_year'] = datascience2['work_year'].astype(np.int64)
label_encoder3 = LabelEncoder()
datascience2_encoded = datascience2.copy()  # Créer une copie du DataFrame pour ne pas modifier l'original
categorical_cols = ['experience_level', 'employment_type', 'job_title', 'salary_currency', 
                    'employee_residence', 'company_location', 'company_size']
for col in categorical_cols:
    datascience2_encoded[col] = label_encoder3.fit_transform(datascience2[col])

matrice_corrélation = datascience2_encoded.corr()
mask = np.triu(np.ones_like(matrice_corrélation, dtype=bool))
plt.figure(figsize=(12, 8))
sns.heatmap(matrice_corrélation, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Matrice de corrélation entre les variables')
plt.show()

sns.heatmap(datascience2_encoded.corr()[['employee_residence']].sort_values(by='employee_residence', ascending=False), vmin=-1, vmax=1, annot=True,)


#INTERPRETATION: VOIR SLIDE DANS PLOTS
fig, ax = plt.subplots(figsize=(10, 10))
plt.title('Calcul de corrélations pertinentes pour ce dataset', fontsize=30, color='red')
# Ajout des lignes de texte
text_lines = [
"On a donc 3 corrélations intéressantes :", 
"1ere : employee_residence et company_location : 0.95",
"2eme : salary_currency et company_location : 0.76",
"3eme : employee_residence et salary_currency : 0.76",
"On remarque que ce sont les 3 memes colonnes qui reviennent",
"Les nuages de points et regressions linéaires seront axées sur ces 3 variables",
]

# Positionnement du texte
for i, line in enumerate(text_lines):
    ax.text(0.1, 0.9 - i * 0.1, line, ha='left', va='center', color='black', fontsize=16)

ax.axis('off')
plt.show()


# Données de corrélations
correlations = {
    "employee_residence - company_location": 0.95,
    "salary_currency - company_location": 0.76,
    "employee_residence - salary_currency": 0.76
}

#2 : VISUALISATION 2D OU 3D : Création des nuages de points

# Nuage de points pour employee_residence - company_location
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Nuage de points pour employee_residence - company_location
axes[0].scatter(datascience2['employee_residence'], datascience2['company_location'], alpha=0.5)
axes[0].set_xlabel('Employee Residence')
axes[0].set_ylabel('Company Location')
axes[0].set_title('Nuage de points : employee_residence - company_location\nCorrélation : 0.945676')
axes[0].tick_params(axis='x', labelrotation=45, labelsize=8)
# Nuage de points pour salary_currency - company_location
axes[1].scatter(datascience2['salary_currency'], datascience2['company_location'], alpha=0.5)
axes[1].set_xlabel('Salary Currency')
axes[1].set_ylabel('Company Location')
axes[1].set_title('Nuage de points : salary_currency - company_location\nCorrélation : 0.764744')
axes[1].tick_params(axis='x', labelrotation=45, labelsize=8)
# Nuage de points pour employee_residence - salary_currency
axes[2].scatter(datascience2['employee_residence'], datascience2['salary_currency'], alpha=0.5)
axes[2].set_xlabel('Employee Residence')
axes[2].set_ylabel('Salary Currency')
axes[2].set_title('Nuage de points : employee_residence - salary_currency\nCorrélation : 0.755836')
axes[2].tick_params(axis='x', labelrotation=45, labelsize=8)

plt.tight_layout()
plt.show()



# Réduction de la dimensionalité avec PCA
# Charger les données
df = datascience2_encoded # juste pour ecrire plus rapidement
colonnes = ['work_year', 'experience_level', 'employment_type', 'job_title',
            'salary', 'salary_currency', 'salary_in_usd', 'employee_residence',
            'remote_ratio', 'company_location', 'company_size', 'employee_residence_encoded', 'company_location_encoded']


X = df[colonnes]
# Centrage et mise à l'échelle des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Création d'un objet PCA + Application aux données
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
# Visualisation de la variance expliquée
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()
variance_df = pd.DataFrame({'Explained Variance Ratio': explained_variance_ratio,
                            'Cumulative Variance Ratio': cumulative_variance_ratio})
print(variance_df)


# Création du graphique pour illustrer
plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='-')
plt.title('Cumulative Variance Explained')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.grid(True)
plt.show()

# INTERPRETATION : 8 dimensions suffisent à capturer plus de 90% des informations


# Réduction de la dimensionnalité avec t-SNE, surtout pour avoir une représentation graphique
# Extraction des caractéristiques
X = df[colonnes]
Y = df['salary_in_usd']
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, cmap='viridis', alpha=0.5)
plt.title('Visualisation t-SNE des données avec la cible (salaire en USD)')
plt.xlabel('Composante 1')
plt.ylabel('Composante 2')
plt.colorbar(label='Salaire en USD')
plt.show()


# 3 : PROCESSUS DE VALIDATION EXTERNE : RÉGRESSIONS
# On choisit la regréssion car la seule variable intéressante à prédire et le salaire en $
# Sachant que c'est une variable numérique, nous allons déterminer quelle régression est la plus pertinente

# Définition des modèles
models = {
    "Linear Regression": (LinearRegression(), {'normalize': [True, False]}),
    "Ridge Regression": (Ridge(), {'max_iter': [10, 20, 30], 'solver': ['auto', 'svd', 'cholesky']}),
    "XGBoost Regressor": (XGBRegressor(), {}),
    "Random Forest Regressor": (RandomForestRegressor(), {'n_estimators': [5, 10, 20,30,50], 'criterion': ['mse', 'mae'], 'max_depth': [None, 5, 10,15,25], 'random_state': 42}),
    "Gradient Boosting Regressor": (GradientBoostingRegressor(), {'max_depth': [5, 10, 20], 'loss': ['ls', 'lad'], 'n_estimators': [5, 10, 20], 'random_state': 42}),
    "SVR": (SVR(), {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10]}),
    "ElasticNet": (ElasticNet(), {'alpha': [0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]}),
    "Lasso Regression": (Lasso(), {'alpha': [0.1, 1, 10]}),
    "Decision Tree Regressor": (DecisionTreeRegressor(), {'max_depth': [None, 5, 10, 15]})
}

# Sélection du meilleur modèle
best_r2 = -float('inf')
best_model_name = None
best_model = None
for name, (model, params) in models.items():
    pipe = Pipeline([('Model', model)])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.20)
    model = pipe.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)
    mae = mean_absolute_error(Y_test, y_pred)
    print(f'{name} - MSE: {mse:.4f}')
    print(f'{name} - R2: {r2:.4f}')
    print(f'{name} - MAE: {mae:.4f}\n')
    
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_model = model

print('The best model is:', best_model_name)
print('The best model R2 is' , best_r2)


# Utiliser la régression linéaire comme le meilleur modèle, résultat du programme précédent
# Notre variance ciblée : salary_in_usd
best_model = LinearRegression()
best_model.fit(X_train, Y_train)
y_pred = best_model.predict(X_test)
#Calculer les métriques d'évaluation (MSE, R², MAE)
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)
mae = mean_absolute_error(Y_test, y_pred)
# Afficher les résultats
print("Best Model: Linear Regression")
print("MSE:", mse)
print("R2:", r2)
print("MAE:", mae)

# INTERPRETATION
#MSE : 4.73246108596992e-10 , hyper proche de 0, la regression linéaire pour nos données prédit les valeurs de manière très précise
#R2 : 1.0 , le modèle explique parfaitement la variance de la variable cible à l'aide des caractéristiques
#MAE : 1.1574754780704049e-05 , proche de 0 , précision élevée de la reg lin


# Définir les colonnes à utiliser pour la régression linéaire
colonnes = ['work_year', 'experience_level', 'employment_type', 'job_title', 'salary', 'salary_currency', 'employee_residence', 'remote_ratio', 'company_location', 'company_size', 'employee_residence_encoded', 'company_location_encoded']
# Initialiser une figure
plt.figure(figsize=(15, 10))
# Boucle sur chaque colonne
for i, col in enumerate(colonnes):
    # Créer un sous-ensemble de données avec la colonne actuelle et la variable cible
    X = df[[col]]
    Y = df['salary_in_usd']
    # Ajuster une régression linéaire
    model = LinearRegression()
    model.fit(X, Y)
    # Faire des prédictions
    Y_pred = model.predict(X)
    # Tracer la régression linéaire et les points de données
    plt.subplot(3, 4, i+1)
    sns.scatterplot(x=X[col], y=Y, color='blue', alpha=0.5)
    sns.lineplot(x=X[col], y=Y_pred, color='red')
    plt.title(f'salary_in_usd vs {col}')
    plt.xlabel(col)
    plt.ylabel('Salary in USD')

plt.tight_layout()
plt.show()

# ANALYSES & INTERPRÉTATION 
# Les regréssions sur l'ensemble du tableau ne sont pas forcément pertinentes et bien lisibles
# L'utilisation de plusieurs variables explicatives peut améliorer la capacité du modèle à expliquer la variation dans la variable cible (salaire en USD) 
# En prenant en compte plusieurs aspects qui peuvent influencer le salaire, on pourrait faire une reg lin multiple avec des études plus précises
# On remarque néanmoins que certaines régressions ont une pente positive
# work_year, experience_level, job_title, employee_residence_encoded, company_location_encoded





## PARTIE 3 : APPRENTISSAGE ARTIFICIEL
# Nous aimerons faire plusieurs régressiosns plus ciblées
# salaire des cadres aux USA, des datascientist en France+Allemagne+Espagne, salaire des débutants dans le monde
# Code pour : débutants = 0 dans colonne 'experience_level', cadres = 1 dans colonne 'experience_level' , 
# dans la colonne 'company_location' : France = 27, USA = 70, Espagne = 25, Allemagne = 20
# datascientist = 47 dans la colonne 'job_title'

#Création des sous-tableaux
cadresUS_regr = df[(df['experience_level'] == 1) & (df['company_location'] == 70)] # ce nom pour distinguer du cadres_USA 
datascientist_FR = df[(df['company_location'] == 27) & (df['job_title'] == 47)]
datascientist_ES =  df[(df['company_location'] == 25) & (df['job_title'] == 47)]
datascientist_DE =  df[(df['company_location'] == 20 ) & (df['job_title'] == 47)]
datascientist_FRESDE = pd.concat([datascientist_FR, datascientist_ES, datascientist_DE], ignore_index=True)
debutants_monde = df[df['experience_level'] == 0]

# REGRESSIONS LINÉAIRES SIMPLES POUR CHACUN DES 3 TABLEAUX + MSE R2 ET MAE DE CHAQUE GRAPHIQUE


colonnes_explicatives = ['work_year', 'experience_level', 'employment_type', 'job_title', 'salary', 'salary_currency', 'employee_residence', 'remote_ratio', 'company_location', 'company_size', 'employee_residence_encoded', 'company_location_encoded']
variable_cible = 'salary_in_usd'
data1 = cadresUS_regr[colonnes_explicatives + [variable_cible]]

plt.figure(figsize=(20, 14))
for i, col in enumerate(colonnes_explicatives):
    X1 = data1[[col]]
    Y1 = data1[variable_cible]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, random_state=42, test_size=0.20)
    
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions) 
    mae = mean_absolute_error(Y_test, predictions)
    
    plt.subplot(3, 4, i+1)
    sns.scatterplot(x=X_test[col], y=Y_test, color='blue', alpha=0.5)
    sns.lineplot(x=X_test[col], y=predictions, color='red')
    plt.title(f'{variable_cible} vs {col}\nMSE: {mse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}')
    plt.xlabel(col)
    plt.ylabel(variable_cible)

plt.tight_layout()
plt.show()


colonnes_explicatives = ['work_year', 'experience_level', 'employment_type', 'job_title', 'salary', 'salary_currency', 'employee_residence', 'remote_ratio', 'company_location', 'company_size', 'employee_residence_encoded', 'company_location_encoded']
variable_cible = 'salary_in_usd'
data2 = datascientist_FRESDE[colonnes_explicatives + [variable_cible]]

plt.figure(figsize=(20, 14))
for i, col in enumerate(colonnes_explicatives):
    X2 = data2[[col]]
    Y2 = data2[variable_cible]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X2, Y2, random_state=42, test_size=0.20)
    
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)
    mae = mean_absolute_error(Y_test, predictions)
    
    plt.subplot(3, 4, i+1)
    sns.scatterplot(x=X_test[col], y=Y_test, color='blue', alpha=0.5)
    sns.lineplot(x=X_test[col], y=predictions, color='red')
    plt.title(f'{variable_cible} vs {col}\nMSE: {mse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}')
    plt.xlabel(col)
    plt.ylabel(variable_cible)

plt.tight_layout()
plt.show()


colonnes_explicatives = ['work_year', 'experience_level', 'employment_type', 'job_title', 'salary', 'salary_currency', 'employee_residence', 'remote_ratio', 'company_location', 'company_size', 'employee_residence_encoded', 'company_location_encoded']
variable_cible = 'salary_in_usd'
data3 = debutants_monde[colonnes_explicatives + [variable_cible]]


plt.figure(figsize=(20, 14))
for i, col in enumerate(colonnes_explicatives):
    X3 = data3[[col]]
    Y3 = data3[variable_cible]

    X_train, X_test, Y_train, Y_test = train_test_split(X3, Y3, random_state=42, test_size=0.20)
    
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)
    mae = mean_absolute_error(Y_test, predictions)
    
    plt.subplot(3, 4, i+1)
    sns.scatterplot(x=X_test[col], y=Y_test, color='blue', alpha=0.5)
    sns.lineplot(x=X_test[col], y=predictions, color='red')
    plt.title(f'{variable_cible} vs {col}\nMSE: {mse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}')
    plt.xlabel(col)
    plt.ylabel(variable_cible)

plt.tight_layout()
plt.show()



# REGRESSIONS LINÉAIRES MULTIPLES POUR CHACUN DES 3 TABLEAUX + MSE R2 MAE DE CHAQUE TABLEAU


colonnes_explicatives = ['salary', 'employment_type', 'job_title', 'employee_residence', 'remote_ratio', 'experience_level']
variable_cible = 'salary_in_usd'
data1 = cadresUS_regr[colonnes_explicatives + [variable_cible]]

X = data1[colonnes_explicatives]
Y = data1[variable_cible]
model = LinearRegression()
model.fit(X, Y)
predictions = model.predict(X)

mse = mean_squared_error(Y, predictions)
r2 = r2_score(Y, predictions)
mae = mean_absolute_error(Y, predictions)

print("MSE cadres USA :", mse)
print("R2 ccadres USA:", r2)
print("MAE cadres USA:", mae)

plt.figure(figsize=(15, 10))
for i, col in enumerate(colonnes_explicatives):
    plt.subplot(2, 3, i+1)
    sns.scatterplot(x=data1[col], y=data1[variable_cible], color='blue', alpha=0.5)
    sns.lineplot(x=data1[col], y=predictions, color='red')
    plt.title(f'{variable_cible} vs {col}')
    plt.xlabel(col)
    plt.ylabel(variable_cible)

plt.tight_layout()
plt.show()


colonnes_explicatives = ['salary', 'employment_type', 'job_title', 'employee_residence', 'remote_ratio', 'experience_level']
variable_cible = 'salary_in_usd'
data2 = datascientist_FRESDE[colonnes_explicatives + [variable_cible]]

X2 = data2[colonnes_explicatives]
Y2 = data2[variable_cible]
model = LinearRegression()
model.fit(X2, Y2)
predictions = model.predict(X2)

mse = mean_squared_error(Y2, predictions)
r2 = r2_score(Y2, predictions)
mae = mean_absolute_error(Y2, predictions)

print("MSE datascientists FR ES DE :", mse)
print("R2 datascientists FR ES DE:", r2)
print("MAE datascientists FR ES DE:", mae)

plt.figure(figsize=(15, 10))
for i, col in enumerate(colonnes_explicatives):
    plt.subplot(2, 3, i+1)
    sns.scatterplot(x=data2[col], y=data2[variable_cible], color='blue', alpha=0.5)
    sns.lineplot(x=data2[col], y=predictions, color='red')
    plt.title(f'{variable_cible} vs {col}')
    plt.xlabel(col)
    plt.ylabel(variable_cible)

plt.tight_layout()
plt.show()


colonnes_explicatives = ['salary', 'employment_type', 'job_title', 'employee_residence', 'remote_ratio', 'experience_level']
variable_cible = 'salary_in_usd'
data3 = debutants_monde[colonnes_explicatives + [variable_cible]]

X3 = data3[colonnes_explicatives]
Y3 = data3[variable_cible]
model = LinearRegression()
model.fit(X3, Y3)
predictions = model.predict(X3)

mse = mean_squared_error(Y3, predictions)
r2 = r2_score(Y3, predictions)
mae = mean_absolute_error(Y3, predictions)

print("MSE debutants monde :", mse)
print("R2 debutants monde :", r2)
print("MAE debutants monde :", mae)

plt.figure(figsize=(15, 10))
for i, col in enumerate(colonnes_explicatives):
    plt.subplot(2, 3, i+1)
    sns.scatterplot(x=data3[col], y=data3[variable_cible], color='blue', alpha=0.5)
    sns.lineplot(x=data3[col], y=predictions, color='red')
    plt.title(f'{variable_cible} vs {col}')
    plt.xlabel(col)
    plt.ylabel(variable_cible)

plt.tight_layout()
plt.show()

# MSE cadres USA : 2.5929580017988785e-23 R2 : 1.0  MAE : 1.7818671708204308e-12
# Regression linéaire multiple strès bien adaptée pour les cadres USA vu les résultats
# MSE datascientists FR ES DE : 6809091.542730387 R2 : 0.9777121196535893 MAE : 1973.8622315023613
# R2 très proche de 1 pour les datascientists
# MSE debutants monde : 1973645862.3607528 R2  : 0.27412043634929717 MAE : 32468.709555687605
# Valeurs pas top pour les débutants monde



# CLASSIFICATION : ON EFFECTUE LA CLASSIFICATION AVEC LE TABLEAU SANS 63 VAL ABERRANTES

nouveau_tableau = datascience2_sans_top63
nouveau_tableau_encoded = nouveau_tableau.copy()
colonnes_catégoriques = ['work_year', 'experience_level', 'employment_type', 'job_title','salary_currency', 'employee_residence', 'company_location', 'company_size']
label_encoder = LabelEncoder()
for colonne in colonnes_catégoriques:
    nouveau_tableau_encoded[colonne] = label_encoder.fit_transform(nouveau_tableau_encoded[colonne])


# CLASSIFICATION UNIQUEMENT GRACE AUX SALARIÉS DE 2023 QUI SONT SUFFISAMENT NOMBREUX (1747) POUR DÉCRIRE LES SALAIRES DE 2024

nouveau_tableau_2023 = nouveau_tableau_encoded[nouveau_tableau_encoded['work_year'] == 3]
print(nouveau_tableau_2023.shape)



seuil_1 = 95000
seuil_2 = 135000
seuil_3 = 175000
seuil_4 = 250000

# Fonction pour catégoriser les salaires en fonction des seuils
def categoriser_salaire(salaire):
    if salaire < seuil_1:
        return 'Moins de 95k$ annuel'
    elif seuil_1 <= salaire < seuil_2:
        return 'Entre 95k$ et 135k$ annuel'
    elif seuil_2 <= salaire < seuil_3:
        return 'Entre 135k$ et 175k$ annuel'
    elif seuil_3 <= salaire < seuil_4:
        return 'Entre 175k$ et 250k$ annuel'
    else:
        return '+ de 250k$ annuel'

# Appliquer la fonction de catégorisation pour créer une nouvelle colonne 'salary_category'
nouveau_tableau_2023['salary_category'] = nouveau_tableau_2023['salary_in_usd'].apply(categoriser_salaire)
nouveau_tableau_2023_encoded = nouveau_tableau_2023.copy()
colonnes_catégoriques = ['salary_category']
label_encoder = LabelEncoder()
for colonne in colonnes_catégoriques:
    nouveau_tableau_2023_encoded[colonne] = label_encoder.fit_transform(nouveau_tableau_2023_encoded[colonne])


# Afficher les premières lignes du DataFrame pour vérifier la nouvelle colonne
print(nouveau_tableau_2023_encoded.head())


X = nouveau_tableau_2023_encoded.drop('salary_in_usd', axis=1)
y = nouveau_tableau_2023_encoded['salary_category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  VISUALISATION : MATRICE DE CONFUSION

naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)
predictions = naive_bayes_model.predict(X_test)

print(classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))

# Afficher la matrice de confusion
conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Classe prédite')
plt.ylabel('Classe réelle')
plt.title('Matrice de confusion')
plt.show()

#
#
#












