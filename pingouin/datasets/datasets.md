# Datasets

**********************************************************

## bland1995
A dataset containing the repeated measurements of intramural pH and PaCO2 for eight subjects.

#### Reference
Bland & Altman (1995). Used in the R *rmcorr* package.

#### Useful for
Repeated measures correlation (`rm_corr`)

#### Structure
- 3 columns (*Subject*, *pH*, *PaCO2*)
- 47 rows

**********************************************************

## mcclave1991
A dataset providing pain threshold for people with different hair color.

#### Reference
McClave and Dietrich (1991). Used in JASP.

#### Useful for
anova (`anova`) and pairwise Tukey tests (`pairwise_tukey`)

#### Structure
- 3 columns (*Subject*, *Hair color*, *Pain threshold*)
- 19 rows

**********************************************************

## dolan2009
This data set, "Big Five Personality Traits", provides scores of 500 participants on a Big Five personality questionnaire.

#### Reference
Dolan et al. 2009. Used in JASP.

#### Useful for
`corr` and `pairwise_corr` functions

#### Structure
- 6 columns (*Subject, Neuroticism, Extraversion, Openness, Agreeableness, Conscientiousness*)
- 500 rows

**********************************************************

## berens2009
A dataset providing the orientation tuning properties of three neurons recorded from the primary visual cortex of awake macaques. The number of action potentials such neurons fire is modulated by the orientation of a visual stimulus such as an oriented grating.  The main variables are (1) the stimulus orientations (spaced 22.5 degrees apart) and (2) the number of spikes fired in response to each orientation of the stimulus.

#### Reference
Berens (2009). Used in the CircStats Matlab toolbox.

#### Useful for
Circular statistics.

#### Structure
- 3 columns (*Orientation*, *N1_Spikes*, *N2_Spikes*, *N3_Spikes*)
- 8 rows

**********************************************************
