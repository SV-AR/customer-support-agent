# Top Failure Modes -- AWSSupport Intent Classifier

## Failure 1: Cluster_0_(support) -> predicted Cluster_7_(link)

**Example:** @123644 Node LTS Carbon, plz.

**Prediction:** Cluster_7_(link)  
**Expected:** Cluster_0_(support)

**Why it failed:** 'Cluster_0_(support)' and 'Cluster_7_(link)' share overlapping vocabulary (3 such confusions in the eval set), suggesting the model relies on surface keywords rather than deeper intent semantics.

**Possible fix:** Add more contrastive training examples that distinguish 'Cluster_0_(support)' from 'Cluster_7_(link)', or add a rule-based tiebreaker for their most confusable keywords.

---

## Failure 2: Cluster_7_(link) -> predicted Cluster_0_(support)

**Example:** @123644 I can't submit support tickets for technical issues sooo... here you go. API Gateway won't let me configure... APIs...

**Prediction:** Cluster_0_(support)  
**Expected:** Cluster_7_(link)

**Why it failed:** 'Cluster_7_(link)' and 'Cluster_0_(support)' share overlapping vocabulary (3 such confusions in the eval set), suggesting the model relies on surface keywords rather than deeper intent semantics.

**Possible fix:** Add more contrastive training examples that distinguish 'Cluster_7_(link)' from 'Cluster_0_(support)', or add a rule-based tiebreaker for their most confusable keywords.

---

## Failure 3: Password Reset -> predicted Cluster_7_(link)

**Example:** @AWSSupport Hi, Your Win Beenstalk create is broke. Tried to let you know via support ticket but have to pay extra to go that route aprntly.

**Prediction:** Cluster_7_(link)  
**Expected:** Password Reset

**Why it failed:** 'Password Reset' and 'Cluster_7_(link)' share overlapping vocabulary (3 such confusions in the eval set), suggesting the model relies on surface keywords rather than deeper intent semantics.

**Possible fix:** Add more contrastive training examples that distinguish 'Password Reset' from 'Cluster_7_(link)', or add a rule-based tiebreaker for their most confusable keywords.

---

## Failure 4: Cluster_4_(aws) -> predicted Cluster_5_(aws)

**Example:** @123644 it takes too long for resources to be deleted! Created an S3 bucket in the wrong region and now I'm sitting here paying for it :/

**Prediction:** Cluster_5_(aws)  
**Expected:** Cluster_4_(aws)

**Why it failed:** 'Cluster_4_(aws)' and 'Cluster_5_(aws)' share overlapping vocabulary (1 such confusions in the eval set), suggesting the model relies on surface keywords rather than deeper intent semantics.

**Possible fix:** Add more contrastive training examples that distinguish 'Cluster_4_(aws)' from 'Cluster_5_(aws)', or add a rule-based tiebreaker for their most confusable keywords.

---

## Failure 5: Cluster_6_(lambda) -> predicted Cluster_7_(link)

**Example:** The @123644 docs are also misleading; if you use pip3, you need that path. Lots of mucking around :(

**Prediction:** Cluster_7_(link)  
**Expected:** Cluster_6_(lambda)

**Why it failed:** 'Cluster_6_(lambda)' and 'Cluster_7_(link)' share overlapping vocabulary (1 such confusions in the eval set), suggesting the model relies on surface keywords rather than deeper intent semantics.

**Possible fix:** Add more contrastive training examples that distinguish 'Cluster_6_(lambda)' from 'Cluster_7_(link)', or add a rule-based tiebreaker for their most confusable keywords.

---

