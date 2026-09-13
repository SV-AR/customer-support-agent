# Top Failure Modes -- AWSSupport Intent Classifier

## Failure 1: Cluster_5_(aws) -> predicted Cluster_4_(ec2)

**Example:** @123644 @AWSSupport we're getting various API errors in eu-central (s3 timeouts, ec2 inconsistent responses, web gui hiccups, ie. it shows deleted resources or timeouts). There's nothing on status page, all green.

**Prediction:** Cluster_4_(ec2)  
**Expected:** Cluster_5_(aws)

**Why it failed:** 'Cluster_5_(aws)' and 'Cluster_4_(ec2)' share overlapping vocabulary (2 such confusions in the eval set), suggesting the model relies on surface keywords rather than deeper intent semantics.

**Possible fix:** Add more contrastive training examples that distinguish 'Cluster_5_(aws)' from 'Cluster_4_(ec2)', or add a rule-based tiebreaker for their most confusable keywords.

---

## Failure 2: Cluster_5_(aws) -> predicted Cluster_6_(link)

**Example:** @awssupport please if you have a link on how to use amazon ssl for a ghost blog hosted on aws thank you

**Prediction:** Cluster_6_(link)  
**Expected:** Cluster_5_(aws)

**Why it failed:** 'Cluster_5_(aws)' and 'Cluster_6_(link)' share overlapping vocabulary (2 such confusions in the eval set), suggesting the model relies on surface keywords rather than deeper intent semantics.

**Possible fix:** Add more contrastive training examples that distinguish 'Cluster_5_(aws)' from 'Cluster_6_(link)', or add a rule-based tiebreaker for their most confusable keywords.

---

## Failure 3: Cluster_5_(aws) -> predicted Password Reset

**Example:** @123644 Unable to sign into any root account. Clicking “Sign-in using root account credentials” prompts for account ID followed by asking for `IAM` user name. Nowhere to enter root email. Seeing across multiple browsers.

**Prediction:** Password Reset  
**Expected:** Cluster_5_(aws)

**Why it failed:** 'Cluster_5_(aws)' and 'Password Reset' share overlapping vocabulary (2 such confusions in the eval set), suggesting the model relies on surface keywords rather than deeper intent semantics.

**Possible fix:** Add more contrastive training examples that distinguish 'Cluster_5_(aws)' from 'Password Reset', or add a rule-based tiebreaker for their most confusable keywords.

---

## Failure 4: Cluster_3_(help) -> predicted Cluster_4_(ec2)

**Example:** @AWSSupport EC2 RunInstance with TagSpecifications actually requires CreateTags too. Neither of Policy generator and simulator may not help for it.

**Prediction:** Cluster_4_(ec2)  
**Expected:** Cluster_3_(help)

**Why it failed:** 'Cluster_3_(help)' and 'Cluster_4_(ec2)' share overlapping vocabulary (1 such confusions in the eval set), suggesting the model relies on surface keywords rather than deeper intent semantics.

**Possible fix:** Add more contrastive training examples that distinguish 'Cluster_3_(help)' from 'Cluster_4_(ec2)', or add a rule-based tiebreaker for their most confusable keywords.

---

## Failure 5: Cluster_5_(aws) -> predicted Cluster_3_(help)

**Example:** @AWSSupport - I am trying to assist someone set up the AWS account with email __email__. The address in in South Sudan. We see this error

**Prediction:** Cluster_3_(help)  
**Expected:** Cluster_5_(aws)

**Why it failed:** 'Cluster_5_(aws)' and 'Cluster_3_(help)' share overlapping vocabulary (1 such confusions in the eval set), suggesting the model relies on surface keywords rather than deeper intent semantics.

**Possible fix:** Add more contrastive training examples that distinguish 'Cluster_5_(aws)' from 'Cluster_3_(help)', or add a rule-based tiebreaker for their most confusable keywords.

---

