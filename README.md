# Credit Scoring Business Understanding

How does Basel II influence model design?

1,Basel II requires banks to measure credit risk accurately.
Models must be explainable.
Decisions must be documented.
Regulators should understand why a customer is classified as risky.
Model monitoring and validation are required.

2,Why is a proxy variable needed?

The dataset contains transaction history but no actual loan repayment information.

Therefore:
-No default labels exist.
-Machine learning requires target labels.
-We must create a proxy target.

Example:

-Customers who:
      purchase rarely
      spend very little
      become inactive
may be considered higher risk.

-Business risk:
    The proxy is not actual default behavior.

-Possible errors:
      Good customers labeled risky
      Risky customers labeled safe

3,Logistic Regression vs Gradient Boosting

###Logistic Regression    	###Gradient Boosting
Explainable          	Less Explainable
Regulatory Friendly  	Harder to justify
Faster                  	Slower
Lower accuracy	         Higher accuracy
Easy documentation	     More complex

Conclusion:

For Basel II:Logistic Regression preferred for transparency
Gradient Boosting useful if performance gain is significant