### Implementation Note
***
#### TD Lambda forward-view
The implmentation uses forward-view and discount factor of one. Learning rate and lambda are set to 0.1 and 0.9 respectively.

The value function update is defined as follow:
![nstep-return](pictures/lambda_return.png)
Note that G_t is defined as follow :
![G_t](pictures/G_t.png)
***
#### TD Lambda backward-view (eligibility trace)
The implmentation uses backward-view and discount factor of one. Learning rate and lambda are set to 0.1 and 0.9 respectively.

The value function update is defined as follow:
![nstep-return](pictures/lambda_return_bk.png)
Where a lambda dicounte is applied to the eligibility trace:
![lambda-discount](pictures/lambda_discount.png)