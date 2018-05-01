data {
  int<lower=0> n;
  vector[n] X;
  vector[n] y;
}
parameters {
  real theta_0;
  real theta_1;
}
model {
  y ~ normal(theta_0 + theta_1 * X, 1);
}
