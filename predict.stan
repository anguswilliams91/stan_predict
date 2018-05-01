data {
  int<lower=0> n;
  vector[n] theta_0;
  vector[n] theta_1;
  real X;
}
model {
}
generated quantities {
  vector[n] y_pred;
  for (i in 1:n) y_pred[i] = normal_rng(theta_0[n] + theta_1[n]*X, 1);
}

// The commented out version below is the model I wanted to use, where
// we can predict for multiple datapoints simultaneously. However, pystan's
// fit.extract() method was very slow when I tried this (the stan code itself
// ran very quickly). Another workaround might be to define y as a parameter
// and put it in the model block, but this will mean that we use HMC for the
// sampling, when we should be using simple monte carlo.

// data {
//   int<lower=0> m;
//   int<lower=0> n;
//   vector[m] theta_0;
//   vector[m] theta_1;
//   vector[n] X;
// }
// parameters {
// }
// model {
// }
// generated quantities {
//   vector[n] y_pred[m];
//   for (i in 1:m) {
//     for (j in 1:n) {
//       y_pred[i][j] = normal_rng(theta_0[i] + theta_1[i]*X[j], 1);
//     }
//   }
// }
