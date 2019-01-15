return function f(vegas,pred,bankroll) {
  if (vegas > pred) {
    return -1*bankroll/20;
  }
  else {
    return bankroll/20;
  }
};
