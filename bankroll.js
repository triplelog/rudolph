return function f(vegas,pred,bankroll) {
  if (vegas > pred) {
    return 0;
  }
  else {
    return bankroll;
  }
};
