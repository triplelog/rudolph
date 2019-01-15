return function f(vegas,pred,bankroll) {
  if (pred > vegas) {
    return 0;
  }
  else {
    return bankroll;
  }
};
