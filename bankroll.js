return function f(vegas,pred,bankroll) {
  
  if (vegas > pred) {
    //Bet on away team: so negative.
    var b = 1/(1-vegas)-1;
    var p = 1-pred;
    return -1*bankroll*(p*(b+1)-1)/b/5;
  }
  else {
    var b = 1/(vegas)-1;
    var p = pred;
    return bankroll*(p*(b+1)-1)/b/5;
  }
};
