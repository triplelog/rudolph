return function f(awin,aloss,hwin,hloss) {
  var probAA = (awin+10)/(awin+aloss+20);
  var probHH = (hwin+10)/(hwin+hloss+20);
  return (probHH-probAA*probHH)/(probAA+probHH-2*probAA*probHH);
};
