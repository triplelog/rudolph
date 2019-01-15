return function f(awin,aloss,hwin,hloss) {
  var probAA = (awin+10)/(awin+aloss+10);
  var probHH = (hwin+10)/(hwin+hloss+10);
  return (probHH-probAA*probHH)/(probAA+probHH-2*probAA*probHH);
};
