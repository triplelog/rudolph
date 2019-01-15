return function f(awin,aloss,hwin,hloss) {
  var probA = (awin+10)/(awin+aloss+10);
  var probH = (hwin+10)/(hwin+hloss+10);
  return probA;
  return (probH-probA*probH)/(probA+probH-2*probA*probH);
};
